"""Optuna hyperparameter tuning for LightGBM."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TuningConfig:
    n_trials: int = 100
    timeout_minutes: int = 120
    cv_folds: int = 3
    n_seeds: int = 1
    early_stopping_rounds: int = 50
    pruning: bool = True
    seed: int = 42
    cache_dir: str | Path | None = None


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_cv_rmse: float
    cv_rmse_std: float
    study: Any  # optuna.Study
    trials_df: pd.DataFrame

    @property
    def n_trials_completed(self) -> int:
        return len(self.trials_df)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

SEARCH_SPACE_SPEC: dict[str, dict[str, Any]] = {
    "learning_rate":       {"type": "float",  "low": 1e-3, "high": 0.1,  "log": True},
    "num_leaves":          {"type": "int",    "low": 16,   "high": 256,  "step": 16},
    "min_child_samples":   {"type": "int",    "low": 5,    "high": 100,  "step": 5},
    "subsample":           {"type": "float",  "low": 0.5,  "high": 1.0,  "log": False},
    "colsample_bytree":    {"type": "float",  "low": 0.5,  "high": 1.0,  "log": False},
    "reg_alpha":           {"type": "float",  "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda":          {"type": "float",  "low": 1e-8, "high": 10.0, "log": True},
    "min_child_weight":    {"type": "float",  "low": 1e-8, "high": 10.0, "log": True},
}


def suggest_params(trial: Any, spec: dict | None = None) -> dict[str, Any]:
    spec = spec or SEARCH_SPACE_SPEC
    params: dict[str, Any] = {}
    for name, cfg in spec.items():
        kind = cfg["type"]
        if kind == "float":
            params[name] = trial.suggest_float(
                name, cfg["low"], cfg["high"], log=cfg.get("log", False),
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, cfg["low"], cfg["high"], step=cfg.get("step", 1),
            )
    return params


# ---------------------------------------------------------------------------
# Objective (with TrainData caching between trials)
# ---------------------------------------------------------------------------

def _make_objective(
    data_dir: str | Path,
    feature_flags: dict[str, bool],
    fixed_params: dict[str, Any],
    tuning_config: TuningConfig,
):
    import optuna
    from rogii.train import TrainData, run_train

    # Mutable container: first trial populates, all subsequent reuse
    _cache: list[TrainData | None] = [None]

    def _obj(trial: optuna.Trial) -> float:
        nonlocal _cache  # pylint: disable=used-before-assignment  # noqa: F824

        trial_params = suggest_params(trial)
        model_params: dict[str, Any] = {**fixed_params, **trial_params}
        if "n_estimators" not in model_params:
            model_params["n_estimators"] = 3000

        preloaded = _cache[0]
        is_first = preloaded is None

        result = run_train(
            data_dir=data_dir,
            n_splits=tuning_config.cv_folds,
            seed_list=[tuning_config.seed] if tuning_config.n_seeds <= 1
            else [tuning_config.seed + i for i in range(tuning_config.n_seeds)],
            model_params=model_params,
            early_stopping_rounds=tuning_config.early_stopping_rounds,
            progress=False,
            preloaded_data=preloaded,
            cache_dir=tuning_config.cache_dir,
            profile=is_first,  # profile only once
            **feature_flags,
        )

        if is_first:
            td = getattr(result, "_train_data", None)
            if td is not None:
                _cache[0] = td

        trial.set_user_attr("cv_rmse_std", float(result.cv_rmse_std))
        trial.set_user_attr("n_features", len(result.feature_columns))
        return float(result.cv_rmse_mean)

    return _obj


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_tuning(
    data_dir: str | Path,
    feature_flags: dict[str, bool],
    fixed_params: dict[str, Any] | None = None,
    tuning_config: TuningConfig | None = None,
) -> TuningResult:
    import optuna

    cfg = tuning_config or TuningConfig()
    fixed = dict(fixed_params or {})

    # Setup sampler and pruner
    sampler = optuna.samplers.TPESampler(seed=cfg.seed)
    pruner = None
    if cfg.pruning:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10,
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"lgbm_tuning_{cfg.seed}",
    )

    objective_fn = _make_objective(data_dir, feature_flags, fixed, cfg)

    # Progress callback
    def _cb(study: optuna.Study, trial: optuna.Trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            best = study.best_value
            current = trial.value
            n_done = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            print(
                f"  trial {n_done:>3}/{cfg.n_trials}  "
                f"rmse={current:.4f}  best={best:.4f}  "
                f"params: {trial.params}"
            )

    print(f"\n  [Tuning] Trial 1 will profile data loading; trials 2+ reuse cached TrainData\n")

    try:
        study.optimize(
            objective_fn,
            n_trials=cfg.n_trials,
            timeout=cfg.timeout_minutes * 60 if cfg.timeout_minutes else None,
            callbacks=[_cb],
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTuning interrupted. Returning best result so far.")

    best_params = dict(study.best_params)
    best_cv = study.best_value

    # Std from best trial
    best_trial = study.best_trial
    cv_std = float(best_trial.user_attrs.get("cv_rmse_std", 0.0))

    # Build trials dataframe
    trials_data = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = {"trial": t.number, "value": t.value, **t.params}
            trials_data.append(row)
    trials_df = pd.DataFrame(trials_data)

    return TuningResult(
        best_params=best_params,
        best_cv_rmse=float(best_cv),
        cv_rmse_std=cv_std,
        study=study,
        trials_df=trials_df,
    )
