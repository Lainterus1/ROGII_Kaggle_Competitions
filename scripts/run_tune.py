"""Optuna hyperparameter tuning CLI for LightGBM."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.config import load_yaml_config
from rogii.mlflow_utils import (
    default_experiment_name, end_run, log_metrics,
    log_params, setup_tracking, start_run,
)
from rogii.tuning import TuningConfig, run_tuning

_FEATURE_KEYS = [
    "include_tvt_input", "include_geometry", "include_gr", "include_gr_dwt",
    "include_trajectory", "include_typewell", "include_spatial", "include_dtw",
    "include_geology", "include_beam", "include_formation_plane", "include_z_drift",
]


def _section(config: dict, name: str) -> dict:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return value


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Path to tuning config YAML")
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=None, help="Override timeout (minutes)")
    parser.add_argument("--cv-folds", type=int, default=None, help="Override CV folds for tuning")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--no-pruning", action="store_true", help="Disable MedianPruner")
    parser.add_argument("--output-best", default=None, help="Path to save best params YAML")
    parser.add_argument("--residual-target", action="store_true", help="Use residual target")
    parser.add_argument("--baseline-method", default="flat",
                        choices=["flat", "slope_md", "slope_z", "slope_recent", "wls"])
    parser.add_argument("--include-tvt-input", action="store_true")
    parser.add_argument("--include-geometry", action="store_true")
    parser.add_argument("--include-gr", action="store_true")
    parser.add_argument("--include-gr-dwt", action="store_true")
    parser.add_argument("--include-trajectory", action="store_true")
    parser.add_argument("--include-typewell", action="store_true")
    parser.add_argument("--include-spatial", action="store_true")
    parser.add_argument("--include-dtw", action="store_true")
    parser.add_argument("--include-geology", action="store_true")
    parser.add_argument("--include-beam", action="store_true")
    parser.add_argument("--include-formation-plane", action="store_true")
    parser.add_argument("--include-z-drift", action="store_true")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    config = load_yaml_config(args.config) if args.config else {}
    run_cfg = _section(config, "run")
    feat_cfg = _section(config, "features")
    tune_cfg = _section(config, "tuning")

    # Feature flags
    ff: dict[str, bool | str] = {}
    for k in _FEATURE_KEYS:
        ff[k] = bool(getattr(args, k, False) or feat_cfg.get(k, False))
    ff["residual_target"] = bool(args.residual_target or feat_cfg.get("residual_target", False))
    ff["baseline_method"] = args.baseline_method or feat_cfg.get("baseline_method", "flat")

    residual_target = ff.pop("residual_target")
    baseline_method = ff.pop("baseline_method")
    feature_flags = {k: v for k, v in ff.items() if isinstance(v, bool)}
    feature_flags["residual_target"] = bool(residual_target)
    feature_flags["baseline_method"] = str(baseline_method)

    # Fixed model params (not tuned)
    model_cfg = _section(config, "model")
    fixed_params = dict(model_cfg.get("params", {}))
    if not isinstance(fixed_params, dict):
        fixed_params = {}

    objective_override = fixed_params.pop("objective", None) or "regression"
    fixed_params["objective"] = objective_override
    fixed_params["verbose"] = -1

    # Tuning config
    tuning_config = TuningConfig(
        n_trials=args.n_trials or tune_cfg.get("n_trials", 100),
        timeout_minutes=args.timeout or tune_cfg.get("timeout_minutes", 120),
        cv_folds=args.cv_folds or tune_cfg.get("cv_folds", 3),
        n_seeds=tune_cfg.get("n_seeds", 1),
        early_stopping_rounds=tune_cfg.get("early_stopping_rounds", 50),
        pruning=not args.no_pruning,
        seed=args.seed or tune_cfg.get("seed", 42),
        cache_dir=tune_cfg.get("cache_dir", None),
    )

    # --- MLflow ---
    run_name = run_cfg.get("name", "lgbm-tuning")
    exp_name = run_cfg.get("experiment_name", default_experiment_name())
    print(f"MLflow experiment: {exp_name}")
    print(f"MLflow run: {run_name}")

    setup_tracking()
    mlflow_run = start_run(run_name=run_name, experiment_name=exp_name)
    log_params({
        "tuning_method": "optuna",
        "n_trials": tuning_config.n_trials,
        "timeout_minutes": tuning_config.timeout_minutes,
        "cv_folds": tuning_config.cv_folds,
        "n_seeds": tuning_config.n_seeds,
        "seed": tuning_config.seed,
        "pruning": tuning_config.pruning,
        "early_stopping_rounds": tuning_config.early_stopping_rounds,
        "cache_dir": str(tuning_config.cache_dir) if tuning_config.cache_dir else "none",
        "objective": objective_override,
        "fixed_params": str(fixed_params),
        "feature_flags": str(feature_flags),
        "config_file": args.config or "none",
    }, run=mlflow_run)

    # --- Run tuning ---
    print(f"\n{'='*70}")
    print(f"Optuna LightGBM Tuning")
    print(f"  Trials: {tuning_config.n_trials}")
    print(f"  Timeout: {tuning_config.timeout_minutes} min")
    print(f"  CV folds: {tuning_config.cv_folds}")
    print(f"  Pruning: {'on' if tuning_config.pruning else 'off'}")
    print(f"  Objective: {objective_override}")
    print(f"  Data dir: {args.data_dir}")
    print(f"{'='*70}\n")

    result = run_tuning(
        data_dir=args.data_dir,
        feature_flags=feature_flags,
        fixed_params=fixed_params,
        tuning_config=tuning_config,
    )

    # --- Results ---
    print(f"\n{'='*70}")
    print(f"Tuning Complete")
    print(f"  Trials completed: {result.n_trials_completed}")
    print(f"  Best CV RMSE: {result.best_cv_rmse:.6f}")
    print(f"  Best CV std:  {result.cv_rmse_std:.6f}")
    print(f"\nBest params:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print(f"{'='*70}")

    # Top 5 trials
    top5 = result.trials_df.nsmallest(5, "value")
    print("\nTop 5 trials:")
    for _, row in top5.iterrows():
        params_str = ", ".join(
            f"{k}={v}" for k, v in row.items()
            if k not in ("trial", "value")
        )
        print(f"  #{int(row['trial']):>3}  rmse={row['value']:.4f}  [{params_str}]")

    # --- MLflow logging ---
    log_metrics({
        "best_cv_rmse": result.best_cv_rmse,
        "best_cv_std": result.cv_rmse_std,
        "n_trials_completed": result.n_trials_completed,
    }, run=mlflow_run)
    log_params(result.best_params, run=mlflow_run)
    end_run(mlflow_run)

    # --- Save best params ---
    output_path = args.output_best or tune_cfg.get("output_best_params")
    if output_path:
        import yaml
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump({"model": {"params": result.best_params}}, f, allow_unicode=True)
        print(f"\nBest params saved: {out}")


if __name__ == "__main__":
    main()
