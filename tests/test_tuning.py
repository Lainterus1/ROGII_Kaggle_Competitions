"""Tests for Optuna hyperparameter tuning."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.tuning import TuningConfig, TuningResult, suggest_params, run_tuning


def _write_synthetic_train_data(root: Path, n_wells: int = 10) -> None:
    (root / "train").mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    for wi in range(n_wells):
        n_rows = 20
        well_id = f"well{chr(ord('a') + wi)}"
        md = np.linspace(1000, 1019, n_rows)
        tvt_input = np.array([float(10000 + i) for i in range(10)] + [np.nan] * 10)
        tvt = np.array([float(10000 + i) for i in range(10)] + [float(10010 + i) for i in range(10)])
        gr = np.random.uniform(80, 120, n_rows)
        x = np.linspace(float(wi * 10), float(wi * 10 + 1), n_rows)
        y = np.linspace(float(wi * 5), float(wi * 5 + 1), n_rows)
        z = np.linspace(float(-100 - wi * 5), float(-100 - wi * 5 - 5), n_rows)
        df = pd.DataFrame({"MD": md, "X": x, "Y": y, "Z": z, "GR": gr, "TVT": tvt, "TVT_input": tvt_input})
        df.to_csv(root / "train" / f"{well_id}__horizontal_well.csv", index=False)


# ---------------------------------------------------------------------------
# TuningConfig
# ---------------------------------------------------------------------------

def test_tuning_config_defaults():
    cfg = TuningConfig()
    assert cfg.n_trials == 100
    assert cfg.cv_folds == 3
    assert cfg.pruning is True


def test_tuning_config_custom():
    cfg = TuningConfig(n_trials=10, cv_folds=5, pruning=False)
    assert cfg.n_trials == 10
    assert cfg.cv_folds == 5
    assert cfg.pruning is False


# ---------------------------------------------------------------------------
# suggest_params
# ---------------------------------------------------------------------------

def test_suggest_params_returns_all_keys():
    import optuna
    study = optuna.create_study(direction="minimize")

    def _obj(trial):
        params = suggest_params(trial)
        expected = {
            "learning_rate", "num_leaves", "min_child_samples",
            "subsample", "colsample_bytree", "reg_alpha",
            "reg_lambda", "min_child_weight",
        }
        assert set(params.keys()) == expected
        assert 1e-3 <= params["learning_rate"] <= 0.1
        assert 16 <= params["num_leaves"] <= 256
        assert 5 <= params["min_child_samples"] <= 100
        return 1.0

    study.optimize(_obj, n_trials=1)


def test_suggest_params_in_ranges():
    import optuna
    study = optuna.create_study(direction="minimize")

    for _ in range(20):
        def _obj(trial):
            p = suggest_params(trial)
            assert 0.5 <= p["subsample"] <= 1.0, f"subsample={p['subsample']}"
            assert 0.5 <= p["colsample_bytree"] <= 1.0, f"colsample={p['colsample_bytree']}"
            assert 1e-8 <= p["reg_alpha"] <= 10.0, f"reg_alpha={p['reg_alpha']}"
            return 1.0
        study.optimize(_obj, n_trials=1)


# ---------------------------------------------------------------------------
# run_tuning smoke
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_run_tuning_smoke(tmp_path: Path):
    _write_synthetic_train_data(tmp_path, n_wells=10)

    feature_flags = {"residual_target": True, "baseline_method": "flat"}
    fixed_params = {"objective": "regression", "verbose": -1}
    tuning_config = TuningConfig(
        n_trials=3,
        timeout_minutes=5,
        cv_folds=3,
        n_seeds=1,
        early_stopping_rounds=10,
        pruning=True,
        seed=42,
    )

    result = run_tuning(
        data_dir=str(tmp_path),
        feature_flags=feature_flags,
        fixed_params=fixed_params,
        tuning_config=tuning_config,
    )

    assert isinstance(result, TuningResult)
    assert result.n_trials_completed >= 1
    assert result.best_cv_rmse > 0
    assert len(result.best_params) > 0
    assert len(result.trials_df) >= 1
    assert "value" in result.trials_df.columns


@pytest.mark.slow
def test_run_tuning_without_pruning(tmp_path: Path):
    _write_synthetic_train_data(tmp_path, n_wells=10)

    cfg = TuningConfig(n_trials=3, pruning=False, cv_folds=3, early_stopping_rounds=10)
    result = run_tuning(
        data_dir=str(tmp_path),
        feature_flags={"residual_target": True, "baseline_method": "flat"},
        fixed_params={"objective": "regression", "verbose": -1},
        tuning_config=cfg,
    )

    assert result.best_cv_rmse > 0


def test_tuning_result_property():
    trials_df = pd.DataFrame({"trial": [0, 1], "value": [14.0, 13.5]})
    result = TuningResult(
        best_params={"learning_rate": 0.05},
        best_cv_rmse=13.5,
        cv_rmse_std=0.3,
        study=None,
        trials_df=trials_df,
    )
    assert result.n_trials_completed == 2
    assert result.best_cv_rmse == 13.5
