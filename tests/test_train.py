"""Tests for multi-seed training and CV strategy."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.train import run_train, TrainResult


def _write_synthetic_train_data(root: Path, n_wells: int = 8) -> None:
    """Write minimal synthetic train data with multiple wells."""
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


def test_run_train_single_seed(tmp_path: Path) -> None:
    """Single seed training returns one model."""
    _write_synthetic_train_data(tmp_path, n_wells=8)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
    )
    assert isinstance(result, TrainResult)
    assert len(result.models) == 1
    assert result.seed_list == [42]
    assert len(result.cv_rmse_folds) == 3
    assert result.cv_rmse_mean > 0


def test_run_train_multi_seed(tmp_path: Path) -> None:
    """Multi-seed training returns one model per seed."""
    _write_synthetic_train_data(tmp_path, n_wells=8)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42, 7],
        residual_target=True,
    )
    assert isinstance(result, TrainResult)
    assert len(result.models) == 2
    assert result.seed_list == [42, 7]
    assert len(result.cv_rmse_folds) == 3


def test_run_train_default_seed(tmp_path: Path) -> None:
    """No seed_list defaults to [42]."""
    _write_synthetic_train_data(tmp_path, n_wells=8)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        residual_target=True,
    )
    assert result.seed_list == [42]
    assert len(result.models) == 1


def test_run_train_stores_feature_columns(tmp_path: Path) -> None:
    """TrainResult includes feature column names."""
    _write_synthetic_train_data(tmp_path, n_wells=8)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        include_geometry=True,
    )
    assert "MD" in result.feature_columns
    assert "md_since_ps" in result.feature_columns
    assert "TVT" not in result.feature_columns


def test_multi_seed_ensemble_better_or_equal(tmp_path: Path) -> None:
    """CV std with multi-seed should be <= single seed std (usually better)."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    result1 = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
    )
    result2 = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42, 7, 123],
        residual_target=True,
    )
    assert len(result1.models) == 1
    assert len(result2.models) == 3
    # Multi-seed should have lower or equal CV std
    assert result2.cv_rmse_std <= result1.cv_rmse_std + 0.01
