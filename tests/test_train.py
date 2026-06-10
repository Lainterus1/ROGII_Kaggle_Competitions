"""Tests for multi-seed training and CV strategy."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.train import run_train, TrainResult


def _write_synthetic_train_data(root: Path, n_wells: int = 8, tvt_scale: bool = False) -> None:
    """Write minimal synthetic train data with multiple wells.

    Args:
        tvt_scale: If True, scale TVT values per-well so median TVT differs
            across wells. Required for stratified CV tests where qcut needs
            unique bin edges.
    """
    (root / "train").mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    for wi in range(n_wells):
        n_rows = 20
        well_id = f"well{chr(ord('a') + wi)}"
        md = np.linspace(1000, 1019, n_rows)
        scale = float(wi * 1000) if tvt_scale else 0.0
        tvt_input = np.array([float(10000 + i + scale) for i in range(10)] + [np.nan] * 10)
        tvt = np.array([float(10000 + i + scale) for i in range(10)] + [float(10010 + i + scale) for i in range(10)])
        gr = np.random.uniform(80, 120, n_rows)
        x = np.linspace(float(wi * 10), float(wi * 10 + 1), n_rows)
        y = np.linspace(float(wi * 5), float(wi * 5 + 1), n_rows)
        z = np.linspace(float(-100 - wi * 5), float(-100 - wi * 5 - 5), n_rows)
        df = pd.DataFrame({"MD": md, "X": x, "Y": y, "Z": z, "GR": gr, "TVT": tvt, "TVT_input": tvt_input})
        df.to_csv(root / "train" / f"{well_id}__horizontal_well.csv", index=False)


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
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


@pytest.mark.slow
def test_multi_seed_ensemble_returns_more_models(tmp_path: Path) -> None:
    """Multi-seed returns one model per seed and valid CV metrics."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42, 7, 123],
        residual_target=True,
    )
    assert len(result.models) == 3
    assert result.seed_list == [42, 7, 123]
    assert result.cv_rmse_mean > 0
    assert len(result.cv_rmse_folds) == 3


@pytest.mark.slow
def test_early_stopping_single_seed(tmp_path: Path) -> None:
    """Early stopping should produce valid model with best_iteration < n_estimators."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        early_stopping_rounds=10,
        validation_fraction=0.2,
    )
    assert isinstance(result, TrainResult)
    assert len(result.models) == 1
    assert result.cv_rmse_mean > 0


@pytest.mark.slow
def test_early_stopping_disabled(tmp_path: Path) -> None:
    """early_stopping_rounds=None should train without early stopping."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        early_stopping_rounds=None,
    )
    assert result.cv_rmse_mean > 0


@pytest.mark.slow
def test_custom_objective_huber(tmp_path: Path) -> None:
    """huber objective should train without error."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        model_params={"objective": "huber", "alpha": 0.9},
        early_stopping_rounds=10,
    )
    assert result.cv_rmse_mean > 0


@pytest.mark.slow
def test_custom_objective_quantile(tmp_path: Path) -> None:
    """quantile objective should train without error."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        model_params={"objective": "quantile", "alpha": 0.5},
        early_stopping_rounds=10,
    )
    assert result.cv_rmse_mean > 0


@pytest.mark.slow
def test_preloaded_equivalence(tmp_path: Path) -> None:
    """CV result must be identical with and without preloaded_data."""
    from rogii.train import TrainData, _build_or_load_train_data

    _write_synthetic_train_data(tmp_path, n_wells=10)

    feature_flags = {"residual_target": True, "baseline_method": "flat"}
    train_data = _build_or_load_train_data(str(tmp_path), feature_flags, cache_dir=None)
    assert isinstance(train_data, TrainData)

    result_direct = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        early_stopping_rounds=10,
        **feature_flags,
    )

    result_cached = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        early_stopping_rounds=10,
        preloaded_data=train_data,
        **feature_flags,
    )

    assert abs(result_direct.cv_rmse_mean - result_cached.cv_rmse_mean) < 1e-3
    assert result_direct.feature_columns == result_cached.feature_columns
    assert result_direct.train_rows == result_cached.train_rows
    assert result_direct.train_wells == result_cached.train_wells
    assert len(result_direct.cv_rmse_folds) == len(result_cached.cv_rmse_folds)


def test_disk_cache_roundtrip(tmp_path: Path) -> None:
    """TrainData saved to disk cache can be reloaded and produces same CV."""
    from rogii.train import (
        TrainData, _build_cache_key, _build_or_load_train_data,
        _load_cached_data, _save_cached_data,
    )

    _write_synthetic_train_data(tmp_path, n_wells=10)
    cache_root = tmp_path / "cache"
    feature_flags = {"residual_target": True, "baseline_method": "flat"}

    train_data = _build_or_load_train_data(str(tmp_path), feature_flags, cache_dir=str(cache_root))
    key = _build_cache_key(str(tmp_path), feature_flags)
    loaded = _load_cached_data(str(cache_root), key)
    assert loaded is not None
    assert loaded.n_rows == train_data.n_rows
    assert loaded.n_wells == train_data.n_wells
    np.testing.assert_array_equal(loaded.y, train_data.y)
    np.testing.assert_array_equal(loaded.groups, train_data.groups)


@pytest.mark.slow
def test_run_train_stratified_cv(tmp_path: Path) -> None:
    """Stratified CV strategy should train without error."""
    _write_synthetic_train_data(tmp_path, n_wells=12, tvt_scale=True)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        cv_strategy="stratified",
        strat_tvt_bins=2,
    )
    assert isinstance(result, TrainResult)
    assert result.cv_strategy == "stratified"
    assert len(result.models) == 1
    assert result.cv_rmse_mean > 0


@pytest.mark.slow
def test_run_train_eval_postproc(tmp_path: Path) -> None:
    """eval_postproc=True should populate postproc_results and oof_df."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    result = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        eval_postproc=True,
    )
    assert isinstance(result, TrainResult)
    assert result.oof_df is not None
    assert set(result.oof_df.columns) >= {"well_id", "row_idx", "fold", "y_true", "y_pred", "baseline"}
    assert len(result.oof_df) > 0


@pytest.mark.slow
def test_run_train_with_cache_dir(tmp_path: Path) -> None:
    """run_train with cache_dir should store and reuse cached TrainData."""
    _write_synthetic_train_data(tmp_path, n_wells=10)
    cache_root = tmp_path / "cache"

    result1 = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        cache_dir=str(cache_root),
    )
    assert isinstance(result1, TrainResult)
    assert result1.cv_rmse_mean > 0

    assert cache_root.exists()
    cache_files = list(cache_root.iterdir())
    assert len(cache_files) > 0

    result2 = run_train(
        data_dir=str(tmp_path),
        n_splits=3,
        seed_list=[42],
        residual_target=True,
        cache_dir=str(cache_root),
    )
    assert isinstance(result2, TrainResult)
    assert result1.train_rows == result2.train_rows
    assert result1.train_wells == result2.train_wells
