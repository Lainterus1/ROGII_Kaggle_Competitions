"""Tests for src/rogii/diagnostics.py — Phase 0 TCN diagnostics."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.diagnostics import (
    compute_error_correlation,
    compute_per_well_rmse,
    compute_prediction_dispersion,
    compute_rmse_by_fold,
    compute_rmse_by_position,
    compute_rmse_by_well_length,
    generate_report,
    load_oof_with_metadata,
)
from rogii.metrics import rmse


def _make_temp_parquet(df: pd.DataFrame, tmp_path: Path) -> Path:
    path = tmp_path / "oof.parquet"
    df.to_parquet(path, index=False)
    return path


def _make_temp_data_dir(tmp_path: Path, wells: list[dict]) -> Path:
    """Create a minimal data/train/ directory with synthetic wells.

    Each well dict: {"well_id": str, "n_pre": int, "n_post": int, "tvt_base": float}
    """
    data_dir = tmp_path / "data"
    train_dir = data_dir / "train"
    train_dir.mkdir(parents=True)

    for w in wells:
        n_total = w["n_pre"] + w["n_post"]
        tvt_input = [w["tvt_base"]] * w["n_pre"] + [np.nan] * w["n_post"]
        tvt = [np.nan] * w["n_pre"] + [w["tvt_base"] + i * 10 for i in range(w["n_post"])]
        df = pd.DataFrame({
            "MD": np.arange(n_total, dtype=float),
            "X": np.zeros(n_total),
            "Y": np.zeros(n_total),
            "Z": np.zeros(n_total),
            "GR": np.ones(n_total) * 50,
            "TVT_input": tvt_input,
            "TVT": tvt,
        })
        df.to_csv(train_dir / f"{w['well_id']}__horizontal_well.csv", index=False)

    return data_dir


# ---------------------------------------------------------------------------
# RMSE by fold
# ---------------------------------------------------------------------------

def test_rmse_by_fold_with_fold_column():
    df = pd.DataFrame({
        "well_id": ["w1", "w1", "w2", "w2"],
        "row_idx": [5, 6, 5, 6],
        "fold": [0, 0, 1, 1],
        "y_true": [10.0, 20.0, 10.0, 20.0],
        "y_pred": [12.0, 18.0, 8.0, 24.0],
        "baseline": [100.0] * 4,
    })
    result = compute_rmse_by_fold(df)
    assert len(result) == 2
    assert result.loc[result["fold"] == 0, "n_rows"].values[0] == 2


def test_rmse_by_fold_without_fold_column():
    df = pd.DataFrame({
        "well_id": ["w1"],
        "row_idx": [5],
        "y_true": [10.0],
        "y_pred": [12.0],
        "baseline": [100.0],
    })
    result = compute_rmse_by_fold(df)
    assert len(result) == 0


def test_rmse_by_fold_all_nan_fold():
    df = pd.DataFrame({
        "well_id": ["w1"],
        "row_idx": [5],
        "fold": [np.nan],
        "y_true": [10.0],
        "y_pred": [12.0],
        "baseline": [100.0],
    })
    result = compute_rmse_by_fold(df)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# RMSE by position
# ---------------------------------------------------------------------------

def test_rmse_by_position():
    n = 100
    df = pd.DataFrame({
        "well_id": ["w1"] * n,
        "row_idx": np.arange(n),
        "fold": [0] * n,
        "y_true": np.linspace(0, 50, n),
        "y_pred": np.linspace(2, 48, n),
        "baseline": [5000.0] * n,
        "frac_after_ps": np.linspace(0, 1, n),
        "rows_since_ps": np.arange(n),
        "well_post_ps_length": [n] * n,
    })
    result = compute_rmse_by_position(df, n_bins=5)
    assert len(result) >= 5  # 5 bins + 4 special entries
    assert "frac_bin" in result.columns


def test_rmse_by_position_no_frac_column():
    df = pd.DataFrame({
        "well_id": ["w1"],
        "row_idx": [5],
        "y_true": [10.0],
        "y_pred": [12.0],
        "baseline": [100.0],
    })
    with pytest.raises(ValueError, match="frac_after_ps"):
        compute_rmse_by_position(df)


# ---------------------------------------------------------------------------
# RMSE by well length
# ---------------------------------------------------------------------------

def test_rmse_by_well_length():
    df = pd.DataFrame({
        "well_id": ["short"] * 30 + ["medium"] * 300 + ["long"] * 3000,
        "row_idx": list(range(30)) + list(range(300)) + list(range(3000)),
        "y_true": np.random.default_rng(42).normal(0, 50, 3330),
        "y_pred": np.random.default_rng(42).normal(0, 50, 3330),
        "baseline": [5000.0] * 3330,
        "well_post_ps_length": [30] * 30 + [300] * 300 + [3000] * 3000,
        "frac_after_ps": np.linspace(0, 1, 3330),
        "rows_since_ps": list(range(30)) + list(range(300)) + list(range(3000)),
    })
    result = compute_rmse_by_well_length(df, n_bins=3)
    assert len(result) == 3
    assert "length_bin" in result.columns
    assert "n_wells" in result.columns


# ---------------------------------------------------------------------------
# Prediction dispersion
# ---------------------------------------------------------------------------

def test_prediction_dispersion_no_flattening():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    df = pd.DataFrame({
        "well_id": ["w1"] * 5,
        "row_idx": range(5),
        "y_true": y,
        "y_pred": y + 1.0,  # slight offset, same dispersion
        "baseline": [100.0] * 5,
    })
    disp = compute_prediction_dispersion(df)
    assert disp["std_ratio"] > 0.8


def test_prediction_dispersion_flattening():
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=float)
    df = pd.DataFrame({
        "well_id": ["w1"] * 5,
        "row_idx": range(5),
        "y_true": y,
        "y_pred": np.full(5, np.mean(y)),  # collapsed to mean
        "baseline": [100.0] * 5,
    })
    disp = compute_prediction_dispersion(df)
    assert disp["std_ratio"] < 0.1


def test_prediction_dispersion_per_well():
    df = pd.DataFrame({
        "well_id": ["w1"] * 20 + ["w2"] * 20,
        "row_idx": list(range(20)) + list(range(20)),
        "y_true": np.random.default_rng(0).normal(0, 50, 40),
        "y_pred": np.random.default_rng(1).normal(0, 50, 40),
        "baseline": [5000.0] * 40,
    })
    disp = compute_prediction_dispersion(df)
    assert "per_well_ratio_median" in disp
    assert "flattening_wells" in disp


# ---------------------------------------------------------------------------
# Error correlation
# ---------------------------------------------------------------------------

def test_error_correlation_different_errors():
    """TCN errs on even rows, LGBM on odd rows — correlation ~0, blend should help."""
    n = 20
    tcn = pd.DataFrame({
        "well_id": ["w1"] * n,
        "row_idx": np.arange(n),
        "y_true": np.zeros(n),
        "y_pred": np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]),
        "baseline": [0.0] * n,
    })
    lgbm = pd.DataFrame({
        "well_id": ["w1"] * n,
        "row_idx": np.arange(n),
        "y_true": np.zeros(n),
        "y_pred": np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]),
        "baseline": [0.0] * n,
    })
    corr = compute_error_correlation(tcn, lgbm)
    assert corr["blend_gain_vs_best"] < 0  # blend helps (anti-correlated errors cancel)


def test_error_correlation_same_errors():
    """Identical errors — correlation 1.0, blend gives no gain."""
    n = 20
    rng = np.random.default_rng(42)
    err = rng.normal(0, 10, n)
    base = pd.DataFrame({
        "well_id": ["w1"] * n,
        "row_idx": np.arange(n),
        "y_true": np.zeros(n),
        "y_pred": err,
        "baseline": [0.0] * n,
    })
    corr = compute_error_correlation(base, base.copy())
    assert corr["global_error_corr"] > 0.99
    assert corr["blend_gain_vs_best"] >= 0  # blend does NOT help


# ---------------------------------------------------------------------------
# Per-well RMSE
# ---------------------------------------------------------------------------

def test_per_well_rmse():
    df = pd.DataFrame({
        "well_id": [f"w{i}" for i in range(15)],
        "row_idx": range(15),
        "y_true": np.linspace(0, 140, 15),
        "y_pred": np.linspace(5, 135, 15),
        "baseline": [5000.0] * 15,
    })
    result = compute_per_well_rmse(df, top_n=5)
    assert len(result) == 10  # 5 worst + 5 best


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def test_generate_report_basic():
    df = pd.DataFrame({
        "well_id": ["w1", "w1", "w2"],
        "row_idx": [10, 11, 10],
        "fold": [0, 0, 1],
        "y_true": [10.0, 20.0, 15.0],
        "y_pred": [12.0, 18.0, 14.0],
        "baseline": [5000.0] * 3,
        "frac_after_ps": [0.0, 1.0, 0.0],
        "rows_since_ps": [0, 1, 0],
        "well_post_ps_length": [2, 2, 1],
    })
    report = generate_report(df)
    assert "Diagnostic Report" in report
    assert "w1" in report


# ---------------------------------------------------------------------------
# load_oof_with_metadata integration test
# ---------------------------------------------------------------------------

def test_load_oof_with_metadata(tmp_path: Path):
    wells = [
        {"well_id": "WELL_A", "n_pre": 5, "n_post": 10, "tvt_base": 6000.0},
        {"well_id": "WELL_B", "n_pre": 3, "n_post": 7, "tvt_base": 7000.0},
    ]
    data_dir = _make_temp_data_dir(tmp_path, wells)

    oof_df = pd.DataFrame({
        "well_id": ["WELL_A", "WELL_A", "WELL_B"],
        "row_idx": [5, 14, 3],
        "fold": [0, 0, 1],
        "y_true": [10.0, 50.0, 5.0],
        "y_pred": [12.0, 48.0, 4.0],
        "baseline": [6000.0, 6000.0, 7000.0],
    })
    oof_path = _make_temp_parquet(oof_df, tmp_path)

    enriched = load_oof_with_metadata(oof_path, data_dir)
    assert "frac_after_ps" in enriched.columns
    assert "rows_since_ps" in enriched.columns
    assert "well_post_ps_length" in enriched.columns

    # WELL_A: row_idx 5 = first post-PS row (n_pre=5) → frac ~0
    row_a0 = enriched.loc[enriched["row_idx"] == 5].iloc[0]
    assert row_a0["frac_after_ps"] == pytest.approx(0.0, abs=0.01)
    assert row_a0["rows_since_ps"] == 0
    assert row_a0["well_post_ps_length"] == 10

    # WELL_A: row_idx 14 = last post-PS row → frac ~1
    row_a1 = enriched.loc[enriched["row_idx"] == 14].iloc[0]
    assert row_a1["frac_after_ps"] == pytest.approx(1.0, abs=0.01)
    assert row_a1["rows_since_ps"] == 9

    # WELL_B: row_idx 3 = first post-PS row (n_pre=3)
    row_b = enriched.loc[enriched["well_id"] == "WELL_B"].iloc[0]
    assert row_b["frac_after_ps"] == pytest.approx(0.0, abs=0.01)
    assert row_b["well_post_ps_length"] == 7
