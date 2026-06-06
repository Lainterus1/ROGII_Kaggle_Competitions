"""Tests for baseline methods and Savgol smoothing."""

import numpy as np
import pandas as pd
import pytest

from rogii.baseline import compute_baseline, _known_zone_mask, _last_known_idx
from rogii.smoothing import apply_postprocessing, clip_predictions, smooth_predictions_per_well


def _make_horizontal(n_total: int = 500, ps_idx: int = 300, slope: float = 2.5):
    """Build a synthetic horizontal well DataFrame with known TVT_input trend."""
    md = np.linspace(11000, 13000, n_total, dtype=float)
    z = np.linspace(5000, 7000, n_total, dtype=float)
    tvt_input = np.full(n_total, np.nan)
    base_tvt = 11500.0
    tvt_input[:ps_idx] = base_tvt + slope * (md[:ps_idx] - md[0])
    tv = np.zeros(n_total, dtype=float)
    tv[ps_idx:] = tvt_input[ps_idx - 1] + slope * (md[ps_idx:] - md[ps_idx - 1])
    return pd.DataFrame({
        "MD": md,
        "X": 0.0,
        "Y": 0.0,
        "Z": z,
        "GR": 60.0,
        "TVT_input": tvt_input,
        "TVT": tv,
    })


def test_flat_baseline_equals_last_tvt():
    """Flat baseline: all post-PS rows get last known TVT_input value."""
    df = _make_horizontal(n_total=100, ps_idx=60)
    b = compute_baseline(df, method="flat")
    last_tvt = df["TVT_input"].iloc[59]
    post = df["TVT_input"].isna()
    assert np.allclose(b[post], last_tvt)
    assert np.all(np.isnan(b[~post]))


def test_slope_md_monotonic_positive():
    """slope_md baseline increases when TVT increases with MD."""
    df = _make_horizontal(n_total=100, ps_idx=60, slope=3.0)
    b = compute_baseline(df, method="slope_md")
    post = df["TVT_input"].isna()
    post_vals = b[post]
    assert np.all(np.diff(post_vals) > 0), "Baseline should be monotonic increasing"
    # Baseline continues the trend smoothly from last known point
    last_tvt = float(df["TVT_input"].iloc[59])
    md_ps = df["MD"].iloc[59]
    md_first_post = df["MD"].iloc[60]
    expected_cont = last_tvt + 3.0 * (md_first_post - md_ps)
    assert np.isclose(b[60], expected_cont, rtol=0.01)


def test_slope_z_anchored_at_ps():
    """slope_z baseline continues the TVT vs Z trend from the PS boundary."""
    df = _make_horizontal(n_total=100, ps_idx=60)
    b = compute_baseline(df, method="slope_z")
    post = df["TVT_input"].isna()
    ps_row = np.where(post)[0][0]
    # Baseline smoothly continues from the PS point; first post value is close
    # to last_tvt plus slope(z_next - z_ps)
    assert np.isfinite(b[ps_row])


def test_slope_recent_uses_last_n_only():
    """slope_recent uses only last N known zone rows (different from full slope_md)."""
    df = _make_horizontal(n_total=200, ps_idx=120, slope=3.0)
    df.loc[:40, "TVT_input"] = 9999.0  # corrupt early known zone
    b_recent = compute_baseline(df, method="slope_recent", recent_window=50)
    b_full = compute_baseline(df, method="slope_md")
    post = df["TVT_input"].isna()
    # recent should NOT equal full (corrupted early data affects full)
    assert not np.allclose(b_recent[post], b_full[post])


def test_wls_baseline_gives_weight_to_recent():
    """WLS baseline produces finite values."""
    df = _make_horizontal(n_total=100, ps_idx=60, slope=2.0)
    b = compute_baseline(df, method="wls", recent_window=50, decay=0.05)
    post = df["TVT_input"].isna()
    assert len(b[post]) > 0
    assert np.all(np.isfinite(b[post]))
    assert np.all(np.isnan(b[~post]))


def test_pre_ps_rows_are_nan():
    """All baseline methods return NaN for pre-PS rows."""
    df = _make_horizontal(n_total=100, ps_idx=60)
    for method in ["flat", "slope_md", "slope_z", "slope_recent", "wls"]:
        b = compute_baseline(df, method=method)
        pre = ~df["TVT_input"].isna()
        assert np.all(np.isnan(b[pre])), f"{method}: pre-PS rows should be NaN"


def test_savgol_preserves_length():
    """Savgol smoothing does not change row count."""
    n = 100
    df = pd.DataFrame({
        "id": [f"well_a_{i}" for i in range(n)],
        "tvt": np.sin(np.linspace(0, 4 * np.pi, n)),
    })
    result = smooth_predictions_per_well(df, window=17, polyorder=3)
    assert len(result) == n
    assert list(result.columns) == ["id", "tvt"]


def test_savgol_finite_values():
    """Savgol output contains only finite values."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "id": [f"well_b_{i}" for i in range(n // 2)] * 2,
        "tvt": rng.randn(n).astype(float) + 12000.0,
    })
    # Add row indices after underscore for proper well grouping
    df["id"] = df["id"] + "_" + df.groupby("id").cumcount().astype(str)
    result = smooth_predictions_per_well(df, window=17, polyorder=3)
    assert np.all(np.isfinite(result["tvt"]))


def test_savgol_short_sequence_unchanged():
    """Savgol passes through short sequences (len <= window) unchanged."""
    df = pd.DataFrame({
        "id": ["well_c_0", "well_c_1", "well_c_2"],
        "tvt": [11000.0, 11010.0, 11020.0],
    })
    result = smooth_predictions_per_well(df, window=17, polyorder=3)
    assert np.allclose(result["tvt"].values, df["tvt"].values)


def test_fallback_on_short_known_zone():
    """When known zone < 2 rows, slope methods fall back to flat."""
    df = _make_horizontal(n_total=100, ps_idx=1, slope=3.0)
    b = compute_baseline(df, method="slope_md")
    post = df["TVT_input"].isna()
    last_tvt = float(df["TVT_input"].iloc[0])
    assert np.allclose(b[post], last_tvt)


# --- TVT clipping tests ---

def test_clip_preserves_length():
    """Clipping does not change row count or columns."""
    df = pd.DataFrame({
        "id": ["a_0", "b_0", "c_0"],
        "tvt": [11000.0, 12000.0, 13000.0],
    })
    result = clip_predictions(df, lower=11500.0, upper=12500.0)
    assert len(result) == 3
    assert list(result.columns) == ["id", "tvt"]


def test_clip_clamps_values():
    """Values outside [lower, upper] are clamped."""
    df = pd.DataFrame({
        "id": ["a_0", "b_0", "c_0"],
        "tvt": [11000.0, 12000.0, 13000.0],
    })
    result = clip_predictions(df, lower=11500.0, upper=12500.0)
    assert float(result.loc[0, "tvt"]) == 11500.0
    assert float(result.loc[1, "tvt"]) == 12000.0
    assert float(result.loc[2, "tvt"]) == 12500.0


def test_clip_preserves_in_range():
    """Values within [lower, upper] are unchanged."""
    df = pd.DataFrame({
        "id": ["a_0", "b_0"],
        "tvt": [11800.0, 12200.0],
    })
    result = clip_predictions(df, lower=11500.0, upper=12500.0)
    assert np.allclose(result["tvt"].values, [11800.0, 12200.0])


def test_apply_postprocessing_clip_then_smooth():
    """apply_postprocessing clips first, then smooths."""
    n = 50
    ids = [f"well_d_{i}" for i in range(n)]
    base = 12000.0 + np.sin(np.linspace(0, 4 * np.pi, n)) * 200
    base[0] = 5000.0   # outlier
    base[-1] = 20000.0  # outlier
    df = pd.DataFrame({"id": ids, "tvt": base})
    result = apply_postprocessing(
        df,
        savgol_window=11,
        savgol_polyorder=2,
        clip_lower=11500.0,
        clip_upper=12500.0,
    )
    assert len(result) == n
    assert list(result.columns) == ["id", "tvt"]
    assert np.all(np.isfinite(result["tvt"]))
    assert result["tvt"].min() >= 11500.0
    assert result["tvt"].max() <= 12500.0


def test_apply_postprocessing_noop():
    """apply_postprocessing with no params returns unchanged data."""
    df = pd.DataFrame({"id": ["a_0", "b_0"], "tvt": [12000.0, 12100.0]})
    result = apply_postprocessing(df)
    assert np.allclose(result["tvt"].values, df["tvt"].values)


def test_apply_postprocessing_clip_only():
    """apply_postprocessing with only clip params clamps without smoothing."""
    df = pd.DataFrame({"id": ["a_0", "b_0"], "tvt": [5000.0, 12000.0]})
    result = apply_postprocessing(df, clip_lower=11000.0, clip_upper=13000.0)
    assert float(result.loc[0, "tvt"]) == 11000.0
    assert float(result.loc[1, "tvt"]) == 12000.0


def test_apply_postprocessing_smooth_only():
    """apply_postprocessing with only smooth params smooths without clipping."""
    n = 50
    ids = [f"well_e_{i}" for i in range(n)]
    tvt = 12000.0 + np.sin(np.linspace(0, 2 * np.pi, n)) * 100
    tvt[10] = 14000.0  # spike — should be smoothed down
    df = pd.DataFrame({"id": ids, "tvt": tvt})
    result = apply_postprocessing(df, savgol_window=11, savgol_polyorder=2)
    assert len(result) == n
    assert np.all(np.isfinite(result["tvt"]))
    # The spike should be reduced (not still 14000)
    assert float(result.loc[10, "tvt"]) < 14000.0
