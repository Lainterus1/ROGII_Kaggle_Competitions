"""Tests for DTW typewell alignment features."""

import numpy as np
import pandas as pd
import pytest

from rogii.features import DTW_FEATURES, build_features, SAFE_NUMERIC_FEATURES
from rogii.typewell_alignment import build_dtw_features, _dtw_path_from_cost

pytestmark = pytest.mark.experimental


def _make_horizontal(n_rows: int = 100, ps_at: int | None = None) -> pd.DataFrame:
    if ps_at is None:
        ps_at = n_rows // 2
    gr = 100 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_rows)) + np.random.randn(n_rows) * 5
    tvt_in = np.linspace(5000, 5000 + n_rows, n_rows)
    tvt_in[ps_at:] = np.nan
    return pd.DataFrame({
        "MD": np.linspace(1000, 2000, n_rows),
        "X": np.linspace(0, 100, n_rows),
        "Y": np.linspace(0, 0, n_rows),
        "Z": np.linspace(-1000, -900, n_rows),
        "GR": gr,
        "TVT_input": tvt_in,
    })


def _make_typewell(n_rows: int = 150) -> pd.DataFrame:
    gr = 100 + 20 * np.sin(np.linspace(0, 3 * np.pi, n_rows)) + np.random.randn(n_rows) * 3
    return pd.DataFrame({
        "TVT": np.linspace(4800, 5200, n_rows),
        "GR": gr,
    })


def test_dtw_features_constants() -> None:
    assert isinstance(DTW_FEATURES, list)
    assert len(DTW_FEATURES) == 2
    assert DTW_FEATURES[0] == "dtw_optimal_tvt"
    assert DTW_FEATURES[1] == "dtw_cost_cumulative"


def test_dtw_features_columns() -> None:
    h = _make_horizontal(100)
    t = _make_typewell(150)
    feats = build_dtw_features(h, t)
    assert list(feats.columns) == DTW_FEATURES
    assert len(feats) == len(h)


def test_dtw_features_no_nan() -> None:
    h = _make_horizontal(100)
    t = _make_typewell(150)
    feats = build_dtw_features(h, t)
    assert not feats.isna().any().any()


def test_dtw_no_tvt_leak() -> None:
    h = _make_horizontal(100)
    t = _make_typewell(150)
    h["TVT"] = np.random.randn(100) * 100 + 5000
    feats = build_dtw_features(h, t)
    assert "TVT" not in feats.columns


def test_dtw_single_row() -> None:
    h = _make_horizontal(1)
    t = _make_typewell(50)
    feats = build_dtw_features(h, t)
    assert len(feats) == 1
    assert feats.loc[0, "dtw_optimal_tvt"] == 0.0


def test_dtw_cost_cumulative_increasing() -> None:
    np.random.seed(42)
    h = _make_horizontal(80, ps_at=40)
    t = _make_typewell(100)
    feats = build_dtw_features(h, t)
    cost = feats["dtw_cost_cumulative"].values
    diffs = np.diff(cost)
    assert (diffs >= -1e-10).all(), "Cumulative cost must be non-decreasing"


def test_dtw_optimal_tvt_post_ps_constant() -> None:
    np.random.seed(42)
    n = 80
    ps_at = 40
    h = _make_horizontal(n, ps_at=ps_at)
    t = _make_typewell(100)
    feats = build_dtw_features(h, t)
    post_vals = feats.loc[ps_at:, "dtw_optimal_tvt"].values
    assert np.allclose(post_vals, post_vals[0]), "Post-PS mapped TVT should be constant"


def test_dtw_uses_only_pre_ps_gr() -> None:
    np.random.seed(42)
    n = 80
    ps_at = 40
    h = _make_horizontal(n, ps_at=ps_at)
    orig_post_gr = h.loc[ps_at:, "GR"].copy()
    t = _make_typewell(100)
    feats1 = build_dtw_features(h, t)
    h.loc[ps_at:, "GR"] = 9999.0
    feats2 = build_dtw_features(h, t)
    assert np.allclose(feats1["dtw_optimal_tvt"].values, feats2["dtw_optimal_tvt"].values)


def test_build_features_with_dtw() -> None:
    h = _make_horizontal(80)
    t = _make_typewell(100)
    features = build_features(h, typewell=t, include_dtw=True)
    expected_cols = SAFE_NUMERIC_FEATURES + DTW_FEATURES
    assert list(features.columns) == expected_cols
    assert "dtw_optimal_tvt" in features.columns
    assert "dtw_cost_cumulative" in features.columns
