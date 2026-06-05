"""Tests for formation geology features v2 — per-row GR z-scores."""

import numpy as np
import pandas as pd

from rogii.features import GEOLOGY_FEATURES, build_features, SAFE_NUMERIC_FEATURES
from rogii.geology_features import build_geology_features, _FORMATION_GR


def _make_horizontal(n_rows: int = 100) -> pd.DataFrame:
    tvt_in = np.linspace(5000, 5000 + n_rows, n_rows)
    tvt_in[n_rows // 2:] = np.nan
    return pd.DataFrame({
        "MD": np.linspace(1000, 2000, n_rows),
        "X": np.linspace(0, 100, n_rows),
        "Y": np.linspace(0, 0, n_rows),
        "Z": np.linspace(-1000, -900, n_rows),
        "GR": np.linspace(80, 120, n_rows),
        "TVT_input": tvt_in,
    })


def _make_typewell() -> pd.DataFrame:
    n = 200
    tvt = np.linspace(4800, 5200, n)
    gr = 100 + 20 * np.sin(np.linspace(0, 4 * np.pi, n))
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def test_geology_v2_constants() -> None:
    assert isinstance(GEOLOGY_FEATURES, list)
    assert len(GEOLOGY_FEATURES) == 9
    assert GEOLOGY_FEATURES[0] == "geol_gr_zscore_egfdl"
    assert GEOLOGY_FEATURES[-1] == "geol_gr_zscore_roll20"


def test_geology_v2_columns() -> None:
    h = _make_horizontal(100)
    t = _make_typewell()
    feats = build_geology_features(h, t)
    assert list(feats.columns) == GEOLOGY_FEATURES
    assert len(feats) == 100


def test_geology_v2_no_nan() -> None:
    h = _make_horizontal(100)
    t = _make_typewell()
    feats = build_geology_features(h, t)
    assert not feats.isna().any().any()


def test_geology_v2_no_tvt_leak() -> None:
    h = _make_horizontal(100)
    t = _make_typewell()
    h["TVT"] = np.random.randn(100) * 100 + 5000
    feats = build_geology_features(h, t)
    assert "TVT" not in feats.columns


def test_geology_v2_per_row_changes() -> None:
    """At least the z-score features vary between rows (not all constant)."""
    h = _make_horizontal(100)
    h["GR"] = np.linspace(50, 150, 100)  # Varying GR
    t = _make_typewell()
    feats = build_geology_features(h, t)
    varying = sum(1 for col in GEOLOGY_FEATURES if feats[col].nunique() > 1)
    assert varying >= 2, f"Expected >=2 varying per-row features, got {varying}"


def test_geology_v2_zscore_egfdl_high_gr() -> None:
    h = _make_horizontal(50)
    h["GR"] = 200.0
    t = _make_typewell()
    feats = build_geology_features(h, t)
    # GR=200, EGFDL mean=93, std=22 → zscore = (200-93)/22 ≈ +4.86
    assert (feats["geol_gr_zscore_egfdl"] > 0).all()


def test_geology_v2_zscore_at_mean() -> None:
    h = _make_horizontal(50)
    egfdl_mean = _FORMATION_GR["EGFDL"][0]
    h["GR"] = egfdl_mean
    t = _make_typewell()
    feats = build_geology_features(h, t)
    assert np.allclose(feats["geol_gr_zscore_egfdl"].values, 0.0, atol=0.1)


def test_geology_v2_build_features_integration() -> None:
    h = _make_horizontal(80)
    t = _make_typewell()
    features = build_features(h, typewell=t, include_geology=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOLOGY_FEATURES
    assert list(features.columns) == expected_cols
    assert "geol_gr_zscore_egfdl" in features.columns
    assert "geol_gr_zscore_roll20" in features.columns
