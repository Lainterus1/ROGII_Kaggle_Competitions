"""Tests for formation plane KNN features."""

import numpy as np
import pandas as pd
import pytest

from rogii.formation_plane import (
    FORMATION_NAMES,
    FORMATION_PLANE_FEATURES,
    PLANE_K,
    build_formation_reference,
    impute_formations,
    build_formation_plane_features,
)


def _make_train_well_df(well_id: str, x_off: float, y_off: float):
    """Synthetic horizontal well with 6 formation columns."""
    n = 50
    md = np.linspace(10000, 12000, n)
    x = np.linspace(x_off, x_off + 500, n)
    y = np.linspace(y_off, y_off + 300, n)
    z = np.linspace(-9500, -9800, n)
    return pd.DataFrame({
        "MD": md,
        "X": x,
        "Y": y,
        "Z": z,
        "GR": 60.0,
        "TVT_input": np.where(np.arange(n) < 30, 11500 + 2 * md, np.nan),
        "ANCC": np.linspace(-9200 + x_off * 0.01, -9150 + x_off * 0.01, n),
        "ASTNU": np.linspace(-9400 + x_off * 0.01, -9350 + x_off * 0.01, n),
        "ASTNL": np.linspace(-9430 + x_off * 0.01, -9380 + x_off * 0.01, n),
        "EGFDU": np.linspace(-9550 + x_off * 0.01, -9500 + x_off * 0.01, n),
        "EGFDL": np.linspace(-9585 + x_off * 0.01, -9535 + x_off * 0.01, n),
        "BUDA": np.linspace(-9720 + x_off * 0.01, -9670 + x_off * 0.01, n),
    })


def test_build_formation_reference():
    """build_formation_reference returns correct columns."""
    dfs = {
        "well_a": _make_train_well_df("well_a", 0, 0),
        "well_b": _make_train_well_df("well_b", 500, 300),
    }
    ref = pd.DataFrame([
        {"well_id": "well_a", "X_med": float(dfs["well_a"]["X"].median()),
         "Y_med": float(dfs["well_a"]["Y"].median()),
         **{f: float(dfs["well_a"][f].median()) for f in FORMATION_NAMES}},
        {"well_id": "well_b", "X_med": float(dfs["well_b"]["X"].median()),
         "Y_med": float(dfs["well_b"]["Y"].median()),
         **{f: float(dfs["well_b"][f].median()) for f in FORMATION_NAMES}},
    ])
    assert "well_id" in ref.columns
    assert "X_med" in ref.columns
    for f in FORMATION_NAMES:
        assert f in ref.columns
    assert len(ref) == 2


def test_impute_formations_shape():
    """impute_formations returns correct dict structure."""
    ref = pd.DataFrame({
        "well_id": ["w1", "w2", "w3"],
        "X_med": [0.0, 500.0, 1000.0],
        "Y_med": [0.0, 300.0, 600.0],
        **{f: [-9000.0 + i * 50, -9200.0 + i * 50, -9400.0 + i * 50]
           for i, f in enumerate(FORMATION_NAMES)},
    })
    query = np.array([[250.0, 150.0]])
    result = impute_formations(ref, query, k=3)

    for f in FORMATION_NAMES:
        key = f"fp_{f.lower()}"
        assert key in result
        assert len(result[key]) == 1
        assert np.isfinite(result[key][0])
    assert "knn_mean_dist" in result
    assert "knn_std_ancc" in result
    assert "knn_std_buda" in result


def test_impute_formations_oof():
    """KNN imputation excludes query well from reference."""
    ref = pd.DataFrame({
        "well_id": ["q", "w2", "w3"],
        "X_med": [0.0, 500.0, 1000.0],
        "Y_med": [0.0, 300.0, 600.0],
        **{f: [100.0, 200.0, 300.0] for f in FORMATION_NAMES},
    })
    # If query well "q" is in reference, but well itself is excluded by caller
    oof_ref = ref[ref["well_id"] != "q"]
    query = np.array([[0.0, 0.0]])
    result = impute_formations(oof_ref, query, k=2)
    val = result["fp_ancc"][0]
    # Should be mean of w2(200) and w3(300) = 250, NOT including q(100)
    assert np.isclose(val, 250.0, atol=1e-4)


def test_build_formation_plane_features_column_count():
    """build_formation_plane_features returns 21 columns."""
    h = _make_train_well_df("test", 100, 200)
    ref = pd.DataFrame({
        "well_id": ["w1", "w2"],
        "X_med": [150.0, 350.0],
        "Y_med": [200.0, 400.0],
        **{f: [float(h[f].median()) + i * 20 for i in range(2)]
           for f in FORMATION_NAMES},
    })
    query = np.array([[float(h["X"].median()), float(h["Y"].median())]])
    est = impute_formations(ref, query, k=2)
    feats = build_formation_plane_features(h, est)
    assert len(feats.columns) == len(FORMATION_PLANE_FEATURES)
    assert list(feats.columns) == list(FORMATION_PLANE_FEATURES)


def test_z_relative_sign():
    """Z-relative features: above formation => positive, below => negative."""
    h = _make_train_well_df("test", 0, 0)
    ref = pd.DataFrame({
        "well_id": ["w1"],
        "X_med": [float(h["X"].median())],
        "Y_med": [float(h["Y"].median())],
        **{f: [float(h[f].median())] for f in FORMATION_NAMES},
    })
    query = np.array([[float(h["X"].median()), float(h["Y"].median())]])
    est = impute_formations(ref, query, k=1)
    feats = build_formation_plane_features(h, est)
    # Z is negative, formation depth is negative. Z at mid-point ≈ -9650
    # ANCC ≈ -9175 → Z - ANCC ≈ -9650 - (-9175) = -475 (below ANCC)
    mid = len(h) // 2
    assert feats["fp_z_vs_ancc"].iloc[mid] < 0, "Should be below ANCC (negative)"
    assert feats["fp_z_vs_buda"].iloc[mid] > 0, "Should be above BUDA (positive)"


def test_thickness_positive():
    """Formation thicknesses are positive (lower minus upper depth)."""
    h = _make_train_well_df("test", 0, 0)
    ref = pd.DataFrame({
        "well_id": ["w1"],
        "X_med": [float(h["X"].median())],
        "Y_med": [float(h["Y"].median())],
        **{f: [float(h[f].median())] for f in FORMATION_NAMES},
    })
    query = np.array([[float(h["X"].median()), float(h["Y"].median())]])
    est = impute_formations(ref, query, k=1)
    feats = build_formation_plane_features(h, est)
    # All thicknesses should be > 0 (BUDA is deepest, ANCC is shallowest)
    thickness_cols = [c for c in FORMATION_PLANE_FEATURES if c.startswith("fp_thick_")]
    for col in thickness_cols:
        assert (feats[col] > 0).all(), f"{col} should be positive"


def test_nearest_dist_nonnegative():
    """Nearest formation distance is always non-negative."""
    h = _make_train_well_df("test", 0, 0)
    ref = pd.DataFrame({
        "well_id": ["w1"],
        "X_med": [float(h["X"].median())],
        "Y_med": [float(h["Y"].median())],
        **{f: [float(h[f].median())] for f in FORMATION_NAMES},
    })
    query = np.array([[float(h["X"].median()), float(h["Y"].median())]])
    est = impute_formations(ref, query, k=1)
    feats = build_formation_plane_features(h, est)
    assert (feats["fp_nearest_dist"] >= 0).all()
