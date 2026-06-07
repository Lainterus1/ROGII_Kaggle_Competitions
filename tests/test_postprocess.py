"""Tests for 3-strategy blend post-processing: Z-physics, GR matcher, and blend."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rogii.gr_matcher import apply_dtw_matching
from rogii.postprocess import (
    _blend_row,
    apply_postprocess_blend,
    parse_blend_weights,
)
from rogii.z_physics import apply_z_physics


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_horizontal(n_total: int = 100, ps_idx: int = 60) -> pd.DataFrame:
    md = np.linspace(11000, 13000, n_total, dtype=float)
    z = 5000 + (md - md[0]) * 0.05
    gr = 100 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_total))
    gr = gr + np.random.RandomState(42).randn(n_total) * 3

    tvt_input = np.full(n_total, np.nan)
    base_tvt = 11500.0
    slope = 2.5
    tvt_input[:ps_idx] = base_tvt + slope * (md[:ps_idx] - md[0])

    return pd.DataFrame({
        "MD": md,
        "X": 0.0,
        "Y": 0.0,
        "Z": z,
        "GR": gr,
        "TVT_input": tvt_input,
    })


def _make_typewell(n_rows: int = 200) -> pd.DataFrame:
    rng = np.random.RandomState(42)
    tvt = np.linspace(11000, 13000, n_rows)
    gr = 100 + 20 * np.sin(np.linspace(0, 3 * np.pi, n_rows))
    gr = gr + rng.randn(n_rows) * 2
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def _write_test_data(root: Path) -> None:
    (root / "test").mkdir(parents=True)
    (root / "sample_submission.csv").write_text(
        "id,tvt\nwella_2,0.0\nwella_3,0.0\n", encoding="utf-8"
    )
    (root / "test" / "wella__horizontal_well.csv").write_text(
        "MD,X,Y,Z,GR,TVT_input\n"
        "1,0,0,5000,90,11500\n"
        "2,1,0,5003,92,11503\n"
        "3,2,0,5005,91,\n"
        "4,3,0,5008,88,\n",
        encoding="utf-8",
    )
    (root / "test" / "wella__typewell.csv").write_text(
        "TVT,GR\n11000,85\n11500,90\n12000,95\n12500,100\n13000,105\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Z-physics tests
# ---------------------------------------------------------------------------

def test_z_physics_formula() -> None:
    """Z-physics returns Z + offset for post-PS rows."""
    h = _make_horizontal(100, ps_idx=60)
    preds = apply_z_physics(h)

    last_tvt = float(h["TVT_input"].iloc[59])
    z_at_ps = float(h["Z"].iloc[60])
    expected_offset = last_tvt - z_at_ps

    for i in range(60, 100):
        expected = float(h["Z"].iloc[i]) + expected_offset
        assert np.isclose(preds[i], expected), f"Row {i}: {preds[i]} != {expected}"


def test_z_physics_pre_ps_nan() -> None:
    """Pre-PS rows are NaN."""
    h = _make_horizontal(100, ps_idx=60)
    preds = apply_z_physics(h)
    assert np.all(np.isnan(preds[:60]))


def test_z_physics_no_tvt_input() -> None:
    """All NaN when TVT_input column is missing."""
    h = pd.DataFrame({
        "MD": [1, 2, 3],
        "X": [0, 0, 0],
        "Y": [0, 0, 0],
        "Z": [5000, 5005, 5010],
        "GR": [90, 91, 92],
    })
    preds = apply_z_physics(h)
    assert np.all(np.isnan(preds))


def test_z_physics_single_row() -> None:
    """Single post-PS row still gets a prediction."""
    h = pd.DataFrame({
        "MD": [1, 2],
        "X": [0, 0],
        "Y": [0, 0],
        "Z": [5000, 5010],
        "GR": [90, 91],
        "TVT_input": [11500, np.nan],
    })
    preds = apply_z_physics(h)
    assert np.isnan(preds[0])
    assert np.isfinite(preds[1])


# ---------------------------------------------------------------------------
# GR matcher tests
# ---------------------------------------------------------------------------

def test_dtw_matches_identical_gr() -> None:
    """When GR pattern matches exactly, the best TVT is found."""
    rng = np.random.RandomState(42)
    n = 80
    ps_idx = 40
    h = _make_horizontal(n, ps_idx=ps_idx)

    # Build a typewell where a known segment matches
    tw_tvt = np.linspace(10000, 14000, 500)
    tw_gr = np.zeros(500, dtype=float)
    tw_gr[:] = 100.0

    # Insert a matching GR pattern at a known position
    gr_pattern = h["GR"].iloc[ps_idx:ps_idx+20].values
    insert_pos = 200
    tw_gr[insert_pos:insert_pos+20] = gr_pattern

    tw = pd.DataFrame({"TVT": tw_tvt, "GR": tw_gr})

    anchors = np.full(n, np.nan, dtype=float)
    anchors[ps_idx:] = tw_tvt[insert_pos + 10]  # anchor near the match

    preds = apply_dtw_matching(h, tw, anchors, window=5, search_band=500)

    # The matched TVT should be close to the known position
    for i in range(ps_idx, min(ps_idx + 15, n)):
        assert np.isfinite(preds[i])
        # Should be within a few indices of the true match
        assert abs(preds[i] - tw_tvt[insert_pos + 10]) < 200.0


def test_dtw_respects_search_band() -> None:
    """DTW matching does not return TVT outside the search band."""
    h = _make_horizontal(100, ps_idx=60)
    tw = _make_typewell(300)
    anchors = np.full(100, 12000.0, dtype=float)
    anchors[:60] = np.nan

    preds = apply_dtw_matching(h, tw, anchors, window=10, search_band=300)
    for i in range(60, 100):
        assert np.isfinite(preds[i])
        assert 11700 <= preds[i] <= 12300.0


def test_dtw_nan_gr_handling() -> None:
    """NaN in horizontal GR does not cause crash or NaN output."""
    h = _make_horizontal(80, ps_idx=40)
    h.loc[50:55, "GR"] = np.nan
    tw = _make_typewell(200)
    anchors = np.full(80, 12000.0, dtype=float)
    anchors[:40] = np.nan

    preds = apply_dtw_matching(h, tw, anchors)
    for i in range(40, 80):
        assert np.isfinite(preds[i])


def test_dtw_short_typewell() -> None:
    """Typewell with < 3 rows falls back to anchors."""
    h = _make_horizontal(50, ps_idx=20)
    tw = pd.DataFrame({"TVT": [10000, 10100], "GR": [90, 95]})
    anchors = np.full(50, 12000.0, dtype=float)
    anchors[:20] = np.nan

    preds = apply_dtw_matching(h, tw, anchors)
    for i in range(20, 50):
        assert np.isclose(preds[i], 12000.0)


def test_dtw_anchors_length_mismatch() -> None:
    """Anchors array with wrong length raises ValueError."""
    h = _make_horizontal(50, ps_idx=20)
    tw = _make_typewell(100)
    anchors = np.full(30, 12000.0, dtype=float)
    with pytest.raises(ValueError, match="does not match"):
        apply_dtw_matching(h, tw, anchors)


# ---------------------------------------------------------------------------
# Blend tests
# ---------------------------------------------------------------------------

def test_blend_median() -> None:
    """Median blend of 3 values."""
    assert _blend_row(100.0, 110.0, 120.0) == 110.0
    assert _blend_row(10.0, 50.0, 30.0) == 30.0


def test_blend_weighted() -> None:
    """Weighted average with explicit weights."""
    result = _blend_row(100.0, 110.0, 120.0, weights=(0.5, 0.25, 0.25))
    expected = 100.0 * 0.5 + 110.0 * 0.25 + 120.0 * 0.25
    assert np.isclose(result, expected)


def test_blend_nan_value() -> None:
    """Median works when one strategy is NaN."""
    result = _blend_row(100.0, np.nan, 120.0)
    assert result == 110.0


def test_blend_all_nan() -> None:
    """All NaN returns NaN."""
    assert np.isnan(_blend_row(np.nan, np.nan, np.nan))


def test_parse_blend_weights() -> None:
    """Correct parsing of weight strings."""
    assert parse_blend_weights(None) is None
    assert parse_blend_weights("0.5,0.25,0.25") == (0.5, 0.25, 0.25)
    with pytest.raises(ValueError):
        parse_blend_weights("0.5,0.5")


def test_apply_postprocess_blend_smoke(tmp_path: Path) -> None:
    """Full blend pipeline on synthetic data produces valid output."""
    _write_test_data(tmp_path)

    submission = pd.DataFrame({
        "id": ["wella_2", "wella_3"],
        "tvt": [11600.0, 11700.0],
    })

    result = apply_postprocess_blend(submission, tmp_path)
    assert list(result.columns) == ["id", "tvt"]
    assert len(result) == 2
    assert np.isfinite(result["tvt"]).all()


def test_apply_postprocess_blend_missing_typewell(tmp_path: Path) -> None:
    """Blend works when typewell file is missing (falls back to model+Z)."""
    (tmp_path / "test").mkdir(parents=True)
    (tmp_path / "sample_submission.csv").write_text(
        "id,tvt\nwella_2,0.0\n", encoding="utf-8"
    )
    (tmp_path / "test" / "wella__horizontal_well.csv").write_text(
        "MD,X,Y,Z,GR,TVT_input\n"
        "1,0,0,5000,90,11500\n"
        "2,1,0,5003,92,11503\n"
        "3,2,0,5005,91,\n",
        encoding="utf-8",
    )

    submission = pd.DataFrame({"id": ["wella_2"], "tvt": [11600.0]})
    result = apply_postprocess_blend(submission, tmp_path)
    assert np.isfinite(result["tvt"].iloc[0])
