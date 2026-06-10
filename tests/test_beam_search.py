"""Tests for Numba JIT beam search stratigraphic alignment."""

import numpy as np
import pandas as pd
import pytest

from rogii.beam_search import (
    BEAM_CONFIGS,
    _beam_jit,
    beam_search,
    build_beam_features,
    _nn_idx,
    _smooth_gr,
)

pytestmark = pytest.mark.experimental


def test_beam_jit_basic():
    """Beam search returns path of correct length."""
    sgr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    tw_gr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    si = 2
    bs = 8
    mc = 15.0
    es = 100.0

    path = _beam_jit(sgr, tw_gr, si, bs, mc, es)
    assert len(path) == len(sgr)
    assert path.dtype == np.int64
    assert all(0 <= p < len(tw_gr) for p in path)


def test_beam_jit_monotonic():
    """Beam path stays close to start when GR matches perfectly."""
    n = 30
    sgr = np.ones(n, dtype=np.float64) * 5.0
    tw_gr = np.arange(1, 21, dtype=np.float64)
    si = 5
    bs = 8

    path = _beam_jit(sgr, tw_gr, si, bs, 15.0, 100.0)
    assert all(abs(p - 5) <= 2 for p in path)


def test_beam_jit_deterministic():
    """Same inputs produce same output."""
    rng = np.random.RandomState(42)
    sgr = rng.randn(50).astype(np.float64)
    tw_gr = rng.randn(100).astype(np.float64)
    p1 = _beam_jit(sgr, tw_gr, 25, 10, 20.0, 144.0)
    p2 = _beam_jit(sgr, tw_gr, 25, 10, 20.0, 144.0)
    np.testing.assert_array_equal(p1, p2)


def test_beam_search_returns_finite_tvt():
    """beam_search returns finite TVT estimates."""
    rng = np.random.RandomState(42)
    n_eval = 100
    n_typewell = 200
    gr_h = rng.randn(n_eval).astype(np.float32) + 50.0
    tw_tvt = np.linspace(11000, 13000, n_typewell, dtype=np.float32)
    tw_gr = rng.randn(n_typewell).astype(np.float32) + 50.0
    start_tvt = 11500.0

    tvt = beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs=10, mc=20.0, es=144.0, smooth_r=2)
    assert len(tvt) == n_eval
    assert np.all(np.isfinite(tvt))
    assert tvt.min() >= tw_tvt[0] - 100
    assert tvt.max() <= tw_tvt[-1] + 100


def test_beam_configs_produce_different_signals():
    """Different beam configs produce meaningfully distinct TVT estimates."""
    rng = np.random.RandomState(123)
    n_eval = 80
    n_typewell = 150
    gr_h = rng.randn(n_eval).astype(np.float32) * 15.0 + 60.0
    tw_tvt = np.linspace(11000, 13000, n_typewell, dtype=np.float32)
    tw_gr = rng.randn(n_typewell).astype(np.float32) * 12.0 + 55.0
    start_tvt = 11800.0

    results = {}
    for (bs, mc, es, r, tag) in BEAM_CONFIGS:
        results[tag] = beam_search(gr_h, tw_tvt, tw_gr, start_tvt, bs, mc, es, r)

    arrays = np.column_stack(list(results.values()))
    assert arrays.std(axis=1).mean() > 0, "All beam configs produce identical results"


def test_build_beam_features_shape():
    """build_beam_features returns correct number of columns for eval rows."""
    n_total = 300
    ps_idx = 200
    tvt_input = np.full(n_total, np.nan)
    tvt_input[:ps_idx] = np.linspace(11000, 11500, ps_idx)
    horizontal = pd.DataFrame({
        "MD": np.arange(n_total, dtype=float),
        "GR": np.random.RandomState(7).randn(n_total).astype(float) + 60.0,
        "TVT_input": tvt_input,
        "X": 0.0,
        "Y": 0.0,
        "Z": 0.0,
    })
    typewell = pd.DataFrame({
        "TVT": np.linspace(11000, 13000, 200, dtype=np.float32),
        "GR": np.random.RandomState(7).randn(200).astype(np.float32) * 12.0 + 55.0,
    }).sort_values("TVT")

    feats = build_beam_features(horizontal, typewell)
    assert len(feats) == n_total
    assert len(feats.columns) >= 7 + 2 + 10
    eval_rows = feats.iloc[ps_idx:]
    assert not eval_rows["beam_consensus"].isna().all(), "Consensus should be filled for eval rows"
    assert not eval_rows["beam_std"].isna().all()


def test_beam_causal_construction():
    """Beam search is causal: path at step i depends only on GR up to i."""
    n = 50
    nt = 100
    rng = np.random.RandomState(99)
    sgr = rng.randn(n).astype(np.float64)

    def _beam_on_truncated(sgr_full, tw_gr, si, bs, mc, es, cutoff):
        truncated = sgr_full[:cutoff + 1].copy()
        return _beam_jit(truncated, tw_gr, si, bs, mc, es)[cutoff]

    tw_gr = rng.randn(nt).astype(np.float64)
    full_path = _beam_jit(sgr, tw_gr, 50, 10, 20.0, 144.0)

    for step in range(10, n, 5):
        truncated_val = _beam_on_truncated(sgr, tw_gr, 50, 10, 20.0, 144.0, step)
        assert truncated_val == full_path[step], f"Causal violation at step {step}"


def test_nn_idx():
    """_nn_idx returns nearest index in sorted array."""
    arr = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    assert _nn_idx(arr, 16.0) == 1
    assert _nn_idx(arr, 9.0) == 0
    assert _nn_idx(arr, 45.0) == 3
    assert _nn_idx(arr, 10.0) == 0


def test_smooth_gr():
    """_smooth_gr interpolates NaN and returns float64."""
    gr = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float32)
    fb = 0.0
    result = _smooth_gr(gr, radius=1, fb=fb)
    assert len(result) == 5
    assert not np.any(np.isnan(result))
    assert result.dtype == np.float64
