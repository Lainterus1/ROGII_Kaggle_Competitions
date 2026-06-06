"""Beam search stratigraphic alignment via Numba JIT.

Aligns horizontal well GR against typewell GR using a beam search that
minimises squared GR mismatch plus a move-cost penalty. The resulting
typewell-depth path gives a TVT estimate for every evaluation-interval row.

All features are causal: the alignment at row i depends only on GR[0..i].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


BEAM_CONFIGS: list[tuple[int, float, float, int, str]] = [
    (10, 20.0, 144.0, 2, "cons"),
    (10, 8.0, 64.0, 2, "loose"),
    (8, 35.0, 220.0, 1, "vcons"),
    (10, 14.0, 90.0, 5, "sm5"),
    (20, 4.0, 36.0, 3, "vloose"),
    (12, 12.0, 100.0, 3, "mid"),
    (15, 25.0, 180.0, 2, "stiff"),
]

BEAM_OFFSETS = np.array(
    [-40, -20, -10, -5, -3, 0, 3, 5, 10, 20, 40], dtype=np.float32
)

BEAM_FEATURES: list[str] = []


def _register_beam_features() -> None:
    if BEAM_FEATURES:
        return
    tags = [tag for (_, _, _, _, tag) in BEAM_CONFIGS]
    for tag in tags:
        BEAM_FEATURES.append(f"beam_{tag}")
    BEAM_FEATURES.append("beam_consensus")
    BEAM_FEATURES.append("beam_std")
    for o in BEAM_OFFSETS:
        if int(o) == 0:
            continue
        suffix = f"p{int(o)}" if o > 0 else f"m{int(abs(o))}"
        BEAM_FEATURES.append(f"beam_consensus_{suffix}")


@njit(cache=True)
def _beam_jit(
    sgr: np.ndarray,
    tw_gr: np.ndarray,
    si: int,
    bs: int,
    mc: float,
    es: float,
) -> np.ndarray:
    """Numba JIT beam search kernel.

    Args:
        sgr: Smoothed GR of horizontal well evaluation interval (float64).
        tw_gr: Typewell GR (float64).
        si: Starting index in typewell (closest to last_known_tvt).
        bs: Beam width (number of paths kept per step).
        mc: Move cost (penalty for shifting 1 step in typewell).
        es: Emit scale (denominator for squared GR mismatch).

    Returns:
        Array of typewell indices (int64), one per evaluation-interval row.
    """
    n = len(sgr)
    nt = len(tw_gr)
    max_candidates = bs * 5

    bidx = np.zeros(bs, np.int64)
    bidx[0] = si
    bcost = np.full(bs, 1e30)
    bcost[0] = 0.0
    bn = np.int64(1)

    hI = np.zeros((n, bs), np.int64)
    hP = np.zeros((n, bs), np.int64)
    cI = np.zeros(max_candidates, np.int64)
    cC = np.full(max_candidates, 1e30)
    cP = np.zeros(max_candidates, np.int64)

    for step in range(n):
        gv = sgr[step]
        nc = np.int64(0)

        for bi in range(bn):
            idx = bidx[bi]
            cost = bcost[bi]
            for d in range(-2, 3):
                ni = idx + d
                if ni < 0 or ni >= nt:
                    continue
                tot = cost + (gv - tw_gr[ni]) ** 2 / es
                if d >= 0:
                    tot += mc * d
                else:
                    tot += mc * (-d)

                fnd = np.int64(-1)
                for ci in range(nc):
                    if cI[ci] == ni:
                        fnd = ci
                        break
                if fnd >= 0:
                    if tot < cC[fnd]:
                        cC[fnd] = tot
                        cP[fnd] = bi
                else:
                    if nc < max_candidates:
                        cI[nc] = ni
                        cC[nc] = tot
                        cP[nc] = bi
                        nc += 1

        kept = min(bs, nc)
        for i2 in range(kept):
            mi = i2
            for j2 in range(i2 + 1, nc):
                if cC[j2] < cC[mi]:
                    mi = j2
            if mi != i2:
                cI[i2], cI[mi] = cI[mi], cI[i2]
                cC[i2], cC[mi] = cC[mi], cC[i2]
                cP[i2], cP[mi] = cP[mi], cP[i2]

        hI[step, :kept] = cI[:kept]
        hP[step, :kept] = cP[:kept]
        bidx[:kept] = cI[:kept]
        bcost[:kept] = cC[:kept]
        bn = kept

    best = np.int64(0)
    for b_idx in range(1, bn):
        if bcost[b_idx] < bcost[best]:
            best = b_idx

    path = np.zeros(n, np.int64)
    b = best
    for s in range(n - 1, -1, -1):
        path[s] = hI[s, b]
        b = hP[s, b]
    return path


def _nn_idx(sorted_arr: np.ndarray, value: float) -> int:
    """Return index of nearest value in a sorted array."""
    i = int(np.searchsorted(sorted_arr, value, side="left"))
    if i >= len(sorted_arr):
        return len(sorted_arr) - 1
    if i > 0 and abs(sorted_arr[i - 1] - value) <= abs(sorted_arr[i] - value):
        return i - 1
    return i


def _smooth_gr(gr: np.ndarray, radius: int, fb: float) -> np.ndarray:
    """Smooth GR with a rolling mean, filling NaNs with fb."""
    s = pd.Series(gr, dtype="float32")
    s = s.interpolate(limit_direction="both").fillna(fb)
    if radius > 0:
        s = s.rolling(radius * 2 + 1, center=True, min_periods=1).mean()
    return s.to_numpy(dtype=np.float64)


def beam_search(
    gr_h: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    bs: int = 10,
    mc: float = 20.0,
    es: float = 144.0,
    smooth_r: int = 2,
) -> np.ndarray:
    """Run beam search to estimate TVT for each evaluation-interval row.

    Args:
        gr_h: GR of horizontal well (evaluation interval only), float32.
        tw_tvt: TVT column of typewell, sorted ascending.
        tw_gr: GR column of typewell, same order as tw_tvt.
        start_tvt: Last known TVT_input value (anchor point).
        bs: Beam width.
        mc: Move cost.
        es: Emit scale.
        smooth_r: Smoothing radius for GR rolling mean.

    Returns:
        TVT estimate per evaluation-interval row (float32).
    """
    si = _nn_idx(tw_tvt, start_tvt)
    fb = float(np.nanmean(tw_gr))
    sgr = _smooth_gr(gr_h, smooth_r, fb)
    path = _beam_jit(sgr, tw_gr.astype(np.float64), si, bs, float(mc), float(es))
    return tw_tvt[path].astype(np.float32)


def build_beam_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
) -> pd.DataFrame:
    """Build beam-search alignment features for a single well.

    Uses the typewell GR as a reference log and runs 7 diverse beam-search
    configurations. For each config, a TVT estimate is produced per
    evaluation-interval row (post Prediction Start, where TVT_input is NaN).

    Additional aggregate features include consensus (mean of two central
    configs), standard deviation across all configs, and consensus
    differences at 11 offsets.

    Args:
        horizontal: Horizontal well DataFrame with columns GR, TVT_input, MD.
        typewell: Typewell DataFrame with columns TVT, GR (sorted by TVT).

    Returns:
        DataFrame with beam features, indexed like horizontal.
    """
    _register_beam_features()

    kn = horizontal[horizontal["TVT_input"].notna()]
    ev = horizontal[horizontal["TVT_input"].isna()]

    if len(ev) == 0 or len(kn) == 0:
        n = len(horizontal)
        feats = pd.DataFrame(index=horizontal.index)
        for col in BEAM_FEATURES:
            feats[col] = np.nan
        return feats

    last_tvt = float(kn["TVT_input"].iloc[-1])
    tw_tvt = typewell["TVT"].to_numpy(dtype=np.float32)
    tw_gr = typewell["GR"].to_numpy(dtype=np.float32)

    if len(tw_tvt) < 3:
        n = len(horizontal)
        feats = pd.DataFrame(index=horizontal.index)
        for col in BEAM_FEATURES:
            feats[col] = np.nan
        return feats

    gr_full = horizontal["GR"].astype(float)
    gr_full = gr_full.interpolate(method="linear", limit_direction="both")
    gr_full = gr_full.fillna(float(np.nanmean(tw_gr)))
    hgr = gr_full.iloc[ev.index[0] :].to_numpy(dtype=np.float32)

    feats = pd.DataFrame(index=horizontal.index)

    all_paths = {}
    for (bs, mc, es, r, tag) in BEAM_CONFIGS:
        tvt_est = beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
        all_paths[tag] = tvt_est
        col = f"beam_{tag}"
        feats[col] = np.nan
        feats.loc[ev.index, col] = tvt_est

    cons_array = np.column_stack([all_paths["cons"], all_paths["sm5"]])
    consensus = cons_array.mean(axis=1).astype(np.float32)
    feats["beam_consensus"] = np.nan
    feats.loc[ev.index, "beam_consensus"] = consensus

    all_arrays = np.column_stack(list(all_paths.values()))
    beam_std = all_arrays.std(axis=1).astype(np.float32)
    feats["beam_std"] = np.nan
    feats.loc[ev.index, "beam_std"] = beam_std

    md_ev = ev["MD"].to_numpy(dtype=np.float32)
    consensus_series = pd.Series(consensus, index=ev.index)
    for o in BEAM_OFFSETS:
        if int(o) == 0:
            continue
        suffix = f"p{int(o)}" if o > 0 else f"m{int(abs(o))}"
        col = f"beam_consensus_{suffix}"
        shifted = consensus_series.shift(int(o))
        feats[col] = np.nan
        feats.loc[ev.index, col] = consensus - shifted.loc[ev.index].to_numpy(
            dtype=np.float32
        )

    return feats
