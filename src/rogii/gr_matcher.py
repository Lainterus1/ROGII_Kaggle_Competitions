"""Sliding-window GR matching against typewell for TVT estimation.

For each post-PS row, extracts a GR window and finds the best-matching
position in the typewell via minimum SAD (Sum of Absolute Differences).
Uses numpy sliding_window_view for vectorized computation.

Based on Scott Weeden's _dtw_invert pattern.
"""

import numpy as np
import pandas as pd

from rogii.features import _ps_index


def _safe_gr(series: pd.Series) -> np.ndarray:
    values = series.astype(float).values.copy()
    values[np.isnan(values)] = 0.0
    return values


def apply_dtw_matching(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    anchors: np.ndarray,
    window: int = 25,
    search_band: float = 300.0,
) -> np.ndarray:
    """Match post-PS rows to typewell TVT via sliding GR window SAD.

    For each post-PS row, a GR window of ±window points is matched against
    the typewell GR within search_band around the anchor TVT value.

    Args:
        horizontal: DataFrame with GR and TVT_input columns.
        typewell: DataFrame with TVT and GR columns.
        anchors: Array of anchor TVT values per row (typically model predictions).
        window: Half-window size for GR (default 25 → total 51 points).
        search_band: TVT range around anchor to search (default 300 ft).

    Returns:
        Float array of length len(horizontal). NaN for pre-PS rows,
        matched TVT values for post-PS rows.
    """
    n = len(horizontal)
    result = np.full(n, np.nan, dtype=float)

    ps_idx = _ps_index(horizontal)
    if ps_idx >= n:
        return result

    if len(anchors) != n:
        raise ValueError(
            f"Anchors length {len(anchors)} does not match horizontal length {n}"
        )

    gr = _safe_gr(horizontal["GR"])
    tw_tvt = typewell["TVT"].astype(float).values
    tw_gr = _safe_gr(typewell["GR"])
    m = len(tw_tvt)

    if m < 3:
        # Typewell too short — fallback to anchors
        result[ps_idx:] = anchors[ps_idx:]
        return result

    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except ImportError:
        # numpy < 1.20 fallback — use anchors directly
        result[ps_idx:] = anchors[ps_idx:]
        return result

    for i in range(ps_idx, n):
        anchor = anchors[i]
        if not np.isfinite(anchor):
            result[i] = anchor
            continue

        # Build GR window around row i
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        gr_window = gr[lo:hi]
        win_len = len(gr_window)

        if win_len < 3:
            result[i] = anchor
            continue

        # Determine typewell indices within search band
        tvt_lo = anchor - search_band
        tvt_hi = anchor + search_band
        band_mask = (tw_tvt >= tvt_lo) & (tw_tvt <= tvt_hi)
        band_indices = np.where(band_mask)[0]

        if len(band_indices) < win_len:
            result[i] = anchor
            continue

        # Sliding windows over typewell GR within band
        band_tvt = tw_tvt[band_indices]
        band_gr = tw_gr[band_indices]

        tw_windows = sliding_window_view(band_gr, win_len)
        # tw_windows shape: (len(band_indices) - win_len + 1, win_len)

        sads = np.sum(np.abs(tw_windows - gr_window[np.newaxis, :]), axis=1)

        best_local = int(np.argmin(sads))
        center_offset = win_len // 2
        best_idx_in_band = band_indices[best_local + center_offset]
        result[i] = tw_tvt[best_idx_in_band]

    return result
