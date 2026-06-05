"""DTW typewell alignment features.

Aligns horizontal pre-PS GR to typewell GR using Dynamic Time Warping
with Sakoe-Chiba band constraint. Extracts anchor-depth and alignment-cost
features. Does NOT use post-PS TVT — only pre-PS GR and typewell data.
"""

import numpy as np
import pandas as pd

from rogii.features import post_ps_mask


def _dtw_path_from_cost(cost: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute optimal DTW path through cost matrix with Sakoe-Chiba band.

    Returns (path_i, path_j) arrays where path_i[k] maps to path_j[k].
    """
    n, m = cost.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0

    for i in range(1, n + 1):
        lo = max(1, i - window)
        hi = min(m, i + window)
        for j in range(lo, hi + 1):
            d = cost[i - 1, j - 1]
            D[i, j] = d + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])

    path_i, path_j = [], []
    i, j = n, m
    while i > 0 or j > 0:
        path_i.append(i - 1)
        path_j.append(j - 1)
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            diag = D[i - 1, j - 1]
            left = D[i, j - 1]
            up = D[i - 1, j]
            if diag <= left and diag <= up:
                i -= 1
                j -= 1
            elif left <= up:
                j -= 1
            else:
                i -= 1

    return np.array(path_i[::-1]), np.array(path_j[::-1])


def build_dtw_features(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    window: int = 50,
) -> pd.DataFrame:
    """Build DTW alignment features for a horizontal well.

    Aligns pre-PS horizontal GR to typewell GR. Maps each horizontal row
    to the nearest typewell TVT via the DTW path, then fills post-PS rows
    by extrapolating the last alignment mapping.

    Features:
    - dtw_optimal_tvt: mapped typewell TVT for each row
    - dtw_cost_cumulative: cumulative DTW cost along alignment path
    """
    n_h = len(horizontal)
    feats = pd.DataFrame(index=horizontal.index)

    ps_mask = post_ps_mask(horizontal)
    pre_gr = horizontal.loc[~ps_mask, "GR"].dropna().values.astype(float)
    tw_gr = typewell["GR"].dropna().values.astype(float)
    tw_tvt = typewell["TVT"].astype(float).values

    ps_idx = int(np.argmax(ps_mask)) if ps_mask.any() else n_h
    h_len = len(pre_gr)
    t_len = len(tw_gr)

    if h_len < 3 or t_len < 3:
        feats["dtw_optimal_tvt"] = 0.0
        feats["dtw_cost_cumulative"] = 0.0
        return feats

    # Build cost matrix
    x = pre_gr.reshape(-1, 1)
    y = tw_gr.reshape(-1, 1)
    from scipy.spatial.distance import cdist
    cost = cdist(x, y, "euclidean")

    # DTW alignment
    path_i, path_j = _dtw_path_from_cost(cost, window)

    # Map horizontal pre-PS index -> typewell TVT via nearest path entry
    mapped_tvt = np.zeros(n_h, dtype=float)
    cumulative_cost = np.zeros(n_h, dtype=float)

    for h_idx in range(ps_idx):
        # Find DTW path entries that map h_idx -> typewell TVT
        matches = np.where(path_i == h_idx)[0]
        if len(matches) > 0:
            avg_tvt = float(np.mean(tw_tvt[path_j[matches]]))
            avg_cost = float(np.mean(cost[h_idx, path_j[matches]]))
        else:
            # Interpolate from nearest mapped indices
            nearest = np.searchsorted(path_i, h_idx)
            if nearest >= len(path_j):
                nearest = len(path_j) - 1
            avg_tvt = float(tw_tvt[path_j[nearest]])
            avg_cost = 0.0
        mapped_tvt[h_idx] = avg_tvt
        cumulative_cost[h_idx] = avg_cost

    # Extrapolate to post-PS rows: use last mapped TVT as constant
    for h_idx in range(ps_idx, n_h):
        mapped_tvt[h_idx] = mapped_tvt[ps_idx - 1] if ps_idx > 0 else 0.0
        cumulative_cost[h_idx] = cumulative_cost[ps_idx - 1] if ps_idx > 0 else 0.0

    # Cumulative sum of cost
    cumulative_cost = np.cumsum(cumulative_cost)

    feats["dtw_optimal_tvt"] = mapped_tvt
    feats["dtw_cost_cumulative"] = cumulative_cost
    return feats
