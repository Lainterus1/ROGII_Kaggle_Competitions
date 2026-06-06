"""Formation plane KNN features — impute formation depths for wells missing
formation columns (test wells) via KNN on (X, Y) from train wells.

Pattern: same as spatial_features.py — fold-aware OOF, separate module,
not in build_features(). Features are well-level constants broadcast to
all rows of a well; Z-relative transforms make them per-row.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rogii.data_loading import read_horizontal_well

FORMATION_NAMES = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
PLANE_K = 10

FORMATION_PLANE_FEATURES = (
    [f"fp_{f.lower()}" for f in FORMATION_NAMES]                             # 6 raw
    + [f"fp_z_vs_{f.lower()}" for f in FORMATION_NAMES]                       # 6 Z-relative
    + ["fp_nearest_dist"]                                                      # 1 nearest-dist
    + ["fp_thick_ancc_astnu", "fp_thick_astnu_astnl",                         # 5 thickness
       "fp_thick_astnl_egfdu", "fp_thick_egfdu_egfdl", "fp_thick_egfdl_buda"]
    + ["fp_knn_mean_dist", "fp_knn_std_ancc", "fp_knn_std_buda"]             # 3 uncertainty
)


def _well_median_xy_formations(h: pd.DataFrame) -> dict | None:
    """Extract median X, Y and formation depths from a horizontal well."""
    available = [f for f in FORMATION_NAMES if f in h.columns and not h[f].isna().all()]
    if len(available) < 6:
        return None
    result = {"X_med": float(h["X"].median()), "Y_med": float(h["Y"].median())}
    for f in FORMATION_NAMES:
        result[f] = float(h[f].dropna().median())
    return result


def build_formation_reference(
    data_dir, split: str, well_ids: list[str],
) -> pd.DataFrame:
    """Build one reference row per well with median X, Y and formation depths.

    Only wells with all 6 formation columns present are included.
    """
    rows = []
    for wid in well_ids:
        h = read_horizontal_well(data_dir, split, wid)
        info = _well_median_xy_formations(h)
        if info is None:
            continue
        info["well_id"] = wid
        rows.append(info)
    if not rows:
        raise ValueError("No wells with complete formation data in reference")
    return pd.DataFrame(rows)


def impute_formations(
    reference: pd.DataFrame,
    query_xy: np.ndarray,
    k: int = PLANE_K,
) -> dict:
    """Impute formation depths for query (X,Y) points via KNN.

    Args:
        reference: DataFrame with X_med, Y_med, and 6 formation columns.
        query_xy: (n_points, 2) array of (X, Y) coordinates.
        k: Number of nearest neighbors.

    Returns dict with keys:
        - 6 formation depth arrays (float32, shape n_points)
        - 'knn_mean_dist': mean distance to k neighbors
        - 'knn_std_ancc', 'knn_std_buda': std of formation depths among neighbors
    """
    ref_xy = reference[["X_med", "Y_med"]].to_numpy(dtype=float)
    k_eff = min(k, len(ref_xy))

    tree = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree")
    tree.fit(ref_xy)
    dists, indices = tree.kneighbors(query_xy.astype(float))

    result: dict[str, np.ndarray] = {}
    for f in FORMATION_NAMES:
        ref_vals = reference[f].to_numpy(dtype=float)
        result[f"fp_{f.lower()}"] = ref_vals[indices].mean(axis=1).astype(np.float32)

    result["knn_mean_dist"] = dists.mean(axis=1).astype(np.float32)
    result["knn_std_ancc"] = reference["ANCC"].to_numpy(dtype=float)[indices].std(axis=1).astype(np.float32)
    result["knn_std_buda"] = reference["BUDA"].to_numpy(dtype=float)[indices].std(axis=1).astype(np.float32)
    return result


def build_formation_plane_features(
    horizontal: pd.DataFrame,
    formation_estimates: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Build 21 formation plane features from imputed formation depths.

    Args:
        horizontal: DataFrame with Z column for the well.
        formation_estimates: Output of impute_formations for this well
                             (arrays of length 1, well-level).

    Returns:
        DataFrame with FORMATION_PLANE_FEATURES columns, len = len(horizontal).
    """
    n = len(horizontal)
    feats = pd.DataFrame(index=horizontal.index)
    z = horizontal["Z"].to_numpy(dtype=float)

    # Broadcast well-level constants to all rows
    for f in FORMATION_NAMES:
        key = f"fp_{f.lower()}"
        feats[key] = float(formation_estimates[key][0])

    # Z-relative (6)
    for f in FORMATION_NAMES:
        key = f"fp_z_vs_{f.lower()}"
        raw_key = f"fp_{f.lower()}"
        feats[key] = z - feats[raw_key].to_numpy(dtype=float)

    # Distance to nearest formation (1)
    dists = np.full(n, np.inf, dtype=float)
    for f in FORMATION_NAMES:
        raw_key = f"fp_{f.lower()}"
        d = np.abs(z - feats[raw_key].to_numpy(dtype=float))
        dists = np.minimum(dists, d)
    feats["fp_nearest_dist"] = dists.astype(float)

    # Formation thicknesses (5)
    pairs = [
        ("ANCC", "ASTNU"), ("ASTNU", "ASTNL"),
        ("ASTNL", "EGFDU"), ("EGFDU", "EGFDL"), ("EGFDL", "BUDA"),
    ]
    for upper, lower in pairs:
        col = f"fp_thick_{upper.lower()}_{lower.lower()}"
        fp_lower = f"fp_{lower.lower()}"
        fp_upper = f"fp_{upper.lower()}"
        feats[col] = float(formation_estimates[fp_upper][0] - formation_estimates[fp_lower][0])

    # KNN uncertainty (3)
    feats["fp_knn_mean_dist"] = float(formation_estimates["knn_mean_dist"][0])
    feats["fp_knn_std_ancc"] = float(formation_estimates["knn_std_ancc"][0])
    feats["fp_knn_std_buda"] = float(formation_estimates["knn_std_buda"][0])

    return feats
