"""Validation split contracts and stratification helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def validate_no_group_overlap(train_groups: set[object], valid_groups: set[object]) -> None:
    """Raise if a group appears in both train and validation partitions."""
    overlap = train_groups & valid_groups
    if overlap:
        raise ValueError(f"Group leakage detected: {len(overlap)} overlapping groups")


def build_stratification_labels(
    data_dir: str | Path,
    well_ids: list[str],
    n_tvt_bins: int = 4,
    n_spatial_clusters: int = 5,
) -> tuple[np.ndarray, dict]:
    """Build per-well stratification labels for StratifiedGroupKFold.

    Computes three dimensions per well:
      1. signed_azimuth_bin (0/1): updip vs downdip direction.
      2. median_tvt_quintile (0..n_tvt_bins-1): binned median post-PS TVT.
      3. spatial_cluster (0..n_spatial_clusters-1): KMeans on median X/Y.

    Dimensions are combined into a single integer label: az_bin * (n_tvt_bins * n_spatial_clusters) + tvt_bin * n_spatial_clusters + spatial_bin.
    Classes with fewer than `min_class_size` wells are merged into nearest
    neighbor class to satisfy StratifiedGroupKFold's per-class minimum.

    Returns:
        strat_labels: (n_wells,) integer array, one label per well.
        strat_meta: metadata dict with bin edges and cluster info for reproducibility.

    Note:
        Uses ONLY pre-PS data (X, Y, azimuth, TVT_input for PS detection).
        median_tvt is computed from train target TVT — this is standard for
        StratifiedKFold and does NOT leak into features.
    """
    from rogii.data_loading import read_horizontal_well

    n_wells = len(well_ids)
    azimuths = np.empty(n_wells, dtype=float)
    median_tvts = np.empty(n_wells, dtype=float)
    median_xs = np.empty(n_wells, dtype=float)
    median_ys = np.empty(n_wells, dtype=float)

    for i, wid in enumerate(well_ids):
        h = read_horizontal_well(data_dir, "train", wid)

        ps_idx = int(h["TVT_input"].isna().idxmax()) if "TVT_input" in h.columns else len(h) - 1

        x = h["X"].astype(float).values
        y = h["Y"].astype(float).values

        if ps_idx > 0 and ps_idx < len(h):
            dx = x[ps_idx] - x[ps_idx - 1]
            dy = y[ps_idx] - y[ps_idx - 1]
        elif ps_idx > 0:
            dx = x[-1] - x[0]
            dy = y[-1] - y[0]
        else:
            dx = x[1] - x[0] if len(x) > 1 else 0.0
            dy = y[1] - y[0] if len(y) > 1 else 0.0
        azimuths[i] = float(np.arctan2(dy, dx))

        post_mask = h["TVT_input"].isna().to_numpy(dtype=bool)
        tvt_post = h.loc[post_mask, "TVT"].astype(float).dropna()
        median_tvts[i] = float(tvt_post.median()) if len(tvt_post) > 0 else np.nan

        median_xs[i] = float(h["X"].median())
        median_ys[i] = float(h["Y"].median())

    az_bins = (azimuths >= 0).astype(int)

    tvt_labels = pd.qcut(median_tvts, q=n_tvt_bins, labels=False)
    if hasattr(tvt_labels, "to_numpy"):
        tvt_bins = tvt_labels.to_numpy(dtype=int)
    else:
        tvt_bins = np.array(tvt_labels, dtype=int)

    coords = np.column_stack([median_xs, median_ys])
    km = KMeans(n_clusters=n_spatial_clusters, random_state=42, n_init=10)
    spatial_bins = km.fit_predict(coords)

    combined_label = az_bins * (n_tvt_bins * n_spatial_clusters) + tvt_bins * n_spatial_clusters + spatial_bins

    combined_label = _merge_rare_classes(combined_label, min_class_size=5)

    tvt_edges: list[float] | None = None
    try:
        tvt_edges = [float(e) for e in pd.qcut(median_tvts, q=n_tvt_bins, retbins=True)[1]]
    except (ValueError, TypeError):
        tvt_edges = None

    meta: dict = {
        "n_tvt_bins": n_tvt_bins,
        "n_spatial_clusters": n_spatial_clusters,
        "tvt_edges": tvt_edges,
        "cluster_centers": km.cluster_centers_.tolist(),
        "unique_labels": [int(l) for l in np.unique(combined_label)],
        "label_counts": {int(l): int(c) for l, c in zip(*np.unique(combined_label, return_counts=True))},
    }

    return combined_label, meta


def _merge_rare_classes(labels: np.ndarray, min_class_size: int = 5) -> np.ndarray:
    """Merge classes with fewer than min_class_size samples into numerically nearest class."""
    unique, counts = np.unique(labels, return_counts=True)
    rare = unique[counts < min_class_size]
    if len(rare) == 0:
        return labels

    valid = unique[counts >= min_class_size]
    if len(valid) == 0:
        return labels

    result = labels.copy()
    for r in rare:
        mask = result == r
        nearest = int(valid[np.argmin(np.abs(valid - r))])
        result[mask] = nearest
    return result


def create_cv_splitter(
    cv_strategy: str = "group",
    n_splits: int = 5,
) -> GroupKFold | StratifiedGroupKFold:
    """Return a CV splitter for the given strategy.

    Args:
        cv_strategy: "group" for GroupKFold, "stratified" for StratifiedGroupKFold.
        n_splits: number of folds.
    """
    if cv_strategy == "stratified":
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return GroupKFold(n_splits=n_splits)
