"""Strict OOF Spatial KNN features.

Builds KNN features in 3D space (X, Y, Z) using only pre-PS rows from
wells outside the current validation fold. Reference values come from
known pre-PS TVT_input, never from post-PS TVT.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from rogii.data_loading import read_horizontal_well
from rogii.features import last_known_tvt_input_value, post_ps_mask

SPATIAL_NN_K = [5, 10, 50]

SPATIAL_FEATURES = [
    f"spatial_nn{k}_{stat}_tvt"
    for k in SPATIAL_NN_K
    for stat in ["mean", "median", "std"]
]


def build_pre_ps_reference(data_dir, split, well_ids):
    """Build a reference DataFrame of pre-PS rows from specified wells.

    Returns DataFrame with columns X, Y, Z, TVT_input, well_id.
    Only rows where TVT_input is known (pre-PS) are included.
    """
    frames = []
    for wid in well_ids:
        h = read_horizontal_well(data_dir, split, wid)
        if "TVT_input" not in h.columns:
            continue
        mask = ~post_ps_mask(h)
        pre = h.loc[mask, ["X", "Y", "Z", "TVT_input"]].copy()
        pre = pre.dropna(subset=["TVT_input"])
        if pre.empty:
            continue
        pre["well_id"] = wid
        frames.append(pre)
    if not frames:
        raise ValueError("No pre-PS reference rows found")
    return pd.concat(frames, ignore_index=True)


def build_spatial_knn_features(reference, query, k_values=None):
    """Build spatial KNN aggregate features for query rows.

    reference: DataFrame with at least X, Y, Z, TVT_input.
    query: DataFrame with at least X, Y, Z.
    k_values: list of k (default SPATIAL_NN_K).

    Returns DataFrame with spatial_nn{k}_{mean,median,std}_tvt columns.
    """
    if k_values is None:
        k_values = SPATIAL_NN_K

    max_k = max(k_values)
    ref_coords = reference[["X", "Y", "Z"]].astype(float).values
    query_coords = query[["X", "Y", "Z"]].astype(float).values
    ref_tvt = reference["TVT_input"].astype(float).values

    if len(ref_coords) == 0:
        feats = pd.DataFrame(index=query.index)
        for k in k_values:
            for stat in ["mean", "median", "std"]:
                feats[f"spatial_nn{k}_{stat}_tvt"] = 0.0
        return feats

    tree = NearestNeighbors(n_neighbors=max_k, algorithm="ball_tree", n_jobs=-1)
    tree.fit(ref_coords)
    dists, indices = tree.kneighbors(query_coords)

    feats = pd.DataFrame(index=query.index)
    for k in k_values:
        nn_idx = indices[:, :k]
        nn_tvt = ref_tvt[nn_idx]
        feats[f"spatial_nn{k}_mean_tvt"] = np.mean(nn_tvt, axis=1)
        feats[f"spatial_nn{k}_median_tvt"] = np.median(nn_tvt, axis=1)
        feats[f"spatial_nn{k}_std_tvt"] = np.std(nn_tvt, axis=1)
    return feats
