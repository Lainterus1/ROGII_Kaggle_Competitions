"""Prediction-time 3-strategy blend: model + Z-physics + DTW matching."""

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.data_loading import read_horizontal_well, read_typewell
from rogii.gr_matcher import apply_dtw_matching
from rogii.models import parse_submission_id
from rogii.z_physics import apply_z_physics


def _blend_row(
    model_pred: float,
    z_pred: float,
    dtw_pred: float,
    weights: tuple[float, float, float] | None = None,
) -> float:
    preds = np.array([model_pred, z_pred, dtw_pred], dtype=float)
    valid_mask = np.isfinite(preds)

    if not valid_mask.any():
        return np.nan

    if weights is not None:
        w = np.array(weights, dtype=float)
        w[~valid_mask] = 0.0
        total = w.sum()
        if total > 0:
            return float(np.dot(preds, w) / total)

    return float(np.median(preds[valid_mask]))


def parse_blend_weights(raw: str | None) -> tuple[float, float, float] | None:
    """Parse '0.5,0.25,0.25' string into (model, z_physics, dtw) weights."""
    if raw is None:
        return None
    parts = [float(x.strip()) for x in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(
            f"Blend weights must have 3 values (model,z_physics,dtw), got {len(parts)}"
        )
    return tuple(parts)


def apply_postprocess_blend(
    submission_df: pd.DataFrame,
    data_dir: str | Path,
    weights: tuple[float, float, float] | None = None,
) -> pd.DataFrame:
    """Blend model predictions with Z-physics and DTW matching.

    For each well in the submission, loads horizontal and typewell data,
    computes Z-physics and DTW-matched TVT predictions, then blends with
    the model predictions using median (default) or weighted average.

    Args:
        submission_df: DataFrame with "id" and "tvt" columns (model predictions).
        data_dir: Competition data directory.
        weights: Optional (model, z_physics, dtw) blend weights.

    Returns:
        New DataFrame with blended "tvt" column.
    """
    result = submission_df.copy()
    ids = result["id"].astype(str)
    result["_well_id"] = ids.apply(lambda x: parse_submission_id(x)[0])
    result["_row_idx"] = ids.apply(lambda x: parse_submission_id(x)[1])

    for well_id, group in result.groupby("_well_id", sort=False):
        horizontal = read_horizontal_well(data_dir, "test", well_id)
        n_rows = len(horizontal)

        # Model predictions (from submission)
        model_preds = np.full(n_rows, np.nan, dtype=float)
        for _, row in group.iterrows():
            ri = int(row["_row_idx"])
            if 0 <= ri < n_rows:
                model_preds[ri] = float(row["tvt"])

        # Z-physics predictions
        z_preds = apply_z_physics(horizontal)

        # DTW matching predictions
        has_typewell = False
        try:
            typewell = read_typewell(data_dir, "test", well_id)
            has_typewell = True
        except FileNotFoundError:
            pass

        if has_typewell and typewell is not None and len(typewell) >= 3:
            dtw_preds = apply_dtw_matching(
                horizontal, typewell, model_preds.copy()
            )
        else:
            dtw_preds = model_preds.copy()

        # Blend per-row and write back
        for _, row in group.iterrows():
            ri = int(row["_row_idx"])
            blended = _blend_row(
                model_preds[ri], z_preds[ri], dtw_preds[ri], weights
            )
            if not np.isfinite(blended):
                blended = model_preds[ri]
            result.loc[row.name, "tvt"] = float(blended)

    result = result.drop(columns=["_well_id", "_row_idx"])
    if not np.isfinite(result["tvt"]).all():
        raise ValueError("Blended submission contains non-finite values")

    return result
