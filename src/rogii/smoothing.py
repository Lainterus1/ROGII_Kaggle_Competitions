"""Per-well Savitzky-Golay smoothing and TVT clipping of predicted sequences."""

from pathlib import Path

import numpy as np
import pandas as pd


def smooth_predictions_per_well(
    submission_df: pd.DataFrame,
    window: int = 31,
    polyorder: int = 2,
) -> pd.DataFrame:
    """Apply Savgol smoothing to TVT predictions per well.

    Parses well_id from submission id and smooths each well's TVT sequence
    independently. Short sequences (length <= window) are passed through
    unchanged.

    Args:
        submission_df: DataFrame with "id" and "tvt" columns.
        window: Savgol filter window length (must be odd, default 31).
        polyorder: Savgol filter polynomial order (default 2).

    Returns:
        New DataFrame with smoothed "tvt" column.
    """
    result = submission_df.copy()
    # Parse well_id from "id" column (format: <well_id>_<row_index>)
    ids = result["id"].astype(str)
    last_underscore = ids.str.rsplit("_", n=1)
    well_ids = last_underscore.str[0]
    row_indices = last_underscore.str[1].astype(int)

    # Sort by (well_id, row_index) to get contiguous per-well sequences
    result["_well_id"] = well_ids
    result["_row_idx"] = row_indices
    result = result.sort_values(["_well_id", "_row_idx"]).reset_index(drop=True)

    # Smooth per well
    from scipy.signal import savgol_filter
    smoothed = np.zeros(len(result), dtype=float)
    for well_id, group in result.groupby("_well_id", sort=False):
        values = group["tvt"].to_numpy(dtype=float)
        if len(values) > window:
            smoothed[group.index] = savgol_filter(values, window, polyorder)
        else:
            smoothed[group.index] = values

    result["tvt"] = smoothed
    result = result.drop(columns=["_well_id", "_row_idx"])
    return result


def clip_predictions(
    submission_df: pd.DataFrame,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    """Clamp TVT predictions to [lower, upper].

    Args:
        submission_df: DataFrame with "id" and "tvt" columns.
        lower: Minimum allowed TVT value.
        upper: Maximum allowed TVT value.

    Returns:
        New DataFrame with clipped "tvt" column.
    """
    result = submission_df.copy()
    result["tvt"] = result["tvt"].clip(lower, upper)
    return result


def compute_tvt_clip_bounds(
    data_dir: str | Path,
    lower_percentile: float = 0.1,
    upper_percentile: float = 99.9,
) -> tuple[float, float]:
    """Scan train TVT values and return (lower, upper) clip bounds.

    Args:
        data_dir: Competition data directory.
        lower_percentile: Percentile for lower clipping bound.
        upper_percentile: Percentile for upper clipping bound.

    Returns:
        (lower, upper) tuple of float bounds.
    """
    from rogii.data_loading import list_well_ids, read_horizontal_well

    well_ids = list_well_ids(data_dir, "train")
    all_tvt: list[float] = []

    for wid in well_ids:
        df = read_horizontal_well(data_dir, "train", wid)
        if "TVT" not in df.columns:
            continue
        tvt = df["TVT"].dropna().to_numpy(dtype=float)
        all_tvt.extend(tvt.tolist())

    if not all_tvt:
        raise ValueError("No TVT values found in train data")

    arr = np.array(all_tvt, dtype=float)
    lower = float(np.percentile(arr, lower_percentile))
    upper = float(np.percentile(arr, upper_percentile))
    return lower, upper


def apply_postprocessing(
    submission_df: pd.DataFrame,
    savgol_window: int | None = None,
    savgol_polyorder: int = 3,
    clip_lower: float | None = None,
    clip_upper: float | None = None,
) -> pd.DataFrame:
    """Apply clipping then Savgol smoothing to predictions.

    Clipping is applied first to remove outliers that would distort the
    Savgol filter.

    Args:
        submission_df: DataFrame with "id" and "tvt" columns.
        savgol_window: Savgol window length (None = skip smoothing).
        savgol_polyorder: Savgol polynomial order.
        clip_lower: Minimum TVT bound (None = skip clipping).
        clip_upper: Maximum TVT bound (None = skip clipping).

    Returns:
        New DataFrame with post-processed "tvt" column.
    """
    result = submission_df.copy()

    if clip_lower is not None and clip_upper is not None:
        result = clip_predictions(result, clip_lower, clip_upper)

    if savgol_window is not None:
        result = smooth_predictions_per_well(result, window=savgol_window, polyorder=savgol_polyorder)

    return result
