"""Per-row baseline TVT computation for residual target construction.

Replaces the flat `last_tvt_input` baseline with slope-based extrapolation
from the known zone, so the model learns only deviations from the trend.
"""

import numpy as np
import pandas as pd


def _known_zone_mask(horizontal: pd.DataFrame) -> np.ndarray:
    if "TVT_input" not in horizontal.columns:
        return np.zeros(len(horizontal), dtype=bool)
    return horizontal["TVT_input"].notna().to_numpy(dtype=bool)


def _last_known_idx(horizontal: pd.DataFrame) -> int:
    if "TVT_input" not in horizontal.columns:
        return 0
    known = horizontal["TVT_input"].notna()
    if not known.any():
        return 0
    return int(known[known].index[-1])


def compute_baseline(
    horizontal: pd.DataFrame,
    method: str = "flat",
    recent_window: int = 200,
    decay: float = 0.02,
) -> np.ndarray:
    """Compute per-row baseline TVT for residual target construction.

    Pre-PS rows always get NaN (they are not used for training).
    Post-PS rows get a baseline value according to the selected method.

    Args:
        horizontal: DataFrame with MD, Z, TVT_input columns.
        method: One of "flat", "slope_md", "slope_z", "slope_recent", "wls".
        recent_window: Number of recent known-zone rows for slope_recent / wls.
        decay: Exponential decay rate for wls weights.

    Returns:
        Array of baseline values (float64), length = len(horizontal).
        Pre-PS rows are NaN. Post-PS rows are finite when computable.
    """
    n = len(horizontal)
    baseline = np.full(n, np.nan, dtype=float)

    known_mask = _known_zone_mask(horizontal)
    if not known_mask.any():
        return baseline

    last_known = _last_known_idx(horizontal)
    last_tvt = float(horizontal["TVT_input"].iloc[last_known])

    post_mask = ~known_mask
    if not post_mask.any():
        return baseline

    if method == "flat":
        baseline[post_mask] = last_tvt
        return baseline

    known_indices = np.where(known_mask)[0]
    if len(known_indices) < 2:
        baseline[post_mask] = last_tvt
        return baseline

    md = horizontal["MD"].to_numpy(dtype=float)
    z = horizontal["Z"].to_numpy(dtype=float)
    tvt_input = horizontal["TVT_input"].to_numpy(dtype=float)

    if method == "slope_z":
        x_vals = z[known_indices]
        x_at_ps = z[last_known]
    else:
        x_vals = md[known_indices]
        x_at_ps = md[last_known]

    y_vals = tvt_input[known_indices]

    if method in ("slope_md", "slope_z"):
        slope, _intercept = np.polyfit(x_vals, y_vals, 1)
    elif method == "slope_recent":
        cnt = min(recent_window, len(known_indices))
        slope, _intercept = np.polyfit(x_vals[-cnt:], y_vals[-cnt:], 1)
    elif method == "wls":
        cnt = min(recent_window, len(known_indices))
        x_r = x_vals[-cnt:]
        y_r = y_vals[-cnt:]
        weights = np.exp(decay * np.arange(cnt))
        slope, _intercept = np.polyfit(x_r, y_r, 1, w=weights)
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    if method == "slope_z":
        baseline[post_mask] = last_tvt + slope * (z[post_mask] - x_at_ps)
    else:
        baseline[post_mask] = last_tvt + slope * (md[post_mask] - x_at_ps)

    return baseline
