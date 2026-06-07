"""Z-physics TVT correction for the ROGII baseline.

Assumes locally flat formation where TVT − Z ≈ const in the lateral section.
Computes a well-level offset from pre-PS data and applies it row-by-row.
No target leakage: uses only Z and pre-PS TVT_input.
"""

import numpy as np
import pandas as pd

from rogii.features import _ps_index, last_known_tvt_input_value


def apply_z_physics(horizontal: pd.DataFrame) -> np.ndarray:
    """Return Z-physics TVT predictions for a horizontal well.

    For each row: TVT_pred = Z_row + offset
    where offset = last_tvt_input − Z_at_PS (well-level constant).

    Pre-PS rows are returned as NaN.

    Args:
        horizontal: DataFrame with at minimum Z and TVT_input columns.

    Returns:
        Float array of length len(horizontal). NaN for pre-PS rows,
        Z + offset for post-PS rows.
    """
    n = len(horizontal)
    result = np.full(n, np.nan, dtype=float)

    last_tvt = last_known_tvt_input_value(horizontal)
    if np.isnan(last_tvt):
        return result

    ps_idx = _ps_index(horizontal)
    if ps_idx >= n:
        return result

    z_at_ps = float(horizontal["Z"].iloc[ps_idx])
    offset = last_tvt - z_at_ps

    z = horizontal["Z"].astype(float).values
    result[ps_idx:] = z[ps_idx:] + offset

    return result
