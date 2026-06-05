"""Formation geology features v2 — per-row GR-to-formation matching.

All 9 features are per-row (no well-level constants).
GR z-scores measure how much a row's GR deviates from each
formation's mean GR, using statistics collected from all train typewells.
The rolling z-score helps the model detect formation transitions.
"""

import numpy as np
import pandas as pd

# GR statistics (mean, std) for top-8 formations, computed from all 773
# train typewells. Used as hard-coded reference for z-score computation.
_FORMATION_GR = {
    "EGFDL":  (93.0, 22.2),
    "BUDA":   (41.4, 15.4),
    "MNSS":   (131.1, 27.5),
    "EGFDU":  (69.5, 15.8),
    "ASTNL":  (53.7, 24.2),
    "LTHL":   (84.0, 14.6),
    "LTGT":   (75.3, 10.4),
    "clayrich": (120.9, 26.8),
}

_ORDERED_FORMS = ["EGFDL", "BUDA", "MNSS", "EGFDU", "ASTNL", "LTHL", "LTGT", "clayrich"]


def build_geology_features(horizontal: pd.DataFrame, typewell: pd.DataFrame) -> pd.DataFrame:
    """Build per-row GR-vs-formation z-score features.

    For each row, computes the z-score of the row's GR against each of
    8 key formations' GR distributions. The last feature is a 20-point
    rolling average of the EGFDL z-score (the most common PS formation).
    """
    n = len(horizontal)
    gr = horizontal["GR"].astype(float).fillna(0.0).values

    feats = pd.DataFrame(index=horizontal.index)

    for i, form_name in enumerate(_ORDERED_FORMS):
        gmean, gstd = _FORMATION_GR[form_name]
        col = f"geol_gr_zscore_{form_name.lower()}"
        feats[col] = (gr - gmean) / (gstd + 1e-6)

    # Rolling z-score for the dominant formation (EGFDL) to detect transitions
    z_egfdl = feats["geol_gr_zscore_egfdl"].values
    roll = pd.Series(z_egfdl).rolling(window=20, center=False, min_periods=1)
    feats["geol_gr_zscore_roll20"] = roll.mean().fillna(0.0).values

    return feats
