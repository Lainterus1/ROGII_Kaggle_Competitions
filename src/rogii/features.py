"""Feature engineering for the ROGII baseline.

Only columns available in both train and test are used.
TVT_input is intentionally excluded for the pure geometric baseline (Option B).
"""

import numpy as np
import pandas as pd


SAFE_NUMERIC_FEATURES = [
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "GR_is_missing",
    "MD_delta",
    "MD_relative",
    "row_position",
]


def build_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Build safe numeric features from a horizontal well DataFrame.

    The input DataFrame must contain at minimum: MD, X, Y, Z, GR.
    """
    n = len(horizontal)
    features = pd.DataFrame(index=horizontal.index)

    features["MD"] = horizontal["MD"].astype(float)
    features["X"] = horizontal["X"].astype(float)
    features["Y"] = horizontal["Y"].astype(float)
    features["Z"] = horizontal["Z"].astype(float)
    features["GR"] = horizontal["GR"].astype(float)
    features["GR_is_missing"] = horizontal["GR"].isna().astype(float)

    features["MD_delta"] = features["MD"].diff().fillna(0.0).astype(float)

    md_min = features["MD"].min()
    md_max = features["MD"].max()
    md_range = md_max - md_min
    if md_range > 0:
        features["MD_relative"] = ((features["MD"] - md_min) / md_range).astype(float)
    else:
        features["MD_relative"] = 0.0

    if n > 1:
        features["row_position"] = np.linspace(0.0, 1.0, n)
    else:
        features["row_position"] = 0.0

    return features


def post_ps_mask(horizontal: pd.DataFrame) -> np.ndarray:
    """Return a boolean mask for rows after Prediction Start.

    Prediction Start is defined as the first row where TVT_input becomes NaN.
    All rows from that point onward (including the NaN row) are post-PS.
    """
    if "TVT_input" not in horizontal.columns:
        raise ValueError("Horizontal well data must contain TVT_input")
    return horizontal["TVT_input"].isna().to_numpy(dtype=bool)


def get_feature_set_name() -> str:
    """Return the current feature set name."""
    return "safe_numeric_v1"
