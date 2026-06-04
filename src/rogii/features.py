"""Feature engineering for the ROGII baseline.

Only columns available in both train and test are used.
TVT_input is used only via last_known_pre_ps value (well-level constant, available at prediction time).
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

WITH_TVT_INPUT_FEATURES = SAFE_NUMERIC_FEATURES + ["last_tvt_input"]


def last_known_tvt_input_value(horizontal: pd.DataFrame) -> float | np.floating:
    """Return the last non-null TVT_input value (pre-PS well-level constant)."""
    if "TVT_input" not in horizontal.columns:
        return np.nan
    known = horizontal["TVT_input"].dropna()
    if known.empty:
        return np.nan
    return float(known.iloc[-1])


def build_features(horizontal: pd.DataFrame, include_tvt_input: bool = False) -> pd.DataFrame:
    """Build numeric features from a horizontal well DataFrame.

    The input DataFrame must contain at minimum: MD, X, Y, Z, GR.
    When include_tvt_input=True, the last known pre-PS TVT_input value is added
    as a well-level constant feature.
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

    if include_tvt_input:
        features["last_tvt_input"] = last_known_tvt_input_value(horizontal)

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
