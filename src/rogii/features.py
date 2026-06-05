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
    "MD_relative",
]

WITH_TVT_INPUT_FEATURES = SAFE_NUMERIC_FEATURES + ["last_tvt_input"]

GR_FEATURES = [
    "gr_roll_mean_101",
    "gr_roll_std_101",
    "gr_energy",
]

GEOMETRY_FEATURES = [
    "md_since_ps",
    "frac_after_ps",
    "dx_since_ps",
    "dy_since_ps",
    "dz_since_ps",
    "dxy_since_ps",
    "dxdmd",
    "dydmd",
    "dzdmd",
]

TRAJECTORY_FEATURES = [
    "z_local_delta",
    "dip_angle_proxy_10",
    "tortuosity_window_50",
    "sin_azimuth",
    "cos_azimuth",
]

TYPEWELL_OFFSETS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]

TYPEWELL_FEATURES = (
    [f"tw_gr_residual_{o}" for o in TYPEWELL_OFFSETS]
    + ["tw_range", "tw_gr_mean", "tw_gr_std", "tw_gr_at_last_tvt"]
)

TYPEWELL_SUMMARY_FEATURES = ["tw_range", "tw_gr_mean", "tw_gr_std", "tw_gr_at_last_tvt"]


def last_known_tvt_input_value(horizontal: pd.DataFrame) -> float | np.floating:
    """Return the last non-null TVT_input value (pre-PS well-level constant)."""
    if "TVT_input" not in horizontal.columns:
        return np.nan
    known = horizontal["TVT_input"].dropna()
    if known.empty:
        return np.nan
    return float(known.iloc[-1])


def _ps_index(horizontal: pd.DataFrame) -> int:
    """Return the row index of Prediction Start (first NaN TVT_input)."""
    if "TVT_input" not in horizontal.columns:
        return len(horizontal) - 1
    missing = horizontal["TVT_input"].isna()
    if not missing.any():
        return len(horizontal) - 1
    return int(missing.idxmax())


def build_geometry_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Build geometry features relative to Prediction Start.

    Uses TVT_input only to locate PS; does not use TVT.
    All features are row-level and available at prediction time.
    """
    n = len(horizontal)
    feats = pd.DataFrame(index=horizontal.index)
    ps_idx = _ps_index(horizontal)

    md = horizontal["MD"].astype(float)
    x = horizontal["X"].astype(float)
    y = horizontal["Y"].astype(float)
    z = horizontal["Z"].astype(float)

    md_ps = md.iloc[ps_idx]
    x_ps = x.iloc[ps_idx]
    y_ps = y.iloc[ps_idx]
    z_ps = z.iloc[ps_idx]

    feats["md_since_ps"] = md - md_ps

    post_n = n - ps_idx
    if post_n > 1:
        frac = np.zeros(n, dtype=float)
        frac[ps_idx:] = np.linspace(0.0, 1.0, post_n)
        feats["frac_after_ps"] = frac
    else:
        feats["frac_after_ps"] = 0.0

    feats["dx_since_ps"] = x - x_ps
    feats["dy_since_ps"] = y - y_ps
    feats["dz_since_ps"] = z - z_ps
    feats["dxy_since_ps"] = np.sqrt(feats["dx_since_ps"] ** 2 + feats["dy_since_ps"] ** 2)

    md_diff = md.diff()
    safe_diff = md_diff.replace(0.0, np.nan)
    feats["dxdmd"] = (x.diff() / safe_diff).fillna(0.0)
    feats["dydmd"] = (y.diff() / safe_diff).fillna(0.0)
    feats["dzdmd"] = (z.diff() / safe_diff).fillna(0.0)

    return feats


def build_trajectory_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Build 3D trajectory kinematics features from MD, X, Y, Z.

    These capture local dip angle, wellbore curvature (dogleg severity),
    tortuosity, and directional azimuth. Only uses row-level geometry;
    does not depend on TVT or TVT_input beyond PS boundary detection.
    """
    n = len(horizontal)
    feats = pd.DataFrame(index=horizontal.index)
    ps_idx = _ps_index(horizontal)

    md = horizontal["MD"].astype(float).values
    x = horizontal["X"].astype(float).values
    y = horizontal["Y"].astype(float).values
    z = horizontal["Z"].astype(float).values

    pre_ps_z = z[:ps_idx] if ps_idx > 0 else z
    mean_pre_ps_z = float(np.mean(pre_ps_z))
    feats["z_local_delta"] = z - mean_pre_ps_z

    delta = 10
    dz_diff10 = np.empty(n, dtype=float)
    dm_diff10 = np.empty(n, dtype=float)
    for i in range(n):
        if i >= delta:
            dz_diff10[i] = z[i] - z[i - delta]
            dm_diff10[i] = md[i] - md[i - delta]
        elif i > 0:
            dz_diff10[i] = z[i] - z[0]
            dm_diff10[i] = md[i] - md[0]
        else:
            dz_diff10[i] = 0.0
            dm_diff10[i] = 1.0
    dip_mask = np.abs(dm_diff10) > 1e-9
    feats["dip_angle_proxy_10"] = np.divide(
        dz_diff10, dm_diff10, out=np.zeros_like(dz_diff10), where=dip_mask
    )

    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    dz = np.diff(z, prepend=z[0])
    ds = np.sqrt(dx**2 + dy**2 + dz**2)
    tort = np.ones(n, dtype=float)
    window_t = 50
    for i in range(n):
        j = max(0, i - window_t)
        if j >= i:
            tort[i] = 1.0
        else:
            arc = np.sum(ds[j + 1 : i + 1])
            straight = np.sqrt(
                (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
            )
            if straight > 1e-9 and arc > 0:
                tort[i] = float(arc / straight)
            else:
                tort[i] = 1.0
    feats["tortuosity_window_50"] = tort

    ddx = np.diff(x, prepend=x[0])
    ddy = np.diff(y, prepend=y[0])
    azimuth = np.arctan2(ddy, ddx)
    feats["sin_azimuth"] = np.sin(azimuth)
    feats["cos_azimuth"] = np.cos(azimuth)

    return feats


def build_gr_features(horizontal: pd.DataFrame) -> pd.DataFrame:
    """Build GR-derived features: rolling stats, lags/leads, differences, energy, envelope.

    Only computes features listed in GR_FEATURES constant.
    Uses centered rolling windows and forward-looking leads.
    All GR values are available at prediction time (full well profile is known).
    """
    gr = horizontal["GR"].astype(float)
    feats = pd.DataFrame(index=horizontal.index)

    for w in [5, 21, 51, 101]:
        name = f"gr_roll_mean_{w}"
        if name in GR_FEATURES:
            roll = gr.rolling(window=w, center=True, min_periods=1)
            feats[name] = roll.mean().fillna(0.0)

    for w in [5, 21, 51, 101]:
        name = f"gr_roll_std_{w}"
        if name in GR_FEATURES:
            roll = gr.rolling(window=w, center=True, min_periods=1)
            feats[name] = roll.std().fillna(0.0)

    for k in [1, 5, 15, 30]:
        name = f"gr_lag_{k}"
        if name in GR_FEATURES:
            feats[name] = gr.shift(k).fillna(0.0)

    for k in [1, 5, 15, 30]:
        name = f"gr_lead_{k}"
        if name in GR_FEATURES:
            feats[name] = gr.shift(-k).fillna(0.0)

    if "gr_d1" in GR_FEATURES:
        feats["gr_d1"] = gr.diff().fillna(0.0)
    if "gr_d2" in GR_FEATURES:
        feats["gr_d2"] = gr.diff().diff().fillna(0.0)

    if "gr_energy" in GR_FEATURES:
        gr_filled = gr.fillna(0.0)
        cumsum_sq = gr_filled.pow(2).cumsum()
        count = (~gr.isna()).cumsum()
        feats["gr_energy"] = (cumsum_sq / count.replace(0, 1.0)).astype(float)

    if "gr_envelope" in GR_FEATURES:
        roll_max = gr.rolling(window=21, center=True, min_periods=1).max()
        roll_min = gr.rolling(window=21, center=True, min_periods=1).min()
        feats["gr_envelope"] = (roll_max - roll_min).astype(float)

    return feats


def build_typewell_features(horizontal: pd.DataFrame, typewell: pd.DataFrame,
                           summary_only: bool = False) -> pd.DataFrame:
    """Build typewell-reference features: anchor-offset GR residuals and summary stats.

    Uses last_tvt_input to anchor offsets in typewell TVT space.
    All typewell data is available in both train and test.
    When summary_only=True, only well-level summary features are built (no residuals).
    """
    n = len(horizontal)
    feats = pd.DataFrame(index=horizontal.index)

    last_tvt = last_known_tvt_input_value(horizontal)
    if np.isnan(last_tvt):
        for col in TYPEWELL_FEATURES:
            feats[col] = 0.0
        return feats

    tw_tvt = typewell["TVT"].astype(float).values
    tw_gr = typewell["GR"].astype(float).values
    horizontal_gr = horizontal["GR"].astype(float)

    tw_range = float(tw_tvt[-1] - tw_tvt[0])
    tw_gr_mean = float(tw_gr.mean())
    tw_gr_std = float(tw_gr.std())
    tw_gr_at_last_tvt = float(np.interp(last_tvt, tw_tvt, tw_gr))

    if not summary_only:
        for o in TYPEWELL_OFFSETS:
            target_tvt = last_tvt + o
            tw_gr_at_offset = float(np.interp(target_tvt, tw_tvt, tw_gr))
            feats[f"tw_gr_residual_{o}"] = horizontal_gr - tw_gr_at_offset

    feats["tw_range"] = tw_range
    feats["tw_gr_mean"] = tw_gr_mean
    feats["tw_gr_std"] = tw_gr_std
    feats["tw_gr_at_last_tvt"] = tw_gr_at_last_tvt

    return feats


def build_features(horizontal: pd.DataFrame, include_tvt_input: bool = False,
                   include_geometry: bool = False, include_gr: bool = False,
                   include_trajectory: bool = False,
                   typewell: pd.DataFrame | None = None,
                   include_typewell: bool = False,
                   typewell_summary_only: bool = False) -> pd.DataFrame:
    """Build numeric features from a horizontal well DataFrame.

    The input DataFrame must contain at minimum: MD, X, Y, Z, GR.
    When include_tvt_input=True, the last known pre-PS TVT_input value is added
    as a well-level constant feature.
    When include_geometry=True, geometry features relative to Prediction Start
    are added. When include_trajectory=True, trajectory kinematics features
    are added and geometry features are automatically included.
    When include_gr=True, GR-derived rolling/lag/energy/envelope features are added.
    When include_typewell=True, typewell-reference residual and summary features
    are added. Requires a typewell DataFrame.
    When typewell_summary_only=True, only summary features are built (no residuals).
    """
    n = len(horizontal)
    features = pd.DataFrame(index=horizontal.index)

    features["MD"] = horizontal["MD"].astype(float)
    features["X"] = horizontal["X"].astype(float)
    features["Y"] = horizontal["Y"].astype(float)
    features["Z"] = horizontal["Z"].astype(float)
    features["GR"] = horizontal["GR"].astype(float)

    md_min = features["MD"].min()
    md_max = features["MD"].max()
    md_range = md_max - md_min
    if md_range > 0:
        features["MD_relative"] = ((features["MD"] - md_min) / md_range).astype(float)
    else:
        features["MD_relative"] = 0.0

    if include_tvt_input:
        features["last_tvt_input"] = last_known_tvt_input_value(horizontal)

    use_geometry = include_geometry or include_trajectory
    if use_geometry:
        geo = build_geometry_features(horizontal)
        features = pd.concat([features, geo], axis=1)

    if include_trajectory:
        traj = build_trajectory_features(horizontal)
        features = pd.concat([features, traj], axis=1)

    if include_gr:
        gr_feats = build_gr_features(horizontal)
        features = pd.concat([features, gr_feats], axis=1)

    if include_typewell and typewell is not None:
        tw_feats = build_typewell_features(horizontal, typewell, summary_only=typewell_summary_only)
        features = pd.concat([features, tw_feats], axis=1)

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
