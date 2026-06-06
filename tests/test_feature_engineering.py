import numpy as np
import pandas as pd
import pytest

from rogii.features import (
    GEOMETRY_FEATURES,
    GR_DWT_FEATURES,
    GR_FEATURES,
    SAFE_NUMERIC_FEATURES,
    TRAJECTORY_FEATURES,
    TYPEWELL_FEATURES,
    WITH_TVT_INPUT_FEATURES,
    Z_DRIFT_FEATURES,
    build_features,
    build_geometry_features,
    build_gr_features,
    build_trajectory_features,
    build_typewell_features,
    build_z_drift_features,
    get_feature_set_name,
    last_known_tvt_input_value,
    post_ps_mask,
)


def _make_well(n_rows: int = 10, gr_nan_after: int | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "MD": np.linspace(1000, 1000 + n_rows, n_rows),
            "X": np.linspace(0, n_rows, n_rows),
            "Y": np.linspace(0, n_rows, n_rows),
            "Z": np.linspace(-100, -100 + n_rows, n_rows),
            "GR": np.linspace(80, 120, n_rows),
            "TVT_input": np.linspace(5000, 5000 + n_rows, n_rows),
        }
    )
    if gr_nan_after is not None:
        frame.loc[frame.index >= gr_nan_after, "GR"] = np.nan
    ps_start = n_rows // 2
    frame.loc[frame.index >= ps_start, "TVT_input"] = np.nan
    return frame


def test_build_features_columns() -> None:
    frame = _make_well(10)
    features = build_features(frame)
    assert list(features.columns) == SAFE_NUMERIC_FEATURES
    assert len(features) == 10


def test_build_features_no_nan_in_base() -> None:
    frame = _make_well(10)
    features = build_features(frame)
    assert not features[["MD", "X", "Y", "Z"]].isna().any().any()


def test_md_relative_range() -> None:
    frame = _make_well(10)
    features = build_features(frame)
    assert features.loc[0, "MD_relative"] == pytest.approx(0.0)
    assert features.loc[9, "MD_relative"] == pytest.approx(1.0)


def test_single_row_well() -> None:
    frame = _make_well(1)
    features = build_features(frame)
    assert len(features) == 1
    assert features.loc[0, "MD_relative"] == 0.0


def test_post_ps_mask() -> None:
    frame = _make_well(10)
    mask = post_ps_mask(frame)
    assert mask.sum() == 5
    assert mask[4] == False
    assert mask[5] == True


def test_post_ps_mask_all_known() -> None:
    frame = _make_well(5)
    frame["TVT_input"] = 1.0
    mask = post_ps_mask(frame)
    assert mask.sum() == 0


def test_post_ps_mask_all_missing() -> None:
    frame = _make_well(5)
    frame["TVT_input"] = np.nan
    mask = post_ps_mask(frame)
    assert mask.sum() == 5


def test_get_feature_set_name() -> None:
    assert get_feature_set_name() == "safe_numeric_v1"


def test_last_known_tvt_input_value() -> None:
    frame = _make_well(10)
    result = last_known_tvt_input_value(frame)
    assert result == frame.loc[4, "TVT_input"]


def test_last_known_tvt_input_missing_column() -> None:
    frame = pd.DataFrame({"MD": [1, 2, 3]})
    result = last_known_tvt_input_value(frame)
    assert np.isnan(result)


def test_build_features_with_tvt_input() -> None:
    frame = _make_well(10)
    features = build_features(frame, include_tvt_input=True)
    assert list(features.columns) == WITH_TVT_INPUT_FEATURES
    assert "last_tvt_input" in features.columns
    expected = last_known_tvt_input_value(frame)
    assert (features["last_tvt_input"] == expected).all()


def test_build_features_without_tvt_input() -> None:
    frame = _make_well(10)
    features = build_features(frame, include_tvt_input=False)
    assert list(features.columns) == SAFE_NUMERIC_FEATURES
    assert "last_tvt_input" not in features.columns


def test_feature_columns_match_constants() -> None:
    assert len(WITH_TVT_INPUT_FEATURES) == len(SAFE_NUMERIC_FEATURES) + 1
    assert WITH_TVT_INPUT_FEATURES[-1] == "last_tvt_input"


def test_geometry_features_columns() -> None:
    frame = _make_well(10)
    geo = build_geometry_features(frame)
    assert list(geo.columns) == GEOMETRY_FEATURES
    assert len(geo) == 10


def test_geometry_features_md_since_ps() -> None:
    frame = _make_well(10)
    geo = build_geometry_features(frame)
    ps_idx = 5
    assert geo.loc[ps_idx, "md_since_ps"] == pytest.approx(0.0)
    assert geo.loc[ps_idx + 1, "md_since_ps"] > 0.0
    assert geo.loc[ps_idx - 1, "md_since_ps"] < 0.0


def test_geometry_features_frac_after_ps() -> None:
    frame = _make_well(10)
    geo = build_geometry_features(frame)
    ps_idx = 5
    assert geo.loc[ps_idx, "frac_after_ps"] == pytest.approx(0.0)
    assert geo.loc[9, "frac_after_ps"] == pytest.approx(1.0)
    assert geo.loc[ps_idx - 1, "frac_after_ps"] == pytest.approx(0.0)


def test_geometry_features_dxy() -> None:
    frame = _make_well(10)
    geo = build_geometry_features(frame)
    expected = np.sqrt(geo["dx_since_ps"] ** 2 + geo["dy_since_ps"] ** 2)
    assert np.allclose(geo["dxy_since_ps"].values, expected.values)


def test_geometry_features_directional_derivatives() -> None:
    frame = _make_well(10)
    geo = build_geometry_features(frame)
    assert geo.loc[0, "dxdmd"] == pytest.approx(0.0)
    assert geo.loc[0, "dydmd"] == pytest.approx(0.0)
    assert geo.loc[0, "dzdmd"] == pytest.approx(0.0)
    assert not geo[["dxdmd", "dydmd", "dzdmd"]].isna().any().any()


def test_geometry_features_single_row() -> None:
    frame = _make_well(1)
    frame.loc[0, "TVT_input"] = np.nan
    geo = build_geometry_features(frame)
    assert len(geo) == 1
    assert geo.loc[0, "md_since_ps"] == pytest.approx(0.0)
    assert geo.loc[0, "frac_after_ps"] == pytest.approx(0.0)


def test_geometry_features_no_post_ps() -> None:
    frame = _make_well(5)
    frame["TVT_input"] = 1.0
    geo = build_geometry_features(frame)
    assert len(geo) == 5
    assert geo["frac_after_ps"].iloc[-1] == pytest.approx(0.0)


def test_build_features_with_geometry() -> None:
    frame = _make_well(10)
    features = build_features(frame, include_geometry=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES
    assert list(features.columns) == expected_cols
    assert "md_since_ps" in features.columns
    assert "frac_after_ps" in features.columns


def test_geometry_features_constants_match() -> None:
    assert len(GEOMETRY_FEATURES) == 9
    assert GEOMETRY_FEATURES[0] == "md_since_ps"
    assert GEOMETRY_FEATURES[-1] == "dzdmd"


def test_gr_features_columns() -> None:
    frame = _make_well(20)
    gr_feats = build_gr_features(frame)
    assert list(gr_feats.columns) == GR_FEATURES
    assert len(GR_FEATURES) == 3


def test_gr_features_no_nan() -> None:
    frame = _make_well(20)
    gr_feats = build_gr_features(frame)
    assert not gr_feats.isna().any().any()


def test_gr_energy_cumulative() -> None:
    frame = _make_well(5, gr_nan_after=5)
    gr_feats = build_gr_features(frame)
    gr_vals = frame["GR"].astype(float).values[:5]
    expected = sum(v**2 for v in gr_vals) / 5
    assert gr_feats.loc[4, "gr_energy"] == pytest.approx(expected)


def test_gr_features_single_row() -> None:
    frame = _make_well(1)
    gr_feats = build_gr_features(frame)
    assert len(gr_feats) == 1
    assert not gr_feats.isna().any().any()


def test_gr_features_with_nan_in_middle() -> None:
    frame = _make_well(20, gr_nan_after=10)
    gr_feats = build_gr_features(frame)
    assert not gr_feats.isna().any().any()


def test_build_features_with_gr() -> None:
    frame = _make_well(10)
    features = build_features(frame, include_gr=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GR_FEATURES
    assert list(features.columns) == expected_cols
    assert "gr_roll_mean_101" in features.columns
    assert "gr_energy" in features.columns


def test_build_features_full_r1() -> None:
    frame = _make_well(10)
    features = build_features(frame, include_geometry=True, include_gr=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES + GR_FEATURES
    assert list(features.columns) == expected_cols
    assert "md_since_ps" in features.columns
    assert "gr_roll_std_101" in features.columns


def test_gr_features_constants() -> None:
    assert isinstance(GR_FEATURES, list)
    assert len(GR_FEATURES) == 3
    assert "gr_roll_mean_101" in GR_FEATURES
    assert "gr_roll_std_101" in GR_FEATURES
    assert "gr_energy" in GR_FEATURES


def _make_typewell(tvt_start: float = 5000.0, tvt_step: float = 0.5, n: int = 100) -> pd.DataFrame:
    tvt = np.arange(tvt_start, tvt_start + n * tvt_step, tvt_step)
    gr = 100.0 + 20.0 * np.sin(np.linspace(0, 4 * np.pi, n))
    return pd.DataFrame({"TVT": tvt, "GR": gr})


def test_typewell_features_columns() -> None:
    hw = _make_well(10)
    tw = _make_typewell()
    feats = build_typewell_features(hw, tw)
    assert list(feats.columns) == TYPEWELL_FEATURES
    assert len(feats) == 10
    assert len(TYPEWELL_FEATURES) == 15


def test_typewell_features_summary_constants() -> None:
    hw = _make_well(10)
    tw = _make_typewell()
    feats = build_typewell_features(hw, tw)
    assert feats["tw_range"].nunique() == 1
    assert feats["tw_gr_mean"].nunique() == 1
    assert feats["tw_gr_std"].nunique() == 1
    assert feats["tw_gr_at_last_tvt"].nunique() == 1


def test_typewell_features_residual_range() -> None:
    hw = _make_well(10)
    tw = _make_typewell(tvt_start=5000.0, tvt_step=0.5, n=200)
    feats = build_typewell_features(hw, tw)
    for col in [c for c in feats.columns if c.startswith("tw_gr_residual_")]:
        assert not np.isnan(feats[col]).any()


def test_typewell_features_nan_last_tvt() -> None:
    hw = _make_well(10)
    hw["TVT_input"] = np.nan
    tw = _make_typewell()
    feats = build_typewell_features(hw, tw)
    assert (feats == 0.0).all().all()


def test_build_features_with_typewell() -> None:
    hw = _make_well(10)
    tw = _make_typewell()
    features = build_features(hw, typewell=tw, include_typewell=True)
    expected_cols = SAFE_NUMERIC_FEATURES + TYPEWELL_FEATURES
    assert list(features.columns) == expected_cols
    assert "tw_gr_residual_0" in features.columns
    assert "tw_range" in features.columns


def test_build_features_full_r2() -> None:
    hw = _make_well(10)
    tw = _make_typewell()
    features = build_features(hw, include_geometry=True, include_gr=True, typewell=tw, include_typewell=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES + GR_FEATURES + TYPEWELL_FEATURES
    assert list(features.columns) == expected_cols
    assert len(features.columns) == 6 + 9 + 3 + 15


def test_typewell_features_residual_at_anchor_zero() -> None:
    hw = _make_well(20)
    tw = _make_typewell(tvt_start=5000.0, tvt_step=0.5, n=200)
    feats = build_typewell_features(hw, tw)
    last_tvt = last_known_tvt_input_value(hw)
    tw_gr_at_last = float(np.interp(last_tvt, tw["TVT"].values, tw["GR"].values))
    gr = hw["GR"].astype(float).values
    expected = gr - tw_gr_at_last
    assert np.allclose(feats["tw_gr_residual_0"].values, expected)


def _make_straight_well(n_rows: int = 100) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "MD": np.linspace(1000, 2000, n_rows),
            "X": np.linspace(0, 100, n_rows),
            "Y": np.linspace(0, 0, n_rows),
            "Z": np.linspace(-1000, -900, n_rows),
            "GR": np.linspace(80, 120, n_rows),
            "TVT_input": np.linspace(5000, 5000 + n_rows, n_rows),
        }
    )
    ps_start = n_rows // 2
    frame.loc[frame.index >= ps_start, "TVT_input"] = np.nan
    return frame


def _make_curved_well(n_rows: int = 100) -> pd.DataFrame:
    t = np.linspace(0, 2 * np.pi, n_rows)
    frame = pd.DataFrame(
        {
            "MD": np.linspace(1000, 2000, n_rows),
            "X": 100 * np.cos(t),
            "Y": 100 * np.sin(t),
            "Z": np.linspace(-1000, -900, n_rows),
            "GR": np.linspace(80, 120, n_rows),
            "TVT_input": np.linspace(5000, 5000 + n_rows, n_rows),
        }
    )
    ps_start = n_rows // 2
    frame.loc[frame.index >= ps_start, "TVT_input"] = np.nan
    return frame


def test_trajectory_features_columns() -> None:
    frame = _make_straight_well(100)
    traj = build_trajectory_features(frame)
    assert list(traj.columns) == TRAJECTORY_FEATURES
    assert len(traj) == 100
    assert len(TRAJECTORY_FEATURES) == 5


def test_trajectory_features_no_nan() -> None:
    frame = _make_straight_well(100)
    traj = build_trajectory_features(frame)
    assert not traj.isna().any().any()


def test_trajectory_single_row() -> None:
    frame = _make_straight_well(1)
    traj = build_trajectory_features(frame)
    assert len(traj) == 1
    assert not traj.isna().any().any()


def test_trajectory_straight_well() -> None:
    frame = _make_straight_well(100)
    traj = build_trajectory_features(frame)
    tort = traj["tortuosity_window_50"].values
    assert np.allclose(tort[51:], 1.0, atol=1e-3)


def test_trajectory_curved_well() -> None:
    frame = _make_curved_well(100)
    traj = build_trajectory_features(frame)
    max_tort = traj["tortuosity_window_50"].max()
    assert max_tort >= 1.0


def test_trajectory_dip_angle_proxy_updip() -> None:
    frame = _make_straight_well(100)
    frame["Z"] = np.linspace(-1000, -900, 100)
    traj = build_trajectory_features(frame)
    assert np.all(traj["dip_angle_proxy_10"].iloc[20:] > 0)


def test_trajectory_azimuth_bounded() -> None:
    frame = _make_curved_well(100)
    traj = build_trajectory_features(frame)
    assert (np.abs(traj["sin_azimuth"]) <= 1.0).all()
    assert (np.abs(traj["cos_azimuth"]) <= 1.0).all()
    sq_sum = traj["sin_azimuth"] ** 2 + traj["cos_azimuth"] ** 2
    disp_zero = frame["X"].diff().fillna(0).abs() < 1e-9
    disp_zero &= frame["Y"].diff().fillna(0).abs() < 1e-9
    non_zero_rows = ~disp_zero.values
    if non_zero_rows.any():
        assert np.allclose(sq_sum[non_zero_rows], 1.0, atol=1e-6)


def test_trajectory_z_local_delta_sign() -> None:
    frame = _make_straight_well(100)
    traj = build_trajectory_features(frame)
    ps_idx = 50
    if ps_idx > 0:
        mean_pre_ps_z = frame["Z"].iloc[:ps_idx].mean()
        expected = frame["Z"].iloc[ps_idx] - mean_pre_ps_z
        assert traj.loc[ps_idx, "z_local_delta"] == pytest.approx(expected)


def test_build_features_with_trajectory() -> None:
    frame = _make_straight_well(100)
    features = build_features(frame, include_trajectory=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES + TRAJECTORY_FEATURES
    assert list(features.columns) == expected_cols
    assert "z_local_delta" in features.columns
    assert "sin_azimuth" in features.columns


def test_build_features_trajectory_with_gr() -> None:
    frame = _make_straight_well(100)
    features = build_features(frame, include_trajectory=True, include_gr=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES + TRAJECTORY_FEATURES + GR_FEATURES
    assert list(features.columns) == expected_cols
    assert len(features.columns) == 6 + 9 + 5 + 3


def test_trajectory_features_constants() -> None:
    assert isinstance(TRAJECTORY_FEATURES, list)
    assert len(TRAJECTORY_FEATURES) == 5
    assert "z_local_delta" in TRAJECTORY_FEATURES
    assert "dip_angle_proxy_10" in TRAJECTORY_FEATURES
    assert "tortuosity_window_50" in TRAJECTORY_FEATURES
    assert "sin_azimuth" in TRAJECTORY_FEATURES
    assert "cos_azimuth" in TRAJECTORY_FEATURES


def test_gr_dwt_features_constants() -> None:
    assert isinstance(GR_DWT_FEATURES, list)
    assert len(GR_DWT_FEATURES) == 2
    assert GR_DWT_FEATURES[0] == "gr_dwt_approx"
    assert GR_DWT_FEATURES[1] == "gr_dwt_detail_energy"


def test_gr_dwt_features_columns() -> None:
    from rogii.gr_dwt import build_gr_dwt_features
    frame = _make_well(100)
    feats = build_gr_dwt_features(frame, window=64, min_window=8)
    assert list(feats.columns) == GR_DWT_FEATURES
    assert len(feats) == 100


def test_gr_dwt_features_no_nan() -> None:
    from rogii.gr_dwt import build_gr_dwt_features
    frame = _make_well(100)
    feats = build_gr_dwt_features(frame, window=64, min_window=8)
    assert not feats.isna().any().any()


def test_gr_dwt_causal_no_future_leak() -> None:
    from rogii.gr_dwt import build_gr_dwt_features
    frame = _make_well(200)
    feats = build_gr_dwt_features(frame, window=64, min_window=8)

    gr_original = frame["GR"].astype(float).values.copy()
    for i in range(50, 150):
        original = feats.loc[i, "gr_dwt_approx"]
        gr_modified = gr_original.copy()
        gr_modified[i + 1:] = 9999.0
        frame2 = frame.copy()
        frame2["GR"] = gr_modified
        feats2 = build_gr_dwt_features(frame2, window=64, min_window=8)
        assert feats2.loc[i, "gr_dwt_approx"] == original


def test_gr_dwt_single_row() -> None:
    from rogii.gr_dwt import build_gr_dwt_features
    frame = _make_well(1)
    feats = build_gr_dwt_features(frame, window=64, min_window=8)
    assert len(feats) == 1
    assert not feats.isna().any().any()
    assert feats.loc[0, "gr_dwt_approx"] == frame.loc[0, "GR"]
    assert feats.loc[0, "gr_dwt_detail_energy"] == 0.0


def test_gr_dwt_fallback_for_short_segments() -> None:
    from rogii.gr_dwt import build_gr_dwt_features
    n = 20
    frame = _make_well(n)
    feats = build_gr_dwt_features(frame, window=256, min_window=32)
    for i in range(n):
        assert feats.loc[i, "gr_dwt_detail_energy"] == 0.0
    gr = frame["GR"].astype(float).values
    assert np.allclose(feats["gr_dwt_approx"].values, gr)


def test_build_features_with_gr_dwt() -> None:
    frame = _make_well(50)
    features = build_features(frame, include_gr_dwt=True, dwt_window=32, dwt_min_window=8)
    expected_cols = SAFE_NUMERIC_FEATURES + GR_DWT_FEATURES
    assert list(features.columns) == expected_cols
    assert "gr_dwt_approx" in features.columns
    assert "gr_dwt_detail_energy" in features.columns


def test_build_features_r1_plus_dwt() -> None:
    frame = _make_well(50)
    features = build_features(frame, include_geometry=True, include_gr=True, include_gr_dwt=True,
                             dwt_window=32, dwt_min_window=8)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES + GR_FEATURES + GR_DWT_FEATURES
    assert list(features.columns) == expected_cols


# ---------------------------------------------------------------------------
# Z-Drift physics features
# ---------------------------------------------------------------------------


def test_z_drift_features_constants() -> None:
    assert isinstance(Z_DRIFT_FEATURES, list)
    assert len(Z_DRIFT_FEATURES) == 3
    assert Z_DRIFT_FEATURES[0] == "z_drift_offset_at_anchor"
    assert Z_DRIFT_FEATURES[1] == "z_drift_implied_tvt"
    assert Z_DRIFT_FEATURES[2] == "z_drift_implied_tvt_resid"


def test_z_drift_features_columns() -> None:
    frame = _make_well(20)
    feats = build_z_drift_features(frame)
    assert list(feats.columns) == Z_DRIFT_FEATURES
    assert len(feats) == 20


def test_z_drift_features_no_nan() -> None:
    frame = _make_well(20)
    feats = build_z_drift_features(frame)
    assert not feats.isna().any().any()


def test_z_drift_offset_computation() -> None:
    frame = _make_well(20)
    feats = build_z_drift_features(frame)
    last_tvt = last_known_tvt_input_value(frame)
    ps_idx = 10
    z_at_ps = frame.loc[ps_idx, "Z"]
    expected_offset = last_tvt - z_at_ps
    assert feats.loc[0, "z_drift_offset_at_anchor"] == pytest.approx(expected_offset)
    assert feats["z_drift_offset_at_anchor"].nunique() == 1


def test_z_drift_implied_tvt_formula() -> None:
    frame = _make_well(20)
    feats = build_z_drift_features(frame)
    last_tvt = last_known_tvt_input_value(frame)
    ps_idx = 10
    z_at_ps = frame.loc[ps_idx, "Z"]
    offset = last_tvt - z_at_ps
    for i in range(20):
        expected = frame.loc[i, "Z"] + offset
        assert feats.loc[i, "z_drift_implied_tvt"] == pytest.approx(expected)


def test_z_drift_resid_formula() -> None:
    frame = _make_well(20)
    feats = build_z_drift_features(frame)
    last_tvt = last_known_tvt_input_value(frame)
    implied = feats["z_drift_implied_tvt"].values
    expected_resid = np.clip(implied - last_tvt, -100.0, 100.0)
    assert np.allclose(feats["z_drift_implied_tvt_resid"].values, expected_resid)


def test_z_drift_resid_clipping() -> None:
    frame = _make_well(20)
    feats = build_z_drift_features(frame)
    resid = feats["z_drift_implied_tvt_resid"].values
    assert np.all(np.abs(resid) <= 100.0)


def test_z_drift_all_nan_tvt_input() -> None:
    frame = _make_well(10)
    frame["TVT_input"] = np.nan
    feats = build_z_drift_features(frame)
    assert (feats["z_drift_offset_at_anchor"] == 0.0).all()
    assert (feats["z_drift_implied_tvt"] == 0.0).all()
    assert (feats["z_drift_implied_tvt_resid"] == 0.0).all()


def test_z_drift_zero_offset() -> None:
    frame = _make_well(20)
    ps_idx = 10
    z_at_ps = frame.loc[ps_idx, "Z"]
    frame["TVT_input"] = z_at_ps
    frame.loc[frame.index >= ps_idx, "TVT_input"] = np.nan
    feats = build_z_drift_features(frame)
    assert feats["z_drift_offset_at_anchor"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert np.allclose(feats["z_drift_implied_tvt"].values, frame["Z"].astype(float).values)


def test_build_features_with_z_drift() -> None:
    frame = _make_well(20)
    features = build_features(frame, include_z_drift=True)
    expected_cols = SAFE_NUMERIC_FEATURES + Z_DRIFT_FEATURES
    assert list(features.columns) == expected_cols
    assert "z_drift_offset_at_anchor" in features.columns
    assert "z_drift_implied_tvt" in features.columns
    assert "z_drift_implied_tvt_resid" in features.columns


def test_build_features_r1_plus_z_drift() -> None:
    frame = _make_well(20)
    features = build_features(frame, include_geometry=True, include_gr=True,
                             include_z_drift=True)
    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES + GR_FEATURES + Z_DRIFT_FEATURES
    assert list(features.columns) == expected_cols
    assert len(features.columns) == 6 + 9 + 3 + 3


def test_z_drift_single_row() -> None:
    frame = _make_well(1)
    frame.loc[0, "TVT_input"] = np.nan
    feats = build_z_drift_features(frame)
    assert len(feats) == 1
    assert feats.loc[0, "z_drift_offset_at_anchor"] == 0.0
    assert feats.loc[0, "z_drift_implied_tvt"] == 0.0
    assert feats.loc[0, "z_drift_implied_tvt_resid"] == 0.0
