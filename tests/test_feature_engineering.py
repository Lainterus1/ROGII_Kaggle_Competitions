import numpy as np
import pandas as pd
import pytest

from rogii.features import (
    GEOMETRY_FEATURES,
    GR_FEATURES,
    SAFE_NUMERIC_FEATURES,
    TYPEWELL_FEATURES,
    WITH_TVT_INPUT_FEATURES,
    build_features,
    build_geometry_features,
    build_gr_features,
    build_typewell_features,
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
