import numpy as np
import pandas as pd
import pytest

from rogii.features import (
    SAFE_NUMERIC_FEATURES,
    build_features,
    get_feature_set_name,
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


def test_gr_is_missing_flag() -> None:
    frame = _make_well(10, gr_nan_after=5)
    features = build_features(frame)
    assert (features.loc[0:4, "GR_is_missing"] == 0).all()
    assert (features.loc[5:9, "GR_is_missing"] == 1).all()


def test_md_delta_first_row_zero() -> None:
    frame = _make_well(10)
    features = build_features(frame)
    assert features.loc[0, "MD_delta"] == 0.0


def test_md_relative_range() -> None:
    frame = _make_well(10)
    features = build_features(frame)
    assert features.loc[0, "MD_relative"] == pytest.approx(0.0)
    assert features.loc[9, "MD_relative"] == pytest.approx(1.0)


def test_row_position_range() -> None:
    frame = _make_well(10)
    features = build_features(frame)
    assert features.loc[0, "row_position"] == pytest.approx(0.0)
    assert features.loc[9, "row_position"] == pytest.approx(1.0)


def test_single_row_well() -> None:
    frame = _make_well(1)
    features = build_features(frame)
    assert len(features) == 1
    assert features.loc[0, "MD_delta"] == 0.0
    assert features.loc[0, "MD_relative"] == 0.0
    assert features.loc[0, "row_position"] == 0.0


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
