import numpy as np
import pandas as pd
import pytest

from rogii import schema
from rogii.features import (
    GEOMETRY_FEATURES,
    GR_FEATURES,
    SAFE_NUMERIC_FEATURES,
    TRAJECTORY_FEATURES,
    build_features,
    last_known_tvt_input_value,
)
from rogii.models import first_prediction_index, last_known_tvt_input


def test_schema_uses_confirmed_submission_and_target_columns() -> None:
    assert schema.TARGET_COLUMN == "TVT"
    assert schema.ID_COLUMN == "id"
    assert schema.PREDICTION_COLUMN == "tvt"


def test_last_known_baseline_uses_only_pre_ps_tvt_input() -> None:
    horizontal = pd.DataFrame(
        {
            "TVT_input": [100.0, 101.0, None, None],
            "TVT": [100.0, 101.0, 9999.0, 9998.0],
        }
    )

    assert first_prediction_index(horizontal) == 2
    assert last_known_tvt_input(horizontal) == 101.0


def test_r1_features_exclude_tvt_and_tvt_input_as_feature() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.linspace(1000, 1010, 10),
            "X": np.linspace(0, 10, 10),
            "Y": np.linspace(0, 10, 10),
            "Z": np.linspace(-100, -90, 10),
            "GR": np.linspace(80, 120, 10),
            "TVT": np.linspace(5000, 5010, 10),
            "TVT_input": [100.0, 101.0, 102.0, 103.0, 104.0, None, None, None, None, None],
        }
    )

    feats = build_features(horizontal, include_geometry=True, include_gr=True)

    assert "TVT" not in feats.columns
    assert "TVT_input" not in feats.columns
    assert "last_tvt_input" not in feats.columns


def test_residual_target_produces_valid_delta() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": [1000, 1001, 1002, 1003],
            "X": [0.0, 1.0, 2.0, 3.0],
            "Y": [0.0, 1.0, 2.0, 3.0],
            "Z": [-100.0, -99.0, -98.0, -97.0],
            "GR": [80.0, 90.0, 100.0, 110.0],
            "TVT": [5000.0, 5001.0, 5010.0, 5011.0],
            "TVT_input": [5000.0, 5001.0, None, None],
        }
    )

    last_tvt = last_known_tvt_input_value(horizontal)
    assert last_tvt == 5001.0

    tvt_post = horizontal.loc[2:, "TVT"].values
    delta = tvt_post - last_tvt
    assert delta[0] == pytest.approx(9.0)
    assert delta[1] == pytest.approx(10.0)


def test_typewell_features_exclude_tvt() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.linspace(1000, 1010, 10),
            "X": np.linspace(0, 10, 10),
            "Y": np.linspace(0, 10, 10),
            "Z": np.linspace(-100, -90, 10),
            "GR": np.linspace(80, 120, 10),
            "TVT": np.linspace(5000, 5010, 10),
            "TVT_input": [100.0, 101.0, 102.0, 103.0, 104.0, None, None, None, None, None],
        }
    )
    typewell = pd.DataFrame({
        "TVT": np.linspace(5000, 5100, 100),
        "GR": np.ones(100) * 100.0,
    })

    feats = build_features(horizontal, typewell=typewell, include_typewell=True,
                           include_geometry=True, include_gr=True)

    assert "TVT" not in feats.columns
    assert "TVT_input" not in feats.columns
    assert "last_tvt_input" not in feats.columns


def test_trajectory_features_exclude_tvt() -> None:
    horizontal = pd.DataFrame(
        {
            "MD": np.linspace(1000, 1010, 10),
            "X": np.linspace(0, 10, 10),
            "Y": np.linspace(0, 10, 10),
            "Z": np.linspace(-100, -90, 10),
            "GR": np.linspace(80, 120, 10),
            "TVT": np.linspace(5000, 5010, 10),
            "TVT_input": [100.0, 101.0, 102.0, 103.0, 104.0, None, None, None, None, None],
        }
    )

    feats = build_features(horizontal, include_trajectory=True, include_geometry=True, include_gr=True)

    assert "TVT" not in feats.columns
    assert "TVT_input" not in feats.columns
    assert "last_tvt_input" not in feats.columns

    expected_cols = SAFE_NUMERIC_FEATURES + GEOMETRY_FEATURES + TRAJECTORY_FEATURES + GR_FEATURES
    assert list(feats.columns) == expected_cols
