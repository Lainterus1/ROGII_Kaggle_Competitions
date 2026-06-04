import pandas as pd

from rogii import schema
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
