from pathlib import Path

import numpy as np
import pytest

from rogii.features import SAFE_NUMERIC_FEATURES
from rogii.predict import run_predict


class ConstantModel:
    def __init__(self) -> None:
        self.seen_columns: list[str] | None = None

    def predict(self, frame):
        self.seen_columns = list(frame.columns)
        return np.ones(len(frame)) * 123.0


def _write_predict_data(root: Path) -> None:
    (root / "test").mkdir(parents=True)
    (root / "sample_submission.csv").write_text("id,tvt\nwella_2,0.0\n", encoding="utf-8")
    (root / "test" / "wella__horizontal_well.csv").write_text(
        "MD,X,Y,Z,GR,TVT_input\n"
        "1,0,0,0,90,10\n"
        "2,1,0,0,91,11\n"
        "3,2,0,0,92,\n",
        encoding="utf-8",
    )


def test_run_predict_uses_saved_feature_column_order(tmp_path: Path) -> None:
    _write_predict_data(tmp_path)
    model = ConstantModel()

    result = run_predict(tmp_path, model, feature_columns=SAFE_NUMERIC_FEATURES)

    assert result["tvt"].tolist() == [123.0]
    assert model.seen_columns == SAFE_NUMERIC_FEATURES


def test_run_predict_rejects_feature_column_mismatch(tmp_path: Path) -> None:
    _write_predict_data(tmp_path)

    with pytest.raises(ValueError, match="Feature columns do not match"):
        run_predict(tmp_path, ConstantModel(), feature_columns=["MD", "GR"])
