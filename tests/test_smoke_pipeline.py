from pathlib import Path

import pytest

import rogii
from rogii.data_inventory import build_inventory
from rogii.models import predict_last_known_submission, validate_last_known_baseline
from rogii.submission import validate_submission


def _write_synthetic_data(root: Path) -> None:
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)
    (root / "sample_submission.csv").write_text("id,tvt\nwella_2,0.0\nwella_3,0.0\n", encoding="utf-8")
    (root / "train" / "wella__horizontal_well.csv").write_text(
        "MD,X,Y,Z,TVT,GR,TVT_input\n"
        "1,0,0,0,10,90,10\n"
        "2,1,0,0,11,91,11\n"
        "3,2,0,0,12,92,\n"
        "4,3,0,0,13,93,\n",
        encoding="utf-8",
    )
    (root / "train" / "wella__typewell.csv").write_text(
        "TVT,GR,Geology\n10,90,A\n11,91,A\n",
        encoding="utf-8",
    )
    (root / "test" / "wella__horizontal_well.csv").write_text(
        "MD,X,Y,Z,GR,TVT_input\n"
        "1,0,0,0,90,10\n"
        "2,1,0,0,91,11\n"
        "3,2,0,0,92,\n"
        "4,3,0,0,93,\n",
        encoding="utf-8",
    )
    (root / "test" / "wella__typewell.csv").write_text("TVT,GR\n10,90\n11,91\n", encoding="utf-8")


def test_package_imports() -> None:
    assert rogii.__version__ == "0.1.0"


def test_inventory_and_naive_submission_smoke(tmp_path: Path) -> None:
    _write_synthetic_data(tmp_path)
    inventory = build_inventory(tmp_path)

    assert inventory["sample_submission"] == {"columns": ["id", "tvt"], "rows": 2}
    assert inventory["splits"]["train"]["unique_wells"] == 1

    validation = validate_last_known_baseline(tmp_path)
    assert validation.wells == 1
    assert validation.rows == 2
    assert validation.rmse == pytest.approx(2.5**0.5)

    submission = predict_last_known_submission(tmp_path)
    assert submission["tvt"].tolist() == [11.0, 11.0]

    submission_path = tmp_path / "submission.csv"
    submission.to_csv(submission_path, index=False)
    assert validate_submission(tmp_path / "sample_submission.csv", submission_path).rows == 2
