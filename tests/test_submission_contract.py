from pathlib import Path

import pytest

from rogii.submission import require_submission_file, validate_submission


def test_require_submission_file_accepts_existing_file(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,tvt\n1,0.0\n", encoding="utf-8")

    assert require_submission_file(submission) == submission


def test_validate_submission_accepts_exact_contract(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,tvt\na_1,0.0\na_2,0.0\n", encoding="utf-8")
    submission.write_text("id,tvt\na_1,10.0\na_2,11.0\n", encoding="utf-8")

    result = validate_submission(sample, submission)

    assert result.rows == 2
    assert result.id_column == "id"
    assert result.prediction_column == "tvt"


def test_validate_submission_rejects_id_order_change(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,tvt\na_1,0.0\na_2,0.0\n", encoding="utf-8")
    submission.write_text("id,tvt\na_2,11.0\na_1,10.0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_submission(sample, submission)


def test_validate_submission_rejects_non_numeric_prediction(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    sample.write_text("id,tvt\na_1,0.0\n", encoding="utf-8")
    submission.write_text("id,tvt\na_1,not_a_number\n", encoding="utf-8")

    with pytest.raises(ValueError):
        validate_submission(sample, submission)
