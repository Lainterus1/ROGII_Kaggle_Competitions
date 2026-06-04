from pathlib import Path

from rogii.submission import require_submission_file


def test_require_submission_file_accepts_existing_file(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,prediction\n1,0.0\n", encoding="utf-8")

    assert require_submission_file(submission) == submission
