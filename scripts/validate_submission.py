"""Validate a generated submission file against sample_submission.csv."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.data_loading import sample_submission_path
from rogii.submission import validate_submission


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--submission", required=True, help="Submission CSV to validate")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    result = validate_submission(sample_submission_path(args.data_dir), args.submission)
    print(
        "Submission valid: "
        f"rows={result.rows} id_column={result.id_column} prediction_column={result.prediction_column}"
    )


if __name__ == "__main__":
    main()
