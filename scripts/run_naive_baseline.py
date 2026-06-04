"""Run the last-known-TVT naive baseline and write a valid submission."""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.data_loading import sample_submission_path
from rogii.models import predict_last_known_submission, validate_last_known_baseline
from rogii.submission import validate_submission


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--output", default="outputs/submission.csv", help="Path for generated submission")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    validation = validate_last_known_baseline(args.data_dir)
    submission = predict_last_known_submission(args.data_dir)
    submission.to_csv(output, index=False)
    result = validate_submission(sample_submission_path(args.data_dir), output)

    print(f"Naive validation RMSE: {validation.rmse:.6f}")
    print(f"Naive validation rows: {validation.rows}")
    print(f"Naive validation wells: {validation.wells}")
    print(f"Submission rows: {result.rows}")
    print(f"Wrote submission: {output}")


if __name__ == "__main__":
    main()
