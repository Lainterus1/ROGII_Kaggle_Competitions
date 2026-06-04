"""Generate predictions and a validated submission from a trained model."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.data_loading import sample_submission_path
from rogii.predict import run_predict
from rogii.submission import validate_submission


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--model", default="models/baseline_lgbm.pkl", help="Path to trained model")
    parser.add_argument("--output", default="outputs/submission.csv", help="Path for generated submission")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    submission = run_predict(args.data_dir, model)
    submission.to_csv(output, index=False)

    result = validate_submission(sample_submission_path(args.data_dir), output)
    print(f"Submission rows: {result.rows}")
    print(f"Wrote submission: {output}")


if __name__ == "__main__":
    main()
