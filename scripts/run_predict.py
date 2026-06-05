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
    parser.add_argument("--include-tvt-input", action="store_true", help="Include last-known TVT_input as a feature")
    parser.add_argument("--include-geometry", action="store_true", help="Include geometry features relative to Prediction Start")
    parser.add_argument("--include-gr", action="store_true", help="Include GR-derived rolling/lag/envelope features")
    parser.add_argument("--include-typewell", action="store_true", help="Include typewell-reference GR residual and summary features")
    parser.add_argument("--residual-target", action="store_true", help="Model predicts TVT delta; reconstruct TVT at prediction time")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as f:
        payload = pickle.load(f)

    if isinstance(payload, dict):
        model = payload["model"]
        residual_target = payload.get("residual_target", False)
        include_geometry = args.include_geometry or payload.get("include_geometry", False)
        include_gr = args.include_gr or payload.get("include_gr", False)
        include_typewell = args.include_typewell or payload.get("include_typewell", False)
    else:
        model = payload
        residual_target = False
        include_geometry = args.include_geometry
        include_gr = args.include_gr
        include_typewell = args.include_typewell

    if args.residual_target:
        residual_target = True

    submission = run_predict(
        args.data_dir,
        model,
        include_tvt_input=args.include_tvt_input,
        include_geometry=include_geometry,
        include_gr=include_gr,
        include_typewell=include_typewell,
        residual_target=residual_target,
    )
    submission.to_csv(output, index=False)

    result = validate_submission(sample_submission_path(args.data_dir), output)
    print(f"Prediction mode: {'residual (delta->TVT)' if residual_target else 'direct (TVT)'}")
    print(f"Submission rows: {result.rows}")
    print(f"Wrote submission: {output}")


if __name__ == "__main__":
    main()
