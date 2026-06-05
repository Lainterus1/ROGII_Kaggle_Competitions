"""Generate predictions and a validated submission from a trained model."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.data_loading import sample_submission_path
from rogii.model_io import make_feature_flags, resolve_prediction_contract
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

    contract = resolve_prediction_contract(
        payload,
        cli_feature_flags=make_feature_flags(
            include_tvt_input=args.include_tvt_input,
            include_geometry=args.include_geometry,
            include_gr=args.include_gr,
            include_typewell=args.include_typewell,
        ),
        cli_residual_target=args.residual_target,
    )
    flags = contract.feature_flags

    submission = run_predict(
        args.data_dir,
        contract.model,
        include_tvt_input=flags["include_tvt_input"],
        include_geometry=flags["include_geometry"],
        include_gr=flags["include_gr"],
        include_typewell=flags["include_typewell"],
        residual_target=contract.residual_target,
        feature_columns=contract.feature_columns,
    )
    submission.to_csv(output, index=False)

    result = validate_submission(sample_submission_path(args.data_dir), output)
    print(f"Prediction mode: {'residual (delta->TVT)' if contract.residual_target else 'direct (TVT)'}")
    if contract.feature_columns is not None:
        print(f"Feature columns ({len(contract.feature_columns)}): {contract.feature_columns}")
    print(f"Submission rows: {result.rows}")
    print(f"Wrote submission: {output}")


if __name__ == "__main__":
    main()
