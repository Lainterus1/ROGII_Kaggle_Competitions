"""Run LightGBM training with GroupKFold CV and generate a trained model."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.train import run_train


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-model", default="models/baseline_lgbm.pkl", help="Path for saved model")
    parser.add_argument("--include-tvt-input", action="store_true", help="Include last-known TVT_input as a feature")
    parser.add_argument("--include-geometry", action="store_true", help="Include geometry features relative to Prediction Start")
    parser.add_argument("--include-gr", action="store_true", help="Include GR-derived rolling/lag/envelope features")
    parser.add_argument("--include-typewell", action="store_true", help="Include typewell-reference GR residual and summary features")
    parser.add_argument("--residual-target", action="store_true", help="Train on TVT - last_tvt_input delta instead of raw TVT")
    return parser


def main() -> None:
    args = parse_args().parse_args()
    model_dir = Path(args.output_model).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    result = run_train(
        data_dir=args.data_dir,
        n_splits=args.n_splits,
        seed=args.seed,
        include_tvt_input=args.include_tvt_input,
        include_geometry=args.include_geometry,
        include_gr=args.include_gr,
        include_typewell=args.include_typewell,
        residual_target=args.residual_target,
    )

    payload: dict = {
        "model": result.model,
        "residual_target": result.residual_target,
        "include_geometry": args.include_geometry,
        "include_gr": args.include_gr,
        "include_typewell": args.include_typewell,
    }
    with open(args.output_model, "wb") as f:
        pickle.dump(payload, f)

    print(f"Target mode: {'residual (delta)' if result.residual_target else 'direct (TVT)'}")
    print(f"CV RMSE (mean ± std): {result.cv_rmse_mean:.6f} ± {result.cv_rmse_std:.6f}")
    print(f"CV fold scores: {[round(s, 6) for s in result.cv_rmse_folds]}")
    print(f"Train rows (post-PS): {result.train_rows}")
    print(f"Train wells: {result.train_wells}")
    print(f"Model saved: {args.output_model}")


if __name__ == "__main__":
    main()
