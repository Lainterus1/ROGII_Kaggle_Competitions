"""Run LightGBM training with GroupKFold CV and generate a trained model."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.config import load_yaml_config
from rogii.model_io import build_model_payload, make_feature_flags
from rogii.train import run_train


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional YAML run config")
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--n-splits", type=int, help="Number of GroupKFold splits")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--output-model", help="Path for saved model")
    parser.add_argument("--include-tvt-input", action="store_true", help="Include last-known TVT_input as a feature")
    parser.add_argument("--include-geometry", action="store_true", help="Include geometry features relative to Prediction Start")
    parser.add_argument("--include-gr", action="store_true", help="Include GR-derived rolling/lag/envelope features")
    parser.add_argument("--include-typewell", action="store_true", help="Include typewell-reference GR residual and summary features")
    parser.add_argument("--residual-target", action="store_true", help="Train on TVT - last_tvt_input delta instead of raw TVT")
    return parser


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return value


def _bool_setting(cli_value: bool, section: dict[str, Any], key: str, default: bool = False) -> bool:
    return bool(cli_value or section.get(key, default))


def main() -> None:
    args = parse_args().parse_args()
    config = load_yaml_config(args.config) if args.config else {}
    run_config = _section(config, "run")
    model_config = _section(config, "model")
    feature_config = _section(config, "features")
    validation_config = _section(config, "validation")
    artifact_config = _section(config, "artifacts")

    n_splits = args.n_splits or int(validation_config.get("n_splits", 5))
    seed = args.seed or int(run_config.get("seed", 42))
    output_model = args.output_model or artifact_config.get("model_path", "models/baseline_lgbm.pkl")
    model_params = model_config.get("params", {})
    if not isinstance(model_params, dict):
        raise ValueError("Config section model.params must be a mapping")
    model_params = dict(model_params)
    model_params["random_state"] = seed

    include_tvt_input = _bool_setting(args.include_tvt_input, feature_config, "include_tvt_input")
    include_geometry = _bool_setting(args.include_geometry, feature_config, "include_geometry")
    include_gr = _bool_setting(args.include_gr, feature_config, "include_gr")
    include_typewell = _bool_setting(args.include_typewell, feature_config, "include_typewell")
    residual_target = _bool_setting(args.residual_target, feature_config, "residual_target")

    model_dir = Path(output_model).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    result = run_train(
        data_dir=args.data_dir,
        n_splits=n_splits,
        seed=seed,
        model_params=model_params,
        include_tvt_input=include_tvt_input,
        include_geometry=include_geometry,
        include_gr=include_gr,
        include_typewell=include_typewell,
        residual_target=residual_target,
    )

    feature_flags = make_feature_flags(
        include_tvt_input=include_tvt_input and not residual_target,
        include_geometry=include_geometry,
        include_gr=include_gr,
        include_typewell=include_typewell,
    )
    payload = build_model_payload(
        model=result.model,
        feature_columns=result.feature_columns,
        residual_target=result.residual_target,
        feature_flags=feature_flags,
        model_type=str(model_config.get("type", "lightgbm")),
        run_name=run_config.get("name"),
        seed=seed,
        n_splits=n_splits,
        cv_rmse_mean=result.cv_rmse_mean,
        cv_rmse_std=result.cv_rmse_std,
        cv_rmse_folds=result.cv_rmse_folds,
        train_rows=result.train_rows,
        train_wells=result.train_wells,
        config_path=args.config,
        model_params=model_params,
    )
    with open(output_model, "wb") as f:
        pickle.dump(payload, f)

    print(f"Target mode: {'residual (delta)' if result.residual_target else 'direct (TVT)'}")
    print(f"Feature columns ({len(result.feature_columns)}): {result.feature_columns}")
    print(f"CV RMSE (mean ± std): {result.cv_rmse_mean:.6f} ± {result.cv_rmse_std:.6f}")
    print(f"CV fold scores: {[round(s, 6) for s in result.cv_rmse_folds]}")
    print(f"Train rows (post-PS): {result.train_rows}")
    print(f"Train wells: {result.train_wells}")
    print(f"Model saved: {output_model}")


if __name__ == "__main__":
    main()
