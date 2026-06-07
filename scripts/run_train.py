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
    parser.add_argument("--seed", type=int, default=None, help="Single random seed (use --seed-list for multi-seed)")
    parser.add_argument("--n-seeds", type=int, default=None, help="Number of seeds for multi-seed averaging (uses --seed as base)")
    parser.add_argument("--seed-list", type=str, default=None, help="Comma-separated list of seeds, e.g. '42,7,123'")
    parser.add_argument("--cv-strategy", default=None, choices=["group", "stratified"],
                        help="CV strategy: group (GroupKFold) or stratified (StratifiedGroupKFold)")
    parser.add_argument("--strat-tvt-bins", type=int, default=None,
                        help="Number of TVT bins for stratified CV (default 4)")
    parser.add_argument("--strat-spatial-clusters", type=int, default=None,
                        help="Number of spatial clusters for stratified CV (default 5)")
    parser.add_argument("--output-model", help="Path for saved model")
    parser.add_argument("--include-tvt-input", action="store_true", help="Include last-known TVT_input as a feature")
    parser.add_argument("--include-geometry", action="store_true", help="Include geometry features relative to Prediction Start")
    parser.add_argument("--include-gr", action="store_true", help="Include GR-derived rolling/lag/envelope features")
    parser.add_argument("--include-gr-dwt", action="store_true", help="Include causal GR DWT (wavelet) features")
    parser.add_argument("--include-trajectory", action="store_true", help="Include 3D trajectory kinematics features (automatically includes geometry)")
    parser.add_argument("--include-typewell", action="store_true", help="Include typewell-reference GR residual and summary features")
    parser.add_argument("--include-spatial", action="store_true", help="Include OOF spatial KNN features (pre-PS TVT_input only)")
    parser.add_argument("--include-dtw", action="store_true", help="Include DTW typewell alignment features")
    parser.add_argument("--include-geology", action="store_true", help="Include formation geology features from typewell")
    parser.add_argument("--include-beam", action="store_true", help="Include Numba JIT beam search stratigraphic alignment features")
    parser.add_argument("--include-formation-plane", action="store_true", help="Include KNN-imputed formation plane features")
    parser.add_argument("--include-z-drift", action="store_true", help="Include TVT-Z drift physics features (offset, implied TVT, resid)")
    parser.add_argument("--residual-target", action="store_true", help="Train on TVT - last_tvt_input delta instead of raw TVT")
    parser.add_argument("--baseline-method", default="flat", choices=["flat", "slope_md", "slope_z", "slope_recent", "wls"],
                        help="Baseline construction method for residual target")
    parser.add_argument("--eval-postproc", action="store_true",
                        help="Evaluate Savgol smoothing + TVT clipping on OOF CV predictions")
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

    # Resolve seed list
    if args.seed_list:
        seed_list = [int(s.strip()) for s in args.seed_list.split(",")]
    elif args.n_seeds:
        base = args.seed or int(run_config.get("seed", 42))
        seed_list = [base + i * 1000 for i in range(args.n_seeds)]
    else:
        config_seed_list = run_config.get("seed_list")
        if config_seed_list and isinstance(config_seed_list, list):
            seed_list = [int(s) for s in config_seed_list]
        else:
            seed = args.seed or int(run_config.get("seed", 42))
            seed_list = [seed]

    cv_strategy = args.cv_strategy or validation_config.get("strategy", "group")
    strat_tvt_bins = args.strat_tvt_bins or int(validation_config.get("strat_tvt_bins", 4))
    strat_spatial_clusters = args.strat_spatial_clusters or int(validation_config.get("strat_spatial_clusters", 5))

    output_model = args.output_model or artifact_config.get("model_path", "models/baseline_lgbm.pkl")
    model_params = model_config.get("params", {})
    if not isinstance(model_params, dict):
        raise ValueError("Config section model.params must be a mapping")
    model_params = dict(model_params)

    include_tvt_input = _bool_setting(args.include_tvt_input, feature_config, "include_tvt_input")
    include_geometry = _bool_setting(args.include_geometry, feature_config, "include_geometry")
    include_gr = _bool_setting(args.include_gr, feature_config, "include_gr")
    include_gr_dwt = _bool_setting(args.include_gr_dwt, feature_config, "include_gr_dwt")
    include_typewell = _bool_setting(args.include_typewell, feature_config, "include_typewell")
    include_trajectory = _bool_setting(args.include_trajectory, feature_config, "include_trajectory")
    include_spatial = _bool_setting(args.include_spatial, feature_config, "include_spatial")
    include_dtw = _bool_setting(args.include_dtw, feature_config, "include_dtw")
    include_geology = _bool_setting(args.include_geology, feature_config, "include_geology")
    include_beam = _bool_setting(args.include_beam, feature_config, "include_beam")
    include_formation_plane = _bool_setting(args.include_formation_plane, feature_config, "include_formation_plane")
    include_z_drift = _bool_setting(args.include_z_drift, feature_config, "include_z_drift")
    residual_target = _bool_setting(args.residual_target, feature_config, "residual_target")
    baseline_method = args.baseline_method or feature_config.get("baseline_method", "flat")

    model_dir = Path(output_model).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    result = run_train(
        data_dir=args.data_dir,
        n_splits=n_splits,
        seed_list=seed_list,
        cv_strategy=cv_strategy,
        strat_tvt_bins=strat_tvt_bins,
        strat_spatial_clusters=strat_spatial_clusters,
        model_params=model_params,
        include_tvt_input=include_tvt_input,
        include_geometry=include_geometry,
        include_gr=include_gr,
        include_gr_dwt=include_gr_dwt,
        include_trajectory=include_trajectory,
        include_typewell=include_typewell,
        include_spatial=include_spatial,
        include_dtw=include_dtw,
        include_geology=include_geology,
        include_beam=include_beam,
        include_formation_plane=include_formation_plane,
        include_z_drift=include_z_drift,
        residual_target=residual_target,
        baseline_method=baseline_method,
        eval_postproc=args.eval_postproc,
    )

    feature_flags = make_feature_flags(
        include_tvt_input=include_tvt_input and not residual_target,
        include_geometry=include_geometry,
        include_gr=include_gr,
        include_trajectory=include_trajectory,
        include_typewell=include_typewell,
        include_gr_dwt=include_gr_dwt,
        include_spatial=include_spatial,
        include_dtw=include_dtw,
        include_geology=include_geology,
        include_beam=include_beam,
        include_z_drift=include_z_drift,
    )
    payload = build_model_payload(
        models=result.models,
        feature_columns=result.feature_columns,
        residual_target=result.residual_target,
        baseline_method=result.baseline_method,
        feature_flags=feature_flags,
        model_type=str(model_config.get("type", "lightgbm")),
        run_name=run_config.get("name"),
        seed_list=result.seed_list,
        n_splits=n_splits,
        cv_strategy=result.cv_strategy,
        cv_rmse_mean=result.cv_rmse_mean,
        cv_rmse_std=result.cv_rmse_std,
        cv_rmse_folds=result.cv_rmse_folds,
        train_rows=result.train_rows,
        train_wells=result.train_wells,
        config_path=args.config,
        model_params=model_params,
        clip_lower=result.clip_bounds[0] if result.clip_bounds else None,
        clip_upper=result.clip_bounds[1] if result.clip_bounds else None,
    )
    with open(output_model, "wb") as f:
        pickle.dump(payload, f)

    print(f"CV strategy: {result.cv_strategy}")
    print(f"Seeds ({len(result.seed_list)}): {result.seed_list}")
    print(f"Target mode: {'residual (delta)' if result.residual_target else 'direct (TVT)'}")
    print(f"Feature columns ({len(result.feature_columns)}): {result.feature_columns}")
    print(f"CV RMSE (mean ± std): {result.cv_rmse_mean:.6f} ± {result.cv_rmse_std:.6f}")
    print(f"CV fold scores: {[round(s, 6) for s in result.cv_rmse_folds]}")
    print(f"Train rows (post-PS): {result.train_rows}")
    print(f"Train wells: {result.train_wells}")
    print(f"Model saved: {output_model}")


if __name__ == "__main__":
    main()
