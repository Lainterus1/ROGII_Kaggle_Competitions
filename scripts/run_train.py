"""Run LightGBM training with GroupKFold CV and generate a trained model."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.config import load_yaml_config
from rogii.mlflow_utils import default_experiment_name, end_run, log_artifact, log_metrics, log_params, setup_tracking, start_run
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
    parser.add_argument("--save-oof", action="store_true",
                        help="Save out-of-fold CV predictions to outputs/oof/")
    parser.add_argument("--model-type", default=None, choices=["lightgbm", "tcn"],
                        help="Model type: lightgbm (default) or tcn")
    parser.add_argument("--tcn-channels", default=None, type=str,
                        help="TCN channel sizes, comma-separated (e.g. '64,128,256')")
    parser.add_argument("--tcn-window", type=int, default=None, help="TCN sliding window size")
    parser.add_argument("--tcn-epochs", type=int, default=None, help="TCN max epochs")
    parser.add_argument("--tcn-batch-size", type=int, default=None, help="TCN batch size")
    parser.add_argument("--tcn-lr", type=float, default=None, help="TCN learning rate")
    parser.add_argument("--tcn-weight-decay", type=float, default=None, help="TCN weight decay")
    parser.add_argument("--tcn-kernel-size", "--tcn-kernel", dest="tcn_kernel_size", type=int, default=None,
                        help="TCN convolution kernel size")
    parser.add_argument("--tcn-dropout", type=float, default=None, help="TCN dropout")
    parser.add_argument("--tcn-patience", type=int, default=None, help="TCN early stopping patience")
    parser.add_argument("--tcn-device", default=None, choices=["cuda", "cpu"],
                        help="TCN device (cuda or cpu)")
    parser.add_argument("--tcn-stride", type=int, default=None,
                        help="Sliding window stride (default 1, increase for speed)")
    parser.add_argument("--tcn-workers", type=int, default=None,
                        help="DataLoader num_workers (default 2)")
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
    model_type = args.model_type or str(model_config.get("type", "lightgbm"))
    seed = args.seed or int(run_config.get("seed", 42))

    # --- MLflow tracking ---
    run_name = run_config.get("name", "rogii-run")
    exp_name = run_config.get("experiment_name", default_experiment_name())
    setup_tracking()
    mlflow_run = start_run(run_name=run_name, experiment_name=exp_name)
    mlflow_params = {
        "model_type": model_type,
        "n_splits": n_splits,
        "n_seeds": len(seed_list),
        "seed_list": str(seed_list),
        "cv_strategy": cv_strategy,
        "residual_target": residual_target,
        "baseline_method": baseline_method,
        "include_geometry": include_geometry,
        "include_gr": include_gr,
        "include_gr_dwt": include_gr_dwt,
        "include_tvt_input": include_tvt_input,
        "include_trajectory": include_trajectory,
        "include_typewell": include_typewell,
        "include_spatial": include_spatial,
        "include_dtw": include_dtw,
        "include_geology": include_geology,
        "include_beam": include_beam,
        "include_formation_plane": include_formation_plane,
        "include_z_drift": include_z_drift,
        "config_file": args.config or "none",
    }
    mlflow_params.update({f"model_param_{k}": v for k, v in model_params.items()})
    log_params(mlflow_params, run=mlflow_run)

    model_dir = Path(output_model).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    if model_type == "tcn":
        from rogii.train import train_tcn

        tcn_params = model_config.get("params", {})
        if not isinstance(tcn_params, dict):
            tcn_params = {}
        tcn_params = dict(tcn_params)

        tcn_channels_str = args.tcn_channels
        if tcn_channels_str:
            tcn_channels = tuple(int(x.strip()) for x in tcn_channels_str.split(","))
        else:
            tcn_channels = tuple(tcn_params.get("num_channels", [64, 128, 256, 128]))

        window_size = args.tcn_window or int(tcn_params.get("window_size", 64))
        epochs = args.tcn_epochs or int(tcn_params.get("epochs", 20))
        batch_size = args.tcn_batch_size or int(tcn_params.get("batch_size", 256))
        tcn_lr = args.tcn_lr or float(tcn_params.get("learning_rate", 1e-3))
        tcn_wd = args.tcn_weight_decay or float(tcn_params.get("weight_decay", 1e-4))
        tcn_kernel_size = args.tcn_kernel_size or int(tcn_params.get("kernel_size", 5))
        tcn_dropout = args.tcn_dropout if args.tcn_dropout is not None else float(tcn_params.get("dropout", 0.1))
        tcn_patience = args.tcn_patience or int(tcn_params.get("patience", 5))
        tcn_device = args.tcn_device or str(tcn_params.get("device", "cuda"))
        tcn_stride = args.tcn_stride if args.tcn_stride is not None else int(tcn_params.get("stride", 2))
        tcn_workers = args.tcn_workers if args.tcn_workers is not None else int(tcn_params.get("num_workers", 0))

        result = train_tcn(
            data_dir=args.data_dir,
            window_size=window_size,
            num_channels=tcn_channels,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            batch_size=batch_size,
            epochs=epochs,
            patience=tcn_patience,
            lr=tcn_lr,
            weight_decay=tcn_wd,
            n_splits=n_splits,
            seed=seed,
            device=tcn_device,
            residual_target=residual_target,
            baseline_method=baseline_method,
            stride=tcn_stride,
            num_workers=tcn_workers,
        )

        import torch
        model_cpu = result.models[0].cpu()
        tcn_state = {k: v for k, v in model_cpu.state_dict().items()}
        tcn_meta = result.tcn_metadata or {}
        payload = build_model_payload(
            models=result.models,
            feature_columns=result.feature_columns,
            residual_target=result.residual_target,
            baseline_method=result.baseline_method,
            feature_flags=None,
            model_type="tcn",
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
            tcn_state_dict=tcn_state,
            tcn_target_scaler=tcn_meta.get("y_scaler"),
            tcn_window_size=tcn_meta.get("window_size", window_size),
            tcn_feature_columns=result.feature_columns,
            tcn_num_channels=tcn_meta.get("num_channels", list(tcn_channels)),
            tcn_kernel_size=tcn_meta.get("kernel_size", tcn_kernel_size),
            tcn_dropout=tcn_meta.get("dropout", tcn_dropout),
            tcn_input_scaler=tcn_meta.get("x_scaler"),
            tcn_input_size=tcn_meta.get("input_size"),
        )
    else:
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
            include_formation_plane=include_formation_plane,
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

    # --- Log to MLflow ---
    mlflow_metrics = {
        "cv_rmse_mean": result.cv_rmse_mean,
        "cv_rmse_std": result.cv_rmse_std,
        "train_rows": result.train_rows,
        "train_wells": result.train_wells,
        "n_features": len(result.feature_columns),
    }
    for i, score in enumerate(result.cv_rmse_folds):
        mlflow_metrics[f"cv_rmse_fold_{i + 1}"] = score
    log_metrics(mlflow_metrics, run=mlflow_run)
    log_artifact(output_model, run=mlflow_run)
    end_run(mlflow_run)

    if args.save_oof and result.oof_df is not None:
        from rogii.oof import save_oof
        run_name = run_config.get("name", "unknown")
        strategy = str(model_config.get("type", "lgbm"))
        oof_path = save_oof(result.oof_df, "outputs", f"{run_name}_{strategy}")
        print(f"OOF saved: {oof_path}")


if __name__ == "__main__":
    main()
