"""Run LightGBM/TCN training with GroupKFold CV and generate a trained model."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.config import load_yaml_config
from rogii.mlflow_utils import (
    default_experiment_name, end_run, log_artifact,
    log_metrics, log_params, setup_tracking, start_run,
)
from rogii.model_io import build_model_payload, make_feature_flags
from rogii.train import run_train

# ---------------------------------------------------------------------------
# Feature keys shared by CLI args, config sections, and downstream kwargs
# ---------------------------------------------------------------------------
_FEATURE_KEYS = [
    "include_tvt_input", "include_geometry", "include_gr", "include_gr_dwt",
    "include_trajectory", "include_typewell", "include_spatial", "include_dtw",
    "include_geology", "include_beam", "include_formation_plane", "include_z_drift",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section must be a mapping: {name}")
    return value


def _val(cli, section: dict[str, Any], key: str, default, coerce=None):
    """Return CLI value if provided (not None), else config[key] or default."""
    if cli is not None:
        return coerce(cli) if coerce else cli
    v = section.get(key, default)
    return coerce(v) if coerce and v is not None else v


def _resolve_seed_list(args, run_cfg: dict[str, Any]) -> list[int]:
    if args.seed_list:
        return [int(s.strip()) for s in args.seed_list.split(",")]
    if args.n_seeds:
        base = _val(args.seed, run_cfg, "seed", 42, int)
        return [base + i * 1000 for i in range(args.n_seeds)]
    config_list = run_cfg.get("seed_list")
    if config_list and isinstance(config_list, list):
        return [int(s) for s in config_list]
    return [_val(args.seed, run_cfg, "seed", 42, int)]


def _resolve_feature_flags(args, feat_cfg: dict[str, Any]) -> dict:
    """Return {feature_key: bool, ..., 'residual_target': bool, 'baseline_method': str}."""
    ff = {k: bool(getattr(args, k, False) or feat_cfg.get(k, False)) for k in _FEATURE_KEYS}
    ff["residual_target"] = bool(args.residual_target or feat_cfg.get("residual_target", False))
    ff["baseline_method"] = args.baseline_method or feat_cfg.get("baseline_method", "flat")
    return ff


def _resolve_tcn_config(args, model_cfg: dict[str, Any]) -> dict[str, Any]:
    raw = model_cfg.get("params", {})
    tc = dict(raw) if isinstance(raw, dict) else {}

    def _i(cli, key, default):
        return _val(cli, tc, key, default, int)

    def _f(cli, key, default):
        return _val(cli, tc, key, default, float)

    return {
        "num_channels": (
            tuple(int(x.strip()) for x in args.tcn_channels.split(","))
            if args.tcn_channels else tuple(tc.get("num_channels", [64, 128, 256, 128]))
        ),
        "window_size": _i(args.tcn_window, "window_size", 64),
        "epochs": _i(args.tcn_epochs, "epochs", 20),
        "batch_size": _i(args.tcn_batch_size, "batch_size", 256),
        "lr": _f(args.tcn_lr, "learning_rate", 1e-3),
        "weight_decay": _f(args.tcn_weight_decay, "weight_decay", 1e-4),
        "kernel_size": _i(args.tcn_kernel_size, "kernel_size", 5),
        "dropout": _f(args.tcn_dropout, "dropout", 0.1),
        "patience": _i(args.tcn_patience, "patience", 5),
        "device": args.tcn_device or str(tc.get("device", "cuda")),
        "stride": _i(args.tcn_stride, "stride", 2),
        "num_workers": _i(args.tcn_workers, "num_workers", 0),
    }


def _report_results(result, output_model: str, mlflow_run, run_name: str,
                    model_type: str, save_oof: bool) -> None:
    print(f"CV strategy: {result.cv_strategy}")
    print(f"Seeds ({len(result.seed_list)}): {result.seed_list}")
    print(f"Target mode: {'residual (delta)' if result.residual_target else 'direct (TVT)'}")
    print(f"Feature columns ({len(result.feature_columns)}): {result.feature_columns}")
    print(f"CV RMSE (mean ± std): {result.cv_rmse_mean:.6f} ± {result.cv_rmse_std:.6f}")
    print(f"CV fold scores: {[round(s, 6) for s in result.cv_rmse_folds]}")
    print(f"Train rows (post-PS): {result.train_rows}")
    print(f"Train wells: {result.train_wells}")
    print(f"Model saved: {output_model}")

    metrics = {
        "cv_rmse_mean": result.cv_rmse_mean,
        "cv_rmse_std": result.cv_rmse_std,
        "train_rows": result.train_rows,
        "train_wells": result.train_wells,
        "n_features": len(result.feature_columns),
    }
    for i, score in enumerate(result.cv_rmse_folds):
        metrics[f"cv_rmse_fold_{i + 1}"] = score
    log_metrics(metrics, run=mlflow_run)
    log_artifact(output_model, run=mlflow_run)
    end_run(mlflow_run)

    if save_oof and result.oof_df is not None:
        from rogii.oof import save_oof
        oof_path = save_oof(result.oof_df, "outputs", f"{run_name}_{model_type}")
        print(f"OOF saved: {oof_path}")


def _common_payload(result, n_splits: int, run_cfg: dict[str, Any],
                    config_path: str | None) -> dict[str, Any]:
    return dict(
        models=result.models,
        feature_columns=result.feature_columns,
        residual_target=result.residual_target,
        baseline_method=result.baseline_method or "flat",
        run_name=run_cfg.get("name"),
        seed_list=result.seed_list,
        n_splits=n_splits,
        cv_strategy=result.cv_strategy,
        cv_rmse_mean=result.cv_rmse_mean,
        cv_rmse_std=result.cv_rmse_std,
        cv_rmse_folds=result.cv_rmse_folds,
        train_rows=result.train_rows,
        train_wells=result.train_wells,
        config_path=config_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args().parse_args()
    config = load_yaml_config(args.config) if args.config else {}
    run_cfg = _section(config, "run")
    model_cfg = _section(config, "model")
    feat_cfg = _section(config, "features")
    val_cfg = _section(config, "validation")
    art_cfg = _section(config, "artifacts")

    # --- Resolve config values ---
    n_splits = _val(args.n_splits, val_cfg, "n_splits", 5, int)
    seed_list = _resolve_seed_list(args, run_cfg)
    seed = _val(args.seed, run_cfg, "seed", 42, int)
    cv_strategy = args.cv_strategy or val_cfg.get("strategy", "group")
    strat_tvt_bins = _val(args.strat_tvt_bins, val_cfg, "strat_tvt_bins", 4, int)
    strat_spatial_clusters = _val(args.strat_spatial_clusters, val_cfg, "strat_spatial_clusters", 5, int)
    model_type = args.model_type or str(model_cfg.get("type", "lightgbm"))
    output_model = args.output_model or art_cfg.get("model_path", "models/baseline_lgbm.pkl")
    model_params = model_cfg.get("params", {})
    if not isinstance(model_params, dict):
        raise ValueError("Config section model.params must be a mapping")
    model_params = dict(model_params)

    # Feature flags (dict with all 12 feature keys + residual_target + baseline_method)
    ff = _resolve_feature_flags(args, feat_cfg)
    residual_target = ff.pop("residual_target")
    baseline_method = ff.pop("baseline_method")

    # --- MLflow setup ---
    run_name = run_cfg.get("name", "rogii-run")
    exp_name = run_cfg.get("experiment_name", default_experiment_name())
    setup_tracking()
    mlflow_run = start_run(run_name=run_name, experiment_name=exp_name)
    log_params({
        "model_type": model_type, "n_splits": n_splits, "n_seeds": len(seed_list),
        "seed_list": str(seed_list), "cv_strategy": cv_strategy,
        "residual_target": residual_target, "baseline_method": baseline_method,
        "config_file": args.config or "none",
        **ff,
        **{f"model_param_{k}": v for k, v in model_params.items()},
    }, run=mlflow_run)

    model_dir = Path(output_model).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    if model_type == "tcn":
        from rogii.train import train_tcn

        tcn = _resolve_tcn_config(args, model_cfg)

        result = train_tcn(
            data_dir=args.data_dir,
            n_splits=n_splits,
            seed=seed,
            residual_target=residual_target,
            baseline_method=baseline_method,
            **tcn,
        )

        import torch
        tcn_state = {k: v for k, v in result.models[0].cpu().state_dict().items()}
        tcn_meta = result.tcn_metadata or {}
        payload = build_model_payload(
            **_common_payload(result, n_splits, run_cfg, args.config),
            feature_flags=None,
            model_type="tcn",
            tcn_state_dict=tcn_state,
            tcn_target_scaler=tcn_meta.get("y_scaler"),
            tcn_window_size=tcn_meta.get("window_size", tcn["window_size"]),
            tcn_feature_columns=result.feature_columns,
            tcn_num_channels=tcn_meta.get("num_channels", list(tcn["num_channels"])),
            tcn_kernel_size=tcn_meta.get("kernel_size", tcn["kernel_size"]),
            tcn_dropout=tcn_meta.get("dropout", tcn["dropout"]),
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
            residual_target=residual_target,
            baseline_method=baseline_method,
            eval_postproc=args.eval_postproc,
            **ff,
        )

        ff_lgbm = dict(ff)
        ff_lgbm["include_tvt_input"] = ff["include_tvt_input"] and not residual_target
        feature_flags = make_feature_flags(**ff_lgbm)

        payload = build_model_payload(
            **_common_payload(result, n_splits, run_cfg, args.config),
            feature_flags=feature_flags,
            model_type=model_type,
            model_params=model_params,
            clip_lower=result.clip_bounds[0] if result.clip_bounds else None,
            clip_upper=result.clip_bounds[1] if result.clip_bounds else None,
        )

    with open(output_model, "wb") as f:
        pickle.dump(payload, f)

    _report_results(result, output_model, mlflow_run, run_name, model_type, args.save_oof)


if __name__ == "__main__":
    main()
