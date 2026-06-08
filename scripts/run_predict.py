"""Generate predictions and a validated submission from a trained model."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.data_loading import sample_submission_path
from rogii.model_io import make_feature_flags, resolve_prediction_contract
from rogii.postprocess import apply_postprocess_blend, parse_blend_weights
from rogii.predict import run_predict
from rogii.smoothing import apply_postprocessing
from rogii.submission import validate_submission


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--model", default="models/baseline_lgbm.pkl", help="Path to trained model")
    parser.add_argument("--output", default="outputs/submission.csv", help="Path for generated submission")
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
    parser.add_argument("--residual-target", action="store_true", help="Model predicts TVT delta; reconstruct TVT at prediction time")
    parser.add_argument("--baseline-method", default="flat", choices=["flat", "slope_md", "slope_z", "slope_recent", "wls"],
                        help="Baseline construction method for residual target (auto-detected from payload)")
    parser.add_argument("--savgol-smooth", action="store_true", help="Apply per-well Savgol smoothing to final predictions")
    parser.add_argument("--savgol-window", type=int, default=31, help="Savgol filter window length (odd, default 31)")
    parser.add_argument("--savgol-polyorder", type=int, default=2, help="Savgol filter polynomial order (default 2)")
    parser.add_argument("--tvt-clip", action="store_true",
                        help="Apply TVT clipping (uses bounds from model payload if available)")
    parser.add_argument("--postprocess-blend", action="store_true",
                        help="Apply 3-strategy blend: model + Z-physics + DTW GR matching")
    parser.add_argument("--blend-weights", type=str, default=None,
                        help="Blend weights for model,z_physics,dtw (e.g. '0.5,0.25,0.25'). Default: median")
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
            include_gr_dwt=args.include_gr_dwt,
            include_trajectory=args.include_trajectory,
            include_typewell=args.include_typewell,
            include_spatial=args.include_spatial,
            include_dtw=args.include_dtw,
            include_geology=args.include_geology,
            include_beam=args.include_beam,
            include_formation_plane=args.include_formation_plane,
            include_z_drift=args.include_z_drift,
        ),
        cli_residual_target=args.residual_target,
        cli_baseline_method=args.baseline_method,
    )
    flags = contract.feature_flags

    if contract.model_metadata and contract.model_metadata.get("type") == "tcn":
        from rogii.predict import predict_tcn
        from rogii.tcn_model import TCNModel

        meta = contract.model_metadata
        input_size = meta.get("input_size", len(meta["feature_columns"]))
        tcn_model = TCNModel(
            input_size=input_size,
            num_channels=meta["num_channels"],
            kernel_size=int(meta.get("kernel_size", 5)),
            dropout=float(meta.get("dropout", 0.1)),
        )
        tcn_model.load_state_dict(meta["state_dict"])
        scaler = meta["target_scaler"]
        window_size = meta["window_size"]

        submission = predict_tcn(
            args.data_dir,
            tcn_model,
            scaler,
            window_size,
            meta["feature_columns"],
            input_scaler=meta.get("input_scaler"),
            residual_target=contract.residual_target,
            baseline_method=contract.baseline_method,
        )
    else:
        submission = run_predict(
            args.data_dir,
            contract.models,
            include_tvt_input=flags["include_tvt_input"],
            include_geometry=flags["include_geometry"],
            include_gr=flags["include_gr"],
            include_gr_dwt=flags["include_gr_dwt"],
            include_trajectory=flags["include_trajectory"],
            include_typewell=flags["include_typewell"],
            include_spatial=flags["include_spatial"],
            include_dtw=flags["include_dtw"],
            include_geology=flags["include_geology"],
            include_beam=flags["include_beam"],
            include_formation_plane=flags["include_formation_plane"],
            include_z_drift=flags["include_z_drift"],
            residual_target=contract.residual_target,
            baseline_method=contract.baseline_method,
            feature_columns=contract.feature_columns,
        )

    apply_savgol = args.savgol_smooth
    apply_clip = args.tvt_clip
    blend_weights = parse_blend_weights(args.blend_weights)

    if args.postprocess_blend:
        submission = apply_postprocess_blend(submission, args.data_dir, weights=blend_weights)

    # Auto-detect clip bounds from payload
    clip_lower: float | None = None
    clip_upper: float | None = None
    if apply_clip and isinstance(payload, dict):
        clip_lower = payload.get("clip_lower")
        clip_upper = payload.get("clip_upper")
        if clip_lower is None or clip_upper is None:
            print("Warning: --tvt-clip requested but model payload has no clip bounds. Skipping clipping.")
            apply_clip = False

    if apply_clip or apply_savgol:
        submission = apply_postprocessing(
            submission,
            savgol_window=args.savgol_window if apply_savgol else None,
            savgol_polyorder=args.savgol_polyorder,
            clip_lower=clip_lower if apply_clip else None,
            clip_upper=clip_upper if apply_clip else None,
        )

    submission.to_csv(output, index=False)

    result = validate_submission(sample_submission_path(args.data_dir), output)
    print(f"Prediction mode: {'residual (delta->TVT)' if contract.residual_target else 'direct (TVT)'}")
    if contract.feature_columns is not None:
        print(f"Feature columns ({len(contract.feature_columns)}): {contract.feature_columns}")
    print(f"Baseline method: {contract.baseline_method}")
    if apply_savgol:
        print(f"Savgol smoothing: ON (window={args.savgol_window}, polyorder={args.savgol_polyorder})")
    else:
        print("Savgol smoothing: OFF")
    if apply_clip:
        print(f"TVT clipping: ON [{clip_lower:.2f}, {clip_upper:.2f}]")
    else:
        print("TVT clipping: OFF")
    if args.postprocess_blend:
        if blend_weights is not None:
            print(f"3-strategy blend: ON (weights: model={blend_weights[0]:.2f}, z_physics={blend_weights[1]:.2f}, dtw={blend_weights[2]:.2f})")
        else:
            print("3-strategy blend: ON (median)")
    else:
        print("3-strategy blend: OFF")
    print(f"Submission rows: {result.rows}")
    print(f"Wrote submission: {output}")


if __name__ == "__main__":
    main()
