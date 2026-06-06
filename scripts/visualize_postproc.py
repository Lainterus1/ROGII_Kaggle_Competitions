"""Visualize Savgol smoothing and TVT clipping effect on per-well predictions.

Plots raw vs post-processed predictions for 2-3 wells, checks continuity
constraints, and saves figures to outputs/visualization/.
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
import numpy as np
import pandas as pd

from rogii.baseline import compute_baseline
from rogii.data_loading import list_well_ids, read_horizontal_well
from rogii.features import build_features, post_ps_mask
from rogii.model_io import validate_feature_columns
from rogii.smoothing import apply_postprocessing, compute_tvt_clip_bounds


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="Competition data directory")
    parser.add_argument("--model", default="models/baseline_lgbm.pkl", help="Path to trained model")
    parser.add_argument("--well-id", nargs="+", default=None,
                        help="Specific train well IDs to visualize (default: auto-select 3)")
    parser.add_argument("--output-dir", default="outputs/visualization", help="Output directory for plots")
    return parser


def predict_per_well(model, horizontal, feature_columns, residual_target, baseline_method):
    """Predict TVT for all rows in a single horizontal well."""
    use_tvt = False  # PredictPostPS only
    feats = build_features(
        horizontal,
        include_tvt_input=use_tvt,
        include_geometry=True,
        include_gr=True,
    )
    ordered_cols = validate_feature_columns(list(feats.columns), feature_columns)
    feats = feats.loc[:, ordered_cols]

    preds = model.predict(feats)
    if residual_target:
        baseline = compute_baseline(horizontal, method=baseline_method)
        mask = post_ps_mask(horizontal)
        for i in range(len(preds)):
            if mask[i] and np.isfinite(baseline[i]):
                preds[i] = baseline[i] + preds[i]
    return preds


def check_continuity(tvt_values, max_jump: float = 30.0):
    """Check that adjacent TVT values don't jump by more than max_jump."""
    jumps = np.abs(np.diff(tvt_values))
    violations = jumps > max_jump
    return {
        "max_jump": float(np.max(jumps)),
        "mean_jump": float(np.mean(jumps)),
        "num_violations": int(np.sum(violations)),
        "violation_indices": np.where(violations)[0].tolist()[:10],
    }


def main() -> None:
    if not HAS_MPL:
        print("matplotlib is not installed. Install with: pip install matplotlib")
        sys.exit(1)

    args = parse_args().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.model, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"] if isinstance(payload, dict) else payload
    residual_target = payload.get("residual_target", False) if isinstance(payload, dict) else False
    baseline_method = payload.get("baseline_method", "flat") if isinstance(payload, dict) else "flat"
    feature_columns = payload.get("feature_columns") if isinstance(payload, dict) else None

    # Select wells
    if args.well_id:
        well_ids = args.well_id
    else:
        train_wells = list_well_ids(args.data_dir, "train")
        rng = np.random.RandomState(42)
        well_ids = list(rng.choice(train_wells, min(3, len(train_wells)), replace=False))
    print(f"Visualizing wells: {well_ids}")

    # Post-processing configs to compare
    configs = [
        ("raw", None, 3, None, None),
        ("savgol w=11 p=2", 11, 2, None, None),
        ("savgol w=17 p=3", 17, 3, None, None),
        ("savgol w=25 p=3", 25, 3, None, None),
    ]

    # Add clip configs if bounds available
    try:
        low, high = compute_tvt_clip_bounds(args.data_dir)
        clip_label = f"clip [{low:.0f}, {high:.0f}]"
        configs.append((f"clip {clip_label}", None, 3, low, high))
        configs.append((f"clip {clip_label} + savgol w=17 p=3", 17, 3, low, high))
    except Exception:
        print("Warning: Could not compute clip bounds, skipping clip configs")

    for wid in well_ids:
        horizontal = read_horizontal_well(args.data_dir, "train", wid)
        if "TVT" not in horizontal.columns:
            print(f"  Skipping {wid}: no TVT column")
            continue

        mask = post_ps_mask(horizontal)
        post_ps_rows = horizontal.loc[mask].copy()
        if len(post_ps_rows) == 0:
            print(f"  Skipping {wid}: no post-PS rows")
            continue

        raw_preds = predict_per_well(model, horizontal, feature_columns, residual_target, baseline_method)
        post_preds = raw_preds[mask]

        fig, axes = plt.subplots(len(configs) + 1, 1, figsize=(14, 3.5 * (len(configs) + 1)))
        post_md = post_ps_rows["MD"].to_numpy(dtype=float)
        true_tvt = post_ps_rows["TVT"].to_numpy(dtype=float)

        # Plot 0: ground truth
        ax = axes[0]
        ax.plot(post_md, true_tvt, "g-", alpha=0.7, label="True TVT", linewidth=1.5)
        ax.set_title(f"Well {wid} — Ground Truth TVT (post-PS, {len(post_ps_rows)} rows)")
        ax.set_xlabel("MD (ft)")
        ax.set_ylabel("TVT (ft)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Build submission-like dataframe for post-processing
        post_ids = [f"{wid}_{i}" for i in horizontal.index[mask]]

        for plot_idx, (label, win, poly, cl_low, cl_high) in enumerate(configs):
            ax = axes[plot_idx + 1]
            sub = pd.DataFrame({"id": post_ids, "tvt": post_preds.copy()})

            pp_sub = apply_postprocessing(
                sub,
                savgol_window=win,
                savgol_polyorder=poly,
                clip_lower=cl_low,
                clip_upper=cl_high,
            )
            pp = pp_sub["tvt"].to_numpy(dtype=float)

            ax.plot(post_md, true_tvt, "g-", alpha=0.3, label="True TVT", linewidth=1)
            ax.plot(post_md, post_preds, "b-", alpha=0.5, label="Raw pred", linewidth=0.8)
            ax.plot(post_md, pp, "r-", alpha=0.8, label=f"Postproc: {label}", linewidth=1.5)

            if cl_low is not None:
                ax.axhline(y=cl_low, color="orange", linestyle="--", alpha=0.5, label=f"Clip {cl_low:.0f}")
                ax.axhline(y=cl_high, color="orange", linestyle="--", alpha=0.5, label=f"Clip {cl_high:.0f}")

            # Continuity check
            raw_cont = check_continuity(post_preds)
            pp_cont = check_continuity(pp)

            raw_rmse = np.sqrt(np.mean((post_preds - true_tvt) ** 2))
            pp_rmse = np.sqrt(np.mean((pp - true_tvt) ** 2))

            ax.set_title(f"{label} | RMSE: raw={raw_rmse:.2f} → pp={pp_rmse:.2f} "
                         f"| Max jump: raw={raw_cont['max_jump']:.1f} → pp={pp_cont['max_jump']:.1f}")
            ax.set_xlabel("MD (ft)")
            ax.set_ylabel("TVT (ft)")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = output_dir / f"postproc_{wid}.png"
        plt.savefig(out_path, dpi=100)
        plt.close()
        print(f"  Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
