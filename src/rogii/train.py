"""Training entry points for the ROGII ML baseline."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from rogii.baseline import compute_baseline
from rogii.data_loading import list_well_ids, read_horizontal_well, read_typewell
from rogii.features import SAFE_NUMERIC_FEATURES, build_features, last_known_tvt_input_value, post_ps_mask
from rogii.metrics import rmse


@dataclass(frozen=True)
class PostprocResult:
    raw_rmse: float
    postproc_rmse: float
    savgol_window: int | None
    savgol_polyorder: int
    clip_lower: float | None
    clip_upper: float | None
    label: str


@dataclass(frozen=True)
class TrainResult:
    model: LGBMRegressor
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_rmse_folds: list[float]
    train_rows: int
    train_wells: int
    feature_columns: list[str]
    residual_target: bool = False
    baseline_method: str = "flat"
    postproc_results: list[PostprocResult] | None = None
    clip_bounds: tuple[float, float] | None = None


def _collect_train_post_ps(
    data_dir: str | Path,
    include_tvt_input: bool = False,
    include_geometry: bool = False,
    include_gr: bool = False,
    include_gr_dwt: bool = False,
    include_trajectory: bool = False,
    include_typewell: bool = False,
    include_spatial: bool = False,
    include_dtw: bool = False,
    include_geology: bool = False,
    include_beam: bool = False,
    include_z_drift: bool = False,
    residual_target: bool = False,
    baseline_method: str = "flat",
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[tuple[str, int]], np.ndarray]:
    features_list: list[pd.DataFrame] = []
    targets: list[float] = []
    groups: list[int] = []
    well_ids_used: list[str] = []
    row_metadata: list[tuple[str, int]] = []
    row_baselines: list[float] = []
    well_index = 0

    well_ids = list_well_ids(data_dir, "train")
    total = len(well_ids)
    print(f"[1/3] Loading {total} train wells ...")

    use_tvt_feature = include_tvt_input and not residual_target

    for i, well_id in enumerate(well_ids):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{total} wells loaded, {len(targets)} rows collected")
        horizontal = read_horizontal_well(data_dir, "train", well_id)
        if "TVT" not in horizontal.columns:
            continue
        mask = post_ps_mask(horizontal)
        if not mask.any():
            continue

        last_tvt = last_known_tvt_input_value(horizontal)

        typewell_frame = read_typewell(data_dir, "train", well_id) if (include_typewell or include_dtw or include_geology or include_beam) else None

        feats = build_features(
            horizontal,
            include_tvt_input=use_tvt_feature,
            include_geometry=include_geometry,
            include_gr=include_gr,
            include_gr_dwt=include_gr_dwt,
            include_trajectory=include_trajectory,
            typewell=typewell_frame,
            include_typewell=include_typewell,
            include_dtw=include_dtw,
            include_geology=include_geology,
            include_beam=include_beam,
            include_z_drift=include_z_drift,
        )
        post_feats = feats.loc[mask].copy()
        post_target = horizontal.loc[mask, "TVT"].astype(float)

        valid = post_target.notna() & np.isfinite(post_target)
        if not valid.any():
            continue

        post_feats = post_feats.loc[valid]
        post_target = post_target.loc[valid]

        # Compute baseline for reconstruction during post-processing evaluation
        baseline_vals = np.zeros(len(post_target), dtype=float)
        if residual_target:
            baseline = compute_baseline(horizontal, method=baseline_method)
            baseline_series = pd.Series(baseline, index=horizontal.index)
            baseline_post = baseline_series.loc[post_target.index]
            if baseline_post.isna().any():
                continue
            baseline_vals = baseline_post.to_numpy(dtype=float)
            post_target = post_target - baseline_post

        # Track row index within the well for per-well post-processing
        masked_indices = horizontal.index[mask][valid]

        features_list.append(post_feats)
        targets.extend(post_target.tolist())
        groups.extend([well_index] * len(post_feats))
        well_ids_used.append(well_id)
        for j, row_idx in enumerate(masked_indices):
            row_metadata.append((well_id, int(row_idx)))
            row_baselines.append(float(baseline_vals[j]))
        well_index += 1

    if not features_list:
        raise ValueError("No train post-PS data found")

    print(f"  Done: {well_index} wells, {len(targets)} post-PS rows")
    X = pd.concat(features_list, ignore_index=True)
    y = np.array(targets, dtype=float)
    g = np.array(groups, dtype=int)
    row_baselines_arr = np.array(row_baselines, dtype=float)
    return X, y, g, well_ids_used, row_metadata, row_baselines_arr


def _spatial_fold_features(
    data_dir: str | Path,
    X_base: pd.DataFrame,
    row_mask: np.ndarray,
    oof_well_ids: list[str],
) -> pd.DataFrame:
    """Build spatial KNN features for masked rows using OOF wells as reference."""
    from rogii.spatial_features import build_pre_ps_reference, build_spatial_knn_features
    ref = build_pre_ps_reference(data_dir, "train", oof_well_ids)
    query = X_base.loc[row_mask, ["X", "Y", "Z"]]
    return build_spatial_knn_features(ref, query)


def _formation_plane_fold_features(
    X_fold: pd.DataFrame,
    fold_groups: np.ndarray,
    well_ids_used: list[str],
    oof_reference: pd.DataFrame,
) -> pd.DataFrame:
    """Build formation plane features for a fold using OOF well reference.

    Calls impute_formations once per unique well in the fold, then broadcasts
    well-level imputed values to all rows using per-row Z from X_fold.
    """
    from rogii.formation_plane import (
        FORMATION_NAMES, FORMATION_PLANE_FEATURES, impute_formations,
    )

    unique_wi = np.unique(fold_groups)
    fp_imputed: dict[int, dict[str, float]] = {}

    for wi in unique_wi:
        wid = well_ids_used[wi]
        wy_xy = X_fold.loc[fold_groups == wi, ["X", "Y"]].median().to_numpy(dtype=float).reshape(1, -1)
        est = impute_formations(oof_reference, wy_xy)
        fp_imputed[int(wi)] = {
            f"fp_{f.lower()}": float(est[f"fp_{f.lower()}"][0])
            for f in FORMATION_NAMES
        }
        fp_imputed[int(wi)]["knn_mean_dist"] = float(est["knn_mean_dist"][0])
        fp_imputed[int(wi)]["knn_std_ancc"] = float(est["knn_std_ancc"][0])
        fp_imputed[int(wi)]["knn_std_buda"] = float(est["knn_std_buda"][0])

    z = X_fold["Z"].to_numpy(dtype=float)
    n_rows = len(X_fold)
    feats = pd.DataFrame(index=X_fold.index)

    # Raw imputed depths (6)
    for f in FORMATION_NAMES:
        key = f"fp_{f.lower()}"
        feats[key] = np.array([fp_imputed[int(g)][key] for g in fold_groups], dtype=float)

    # Z-relative (6)
    for f in FORMATION_NAMES:
        key_raw = f"fp_{f.lower()}"
        key_z = f"fp_z_vs_{f.lower()}"
        feats[key_z] = z - feats[key_raw].to_numpy(dtype=float)

    # Distance to nearest formation (1)
    dists = np.full(n_rows, np.inf, dtype=float)
    for f in FORMATION_NAMES:
        key_raw = f"fp_{f.lower()}"
        d = np.abs(z - feats[key_raw].to_numpy(dtype=float))
        dists = np.minimum(dists, d)
    feats["fp_nearest_dist"] = dists

    # Formation thicknesses (5)
    pairs = [
        ("ANCC", "ASTNU"), ("ASTNU", "ASTNL"),
        ("ASTNL", "EGFDU"), ("EGFDU", "EGFDL"), ("EGFDL", "BUDA"),
    ]
    for upper, lower in pairs:
        col = f"fp_thick_{upper.lower()}_{lower.lower()}"
        key_lower = f"fp_{lower.lower()}"
        key_upper = f"fp_{upper.lower()}"
        feats[col] = feats[key_upper].to_numpy(dtype=float) - feats[key_lower].to_numpy(dtype=float)

    # KNN uncertainty (3)
    for knn_key in ["knn_mean_dist", "knn_std_ancc", "knn_std_buda"]:
        fp_key = f"fp_{knn_key}"
        feats[fp_key] = np.array([fp_imputed[int(g)][knn_key] for g in fold_groups], dtype=float)

    # Ensure exactly the expected features in the right order
    return feats[list(FORMATION_PLANE_FEATURES)]


def _build_oof_per_well(
    oof_rows: list[tuple[str, int, float, float, float]],
) -> pd.DataFrame:
    """Build per-well OOF prediction dataframe for post-processing evaluation.

    Columns: well_id, row_idx, y_true (delta or full), y_pred (delta or full), baseline.
    When residual_target is active, y_true/y_pred are deltas and baseline > 0.
    """
    return pd.DataFrame(oof_rows, columns=["well_id", "row_idx", "y_true", "y_pred", "baseline"])


def _reconstruct_full(oof_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct full TVT from delta + baseline."""
    return (
        oof_df["y_true"].to_numpy(dtype=float) + oof_df["baseline"].to_numpy(dtype=float),
        oof_df["y_pred"].to_numpy(dtype=float) + oof_df["baseline"].to_numpy(dtype=float),
    )


def _evaluate_postproc_config(
    oof_df: pd.DataFrame,
    savgol_window: int | None,
    savgol_polyorder: int,
    clip_lower: float | None,
    clip_upper: float | None,
    label: str,
) -> PostprocResult:
    """Evaluate a single post-processing config on OOF predictions.

    Works in full TVT space: reconstructs delta->full, clips, smooths,
    computes RMSE in full TVT.
    """
    oof_copy = oof_df.copy()

    # Reconstruct to full TVT
    oof_copy["y_true_full"] = oof_copy["y_true"] + oof_copy["baseline"]
    oof_copy["y_pred_full"] = oof_copy["y_pred"] + oof_copy["baseline"]

    # Sort by (well_id, row_idx) for per-well processing
    oof_copy = oof_copy.sort_values(["well_id", "row_idx"]).reset_index(drop=True)

    # Per-well clip + smooth
    y_pred_pp = np.zeros(len(oof_copy), dtype=float)
    for wid, group in oof_copy.groupby("well_id", sort=False):
        preds = group["y_pred_full"].to_numpy(dtype=float)
        if clip_lower is not None and clip_upper is not None:
            preds = preds.clip(clip_lower, clip_upper)
        if savgol_window is not None and len(preds) > savgol_window:
            from scipy.signal import savgol_filter
            preds = savgol_filter(preds, savgol_window, savgol_polyorder)
        y_pred_pp[group.index] = preds

    y_true_full = oof_copy["y_true_full"].to_numpy(dtype=float)
    y_pred_raw_full = oof_copy["y_pred_full"].to_numpy(dtype=float)

    raw = rmse(y_true_full, y_pred_raw_full)
    pp = rmse(y_true_full, y_pred_pp)

    return PostprocResult(
        raw_rmse=raw,
        postproc_rmse=pp,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
        label=label,
    )


def evaluate_postprocessing(
    oof_rows: list[tuple[str, int, float, float, float]],
    savgol_windows: list[int],
    savgol_polyorders: list[int],
    clip_configs: list[tuple[float | None, float | None, str]],
) -> list[PostprocResult]:
    """Evaluate multiple post-processing configurations on OOF predictions.

    Args:
        oof_rows: List of (well_id, row_idx, y_true, y_pred, baseline) tuples.
            y_true/y_pred are in delta space (if residual) or full TVT space.
            baseline is the reconstruction constant (0 for non-residual mode).
        savgol_windows: Savgol window sizes to test.
        savgol_polyorders: Savgol polyorders to test.
        clip_configs: List of (lower, upper, label) clipping configs.

    Returns:
        List of PostprocResult sorted by postproc_rmse ascending.
        All RMSE values are in FULL TVT space.
    """
    oof_df = _build_oof_per_well(oof_rows)
    y_true_full, y_pred_full = _reconstruct_full(oof_df)

    results: list[PostprocResult] = []

    # No postprocessing baseline
    raw = rmse(y_true_full, y_pred_full)
    results.append(PostprocResult(
        raw_rmse=raw, postproc_rmse=raw,
        savgol_window=None, savgol_polyorder=0,
        clip_lower=None, clip_upper=None,
        label="raw (no postproc)",
    ))

    # Clip-only configs
    for clip_lower, clip_upper, clip_label in clip_configs:
        if clip_lower is not None:
            r = _evaluate_postproc_config(oof_df, None, 0, clip_lower, clip_upper, f"clip {clip_label}")
            results.append(r)

    # Smooth-only configs
    for w in savgol_windows:
        for p in savgol_polyorders:
            if w <= p:
                continue
            r = _evaluate_postproc_config(oof_df, w, p, None, None, f"savgol w={w} p={p}")
            results.append(r)

    # Smooth + clip configs
    for w in savgol_windows:
        for p in savgol_polyorders:
            if w <= p:
                continue
            for clip_lower, clip_upper, clip_label in clip_configs:
                if clip_lower is None:
                    continue
                r = _evaluate_postproc_config(
                    oof_df, w, p, clip_lower, clip_upper,
                    f"clip {clip_label} + savgol w={w} p={p}",
                )
                results.append(r)

    results.sort(key=lambda r: r.postproc_rmse)
    return results


def run_train(
    data_dir: str | Path,
    n_splits: int = 5,
    seed: int = 42,
    model_params: dict | None = None,
    include_tvt_input: bool = False,
    include_geometry: bool = False,
    include_gr: bool = False,
    include_gr_dwt: bool = False,
    include_trajectory: bool = False,
    include_typewell: bool = False,
    include_spatial: bool = False,
    include_dtw: bool = False,
    include_geology: bool = False,
    include_beam: bool = False,
    include_formation_plane: bool = False,
    include_z_drift: bool = False,
    residual_target: bool = False,
    baseline_method: str = "flat",
    eval_postproc: bool = False,
) -> TrainResult:
    X, y, groups, well_ids_used, row_metadata, row_baselines = _collect_train_post_ps(
        data_dir,
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
        include_z_drift=include_z_drift,
        residual_target=residual_target,
        baseline_method=baseline_method,
    )

    if model_params is None:
        model_params = {}

    params: dict = {
        "objective": "regression",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "random_state": seed,
        "verbose": -1,
    }
    params.update(model_params)

    print(f"[2/3] Training {n_splits}-fold GroupKFold CV on {len(y)} rows ...")
    cv = GroupKFold(n_splits=n_splits)
    cv_scores: list[float] = []
    oof_rows: list[tuple[str, int, float, float, float]] = []

    if include_formation_plane:
        from rogii.formation_plane import build_formation_reference
        global_fp_ref = build_formation_reference(data_dir, "train", well_ids_used)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"  Fold {fold_idx + 1}/{n_splits} ...")
        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr, y_val = y[train_idx], y[val_idx]

        if include_spatial:
            val_groups = set(groups[val_idx])
            oof_wells = [wid for g, wid in enumerate(well_ids_used) if g not in val_groups]
            X_tr_spatial = _spatial_fold_features(data_dir, X, train_idx, oof_wells)
            X_val_spatial = _spatial_fold_features(data_dir, X, val_idx, oof_wells)
            X_tr = pd.concat([X_tr.reset_index(drop=True), X_tr_spatial.reset_index(drop=True)], axis=1)
            X_val = pd.concat([X_val.reset_index(drop=True), X_val_spatial.reset_index(drop=True)], axis=1)

        if include_formation_plane:
            val_groups_fp = set(groups[val_idx])
            oof_wells_fp = [wid for g, wid in enumerate(well_ids_used) if g not in val_groups_fp]
            oof_ref = global_fp_ref[global_fp_ref["well_id"].isin(oof_wells_fp)]
            X_tr_fp = _formation_plane_fold_features(X_tr, groups[train_idx], well_ids_used, oof_ref)
            X_val_fp = _formation_plane_fold_features(X_val, groups[val_idx], well_ids_used, oof_ref)
            X_tr = pd.concat([X_tr.reset_index(drop=True), X_tr_fp.reset_index(drop=True)], axis=1)
            X_val = pd.concat([X_val.reset_index(drop=True), X_val_fp.reset_index(drop=True)], axis=1)

        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = rmse(y_val, y_pred)
        cv_scores.append(score)
        print(f"  Fold {fold_idx + 1} RMSE: {score:.6f}")

        if eval_postproc:
            for fold_i, flat_idx in enumerate(val_idx):
                wid, row_idx = row_metadata[flat_idx]
                oof_rows.append((
                    wid, row_idx,
                    float(y[flat_idx]),
                    float(y_pred[fold_i]),
                    float(row_baselines[flat_idx]),
                ))

    print(f"[3/3] Training final model on all data ...")
    if include_spatial:
        from rogii.spatial_features import build_pre_ps_reference, build_spatial_knn_features
        ref = build_pre_ps_reference(data_dir, "train", well_ids_used)
        spatial_feats = build_spatial_knn_features(ref, X[["X", "Y", "Z"]])
        X = pd.concat([X, spatial_feats], axis=1)

    if include_formation_plane:
        X_fp_final = _formation_plane_fold_features(X, groups, well_ids_used, global_fp_ref)
        X = pd.concat([X, X_fp_final], axis=1)

    final_model = LGBMRegressor(**params)
    final_model.fit(X, y)

    # Compute clipping bounds from train data
    clip_bounds: tuple[float, float] | None = None
    if eval_postproc:
        from rogii.smoothing import compute_tvt_clip_bounds
        try:
            clip_bounds = compute_tvt_clip_bounds(data_dir)
        except Exception:
            clip_bounds = None

    postproc_results: list[PostprocResult] | None = None
    if eval_postproc and oof_rows:
        savgol_windows = [5, 11, 17, 25, 31]
        savgol_polyorders = [2, 3]
        clip_configs: list[tuple[float | None, float | None, str]] = [(None, None, "none")]

        if clip_bounds is not None:
            low, high = clip_bounds
            clip_configs = [
                (None, None, "none"),
                (low, high, f"p0.1-p99.9"),
            ]
            # Also test tighter bounds
            from rogii.smoothing import compute_tvt_clip_bounds
            try:
                low_tight, high_tight = compute_tvt_clip_bounds(data_dir, 0.5, 99.5)
                clip_configs.append((low_tight, high_tight, "p0.5-p99.5"))
                low_wide, high_wide = compute_tvt_clip_bounds(data_dir, 1.0, 99.0)
                clip_configs.append((low_wide, high_wide, "p1-p99"))
            except Exception:
                pass

        postproc_results = evaluate_postprocessing(oof_rows, savgol_windows, savgol_polyorders, clip_configs)

        print("\n--- Post-processing CV evaluation ---")
        print(f"{'Label':<45} {'Raw':>8} {'PostProc':>10} {'Delta':>8}")
        print("-" * 75)
        for r in postproc_results[:15]:  # Top 15
            delta = r.postproc_rmse - postproc_results[0].postproc_rmse
            print(f"{r.label:<45} {r.raw_rmse:8.4f} {r.postproc_rmse:10.4f} {delta:+8.4f}")

    return TrainResult(
        model=final_model,
        cv_rmse_mean=float(np.mean(cv_scores)),
        cv_rmse_std=float(np.std(cv_scores)),
        cv_rmse_folds=cv_scores,
        train_rows=len(y),
        train_wells=len(np.unique(groups)),
        feature_columns=list(X.columns),
        residual_target=residual_target,
        baseline_method=baseline_method,
        postproc_results=postproc_results,
        clip_bounds=clip_bounds,
    )
