"""Training entry points for the ROGII ML baseline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
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
    models: list
    seed_list: list[int]
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_rmse_folds: list[float]
    train_rows: int
    train_wells: int
    feature_columns: list[str]
    cv_strategy: str = "group"
    residual_target: bool = False
    baseline_method: str = "flat"
    postproc_results: list[PostprocResult] | None = None
    clip_bounds: tuple[float, float] | None = None
    oof_df: "pd.DataFrame | None" = None
    tcn_metadata: dict | None = None


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
    oof_rows: list[tuple[str, int, int, float, float, float]],
) -> pd.DataFrame:
    """Build per-well OOF prediction dataframe for post-processing evaluation.

    Columns: well_id, row_idx, fold, y_true (delta or full), y_pred (delta or full), baseline.
    When residual_target is active, y_true/y_pred are deltas and baseline > 0.
    """
    return pd.DataFrame(oof_rows, columns=["well_id", "row_idx", "fold", "y_true", "y_pred", "baseline"])


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
    oof_rows: list[tuple[str, int, int, float, float, float]],
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
    seed_list: list[int] | None = None,
    cv_strategy: str = "group",
    strat_tvt_bins: int = 4,
    strat_spatial_clusters: int = 5,
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

    seeds = seed_list if seed_list else [42]

    base_params: dict = {
        "objective": "regression",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "verbose": -1,
    }
    base_params.update(model_params)

    from rogii.validation import build_stratification_labels, create_cv_splitter

    cv_splitter = create_cv_splitter(cv_strategy, n_splits)

    strat_labels: np.ndarray | None = None
    if cv_strategy == "stratified":
        strat_labels, _ = build_stratification_labels(
            data_dir, well_ids_used,
            n_tvt_bins=strat_tvt_bins,
            n_spatial_clusters=strat_spatial_clusters,
        )
        # Expand per-well labels to per-row
        strat_labels_per_row = np.array([strat_labels[g] for g in groups], dtype=int)
    else:
        strat_labels_per_row = y

    strategy_label = "StratifiedGroupKFold" if cv_strategy == "stratified" else "GroupKFold"
    n_seeds = len(seeds)
    print(f"[2/3] Training {n_splits}-fold {strategy_label} CV ({n_seeds} seed{'s' if n_seeds > 1 else ''}) on {len(y)} rows ...")
    cv_scores: list[float] = []
    oof_rows: list[tuple[str, int, int, float, float, float]] = []

    if include_formation_plane:
        from rogii.formation_plane import build_formation_reference
        global_fp_ref = build_formation_reference(data_dir, "train", well_ids_used)

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X, strat_labels_per_row, groups)):
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

        fold_seed_preds: list[np.ndarray] = []
        for seed_i, seed in enumerate(seeds):
            seed_params = dict(base_params)
            seed_params["random_state"] = seed
            model = LGBMRegressor(**seed_params)
            model.fit(X_tr, y_tr)
            y_pred_seed = model.predict(X_val)
            fold_seed_preds.append(y_pred_seed)
            if n_seeds == 1:
                print(f"    seed={seed}", end="")

        y_pred = np.mean(fold_seed_preds, axis=0)
        score = rmse(y_val, y_pred)
        cv_scores.append(score)
        print(f"  Fold {fold_idx + 1} RMSE: {score:.6f}")

        for fold_i, flat_idx in enumerate(val_idx):
            wid, row_idx = row_metadata[flat_idx]
            oof_rows.append((
                wid, row_idx, fold_idx,
                float(y[flat_idx]),
                float(y_pred[fold_i]),
                float(row_baselines[flat_idx]),
            ))

    print(f"[3/3] Training {n_seeds} final model{'s' if n_seeds > 1 else ''} on all data ...")
    if include_spatial:
        from rogii.spatial_features import build_pre_ps_reference, build_spatial_knn_features
        ref = build_pre_ps_reference(data_dir, "train", well_ids_used)
        spatial_feats = build_spatial_knn_features(ref, X[["X", "Y", "Z"]])
        X = pd.concat([X, spatial_feats], axis=1)

    if include_formation_plane:
        X_fp_final = _formation_plane_fold_features(X, groups, well_ids_used, global_fp_ref)
        X = pd.concat([X, X_fp_final], axis=1)

    final_models: list = []
    for seed_i, seed in enumerate(seeds):
        seed_params = dict(base_params)
        seed_params["random_state"] = seed
        model = LGBMRegressor(**seed_params)
        model.fit(X, y)
        final_models.append(model)

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
        models=final_models,
        seed_list=list(seeds),
        cv_rmse_mean=float(np.mean(cv_scores)),
        cv_rmse_std=float(np.std(cv_scores)),
        cv_rmse_folds=cv_scores,
        train_rows=len(y),
        train_wells=len(np.unique(groups)),
        feature_columns=list(X.columns),
        cv_strategy=cv_strategy,
        residual_target=residual_target,
        baseline_method=baseline_method,
        postproc_results=postproc_results,
        clip_bounds=clip_bounds,
        oof_df=_build_oof_per_well(oof_rows) if oof_rows else None,
    )


def _collect_train_sequences(
    data_dir: str | Path,
    residual_target: bool = True,
    baseline_method: str = "flat",
) -> tuple[list["WellSequence"], list[str]]:
    """Collect post-PS sequences from all train wells for TCN training.

    Returns list of WellSequence and the feature column names.
    """
    from rogii.sequence_data import WellSequence
    from rogii.sequence_features import build_sequence_features

    well_ids = list_well_ids(data_dir, "train")
    sequences: list[WellSequence] = []
    global_baseline = 0.0
    print(f"[1/3] Loading {len(well_ids)} train wells (TCN sequences) ...")

    for i, well_id in enumerate(well_ids):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(well_ids)} wells loaded")
        horizontal = read_horizontal_well(data_dir, "train", well_id)
        if "TVT" not in horizontal.columns:
            continue
        mask = post_ps_mask(horizontal)
        if not mask.any():
            continue

        seq_df = build_sequence_features(horizontal)
        feature_columns = list(seq_df.columns)
        post_seq = seq_df.loc[mask].copy()
        post_target = horizontal.loc[mask, "TVT"].astype(float)

        valid = post_target.notna() & np.isfinite(post_target)
        if not valid.any():
            continue

        post_seq = post_seq.loc[valid]
        post_target = post_target.loc[valid]
        row_indices = horizontal.index[mask][valid].to_numpy(dtype=int)

        last_tvt = last_known_tvt_input_value(horizontal)
        if np.isnan(last_tvt):
            continue
        global_baseline = float(last_tvt)

        baseline_vals = np.zeros(len(post_target), dtype=float)
        if residual_target:
            baseline = compute_baseline(horizontal, method=baseline_method)
            baseline_series = pd.Series(baseline, index=horizontal.index)
            baseline_post = baseline_series.loc[post_target.index]
            if baseline_post.isna().any():
                continue
            baseline_vals = baseline_post.to_numpy(dtype=float)
            post_target = post_target - baseline_post

        T = len(post_seq)
        if T < 2:
            continue

        raw_abs = horizontal[["X", "Y", "Z", "MD"]].loc[post_seq.index].to_numpy(dtype=np.float64)

        sequences.append(WellSequence(
            well_id=well_id,
            X=post_seq.to_numpy(dtype=np.float32),
            y=post_target.to_numpy(dtype=np.float32),
            baseline=float(global_baseline),
            row_indices=row_indices,
            X_abs=raw_abs,
        ))

    if not sequences:
        raise ValueError("No train post-PS sequence data found")
    print(f"  Done: {len(sequences)} wells, {sum(len(s.X) for s in sequences)} post-PS rows")
    return sequences, feature_columns


def _make_dataloader(
    sequences: list["WellSequence"], window_size: int,
    batch_size: int, shuffle: bool, stride: int = 1,
    num_workers: int = 0,
) -> "torch.utils.data.DataLoader":
    from rogii.sequence_data import WellSequenceDataset
    import torch
    ds = WellSequenceDataset(sequences, window_size=window_size, stride=stride)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=False, drop_last=False,
    )


def train_tcn(
    data_dir: str | Path,
    window_size: int = 64,
    num_channels: tuple[int, ...] = (64, 128, 256, 128),
    kernel_size: int = 5,
    dropout: float = 0.1,
    batch_size: int = 8192,
    epochs: int = 20,
    patience: int = 5,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    n_splits: int = 5,
    seed: int = 42,
    device: str = "cuda",
    residual_target: bool = True,
    baseline_method: str = "flat",
    stride: int = 2,
    num_workers: int = 0,
) -> TrainResult:
    from dataclasses import replace
    import torch
    import torch.nn as nn
    import time
    from sklearn.preprocessing import StandardScaler
    from rogii.sequence_data import WellSequence

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    use_amp = device == "cuda"

    def _os_environ_flag(name: str) -> bool:
        import os
        return os.environ.get(name, "").strip() in ("1", "true", "yes")

    # --- Collect sequences ---
    sequences, tcn_feature_columns = _collect_train_sequences(
        data_dir, residual_target=residual_target, baseline_method=baseline_method,
    )
    well_ids_all = [s.well_id for s in sequences]
    n_wells = len(sequences)

    # --- Per-well input normalization (each well gets its own mean/std) ---
    # Also stash raw absolute X/Y/Z/MD for global scaling before normalization destroys them.
    raw_x_abs_list: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    for s in sequences:
        raw_x_abs_list.append(s.X_abs.copy().astype(np.float64))
        s.X_abs = None  # free memory after extraction

        x = s.X.astype(np.float64)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        scaled = ((x - mean) / std).astype(np.float32)
        s.X = torch.from_numpy(scaled).T.contiguous()  # (F, T)
        raw_targets.append(s.y.astype(np.float64).copy())

    def _fit_target_scaler(seq_indices: np.ndarray) -> StandardScaler | None:
        fold_targets = np.concatenate([raw_targets[i] for i in seq_indices])
        if np.std(fold_targets) < 1e-12:
            return None
        target_scaler = StandardScaler()
        target_scaler.fit(fold_targets.reshape(-1, 1))
        return target_scaler

    def _fit_global_x_scaler(seq_indices: np.ndarray) -> StandardScaler | None:
        if _os_environ_flag("ROGII_NO_GLOBAL_ABS"):
            return None
        abs_data = np.concatenate([raw_x_abs_list[i] for i in seq_indices], axis=0)
        if abs_data.shape[0] < 2 or np.std(abs_data, axis=0).min() < 1e-12:
            return None
        x_scaler = StandardScaler()
        x_scaler.fit(abs_data)
        return x_scaler

    def _scaled_sequences(
        seq_indices: np.ndarray,
        target_scaler: StandardScaler | None,
        x_scaler: StandardScaler | None = None,
    ):
        scaled_sequences = []
        for i in seq_indices:
            y = raw_targets[i]
            if target_scaler is not None:
                y = target_scaler.transform(y.reshape(-1, 1)).ravel()

            if x_scaler is not None:
                abs_scaled = x_scaler.transform(raw_x_abs_list[i]).astype(np.float32)  # (T, 4)
                X_abs_tensor = torch.from_numpy(abs_scaled).T.contiguous()  # (4, T)
                X_combined = torch.cat([sequences[i].X, X_abs_tensor], dim=0)  # (65+4, T)
            else:
                X_combined = sequences[i].X  # (65, T)

            scaled_sequences.append(replace(
                sequences[i],
                X=X_combined,
                y=torch.from_numpy(y.astype(np.float32)),
            ))
        return scaled_sequences

    # --- CV ---
    from sklearn.model_selection import GroupKFold
    groups = np.arange(n_wells, dtype=int)
    cv_splitter = GroupKFold(n_splits=n_splits)

    cv_scores: list[float] = []
    oof_rows: list[tuple[str, int, int, float, float, float]] = []

    print(f"[2/3] Training {n_splits}-fold GroupKFold CV (TCN) on {n_wells} wells, device={device}")
    print(f"  model: TCN(ch={list(num_channels)}, k={kernel_size}, w={window_size}, stride={stride})")
    print(f"  lr={lr}, wd={weight_decay}, B={batch_size}, epochs={epochs}, patience={patience}")

    fold_start_time = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(np.zeros(n_wells), groups=groups)):
        fold_t0 = time.time()
        n_train_seqs = len(train_idx)
        n_val_seqs = len(val_idx)
        print(f"  Fold {fold_idx + 1}/{n_splits}: train={n_train_seqs} wells, val={n_val_seqs} wells")

        fold_scaler = _fit_target_scaler(train_idx)
        fold_x_scaler = _fit_global_x_scaler(train_idx)
        train_seqs = _scaled_sequences(train_idx, fold_scaler, fold_x_scaler)
        val_seqs = _scaled_sequences(val_idx, fold_scaler, fold_x_scaler)

        train_loader = _make_dataloader(train_seqs, window_size, batch_size, shuffle=True,
                                         stride=stride, num_workers=num_workers)
        val_loader = _make_dataloader(val_seqs, window_size, batch_size, shuffle=False,
                                       stride=stride, num_workers=num_workers)

        input_size = sequences[0].X.shape[0]
        if fold_x_scaler is not None:
            input_size += fold_x_scaler.n_features_in_
        from rogii.tcn_model import TCNModel
        model = TCNModel(
            input_size=int(input_size),
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
        criterion = nn.MSELoss()
        scaler_amp = torch.amp.GradScaler(device) if use_amp else None

        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        epochs_no_improve = 0

        print(f"    {'ep':>3}  {'train':>8}  {'val':>8}  {'best':>8}  {'pat':>3}  {'lr':>8}  {'time':>6}")
        for epoch in range(epochs):
            ep_t0 = time.time()
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                optimizer.zero_grad()
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = model(xb)
                        loss = criterion(pred, yb)
                    scaler_amp.scale(loss).backward()
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                else:
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                train_loss += loss.item() * xb.size(0)
            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    pred = model(xb)
                    val_loss += criterion(pred, yb).item() * xb.size(0)
            val_loss /= len(val_loader.dataset)

            # ReduceLROnPlateau after epoch
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            ep_elapsed = time.time() - ep_t0

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                marker = "*"
            else:
                epochs_no_improve += 1
                marker = " "

            print(f"    {epoch + 1:>3}{marker}  {train_loss:>8.4f}  {val_loss:>8.4f}  "
                  f"{best_val_loss:>8.4f}  {epochs_no_improve:>3}  {current_lr:>8.2e}  "
                  f"{ep_elapsed:>5.0f}s")

            if epochs_no_improve >= patience:
                print(f"    Early stop: best epoch {best_epoch}, val_loss={best_val_loss:.4f}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # --- OOF inference (batched via DataLoader) ---
        model.eval()
        val_inf_loader = _make_dataloader(val_seqs, window_size, batch_size, shuffle=False,
                                           stride=1, num_workers=num_workers)
        y_pred_scaled_parts: list[np.ndarray] = []
        y_true_scaled_parts: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_inf_loader:
                xb = xb.to(device, non_blocking=True)
                pred_raw = model(xb)
                if torch.isnan(pred_raw).any() or torch.isinf(pred_raw).any():
                    continue
                y_pred_scaled_parts.append(pred_raw.cpu().numpy().ravel())
                y_true_scaled_parts.append(yb.numpy().ravel())

        if y_pred_scaled_parts:
            y_pred_scaled = np.concatenate(y_pred_scaled_parts)
            y_true_scaled = np.concatenate(y_true_scaled_parts)
        else:
            y_pred_scaled = np.array([])
            y_true_scaled = np.array([])

        if fold_scaler is not None:
            y_true_delta = fold_scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).ravel()
            y_pred_delta = fold_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_true_delta = y_true_scaled
            y_pred_delta = y_pred_scaled

        # Collect OOF metadata
        val_inf_ds = val_inf_loader.dataset
        for sample_idx, (seq_idx_in_ds, _t) in enumerate(val_inf_ds._sample_map):
            if sample_idx >= len(y_pred_delta):
                break
            s = val_inf_ds._seqs[seq_idx_in_ds]
            wid = s.well_id
            row_idx = int(s.row_indices[_t + window_size - 1])
            baseline_val = float(s.baseline)
            oof_rows.append((
                wid, row_idx, fold_idx,
                float(y_true_delta[sample_idx]),
                float(y_pred_delta[sample_idx]),
                baseline_val,
            ))

        score = rmse(y_true_delta, y_pred_delta)
        cv_scores.append(score)

        fold_elapsed = time.time() - fold_t0
        folds_done = fold_idx + 1
        avg_fold_time = (time.time() - fold_start_time) / folds_done
        remaining = avg_fold_time * (n_splits - folds_done)

        print(f"  Fold {fold_idx + 1} RMSE_DELTA: {score:.4f}  |  "
              f"fold time: {fold_elapsed:.0f}s  |  ETA: {remaining/60:.0f}min")

        # --- Global early stop: if Fold 2 RMSE > 15, abort ---
        if folds_done >= 2 and score > 15.0:
            print(f"  *** ABORT: Fold {fold_idx + 1} RMSE {score:.1f} > 15.0. Stopping CV. ***")
            cv_scores += [float("nan")] * (n_splits - folds_done)
            break

    # --- Final model on all data ---
    print(f"[3/3] Training final TCN on all {n_wells} wells ...")
    all_indices = np.arange(n_wells, dtype=int)
    final_scaler = _fit_target_scaler(all_indices)
    final_x_scaler = _fit_global_x_scaler(all_indices)
    final_sequences = _scaled_sequences(all_indices, final_scaler, final_x_scaler)
    all_loader = _make_dataloader(final_sequences, window_size, batch_size, shuffle=True,
                                   stride=stride, num_workers=num_workers)
    input_size = sequences[0].X.shape[0]
    if final_x_scaler is not None:
        input_size += final_x_scaler.n_features_in_
    from rogii.tcn_model import TCNModel
    final_model = TCNModel(
        input_size=int(input_size),
        num_channels=list(num_channels),
        kernel_size=kernel_size,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    criterion = nn.MSELoss()
    scaler_amp_final = torch.amp.GradScaler(device) if use_amp else None

    print(f"    {'ep':>3}  {'loss':>10}  {'lr':>8}  {'time':>6}")
    for epoch in range(epochs):
        ep_t0 = time.time()
        final_model.train()
        total_loss = 0.0
        for xb, yb in all_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast("cuda"):
                    pred = final_model(xb)
                    loss = criterion(pred, yb)
                scaler_amp_final.scale(loss).backward()
                scaler_amp_final.step(optimizer)
                scaler_amp_final.update()
            else:
                pred = final_model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
        total_loss /= len(all_loader.dataset)
        scheduler.step(total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        ep_elapsed = time.time() - ep_t0
        print(f"    {epoch + 1:>3}  {total_loss:>10.4f}  {current_lr:>8.2e}  {ep_elapsed:>5.0f}s")

    return TrainResult(
        models=[final_model],
        seed_list=[seed],
        cv_rmse_mean=float(np.mean(cv_scores)),
        cv_rmse_std=float(np.std(cv_scores)),
        cv_rmse_folds=cv_scores,
        train_rows=sum(len(s.X) for s in sequences),
        train_wells=n_wells,
        feature_columns=tcn_feature_columns,
        cv_strategy="group",
        residual_target=residual_target,
        baseline_method=baseline_method,
        oof_df=_build_oof_per_well(oof_rows) if oof_rows else None,
        tcn_metadata={
            "y_scaler": final_scaler,
            "x_scaler": final_x_scaler,
            "window_size": window_size,
            "num_channels": list(num_channels),
            "kernel_size": int(kernel_size),
            "dropout": float(dropout),
            "input_size": int(sequences[0].X.shape[0]) + (4 if final_x_scaler is not None else 0),
        },
    )
