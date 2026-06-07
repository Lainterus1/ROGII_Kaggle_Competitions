"""Evaluate PoP2 3-strategy blend on 5-fold GroupKFold OOF predictions.

Does NOT retrain the final model — only runs CV to collect OOF preds,
then applies PoP2 blend and computes RMSE for all strategies individually.
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rogii.data_loading import read_horizontal_well, read_typewell
from rogii.gr_matcher import apply_dtw_matching
from rogii.metrics import rmse
from rogii.postprocess import _blend_row
from rogii.train import _collect_train_post_ps
from rogii.z_physics import apply_z_physics


def _load_well_data(data_dir, well_ids):
    """Pre-load horizontal + typewell for all train wells (for PoP2 evaluation)."""
    horizontal = {}
    typewell = {}
    for wid in well_ids:
        horizontal[wid] = read_horizontal_well(data_dir, "train", wid)
        try:
            typewell[wid] = read_typewell(data_dir, "train", wid)
        except FileNotFoundError:
            typewell[wid] = None
    return horizontal, typewell


def evaluate_pop2(
    data_dir: str = "data",
    n_splits: int = 5,
    seed: int = 42,
):
    print(f"=== PoP2 OOF Evaluation ({n_splits}-fold GroupKFold) ===\n", flush=True)

    # Phase 1: Collect train data (same as R1)
    print("[1/4] Building features ...", flush=True)
    t0 = time.time()
    X, y, groups, well_ids_used, row_metadata, row_baselines = _collect_train_post_ps(
        data_dir,
        include_tvt_input=False,
        include_geometry=True,
        include_gr=True,
        include_gr_dwt=False,
        include_trajectory=False,
        include_typewell=False,
        include_spatial=False,
        include_dtw=False,
        include_geology=False,
        include_beam=False,
        include_z_drift=False,
        residual_target=True,
        baseline_method="flat",
    )
    print(f"  Features: {list(X.columns)}")
    print(f"  Rows: {len(y)}, Wells: {len(np.unique(groups))}")
    print(f"  Time: {time.time() - t0:.1f}s\n", flush=True)

    # Phase 2: CV loop, collect OOF
    print(f"[2/4] Running {n_splits}-fold CV ...", flush=True)
    t0 = time.time()
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "random_state": seed,
        "verbose": -1,
    }
    cv = GroupKFold(n_splits=n_splits)

    # OOF predictions: {(well_id, row_idx): (y_true_delta, y_pred_delta, baseline)}
    oof_data: dict[tuple, tuple] = {}
    cv_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = rmse(y_val, y_pred)
        cv_scores.append(score)
        print(f"  Fold {fold_idx + 1}: delta RMSE = {score:.4f}", flush=True)

        for fold_i, flat_idx in enumerate(val_idx):
            wid, row_idx = row_metadata[flat_idx]
            oof_data[(wid, row_idx)] = (
                float(y[flat_idx]),
                float(y_pred[fold_i]),
                float(row_baselines[flat_idx]),
            )

    mean_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))
    print(f"  CV delta RMSE: {mean_cv:.4f} ± {std_cv:.4f}")
    print(f"  Time: {time.time() - t0:.1f}s\n", flush=True)

    # Phase 3: Per-well PoP2 blend evaluation
    print(f"[3/4] Loading well data and computing PoP2 blend ...", flush=True)
    used_wells = sorted(set(wid for wid, _ in oof_data.keys()))
    horizontal, typewell = _load_well_data(data_dir, used_wells)

    # Accumulate predictions per well, then blend
    y_true_full: list[float] = []
    y_pred_model: list[float] = []
    y_pred_z: list[float] = []
    y_pred_dtw: list[float] = []
    y_pred_median: list[float] = []

    for wid in used_wells:
        h = horizontal[wid]
        tw = typewell[wid]
        n_rows = len(h)

        # Build full-length arrays for this well
        model_preds = np.full(n_rows, np.nan, dtype=float)
        y_true_arr = np.full(n_rows, np.nan, dtype=float)
        baselines_arr = np.full(n_rows, np.nan, dtype=float)

        for (w, ri), (y_t_delta, y_p_delta, baseline) in oof_data.items():
            if w == wid:
                model_preds[ri] = y_p_delta + baseline
                y_true_arr[ri] = y_t_delta + baseline
                baselines_arr[ri] = baseline

        # Z-physics
        z_preds = apply_z_physics(h)

        # DTW matching
        post_mask = ~np.isnan(model_preds) & ~np.isnan(z_preds)
        if tw is not None and len(tw) >= 3 and post_mask.sum() > 3:
            dtw_preds = apply_dtw_matching(h, tw, model_preds.copy())
        else:
            dtw_preds = model_preds.copy()

        # Collect per-row results
        for ri in range(n_rows):
            if not np.isfinite(y_true_arr[ri]):
                continue

            mp = model_preds[ri]
            zp = z_preds[ri]
            dp = dtw_preds[ri]

            y_true_full.append(y_true_arr[ri])
            y_pred_model.append(mp)
            y_pred_z.append(zp if np.isfinite(zp) else mp)
            y_pred_dtw.append(dp if np.isfinite(dp) else mp)
            y_pred_median.append(_blend_row(mp, zp, dp, weights=None))

        if (used_wells.index(wid) + 1) % 100 == 0:
            print(f"  {used_wells.index(wid) + 1}/{len(used_wells)} wells ...", flush=True)

    # Phase 4: Results
    print(f"\n[4/4] Results ({len(y_true_full)} OOF rows, {len(used_wells)} wells):\n", flush=True)
    print(f"{'Strategy':<30} {'RMSE':>10} {'vs Model':>10}")
    print("-" * 52)

    model_rmse = rmse(y_true_full, y_pred_model)
    print(f"{'Model (R1, delta target)':<30} {model_rmse:10.4f}")

    for name, preds in [
        ("Z-physics only", y_pred_z),
        ("DTW matching only", y_pred_dtw),
        ("PoP2 Median (model+z+dtw)", y_pred_median),
    ]:
        r = rmse(y_true_full, preds)
        delta = r - model_rmse
        print(f"{name:<30} {r:10.4f} {delta:+10.4f}")

    print("-" * 52)
    print(f"\nCorrelations with y_true:")
    for name, preds in [
        ("Model       ", y_pred_model),
        ("Z-physics   ", y_pred_z),
        ("DTW matching", y_pred_dtw),
        ("PoP2 median ", y_pred_median),
    ]:
        corr = np.corrcoef(y_true_full, preds)[0, 1]
        print(f"  {name}: {corr:.6f}")

    inter_corr_model_z = np.corrcoef(y_pred_model, y_pred_z)[0, 1]
    inter_corr_model_dtw = np.corrcoef(y_pred_model, y_pred_dtw)[0, 1]
    inter_corr_z_dtw = np.corrcoef(y_pred_z, y_pred_dtw)[0, 1]
    print(f"\nInter-strategy correlations:")
    print(f"  Model vs Z-physics:   {inter_corr_model_z:.6f}")
    print(f"  Model vs DTW:         {inter_corr_model_dtw:.6f}")
    print(f"  Z-physics vs DTW:     {inter_corr_z_dtw:.6f}")

    # Weighted blend grid search (optional)
    print(f"\n--- Weighted Blend Grid Search ---")
    print(f"{'Weights (model,z,dtw)':<25} {'RMSE':>10} {'vs Model':>10}")
    print("-" * 47)
    candidates = [
        (0.50, 0.25, 0.25),
        (0.60, 0.20, 0.20),
        (0.70, 0.15, 0.15),
        (0.80, 0.10, 0.10),
        (0.40, 0.30, 0.30),
        (0.33, 0.33, 0.34),
        (0.50, 0.30, 0.20),
        (0.60, 0.30, 0.10),
    ]
    results = []
    for w_m, w_z, w_d in candidates:
        blended = [
            _blend_row(y_pred_model[i], y_pred_z[i], y_pred_dtw[i], weights=(w_m, w_z, w_d))
            for i in range(len(y_true_full))
        ]
        r = rmse(y_true_full, blended)
        results.append((r, w_m, w_z, w_d))

    results.sort()
    for r, w_m, w_z, w_d in results:
        delta = r - model_rmse
        print(f"{w_m:.2f},{w_z:.2f},{w_d:.2f} {'':>11} {r:10.4f} {delta:+10.4f}")

    best = results[0]
    print(f"\nBest weighted blend: ({best[1]:.2f}, {best[2]:.2f}, {best[3]:.2f}) RMSE = {best[0]:.4f}")


if __name__ == "__main__":
    evaluate_pop2()
