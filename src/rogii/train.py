"""Training entry points for the ROGII ML baseline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from rogii.data_loading import list_well_ids, read_horizontal_well, read_typewell
from rogii.features import SAFE_NUMERIC_FEATURES, build_features, last_known_tvt_input_value, post_ps_mask
from rogii.metrics import rmse


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
    residual_target: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    features_list: list[pd.DataFrame] = []
    targets: list[float] = []
    groups: list[int] = []
    well_ids_used: list[str] = []
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

        typewell_frame = read_typewell(data_dir, "train", well_id) if (include_typewell or include_dtw or include_geology) else None

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
        )
        post_feats = feats.loc[mask].copy()
        post_target = horizontal.loc[mask, "TVT"].astype(float)

        valid = post_target.notna() & np.isfinite(post_target)
        if not valid.any():
            continue

        post_feats = post_feats.loc[valid]
        post_target = post_target.loc[valid]

        if residual_target:
            if np.isnan(last_tvt):
                continue
            post_target = post_target - last_tvt

        features_list.append(post_feats)
        targets.extend(post_target.tolist())
        groups.extend([well_index] * len(post_feats))
        well_ids_used.append(well_id)
        well_index += 1

    if not features_list:
        raise ValueError("No train post-PS data found")

    print(f"  Done: {well_index} wells, {len(targets)} post-PS rows")
    X = pd.concat(features_list, ignore_index=True)
    y = np.array(targets, dtype=float)
    g = np.array(groups, dtype=int)
    return X, y, g, well_ids_used


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
    residual_target: bool = False,
) -> TrainResult:
    X, y, groups, well_ids_used = _collect_train_post_ps(
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
        residual_target=residual_target,
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

        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = rmse(y_val, y_pred)
        cv_scores.append(score)
        print(f"  Fold {fold_idx + 1} RMSE: {score:.6f}")

    print(f"[3/3] Training final model on all data ...")
    if include_spatial:
        from rogii.spatial_features import build_pre_ps_reference, build_spatial_knn_features
        ref = build_pre_ps_reference(data_dir, "train", well_ids_used)
        spatial_feats = build_spatial_knn_features(ref, X[["X", "Y", "Z"]])
        X = pd.concat([X, spatial_feats], axis=1)

    final_model = LGBMRegressor(**params)
    final_model.fit(X, y)

    return TrainResult(
        model=final_model,
        cv_rmse_mean=float(np.mean(cv_scores)),
        cv_rmse_std=float(np.std(cv_scores)),
        cv_rmse_folds=cv_scores,
        train_rows=len(y),
        train_wells=len(np.unique(groups)),
        feature_columns=list(X.columns),
        residual_target=residual_target,
    )
