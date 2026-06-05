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
    residual_target: bool = False


def _collect_train_post_ps(
    data_dir: str | Path,
    include_tvt_input: bool = False,
    include_geometry: bool = False,
    include_gr: bool = False,
    include_typewell: bool = False,
    residual_target: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features_list: list[pd.DataFrame] = []
    targets: list[float] = []
    groups: list[int] = []
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

        typewell_frame = read_typewell(data_dir, "train", well_id) if include_typewell else None

        feats = build_features(
            horizontal,
            include_tvt_input=use_tvt_feature,
            include_geometry=include_geometry,
            include_gr=include_gr,
            typewell=typewell_frame,
            include_typewell=include_typewell,
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
        well_index += 1

    if not features_list:
        raise ValueError("No train post-PS data found")

    print(f"  Done: {well_index} wells, {len(targets)} post-PS rows")
    X = pd.concat(features_list, ignore_index=True)
    y = np.array(targets, dtype=float)
    g = np.array(groups, dtype=int)
    return X, y, g


def run_train(
    data_dir: str | Path,
    n_splits: int = 5,
    seed: int = 42,
    model_params: dict | None = None,
    include_tvt_input: bool = False,
    include_geometry: bool = False,
    include_gr: bool = False,
    include_typewell: bool = False,
    residual_target: bool = False,
) -> TrainResult:
    X, y, groups = _collect_train_post_ps(
        data_dir,
        include_tvt_input=include_tvt_input,
        include_geometry=include_geometry,
        include_gr=include_gr,
        include_typewell=include_typewell,
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
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = rmse(y_val, y_pred)
        cv_scores.append(score)
        print(f"  Fold {fold_idx + 1} RMSE: {score:.6f}")

    print(f"[3/3] Training final model on all data ...")
    final_model = LGBMRegressor(**params)
    final_model.fit(X, y)

    return TrainResult(
        model=final_model,
        cv_rmse_mean=float(np.mean(cv_scores)),
        cv_rmse_std=float(np.std(cv_scores)),
        cv_rmse_folds=cv_scores,
        train_rows=len(y),
        train_wells=len(np.unique(groups)),
        residual_target=residual_target,
    )
