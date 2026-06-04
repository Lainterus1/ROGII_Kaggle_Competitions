"""Training entry points for the ROGII ML baseline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold

from rogii.data_loading import list_well_ids, read_horizontal_well
from rogii.features import SAFE_NUMERIC_FEATURES, build_features, post_ps_mask
from rogii.metrics import rmse


@dataclass(frozen=True)
class TrainResult:
    model: LGBMRegressor
    cv_rmse_mean: float
    cv_rmse_std: float
    cv_rmse_folds: list[float]
    train_rows: int
    train_wells: int


def _collect_train_post_ps(data_dir: str | Path, include_tvt_input: bool = False) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features_list: list[pd.DataFrame] = []
    targets: list[float] = []
    groups: list[int] = []
    well_index = 0

    for well_id in list_well_ids(data_dir, "train"):
        horizontal = read_horizontal_well(data_dir, "train", well_id)
        if "TVT" not in horizontal.columns:
            continue
        mask = post_ps_mask(horizontal)
        if not mask.any():
            continue

        feats = build_features(horizontal, include_tvt_input=include_tvt_input)
        post_feats = feats.loc[mask].copy()
        post_target = horizontal.loc[mask, "TVT"].astype(float)

        valid = post_target.notna() & np.isfinite(post_target)
        if not valid.any():
            continue

        post_feats = post_feats.loc[valid]
        post_target = post_target.loc[valid]

        features_list.append(post_feats)
        targets.extend(post_target.tolist())
        groups.extend([well_index] * len(post_feats))
        well_index += 1

    if not features_list:
        raise ValueError("No train post-PS data found")

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
) -> TrainResult:
    X, y, groups = _collect_train_post_ps(data_dir, include_tvt_input=include_tvt_input)

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

    cv = GroupKFold(n_splits=n_splits)
    cv_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        score = rmse(y_val, y_pred)
        cv_scores.append(score)

    final_model = LGBMRegressor(**params)
    final_model.fit(X, y)

    return TrainResult(
        model=final_model,
        cv_rmse_mean=float(np.mean(cv_scores)),
        cv_rmse_std=float(np.std(cv_scores)),
        cv_rmse_folds=cv_scores,
        train_rows=len(y),
        train_wells=len(np.unique(groups)),
    )
