"""Prediction entry points for the ROGII ML baseline."""

from pathlib import Path

import numpy as np
import pandas as pd

from rogii.baseline import compute_baseline
from rogii.data_loading import list_well_ids, read_horizontal_well, read_sample_submission, read_typewell
from rogii.features import build_features, post_ps_mask
from rogii.model_io import validate_feature_columns
from rogii.models import parse_submission_id


def run_predict(
    data_dir: str | Path,
    models: list,
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
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    sample = read_sample_submission(data_dir)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")

    use_tvt_feature = include_tvt_input and not residual_target

    if include_spatial or include_formation_plane:
        return _run_predict_with_spatial(
            data_dir, models, sample,
            include_tvt_input=use_tvt_feature,
            include_geometry=include_geometry,
            include_gr=include_gr,
            include_gr_dwt=include_gr_dwt,
            include_trajectory=include_trajectory,
            include_typewell=include_typewell,
            include_dtw=include_dtw,
            include_geology=include_geology,
            include_beam=include_beam,
            include_formation_plane=include_formation_plane,
            include_z_drift=include_z_drift,
            residual_target=residual_target,
            baseline_method=baseline_method,
            feature_columns=feature_columns,
        )

    well_cache: dict[str, pd.DataFrame] = {}
    baseline_cache: dict[str, np.ndarray] = {}
    predictions: list[float] = []

    for submission_id in sample["id"].astype(str):
        well_id, row_index = parse_submission_id(submission_id)
        if well_id not in well_cache:
            horizontal = read_horizontal_well(data_dir, "test", well_id)
            typewell_frame = read_typewell(data_dir, "test", well_id) if (include_typewell or include_dtw or include_geology or include_beam) else None
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
            ordered_columns = validate_feature_columns(list(feats.columns), feature_columns)
            feats = feats.loc[:, ordered_columns]
            well_cache[well_id] = feats
            if residual_target:
                baseline = compute_baseline(horizontal, method=baseline_method)
                baseline_cache[well_id] = baseline
        feats = well_cache[well_id]
        if row_index < 0 or row_index >= len(feats):
            raise IndexError(f"Submission row index out of bounds for {submission_id}")
        row_feats = feats.iloc[row_index : row_index + 1]
        pred = float(np.mean([m.predict(row_feats)[0] for m in models]))
        if residual_target:
            pred = baseline_cache[well_id][row_index] + pred
        if not np.isfinite(pred):
            raise ValueError(f"Non-finite prediction for {submission_id}: {pred}")
        predictions.append(pred)

    return pd.DataFrame({"id": sample["id"].astype(str), "tvt": predictions})


def _run_predict_with_spatial(
    data_dir: str | Path,
    models: list,
    sample: pd.DataFrame,
    include_tvt_input: bool = False,
    include_geometry: bool = False,
    include_gr: bool = False,
    include_gr_dwt: bool = False,
    include_trajectory: bool = False,
    include_typewell: bool = False,
    include_dtw: bool = False,
    include_geology: bool = False,
    include_beam: bool = False,
    include_formation_plane: bool = False,
    include_z_drift: bool = False,
    residual_target: bool = False,
    baseline_method: str = "flat",
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    from rogii.spatial_features import build_pre_ps_reference, build_spatial_knn_features

    test_well_ids = list_well_ids(data_dir, "test")
    train_well_ids = list_well_ids(data_dir, "train")

    spatial_ref = build_pre_ps_reference(data_dir, "train", train_well_ids) if include_spatial else None

    if include_formation_plane:
        from rogii.formation_plane import build_formation_reference, impute_formations, build_formation_plane_features
        fp_ref = build_formation_reference(data_dir, "train", train_well_ids)

    all_features: dict[str, pd.DataFrame] = {}
    baseline_cache: dict[str, np.ndarray] = {}

    for wid in test_well_ids:
        horizontal = read_horizontal_well(data_dir, "test", wid)
        typewell_frame = read_typewell(data_dir, "test", wid) if (include_typewell or include_dtw or include_geology or include_beam) else None
        feats = build_features(
            horizontal,
            include_tvt_input=include_tvt_input,
            include_geometry=include_geometry,
            include_gr=include_gr,
            include_gr_dwt=include_gr_dwt,
            include_trajectory=include_trajectory,
            typewell=typewell_frame,
            include_typewell=include_typewell,
            include_dtw=include_dtw,
            include_geology=include_geology,
            include_beam=include_beam,
        )
        if include_spatial:
            query_coords = feats[["X", "Y", "Z"]]
            spatial_feats = build_spatial_knn_features(spatial_ref, query_coords)
            feats = pd.concat([feats.reset_index(drop=True), spatial_feats.reset_index(drop=True)], axis=1)
        if include_formation_plane:
            xy = np.array([[horizontal["X"].median(), horizontal["Y"].median()]], dtype=float)
            fp_est = impute_formations(fp_ref, xy)
            fp_feats = build_formation_plane_features(horizontal, fp_est)
            feats = pd.concat([feats.reset_index(drop=True), fp_feats.reset_index(drop=True)], axis=1)
        all_features[wid] = feats
        if residual_target:
            baseline = compute_baseline(horizontal, method=baseline_method)
            baseline_cache[wid] = baseline

    # Validate columns against the first well's features
    first_wid = test_well_ids[0] if test_well_ids else None
    if first_wid is not None:
        validate_feature_columns(list(all_features[first_wid].columns), feature_columns)

    predictions: list[float] = []
    for submission_id in sample["id"].astype(str):
        well_id, row_index = parse_submission_id(submission_id)
        feats = all_features[well_id]
        if row_index < 0 or row_index >= len(feats):
            raise IndexError(f"Submission row index out of bounds for {submission_id}")
        row_feats = feats.iloc[row_index : row_index + 1]
        pred = float(np.mean([m.predict(row_feats)[0] for m in models]))
        if residual_target:
            pred = baseline_cache[well_id][row_index] + pred
        if not np.isfinite(pred):
            raise ValueError(f"Non-finite prediction for {submission_id}: {pred}")
        predictions.append(pred)

    return pd.DataFrame({"id": sample["id"].astype(str), "tvt": predictions})


def predict_tcn(
    data_dir: str | Path,
    model,
    scaler,
    window_size: int,
    feature_columns: list[str],
    input_scaler=None,
    residual_target: bool = True,
    baseline_method: str = "flat",
    device: str = "cuda",
) -> pd.DataFrame:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    sample = read_sample_submission(data_dir)
    test_well_ids = list_well_ids(data_dir, "test")

    model.to(device)
    model.eval()

    well_preds: dict[str, np.ndarray] = {}
    well_baselines: dict[str, np.ndarray] = {}

    for wid in test_well_ids:
        horizontal = read_horizontal_well(data_dir, "test", wid)
        from rogii.sequence_features import build_sequence_features
        seq_df = build_sequence_features(horizontal)
        seq_df = seq_df[feature_columns]
        X = seq_df.to_numpy(dtype=np.float64)

        # Per-well normalization (same as training)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        X = ((X - mean) / std).astype(np.float32)

        # Apply global absolute-coordinate scaler (Phase 2 dual normalization)
        if input_scaler is not None:
            X_abs = horizontal[["X", "Y", "Z", "MD"]].to_numpy(dtype=np.float64)
            X_abs_scaled = input_scaler.transform(X_abs).astype(np.float32)
            X = np.concatenate([X, X_abs_scaled], axis=1)  # (T, 65+4)

        if residual_target:
            well_baselines[wid] = compute_baseline(horizontal, method=baseline_method)

        T = len(X)
        if T < window_size:
            pred_delta = np.zeros(T, dtype=float)
            well_preds[wid] = pred_delta
            continue

        windows = np.stack([X[t : t + window_size] for t in range(T - window_size + 1)])
        xb = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)

        batch_size = 256
        pred_scaled_list: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(xb), batch_size):
                batch = xb[start : start + batch_size]
                pred_batch = model(batch).cpu().numpy().ravel()
                pred_scaled_list.append(pred_batch)
        pred_scaled = np.concatenate(pred_scaled_list)
        if scaler is not None:
            pred_delta = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        else:
            pred_delta = pred_scaled

        full_pred = np.full(T, np.nan, dtype=float)
        full_pred[window_size - 1:] = pred_delta
        if window_size - 1 > 0:
            full_pred[:window_size - 1] = pred_delta[0]
        well_preds[wid] = full_pred

    predictions: list[float] = []
    for submission_id in sample["id"].astype(str):
        well_id, row_index = parse_submission_id(submission_id)
        preds = well_preds[well_id]
        if row_index < 0 or row_index >= len(preds):
            raise IndexError(f"Submission row index out of bounds for {submission_id}")
        pred = float(preds[row_index])
        if residual_target:
            pred = well_baselines[well_id][row_index] + pred
        if not np.isfinite(pred):
            raise ValueError(f"Non-finite prediction for {submission_id}: {pred}")
        predictions.append(pred)

    return pd.DataFrame({"id": sample["id"].astype(str), "tvt": predictions})
