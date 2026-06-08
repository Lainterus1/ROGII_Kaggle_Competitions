"""Smoke tests for TCN train/predict/model-IO pipeline."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from rogii.train import train_tcn, TrainResult
from rogii.predict import predict_tcn
from rogii.model_io import build_model_payload, resolve_prediction_contract
from rogii.tcn_model import TCNModel


def _write_synthetic_train_data(root: Path, n_wells: int = 8) -> None:
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    (root / "sample_submission.csv").write_text(
        "id,tvt\n", encoding="utf-8")
    sub_ids: list[str] = []
    rng = np.random.RandomState(0)
    for wi in range(n_wells):
        well_id = f"well{chr(ord('a') + wi)}"
        n_rows = rng.randint(30, 60)
        md = np.linspace(1000, 1000 + n_rows, n_rows)
        x = np.cumsum(rng.randn(n_rows) * 0.5)
        y = np.cumsum(rng.randn(n_rows) * 0.5)
        z = np.linspace(-100 - wi * 5, -100 - wi * 5 - 5, n_rows)
        gr = np.clip(rng.normal(100, 20, n_rows), 0, 200).astype(float)
        ps_pos = rng.randint(8, 14)
        tvt_input = np.array([float(10000 + i) for i in range(ps_pos)] + [np.nan] * (n_rows - ps_pos))
        tvt = np.array(
            [float(10000 + i) for i in range(ps_pos)]
            + [float(10000 + ps_pos + (i - ps_pos) * 0.5 + rng.randn() * 0.1)
               for i in range(ps_pos, n_rows)]
        )
        df = pd.DataFrame({
            "MD": md, "X": x, "Y": y, "Z": z, "GR": gr,
            "TVT": tvt, "TVT_input": tvt_input,
        })
        df.to_csv(root / "train" / f"{well_id}__horizontal_well.csv", index=False)

        # Test files (same structure, no TVT)
        df_test = df.drop(columns=["TVT"])
        df_test.to_csv(root / "test" / f"{well_id}__horizontal_well.csv", index=False)

        # Submission IDs (post-PS rows only)
        ps_idx = ps_pos
        for row_i in range(ps_idx, n_rows):
            sub_ids.append(f"{well_id}_{row_i}")

    (root / "sample_submission.csv").write_text(
        "id,tvt\n" + "\n".join(f"{sid},0.0" for sid in sub_ids),
        encoding="utf-8",
    )


def test_train_tcn_smoke(tmp_path: Path) -> None:
    _write_synthetic_train_data(tmp_path, n_wells=8)
    result = train_tcn(
        data_dir=str(tmp_path),
        window_size=8,
        num_channels=(8, 16),
        epochs=2,
        batch_size=16,
        n_splits=2,
        seed=42,
        device="cpu",
    )
    assert isinstance(result, TrainResult)
    assert len(result.models) == 1
    assert len(result.cv_rmse_folds) == 2
    assert result.cv_rmse_mean > 0
    assert result.oof_df is not None
    assert list(result.oof_df.columns) == ["well_id", "row_idx", "fold", "y_true", "y_pred", "baseline"]


def test_tcn_model_io_roundtrip(tmp_path: Path) -> None:
    _write_synthetic_train_data(tmp_path, n_wells=6)
    result = train_tcn(
        data_dir=str(tmp_path),
        window_size=8,
        num_channels=(8, 16),
        epochs=2,
        batch_size=16,
        n_splits=2,
        seed=42,
        device="cpu",
    )

    model = result.models[0].cpu()
    state_dict = {k: v for k, v in model.state_dict().items()}
    tcn_meta = result.tcn_metadata or {}

    payload = build_model_payload(
        models=result.models,
        feature_columns=result.feature_columns,
        residual_target=result.residual_target,
        model_type="tcn",
        run_name="test_tcn_io",
        seed_list=[42],
        n_splits=2,
        tcn_state_dict=state_dict,
        tcn_target_scaler=tcn_meta.get("y_scaler"),
        tcn_window_size=8,
        tcn_feature_columns=result.feature_columns,
        tcn_num_channels=[8, 16],
        tcn_input_scaler=tcn_meta.get("x_scaler"),
        tcn_input_size=tcn_meta.get("input_size"),
    )

    model_path = tmp_path / "test_tcn.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    contract = resolve_prediction_contract(loaded)
    assert contract.model_metadata is not None
    assert contract.model_metadata["type"] == "tcn"
    assert contract.model_metadata["window_size"] == 8
    assert contract.model_metadata["input_scaler"] is not None  # Phase 2: x_scaler must be present


def test_predict_tcn_smoke(tmp_path: Path) -> None:
    _write_synthetic_train_data(tmp_path, n_wells=6)
    result = train_tcn(
        data_dir=str(tmp_path),
        window_size=8,
        num_channels=(8, 16),
        epochs=3,
        batch_size=16,
        n_splits=2,
        seed=42,
        device="cpu",
    )

    model_cpu = result.models[0].cpu()
    meta = result.tcn_metadata or {}
    submission = predict_tcn(
        data_dir=str(tmp_path),
        model=model_cpu,
        scaler=None,
        window_size=8,
        feature_columns=result.feature_columns,
        input_scaler=meta.get("x_scaler"),
        residual_target=result.residual_target,
        device="cpu",
    )

    assert list(submission.columns) == ["id", "tvt"]
    assert len(submission) > 0
    assert not submission["tvt"].isna().any()


def test_global_x_scaler_saved_in_metadata(tmp_path: Path) -> None:
    """Phase 2: verify that train_tcn stores the global x_scaler in tcn_metadata."""
    _write_synthetic_train_data(tmp_path, n_wells=6)
    result = train_tcn(
        data_dir=str(tmp_path),
        window_size=8,
        num_channels=(8, 16),
        epochs=2,
        batch_size=16,
        n_splits=2,
        seed=42,
        device="cpu",
    )
    meta = result.tcn_metadata or {}
    assert "x_scaler" in meta, "Phase 2: tcn_metadata must contain x_scaler"
    from sklearn.preprocessing import StandardScaler
    assert isinstance(meta["x_scaler"], StandardScaler), "x_scaler should be StandardScaler"
    assert meta["x_scaler"].n_features_in_ == 4, "x_scaler must have 4 features (X,Y,Z,MD)"
    # input_size must reflect the 4 extra channels
    assert meta["input_size"] == 69, f"Expected input_size=69 (65 seq + 4 abs), got {meta['input_size']}"


def test_x_scaler_roundtrip_predict(tmp_path: Path) -> None:
    """Phase 2: train with x_scaler, save payload, load, predict — scaler must survive."""
    _write_synthetic_train_data(tmp_path, n_wells=6)
    result = train_tcn(
        data_dir=str(tmp_path),
        window_size=8,
        num_channels=(8, 16),
        epochs=3,
        batch_size=16,
        n_splits=2,
        seed=42,
        device="cpu",
    )

    meta = result.tcn_metadata or {}
    assert meta.get("x_scaler") is not None
    assert meta["input_size"] == 69

    model_cpu = result.models[0].cpu()
    state_dict = {k: v for k, v in model_cpu.state_dict().items()}

    payload = build_model_payload(
        models=result.models,
        feature_columns=result.feature_columns,
        residual_target=result.residual_target,
        model_type="tcn",
        run_name="test_roundtrip",
        seed_list=[42],
        n_splits=2,
        tcn_state_dict=state_dict,
        tcn_target_scaler=meta.get("y_scaler"),
        tcn_window_size=8,
        tcn_feature_columns=result.feature_columns,
        tcn_num_channels=meta.get("num_channels", [8, 16]),
        tcn_input_scaler=meta.get("x_scaler"),
        tcn_input_size=meta.get("input_size"),
    )

    model_path = tmp_path / "test_tcn.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    with open(model_path, "rb") as f:
        loaded = pickle.load(f)

    contract = resolve_prediction_contract(loaded)
    assert contract.model_metadata is not None
    saved_input_scaler = contract.model_metadata.get("input_scaler")
    assert saved_input_scaler is not None

    input_size = contract.model_metadata.get("input_size", len(contract.model_metadata["feature_columns"]))
    tcn_model = TCNModel(
        input_size=input_size,
        num_channels=contract.model_metadata["num_channels"],
        kernel_size=contract.model_metadata.get("kernel_size", 5),
        dropout=contract.model_metadata.get("dropout", 0.1),
    )
    tcn_model.load_state_dict(contract.model_metadata["state_dict"])

    submission = predict_tcn(
        data_dir=str(tmp_path),
        model=tcn_model,
        scaler=contract.model_metadata.get("target_scaler"),
        window_size=contract.model_metadata["window_size"],
        feature_columns=contract.model_metadata["feature_columns"],
        input_scaler=saved_input_scaler,
        residual_target=contract.residual_target,
        device="cpu",
    )

    assert list(submission.columns) == ["id", "tvt"]
    assert len(submission) > 0
    assert not submission["tvt"].isna().any()
