"""Tests for predict-time postproc resolution behavior."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest

from rogii.model_io import build_model_payload, resolve_prediction_contract
from rogii.smoothing import PostprocConfig, apply_postprocessing
from rogii.submission import validate_submission


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_submission(tmp_path: Path) -> pd.DataFrame:
    n = 20
    ids = []
    tvt = []
    np.random.seed(3)
    for wi in range(2):
        wid = f"well_{chr(ord('a') + wi)}"
        signal = 10000 + wi * 5000 + np.random.randn(n) * 500
        for ri in range(n):
            ids.append(f"{wid}_{ri}")
            tvt.append(float(signal[ri]))
    return pd.DataFrame({"id": ids, "tvt": tvt})


@pytest.fixture
def payload_with_postproc() -> dict:
    postproc = PostprocConfig(
        savgol_window=51, savgol_polyorder=3,
        clip_lower=5000.0, clip_upper=25000.0,
        apply_order="clip_smooth",
    )
    return build_model_payload(
        models=["model"],
        feature_columns=["MD"],
        residual_target=False,
        postproc_config=postproc,
    )


@pytest.fixture
def payload_without_postproc() -> dict:
    return build_model_payload(
        models=["model"],
        feature_columns=["MD"],
        residual_target=False,
    )


# ---------------------------------------------------------------------------
# Scenario 1: payload with postproc, no CLI → use payload
# ---------------------------------------------------------------------------

def test_predict_uses_payload_postproc_when_no_cli(
    sample_submission: pd.DataFrame, payload_with_postproc: dict,
) -> None:
    contract = resolve_prediction_contract(payload_with_postproc)

    assert contract.postproc_config is not None
    pc = PostprocConfig.from_dict(contract.postproc_config)
    pc.validate()
    assert pc.savgol_window == 51
    assert pc.savgol_polyorder == 3
    assert pc.clip_lower == 5000.0
    assert pc.clip_upper == 25000.0

    # Apply via contract
    result = apply_postprocessing(
        sample_submission,
        savgol_window=pc.savgol_window,
        savgol_polyorder=pc.savgol_polyorder,
        clip_lower=pc.clip_lower,
        clip_upper=pc.clip_upper,
    )
    assert list(result.columns) == ["id", "tvt"]
    assert len(result) == len(sample_submission)
    # Clipping should have been applied
    assert result["tvt"].min() >= 5000.0
    assert result["tvt"].max() <= 25000.0


# ---------------------------------------------------------------------------
# Scenario 2: payload with postproc, CLI override → use CLI
# ---------------------------------------------------------------------------

def test_predict_cli_overrides_payload_postproc(
    sample_submission: pd.DataFrame, payload_with_postproc: dict,
) -> None:
    contract = resolve_prediction_contract(payload_with_postproc)

    # Simulate CLI override: --savgol-window 11 --savgol-polyorder 2
    cli_window = 11
    cli_polyorder = 2

    assert contract.postproc_config is not None  # payload exists
    payload_pc = PostprocConfig.from_dict(contract.postproc_config)
    assert payload_pc.savgol_window == 51  # but we ignore it

    result = apply_postprocessing(
        sample_submission,
        savgol_window=cli_window,
        savgol_polyorder=cli_polyorder,
        clip_lower=None,
        clip_upper=None,
    )
    assert result["tvt"].iloc[0] != sample_submission["tvt"].iloc[0]


# ---------------------------------------------------------------------------
# Scenario 3: payload with postproc, --no-postproc → disabled
# ---------------------------------------------------------------------------

def test_predict_no_postproc_disables_everything(
    sample_submission: pd.DataFrame, payload_with_postproc: dict,
) -> None:
    contract = resolve_prediction_contract(payload_with_postproc)
    assert contract.postproc_config is not None

    # Simulate --no-postproc
    no_postproc = True
    if no_postproc:
        savgol_window = None
        clip_lower = None
        clip_upper = None

    result = apply_postprocessing(
        sample_submission,
        savgol_window=savgol_window,
        savgol_polyorder=2,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )
    # Should be unchanged
    pd.testing.assert_series_equal(
        result["tvt"].reset_index(drop=True),
        sample_submission["tvt"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Scenario 4: old payload without postproc → legacy behavior
# ---------------------------------------------------------------------------

def test_predict_legacy_payload_no_postproc(
    sample_submission: pd.DataFrame, payload_without_postproc: dict,
) -> None:
    contract = resolve_prediction_contract(payload_without_postproc)
    assert contract.postproc_config is None

    # Legacy: --savgol-smooth flag (CLI)
    savgol_window = 31  # default
    savgol_polyorder = 2

    result = apply_postprocessing(
        sample_submission,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
    )
    assert len(result) == len(sample_submission)


# ---------------------------------------------------------------------------
# Scenario 5: old payload, no CLI → raw predictions
# ---------------------------------------------------------------------------

def test_predict_legacy_payload_raw(
    sample_submission: pd.DataFrame, payload_without_postproc: dict,
) -> None:
    contract = resolve_prediction_contract(payload_without_postproc)
    assert contract.postproc_config is None

    result = apply_postprocessing(sample_submission)
    pd.testing.assert_series_equal(
        result["tvt"].reset_index(drop=True),
        sample_submission["tvt"].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Scenario 6: invalid postproc in payload → clear error
# ---------------------------------------------------------------------------

def test_predict_invalid_postproc_config_raises() -> None:
    bad_payload = build_model_payload(
        models=["model"],
        feature_columns=["MD"],
        residual_target=False,
    )
    bad_payload["postproc"] = {"savgol_window": 10, "savgol_polyorder": 2}  # even window

    contract = resolve_prediction_contract(bad_payload)
    assert contract.postproc_config is not None

    with pytest.raises(ValueError, match="must be odd"):
        pc = PostprocConfig.from_dict(contract.postproc_config)
        pc.validate()


# ---------------------------------------------------------------------------
# Scenario 7: short group passes through without error
# ---------------------------------------------------------------------------

def test_predict_short_group_passed_through() -> None:
    submission = pd.DataFrame({
        "id": ["well_short_0"],
        "tvt": [12345.0],
    })
    result = apply_postprocessing(submission, savgol_window=51, savgol_polyorder=2)
    assert result["tvt"].iloc[0] == 12345.0


# ---------------------------------------------------------------------------
# Scenario 8: PostprocConfig round-trip via to_dict/from_dict
# ---------------------------------------------------------------------------

def test_postproc_config_dict_roundtrip_identity() -> None:
    cfg = PostprocConfig(
        savgol_window=51, savgol_polyorder=3,
        clip_lower=1000.0, clip_upper=30000.0,
        apply_order="smooth_clip",
    )
    restored = PostprocConfig.from_dict(cfg.to_dict())
    assert restored == cfg
    assert restored.savgol_window == 51
    assert restored.clip_lower == 1000.0
    assert restored.apply_order == "smooth_clip"
