import pytest

from rogii.model_io import (
    build_model_payload,
    make_feature_flags,
    resolve_prediction_contract,
    validate_feature_columns,
)


def test_model_payload_stores_feature_contract() -> None:
    flags = make_feature_flags(include_geometry=True, include_gr=True)
    payload = build_model_payload(
        models=["model"],
        feature_columns=["MD", "GR"],
        residual_target=True,
        feature_flags=flags,
        run_name="test_run",
        seed_list=[42],
        n_splits=5,
    )

    assert payload["payload_version"] == 2
    assert payload["feature_columns"] == ["MD", "GR"]
    assert payload["feature_flags"] == {
        "include_tvt_input": False,
        "include_geometry": True,
        "include_gr": True,
        "include_trajectory": False,
        "include_typewell": False,
        "include_gr_dwt": False,
        "include_spatial": False,
        "include_dtw": False,
        "include_geology": False,
        "include_beam": False,
        "include_formation_plane": False,
        "include_z_drift": False,
    }
    assert payload["residual_target"] is True


def test_resolve_prediction_contract_rejects_v2_feature_override() -> None:
    payload = build_model_payload(
        models=["model"],
        feature_columns=["MD", "GR"],
        residual_target=False,
        feature_flags=make_feature_flags(include_gr=False),
    )

    with pytest.raises(ValueError, match="feature override"):
        resolve_prediction_contract(
            payload,
            cli_feature_flags=make_feature_flags(include_gr=True),
        )


def test_resolve_prediction_contract_allows_legacy_cli_flags() -> None:
    legacy_payload = {"model": "model", "include_geometry": True, "residual_target": True}

    contract = resolve_prediction_contract(
        legacy_payload,
        cli_feature_flags=make_feature_flags(include_tvt_input=True),
    )

    assert contract.models == ["model"]
    assert contract.residual_target is True
    assert contract.feature_flags["include_geometry"] is True
    assert contract.feature_flags["include_tvt_input"] is True
    assert contract.feature_columns is None
    assert contract.is_multi_seed is False


def test_validate_feature_columns_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="Feature columns do not match"):
        validate_feature_columns(["MD", "GR", "extra"], ["MD", "GR"])


def test_validate_feature_columns_returns_expected_order() -> None:
    assert validate_feature_columns(["MD", "GR"], ["MD", "GR"]) == ["MD", "GR"]


def test_tcn_payload_preserves_architecture_metadata() -> None:
    payload = build_model_payload(
        models=["model"],
        feature_columns=["X", "Y"],
        residual_target=True,
        model_type="tcn",
        tcn_state_dict={"weight": "value"},
        tcn_target_scaler="scaler",
        tcn_window_size=64,
        tcn_feature_columns=["X", "Y"],
        tcn_num_channels=[16, 32],
        tcn_kernel_size=3,
        tcn_dropout=0.0,
        tcn_input_scaler="x_scaler",
    )

    contract = resolve_prediction_contract(payload)

    assert contract.model_metadata == {
        "type": "tcn",
        "state_dict": {"weight": "value"},
        "target_scaler": "scaler",
        "window_size": 64,
        "feature_columns": ["X", "Y"],
        "num_channels": [16, 32],
        "kernel_size": 3,
        "dropout": 0.0,
        "input_scaler": "x_scaler",
        "input_size": None,
    }


# --- Multi-model / multi-seed tests ---


def test_multi_model_payload_roundtrip() -> None:
    """Multi-model list is preserved in payload and contract."""
    payload = build_model_payload(
        models=["m1", "m2", "m3"],
        feature_columns=["MD", "GR"],
        residual_target=True,
        seed_list=[42, 7, 123],
    )
    contract = resolve_prediction_contract(payload)
    assert contract.models == ["m1", "m2", "m3"]
    assert contract.is_multi_seed is True


def test_single_model_backward_compat() -> None:
    """Old single-model payload loads as list of one."""
    payload = build_model_payload(
        models=["single_model"],
        feature_columns=["MD"],
        residual_target=False,
        seed_list=[42],
    )
    contract = resolve_prediction_contract(payload)
    assert contract.models == ["single_model"]
    assert contract.is_multi_seed is False


def test_legacy_raw_model_is_wrapped_in_list() -> None:
    """Non-dict payload (raw model) is wrapped in a list."""
    contract = resolve_prediction_contract("raw_model")
    assert contract.models == ["raw_model"]
    assert contract.is_multi_seed is False
