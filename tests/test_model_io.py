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
        model="model",
        feature_columns=["MD", "GR"],
        residual_target=True,
        feature_flags=flags,
        run_name="test_run",
        seed=42,
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
        }
    assert payload["residual_target"] is True


def test_resolve_prediction_contract_rejects_v2_feature_override() -> None:
    payload = build_model_payload(
        model="model",
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

    assert contract.model == "model"
    assert contract.residual_target is True
    assert contract.feature_flags["include_geometry"] is True
    assert contract.feature_flags["include_tvt_input"] is True
    assert contract.feature_columns is None


def test_validate_feature_columns_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="Feature columns do not match"):
        validate_feature_columns(["MD", "GR", "extra"], ["MD", "GR"])


def test_validate_feature_columns_returns_expected_order() -> None:
    assert validate_feature_columns(["MD", "GR"], ["MD", "GR"]) == ["MD", "GR"]
