"""Model payload contracts for train/predict handoff."""

from dataclasses import dataclass
from typing import Any


FEATURE_FLAG_KEYS = ("include_tvt_input", "include_geometry", "include_gr", "include_trajectory", "include_typewell")


@dataclass(frozen=True)
class PredictionContract:
    """Resolved model and feature contract used at prediction time."""

    model: Any
    residual_target: bool
    feature_flags: dict[str, bool]
    feature_columns: list[str] | None


def make_feature_flags(
    *,
    include_tvt_input: bool = False,
    include_geometry: bool = False,
    include_gr: bool = False,
    include_trajectory: bool = False,
    include_typewell: bool = False,
) -> dict[str, bool]:
    """Return normalized feature flags stored in model payloads."""
    return {
        "include_tvt_input": bool(include_tvt_input),
        "include_geometry": bool(include_geometry),
        "include_gr": bool(include_gr),
        "include_trajectory": bool(include_trajectory),
        "include_typewell": bool(include_typewell),
    }


def build_model_payload(
    *,
    model: Any,
    feature_columns: list[str],
    residual_target: bool,
    feature_flags: dict[str, bool],
    model_type: str = "lightgbm",
    run_name: str | None = None,
    seed: int | None = None,
    n_splits: int | None = None,
    cv_rmse_mean: float | None = None,
    cv_rmse_std: float | None = None,
    cv_rmse_folds: list[float] | None = None,
    train_rows: int | None = None,
    train_wells: int | None = None,
    config_path: str | None = None,
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned payload that prevents train/predict feature drift."""
    flags = make_feature_flags(**{key: feature_flags.get(key, False) for key in FEATURE_FLAG_KEYS})
    return {
        "payload_version": 2,
        "model": model,
        "model_type": model_type,
        "run_name": run_name,
        "feature_columns": list(feature_columns),
        "feature_flags": flags,
        "residual_target": bool(residual_target),
        "seed": seed,
        "n_splits": n_splits,
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "cv_rmse_folds": cv_rmse_folds,
        "train_rows": train_rows,
        "train_wells": train_wells,
        "config_path": config_path,
        "model_params": model_params or {},
        # Legacy top-level keys keep older helper code readable.
        "include_tvt_input": flags["include_tvt_input"],
        "include_geometry": flags["include_geometry"],
        "include_gr": flags["include_gr"],
        "include_trajectory": flags["include_trajectory"],
        "include_typewell": flags["include_typewell"],
    }


def resolve_prediction_contract(
    payload: Any,
    *,
    cli_feature_flags: dict[str, bool] | None = None,
    cli_residual_target: bool = False,
) -> PredictionContract:
    """Resolve model metadata and reject unsafe train/predict flag mismatches."""
    cli_flags = make_feature_flags(**(cli_feature_flags or {}))
    if not isinstance(payload, dict):
        return PredictionContract(
            model=payload,
            residual_target=bool(cli_residual_target),
            feature_flags=cli_flags,
            feature_columns=None,
        )

    model = payload["model"]
    has_v2_contract = "feature_flags" in payload or "feature_columns" in payload
    payload_flags = _payload_feature_flags(payload)
    residual_target = bool(payload.get("residual_target", False))

    if has_v2_contract:
        _reject_cli_overrides(payload_flags, cli_flags)
        if cli_residual_target and not residual_target:
            raise ValueError("Model payload was trained without residual target; do not override target mode at predict time")
    else:
        # Legacy dict payloads did not store include_tvt_input, so allow CLI flags to fill missing metadata.
        payload_flags = {key: payload_flags[key] or cli_flags[key] for key in FEATURE_FLAG_KEYS}
        residual_target = residual_target or bool(cli_residual_target)

    feature_columns = payload.get("feature_columns")
    if feature_columns is not None:
        feature_columns = list(feature_columns)

    return PredictionContract(
        model=model,
        residual_target=residual_target,
        feature_flags=payload_flags,
        feature_columns=feature_columns,
    )


def validate_feature_columns(actual_columns: list[str], expected_columns: list[str] | None) -> list[str]:
    """Validate and return prediction column order expected by the trained model."""
    if expected_columns is None:
        return actual_columns
    if actual_columns != expected_columns:
        missing = [col for col in expected_columns if col not in actual_columns]
        extra = [col for col in actual_columns if col not in expected_columns]
        raise ValueError(
            "Feature columns do not match trained model payload. "
            f"missing={missing} extra={extra} expected={expected_columns} actual={actual_columns}"
        )
    return expected_columns


def _payload_feature_flags(payload: dict[str, Any]) -> dict[str, bool]:
    stored = payload.get("feature_flags") or {}
    return make_feature_flags(
        include_tvt_input=stored.get("include_tvt_input", payload.get("include_tvt_input", False)),
        include_geometry=stored.get("include_geometry", payload.get("include_geometry", False)),
        include_gr=stored.get("include_gr", payload.get("include_gr", False)),
        include_trajectory=stored.get("include_trajectory", payload.get("include_trajectory", False)),
        include_typewell=stored.get("include_typewell", payload.get("include_typewell", False)),
    )


def _reject_cli_overrides(payload_flags: dict[str, bool], cli_flags: dict[str, bool]) -> None:
    conflicts = [key for key in FEATURE_FLAG_KEYS if cli_flags[key] and not payload_flags[key]]
    if conflicts:
        raise ValueError(
            "Model payload already defines feature flags; refusing predict-time feature override: "
            + ", ".join(conflicts)
        )
