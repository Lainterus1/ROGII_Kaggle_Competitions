"""Model helpers for naive and classical ML baselines."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rogii.data_loading import list_well_ids, read_horizontal_well, read_sample_submission
from rogii.metrics import rmse


def baseline_model_order() -> list[str]:
    """Return the planned initial model order."""
    return ["naive", "lightgbm", "catboost", "xgboost"]


@dataclass(frozen=True)
class NaiveValidationResult:
    """Summary of local post-PS naive validation."""

    rmse: float
    rows: int
    wells: int


def last_known_tvt_input(horizontal: pd.DataFrame) -> float:
    """Return the final non-null `TVT_input` value before Prediction Start."""
    if "TVT_input" not in horizontal.columns:
        raise ValueError("Horizontal well data must contain TVT_input")
    known = horizontal["TVT_input"].dropna()
    if known.empty:
        raise ValueError("No non-null TVT_input values found")
    return float(known.iloc[-1])


def first_prediction_index(horizontal: pd.DataFrame) -> int:
    """Return the first row index after Prediction Start."""
    if "TVT_input" not in horizontal.columns:
        raise ValueError("Horizontal well data must contain TVT_input")
    missing = horizontal["TVT_input"].isna()
    if not missing.any():
        raise ValueError("No post-PS rows found; TVT_input is never null")
    return int(missing.idxmax())


def train_global_tvt_median(data_dir: str | Path) -> float:
    """Compute a fallback median from train horizontal `TVT` values."""
    values: list[float] = []
    for well_id in list_well_ids(data_dir, "train"):
        horizontal = read_horizontal_well(data_dir, "train", well_id)
        if "TVT" in horizontal.columns:
            values.extend(horizontal["TVT"].dropna().astype(float).tolist())
    if not values:
        raise ValueError("No train TVT values found for fallback median")
    return float(np.median(values))


def parse_submission_id(value: str) -> tuple[str, int]:
    """Parse a sample submission id of the form `<well_id>_<row_index>`."""
    try:
        well_id, row_index = value.rsplit("_", 1)
        return well_id, int(row_index)
    except ValueError as exc:
        raise ValueError(f"Invalid submission id: {value}") from exc


def predict_last_known_submission(data_dir: str | Path) -> pd.DataFrame:
    """Create a naive submission using the last known pre-PS `TVT_input` per test well."""
    sample = read_sample_submission(data_dir)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")

    fallback = train_global_tvt_median(data_dir)
    constants: dict[str, float] = {}
    frames: dict[str, pd.DataFrame] = {}
    predictions: list[float] = []
    for submission_id in sample["id"].astype(str):
        well_id, row_index = parse_submission_id(submission_id)
        if well_id not in constants:
            horizontal = read_horizontal_well(data_dir, "test", well_id)
            frames[well_id] = horizontal
            try:
                constants[well_id] = last_known_tvt_input(horizontal)
            except ValueError:
                constants[well_id] = fallback
        horizontal = frames[well_id]
        if row_index < 0 or row_index >= len(horizontal):
            raise IndexError(f"Submission row index out of bounds for {submission_id}")
        if pd.notna(horizontal.loc[row_index, "TVT_input"]):
            raise ValueError(f"Submission id points to pre-PS row with known TVT_input: {submission_id}")
        predictions.append(constants[well_id])

    return pd.DataFrame({"id": sample["id"].astype(str), "tvt": predictions})


def validate_last_known_baseline(data_dir: str | Path) -> NaiveValidationResult:
    """Evaluate the last-known-`TVT_input` rule on train post-PS rows."""
    y_true: list[float] = []
    y_pred: list[float] = []
    wells = 0
    for well_id in list_well_ids(data_dir, "train"):
        horizontal = read_horizontal_well(data_dir, "train", well_id)
        if "TVT" not in horizontal.columns:
            continue
        try:
            start = first_prediction_index(horizontal)
            constant = last_known_tvt_input(horizontal)
        except ValueError:
            continue
        target = horizontal.loc[start:, "TVT"].dropna().astype(float)
        if target.empty:
            continue
        y_true.extend(target.tolist())
        y_pred.extend([constant] * len(target))
        wells += 1
    return NaiveValidationResult(rmse=rmse(y_true, y_pred), rows=len(y_true), wells=wells)
