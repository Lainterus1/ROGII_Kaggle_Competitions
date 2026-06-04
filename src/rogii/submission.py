"""Submission validation helpers."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def require_submission_file(path: str | Path) -> Path:
    """Return a submission path if it exists, otherwise raise."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Submission file does not exist: {resolved}")
    return resolved


@dataclass(frozen=True)
class SubmissionValidationResult:
    """Submission validation summary."""

    rows: int
    id_column: str
    prediction_column: str


def validate_submission(sample_path: str | Path, submission_path: str | Path) -> SubmissionValidationResult:
    """Validate a candidate submission against `sample_submission.csv`."""
    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(require_submission_file(submission_path))

    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"Submission columns {list(submission.columns)} do not match sample {list(sample.columns)}")
    if len(submission) != len(sample):
        raise ValueError(f"Submission row count {len(submission)} does not match sample {len(sample)}")

    id_column = sample.columns[0]
    prediction_column = sample.columns[1]
    if not submission[id_column].astype(str).equals(sample[id_column].astype(str)):
        raise ValueError("Submission IDs do not match sample_submission.csv exactly and in order")

    predictions = pd.to_numeric(submission[prediction_column], errors="coerce")
    if predictions.isna().any():
        raise ValueError("Submission predictions contain NaN or non-numeric values")
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Submission predictions contain non-finite values")

    return SubmissionValidationResult(rows=len(submission), id_column=id_column, prediction_column=prediction_column)
