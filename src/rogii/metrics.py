"""Metric helpers."""

import numpy as np


def official_metric_name() -> str:
    """Return the known official metric name."""
    return "RMSE"


def rmse(y_true: object, y_pred: object) -> float:
    """Compute root mean squared error for finite numeric arrays."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"Shape mismatch: y_true={true.shape}, y_pred={pred.shape}")
    if true.size == 0:
        raise ValueError("RMSE requires at least one value")
    if not np.isfinite(true).all() or not np.isfinite(pred).all():
        raise ValueError("RMSE inputs must be finite")
    return float(np.sqrt(np.mean((true - pred) ** 2)))
