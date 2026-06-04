"""MLflow tracking helpers for ROGII baseline experiments."""

import os
from pathlib import Path
from typing import Any


def default_experiment_name() -> str:
    """Return the default MLflow experiment name."""
    return "rogii-wellbore-baseline"


def _import_mlflow():
    try:
        import mlflow
    except ImportError:
        raise ImportError("MLflow is required for experiment tracking. Install with: pip install mlflow")
    return mlflow


def setup_tracking(tracking_uri: str | None = None) -> None:
    """Configure MLflow tracking URI."""
    mlflow = _import_mlflow()
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri)


def start_run(run_name: str, experiment_name: str | None = None) -> Any:
    """Start an MLflow run with the given experiment."""
    mlflow = _import_mlflow()
    exp_name = experiment_name or default_experiment_name()
    mlflow.set_experiment(exp_name)
    return mlflow.start_run(run_name=run_name)


def log_params(params: dict[str, Any], run: Any | None = None) -> None:
    """Log parameters to the active MLflow run."""
    mlflow = _import_mlflow()
    if run is not None:
        mlflow.log_params(params, run_id=run.info.run_id)
    else:
        mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], step: int | None = None, run: Any | None = None) -> None:
    """Log metrics to the active MLflow run."""
    mlflow = _import_mlflow()
    if run is not None:
        mlflow.log_metrics(metrics, step=step, run_id=run.info.run_id)
    else:
        mlflow.log_metrics(metrics, step=step)


def log_artifact(local_path: str | Path, run: Any | None = None) -> None:
    """Log a local file as an MLflow artifact."""
    mlflow = _import_mlflow()
    if run is not None:
        mlflow.log_artifact(str(local_path), run_id=run.info.run_id)
    else:
        mlflow.log_artifact(str(local_path))


def end_run(run: Any | None = None) -> None:
    """End the active MLflow run."""
    mlflow = _import_mlflow()
    if run is not None:
        mlflow.end_run(run.info.run_id)
    else:
        mlflow.end_run()
