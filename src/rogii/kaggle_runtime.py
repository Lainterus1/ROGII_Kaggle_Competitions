"""Kaggle runtime path discovery for offline inference notebooks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


DEFAULT_COMPETITION_SLUG = "rogii-wellbore-geology-prediction"
DEFAULT_REPO_DATASET_SLUG = "rogii-repo-v2"
DEFAULT_MODEL_DATASET_SLUG = "rogii-models-v2"
DEFAULT_MODEL_FILE = "baseline_lgbm.pkl"


@dataclass(frozen=True)
class KaggleInferencePaths:
    """Resolved paths needed by the offline Kaggle inference runner."""

    repo_root: Path
    data_dir: Path
    model_path: Path
    output_path: Path


def find_repo_root(input_root: str | Path, preferred_slug: str = DEFAULT_REPO_DATASET_SLUG) -> Path:
    """Find a mounted repository dataset root.

    Supports both flat Kaggle Dataset layouts and nested notebook-output layouts.
    A valid repository root must contain `scripts/run_predict.py` and `src/rogii`.
    """
    root = _require_dir(input_root, "input_root")
    candidates: list[Path] = []
    for current, dirs, _ in os.walk(root):
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__", ".ipynb_checkpoints"}]
        path = Path(current)
        if (path / "scripts" / "run_predict.py").is_file() and (path / "src" / "rogii").is_dir():
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find repo dataset under {root}. Attach {preferred_slug} and ensure it contains scripts/run_predict.py."
        )
    return sorted(candidates, key=lambda path: _path_score(path, preferred_slug))[0]


def find_model_path(
    input_root: str | Path,
    model_file: str = DEFAULT_MODEL_FILE,
    preferred_slug: str = DEFAULT_MODEL_DATASET_SLUG,
) -> Path:
    """Find a mounted model artifact file."""
    root = _require_dir(input_root, "input_root")
    candidates = [path for path in root.rglob(model_file) if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {model_file} under {root}. Attach {preferred_slug} or upload a model dataset."
        )
    return sorted(candidates, key=lambda path: _path_score(path, preferred_slug))[0]


def find_competition_data_dir(
    input_root: str | Path,
    competition_slug: str = DEFAULT_COMPETITION_SLUG,
) -> Path:
    """Find the competition data directory needed for prediction.

    Inference only requires `sample_submission.csv` and `test/`. The train split may
    be absent or shaped differently in Kaggle hidden reruns, so it is not required here.
    """
    root = _require_dir(input_root, "input_root")
    candidates: list[Path] = []
    for current, dirs, files in os.walk(root):
        path = Path(current)
        if "sample_submission.csv" in files and "test" in dirs:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            f"Could not find competition data under {root}. Attach competition source {competition_slug}."
        )
    return sorted(candidates, key=lambda path: _path_score(path, competition_slug))[0]


def resolve_inference_paths(
    input_root: str | Path = "/kaggle/input",
    output_path: str | Path = "/kaggle/working/submission.csv",
    *,
    repo_dataset_slug: str = DEFAULT_REPO_DATASET_SLUG,
    model_dataset_slug: str = DEFAULT_MODEL_DATASET_SLUG,
    model_file: str = DEFAULT_MODEL_FILE,
    competition_slug: str = DEFAULT_COMPETITION_SLUG,
) -> KaggleInferencePaths:
    """Resolve all paths for offline Kaggle prediction."""
    root = _require_dir(input_root, "input_root")
    return KaggleInferencePaths(
        repo_root=find_repo_root(root, preferred_slug=repo_dataset_slug),
        data_dir=find_competition_data_dir(root, competition_slug=competition_slug),
        model_path=find_model_path(root, model_file=model_file, preferred_slug=model_dataset_slug),
        output_path=Path(output_path),
    )


def build_predict_command(paths: KaggleInferencePaths, python_executable: str | None = None) -> list[str]:
    """Build the `run_predict.py` command used by the offline notebook."""
    executable = python_executable or sys.executable
    return [
        executable,
        str(paths.repo_root / "scripts" / "run_predict.py"),
        "--data-dir",
        str(paths.data_dir),
        "--model",
        str(paths.model_path),
        "--output",
        str(paths.output_path),
    ]


def require_nonempty_file(path: str | Path) -> Path:
    """Return path if it exists and is non-empty; otherwise fail loudly."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected output file does not exist: {resolved}")
    if resolved.stat().st_size <= 0:
        raise ValueError(f"Expected output file is empty: {resolved}")
    return resolved


def _require_dir(path: str | Path, name: str) -> Path:
    resolved = Path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {resolved}")
    return resolved


def _path_score(path: Path, preferred_slug: str) -> tuple[int, int, str]:
    text = str(path).replace("\\", "/").lower()
    slug = preferred_slug.lower()
    return (0 if slug in text else 1, len(path.parts), text)
