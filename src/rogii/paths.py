"""Path helpers for local and Kaggle execution."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(*parts: str) -> Path:
    """Build a path relative to the repository root."""
    return PROJECT_ROOT.joinpath(*parts)
