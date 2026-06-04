"""Submission validation contracts."""

from pathlib import Path


def require_submission_file(path: str | Path) -> Path:
    """Return a submission path if it exists, otherwise raise."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Submission file does not exist: {resolved}")
    return resolved
