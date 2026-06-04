"""Data loading contracts.

Actual loaders must be implemented after the Kaggle files and schema are inspected.
"""

from pathlib import Path


def require_data_dir(data_dir: str | Path) -> Path:
    """Validate that a configured data directory exists."""
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    return path
