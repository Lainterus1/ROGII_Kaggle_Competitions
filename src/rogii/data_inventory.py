"""Data inventory helpers.

This module will inspect Kaggle files after data is available.
"""

from pathlib import Path


def list_data_files(data_dir: str | Path) -> list[Path]:
    """List files under a data directory without reading their contents."""
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    return sorted(item for item in path.rglob("*") if item.is_file())
