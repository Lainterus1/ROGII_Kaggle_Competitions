"""Data loading helpers for the ROGII competition files."""

from pathlib import Path

import pandas as pd


HORIZONTAL_SUFFIX = "__horizontal_well.csv"
TYPEWELL_SUFFIX = "__typewell.csv"


def require_data_dir(data_dir: str | Path) -> Path:
    """Validate that a configured data directory exists."""
    path = Path(data_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    return path


def sample_submission_path(data_dir: str | Path) -> Path:
    """Return the sample submission path for a data directory."""
    path = require_data_dir(data_dir) / "sample_submission.csv"
    if not path.is_file():
        raise FileNotFoundError(f"sample_submission.csv does not exist: {path}")
    return path


def split_dir(data_dir: str | Path, split: str) -> Path:
    """Return a train/test split directory."""
    path = require_data_dir(data_dir) / split
    if not path.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {path}")
    return path


def parse_well_id(path: str | Path) -> str:
    """Extract the well id prefix from a ROGII well file name."""
    name = Path(path).name
    if "__" not in name:
        raise ValueError(f"Cannot parse well id from file name: {name}")
    return name.split("__", 1)[0]


def list_well_ids(data_dir: str | Path, split: str) -> list[str]:
    """List well ids with horizontal files for a split."""
    directory = split_dir(data_dir, split)
    return sorted(parse_well_id(path) for path in directory.glob(f"*{HORIZONTAL_SUFFIX}"))


def horizontal_well_path(data_dir: str | Path, split: str, well_id: str) -> Path:
    """Return a horizontal well CSV path."""
    path = split_dir(data_dir, split) / f"{well_id}{HORIZONTAL_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"Horizontal well file does not exist: {path}")
    return path


def typewell_path(data_dir: str | Path, split: str, well_id: str) -> Path:
    """Return a typewell CSV path."""
    path = split_dir(data_dir, split) / f"{well_id}{TYPEWELL_SUFFIX}"
    if not path.is_file():
        raise FileNotFoundError(f"Typewell file does not exist: {path}")
    return path


def read_horizontal_well(data_dir: str | Path, split: str, well_id: str) -> pd.DataFrame:
    """Read one horizontal well file and attach a `well_id` column."""
    frame = pd.read_csv(horizontal_well_path(data_dir, split, well_id))
    frame.insert(0, "well_id", well_id)
    return frame


def read_typewell(data_dir: str | Path, split: str, well_id: str) -> pd.DataFrame:
    """Read one typewell file and attach a `well_id` column."""
    frame = pd.read_csv(typewell_path(data_dir, split, well_id))
    frame.insert(0, "well_id", well_id)
    return frame


def read_sample_submission(data_dir: str | Path) -> pd.DataFrame:
    """Read sample submission."""
    return pd.read_csv(sample_submission_path(data_dir))
