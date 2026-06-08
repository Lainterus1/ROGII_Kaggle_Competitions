"""Out-of-fold prediction persistence for multi-model blending."""

from pathlib import Path

import pandas as pd

OOF_COLUMNS = ["well_id", "row_idx", "fold", "y_true", "y_pred", "baseline"]
OOF_DIR = "outputs/oof"


def save_oof(df: pd.DataFrame, path: str | Path, strategy_name: str) -> Path:
    dir_path = Path(path)
    dir_path = Path(dir_path) if dir_path.suffix else Path(dir_path) / OOF_DIR
    dir_path.mkdir(parents=True, exist_ok=True)
    filepath = dir_path / f"{strategy_name}_oof.parquet"
    df.to_parquet(filepath, index=False)
    return filepath


def load_oof(path: str | Path) -> pd.DataFrame:
    filepath = Path(path)
    if filepath.is_dir():
        parquet_files = list(filepath.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No .parquet files found in {filepath}")
        raise ValueError(
            f"Path is a directory with {len(parquet_files)} parquet files. "
            f"Specify a single .parquet file. Candidates: {[p.name for p in parquet_files]}"
        )
    if not filepath.is_file():
        raise FileNotFoundError(f"OOF file not found: {filepath}")
    return pd.read_parquet(filepath)
