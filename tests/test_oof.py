import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from rogii.oof import OOF_COLUMNS, load_oof, save_oof


def _make_oof_df(n_rows: int = 10, baseline_val: float = 12345.0) -> pd.DataFrame:
    return pd.DataFrame({
        "well_id": [f"well_{i % 3}" for i in range(n_rows)],
        "row_idx": list(range(n_rows)),
        "fold": [i % 5 for i in range(n_rows)],
        "y_true": np.random.randn(n_rows).astype(float),
        "y_pred": np.random.randn(n_rows).astype(float),
        "baseline": [baseline_val if i % 2 == 0 else 0.0 for i in range(n_rows)],
    })


def test_save_load_roundtrip(tmp_path: Path) -> None:
    df = _make_oof_df(20)
    path = save_oof(df, tmp_path, "test_run")
    loaded = load_oof(path)
    assert loaded.equals(df)


def test_column_contract(tmp_path: Path) -> None:
    df = _make_oof_df(5)
    path = save_oof(df, tmp_path, "test_contract")
    loaded = load_oof(path)
    assert list(loaded.columns) == OOF_COLUMNS


def test_delta_space_zero_baseline(tmp_path: Path) -> None:
    df = _make_oof_df(5, baseline_val=0.0)
    path = save_oof(df, tmp_path, "test_delta")
    loaded = load_oof(path)
    assert (loaded["baseline"] == 0.0).all()


def test_delta_space_nonzero_baseline(tmp_path: Path) -> None:
    df = _make_oof_df(5, baseline_val=10050.0)
    path = save_oof(df, tmp_path, "test_nonzero")
    loaded = load_oof(path)
    assert (loaded.loc[0::2, "baseline"] == 10050.0).all()
    assert (loaded.loc[1::2, "baseline"] == 0.0).all()


def test_load_oof_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_oof("nonexistent.parquet")


def test_load_oof_directory_raises(tmp_path: Path) -> None:
    (tmp_path / "subdir").mkdir()
    with pytest.raises((FileNotFoundError, ValueError)):
        load_oof(tmp_path)


def test_oof_dir_convention(tmp_path: Path) -> None:
    df = _make_oof_df(3)
    path = save_oof(df, tmp_path, "my_strategy")
    assert "my_strategy_oof.parquet" == Path(path).name
