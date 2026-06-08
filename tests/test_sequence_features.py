import numpy as np
import pandas as pd

from rogii.sequence_features import RAW_COLS, build_sequence_features


def _make_well_df(n_rows: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    md = np.linspace(1000, 1099, n_rows)
    return pd.DataFrame({
        "MD": md,
        "X": np.cumsum(np.random.randn(n_rows) * 0.5),
        "Y": np.cumsum(np.random.randn(n_rows) * 0.5),
        "Z": np.linspace(-100, -105, n_rows),
        "GR": np.clip(np.random.normal(100, 20, n_rows), 0, 200),
        "TVT": np.linspace(10000, 10050, n_rows),
        "TVT_input": [float(10000 + i) for i in range(5)] + [np.nan] * (n_rows - 5),
    })


def test_output_shape() -> None:
    df = _make_well_df(100)
    feats = build_sequence_features(df)
    assert feats.shape[0] == 100
    expected_cols = 5 + 10 + 20 + 30  # raw + diff + lag + rolling(mean+std)
    assert feats.shape[1] == expected_cols
    for col in RAW_COLS:
        assert col in feats.columns


def test_causal_construction() -> None:
    df = _make_well_df(20)
    # Modify row 15 drastically
    df_mod = df.copy()
    df_mod.loc[df_mod.index[15], "GR"] = 999.0
    feats_orig = build_sequence_features(df)
    feats_mod = build_sequence_features(df_mod)
    # Rows before 15 should be identical (they can't see the future)
    for i in range(15):
        np.testing.assert_array_almost_equal(
            feats_orig.iloc[i].values, feats_mod.iloc[i].values,
            decimal=10,
        )


def test_no_nan_values() -> None:
    df = _make_well_df(50)
    feats = build_sequence_features(df)
    assert not feats.isna().any().any()


def test_diff_features() -> None:
    df = _make_well_df(10)
    feats = build_sequence_features(df)
    assert "MD_diff1" in feats.columns
    assert "MD_diff2" in feats.columns


def test_lag_check() -> None:
    df = _make_well_df(10)
    feats = build_sequence_features(df)
    md_lag1 = feats["MD_lag1"].values
    md_orig = df["MD"].astype(float).values
    np.testing.assert_array_almost_equal(md_lag1[1:], md_orig[:-1], decimal=10)
