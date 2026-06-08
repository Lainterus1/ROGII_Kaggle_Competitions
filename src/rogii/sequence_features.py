"""Causal sequence features from raw wellbore coordinates.

All features at row i depend only on rows 0..i (causal, no look-ahead).
"""

import numpy as np
import pandas as pd

RAW_COLS = ["X", "Y", "Z", "GR", "MD"]


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    feats = pd.DataFrame(index=df.index)

    for col in RAW_COLS:
        series = df[col].astype(float)
        # Fill NaN (e.g. missing GR values) with forward-then-backward fill
        series = series.ffill().bfill()
        if series.isna().any():
            series = series.fillna(0.0)
        feats[col] = series

        feats[f"{col}_diff1"] = series.diff(1).fillna(0.0)
        feats[f"{col}_diff2"] = series.diff(2).fillna(0.0)

        for lag in [1, 2, 3, 5]:
            feats[f"{col}_lag{lag}"] = series.shift(lag).fillna(0.0)

        shifted = series.shift(1).bfill()
        for w in [3, 5, 10]:
            roll = shifted.rolling(window=w, min_periods=1)
            feats[f"{col}_roll_mean{w}"] = roll.mean().fillna(0.0).astype(float)
            feats[f"{col}_roll_std{w}"] = roll.std().fillna(0.0).astype(float)

    return feats
