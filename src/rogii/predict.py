"""Prediction entry points for the ROGII ML baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from rogii.data_loading import read_horizontal_well, read_sample_submission
from rogii.features import build_features, post_ps_mask
from rogii.models import parse_submission_id


def run_predict(data_dir: str | Path, model: LGBMRegressor, include_tvt_input: bool = False) -> pd.DataFrame:
    sample = read_sample_submission(data_dir)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")

    well_cache: dict[str, pd.DataFrame] = {}
    predictions: list[float] = []

    for submission_id in sample["id"].astype(str):
        well_id, row_index = parse_submission_id(submission_id)
        if well_id not in well_cache:
            horizontal = read_horizontal_well(data_dir, "test", well_id)
            feats = build_features(horizontal, include_tvt_input=include_tvt_input)
            well_cache[well_id] = feats
        feats = well_cache[well_id]
        if row_index < 0 or row_index >= len(feats):
            raise IndexError(f"Submission row index out of bounds for {submission_id}")
        row_feats = feats.iloc[row_index : row_index + 1]
        pred = float(model.predict(row_feats)[0])
        if not np.isfinite(pred):
            raise ValueError(f"Non-finite prediction for {submission_id}: {pred}")
        predictions.append(pred)

    return pd.DataFrame({"id": sample["id"].astype(str), "tvt": predictions})
