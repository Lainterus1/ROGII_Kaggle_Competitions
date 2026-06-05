"""Prediction entry points for the ROGII ML baseline."""

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from rogii.data_loading import read_horizontal_well, read_sample_submission, read_typewell
from rogii.features import build_features, last_known_tvt_input_value, post_ps_mask
from rogii.model_io import validate_feature_columns
from rogii.models import parse_submission_id


def run_predict(
    data_dir: str | Path,
    model: LGBMRegressor,
    include_tvt_input: bool = False,
    include_geometry: bool = False,
    include_gr: bool = False,
    include_typewell: bool = False,
    residual_target: bool = False,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    sample = read_sample_submission(data_dir)
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"Unexpected sample submission columns: {list(sample.columns)}")

    use_tvt_feature = include_tvt_input and not residual_target
    well_cache: dict[str, pd.DataFrame] = {}
    last_tvt_cache: dict[str, float] = {}
    predictions: list[float] = []

    for submission_id in sample["id"].astype(str):
        well_id, row_index = parse_submission_id(submission_id)
        if well_id not in well_cache:
            horizontal = read_horizontal_well(data_dir, "test", well_id)
            typewell_frame = read_typewell(data_dir, "test", well_id) if include_typewell else None
            feats = build_features(
                horizontal,
                include_tvt_input=use_tvt_feature,
                include_geometry=include_geometry,
                include_gr=include_gr,
                typewell=typewell_frame,
                include_typewell=include_typewell,
            )
            ordered_columns = validate_feature_columns(list(feats.columns), feature_columns)
            feats = feats.loc[:, ordered_columns]
            well_cache[well_id] = feats
            if residual_target:
                last_tvt = last_known_tvt_input_value(horizontal)
                if np.isnan(last_tvt):
                    raise ValueError(f"Cannot compute residual prediction for {well_id}: no TVT_input")
                last_tvt_cache[well_id] = float(last_tvt)
        feats = well_cache[well_id]
        if row_index < 0 or row_index >= len(feats):
            raise IndexError(f"Submission row index out of bounds for {submission_id}")
        row_feats = feats.iloc[row_index : row_index + 1]
        pred = float(model.predict(row_feats)[0])
        if residual_target:
            pred = last_tvt_cache[well_id] + pred
        if not np.isfinite(pred):
            raise ValueError(f"Non-finite prediction for {submission_id}: {pred}")
        predictions.append(pred)

    return pd.DataFrame({"id": sample["id"].astype(str), "tvt": predictions})
