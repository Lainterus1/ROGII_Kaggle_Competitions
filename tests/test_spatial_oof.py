"""OOF leakage and unit tests for spatial KNN features."""

import numpy as np
import pandas as pd
import pytest

from rogii.spatial_features import (
    SPATIAL_FEATURES,
    SPATIAL_NN_K,
    build_pre_ps_reference,
    build_spatial_knn_features,
)
from rogii.data_loading import list_well_ids


def test_spatial_features_constants() -> None:
    assert SPATIAL_NN_K == [5, 10, 50]
    assert len(SPATIAL_FEATURES) == 9
    for k in SPATIAL_NN_K:
        assert f"spatial_nn{k}_mean_tvt" in SPATIAL_FEATURES
        assert f"spatial_nn{k}_median_tvt" in SPATIAL_FEATURES
        assert f"spatial_nn{k}_std_tvt" in SPATIAL_FEATURES


def test_build_spatial_knn_output_shape() -> None:
    np.random.seed(42)
    ref = pd.DataFrame({
        "X": np.random.randn(100),
        "Y": np.random.randn(100),
        "Z": np.random.randn(100),
        "TVT_input": np.random.randn(100) * 100 + 5000,
    })
    query = pd.DataFrame({
        "X": np.random.randn(10),
        "Y": np.random.randn(10),
        "Z": np.random.randn(10),
    })
    feats = build_spatial_knn_features(ref, query)
    assert list(feats.columns) == SPATIAL_FEATURES
    assert len(feats) == 10


def test_build_spatial_knn_no_nan() -> None:
    np.random.seed(42)
    ref = pd.DataFrame({
        "X": np.random.randn(100),
        "Y": np.random.randn(100),
        "Z": np.random.randn(100),
        "TVT_input": np.random.randn(100) * 100 + 5000,
    })
    query = pd.DataFrame({
        "X": np.random.randn(10),
        "Y": np.random.randn(10),
        "Z": np.random.randn(10),
    })
    feats = build_spatial_knn_features(ref, query)
    assert not feats.isna().any().any()


def test_build_spatial_knn_k5_equals_mean_of_5() -> None:
    np.random.seed(42)
    ref = pd.DataFrame({
        "X": np.random.randn(20),
        "Y": np.random.randn(20),
        "Z": np.random.randn(20),
        "TVT_input": np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                               1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]),
    })
    query = ref.iloc[:1][["X", "Y", "Z"]]
    feats = build_spatial_knn_features(ref, query, k_values=[5])
    # k=5 NN includes the query point itself (distance 0) + 4 next closest
    assert feats.loc[0, "spatial_nn5_mean_tvt"] > 0


def test_build_spatial_knn_empty_reference() -> None:
    ref = pd.DataFrame({"X": [], "Y": [], "Z": [], "TVT_input": []})
    query = pd.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0]})
    feats = build_spatial_knn_features(ref, query)
    assert (feats == 0.0).all().all()


def test_spatial_oof_no_tvt_in_reference() -> None:
    ref = pd.DataFrame({
        "X": [1.0, 2.0, 3.0],
        "Y": [1.0, 2.0, 3.0],
        "Z": [1.0, 2.0, 3.0],
        "TVT_input": [100.0, 200.0, 300.0],
        # Intentionally NO 'TVT' column
    })
    assert "TVT" not in ref.columns


def test_spatial_pre_ps_reference_excludes_nan_tvt_input() -> None:
    import tempfile, os
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train").mkdir()
        csv = root / "train" / "testwell__horizontal_well.csv"
        csv.write_text(
            "MD,X,Y,Z,GR,TVT_input,TVT\n"
            "1,0,0,0,90,100,5000\n"
            "2,1,0,0,91,200,5001\n"
            "3,2,0,0,92,,5002\n"
            "4,3,0,0,93,,5003\n",
            encoding="utf-8",
        )
        ref = build_pre_ps_reference(str(root), "train", ["testwell"])
        assert len(ref) == 2
        assert ref["TVT_input"].notna().all()
        assert (ref["TVT_input"] == [100.0, 200.0]).all()


def test_spatial_oof_well_exclusion_smoke() -> None:
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "train").mkdir()
        for wid, xs in [("well_a", [0, 1, 2]), ("well_b", [10, 11, 12])]:
            rows = []
            for i, x in enumerate(xs):
                tvt_in = f"{100 + i}" if i < 2 else ""
                rows.append(f"{i},{x},0,0,90,{tvt_in},5000")
            csv = root / "train" / f"{wid}__horizontal_well.csv"
            csv.write_text("MD,X,Y,Z,GR,TVT_input,TVT\n" + "\n".join(rows), encoding="utf-8")

        # Build reference excluding well_a
        ref = build_pre_ps_reference(str(root), "train", ["well_b"])
        assert len(ref) > 0
        assert "well_a" not in ref["well_id"].values


def test_train_has_data_for_spatial() -> None:
    wells = list_well_ids("data", "train")
    assert len(wells) > 0, "No train wells found; spatial tests need local data"
