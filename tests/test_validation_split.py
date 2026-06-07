import numpy as np
import pytest
from sklearn.model_selection import StratifiedGroupKFold

from rogii.validation import (
    validate_no_group_overlap,
    build_stratification_labels,
    create_cv_splitter,
    _merge_rare_classes,
)


def test_validate_no_group_overlap_allows_disjoint_groups() -> None:
    validate_no_group_overlap({"well_a"}, {"well_b"})


def test_validate_no_group_overlap_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        validate_no_group_overlap({"well_a"}, {"well_a"})


def test_create_cv_splitter_group() -> None:
    from sklearn.model_selection import GroupKFold
    cv = create_cv_splitter("group", 5)
    assert isinstance(cv, GroupKFold)
    assert cv.n_splits == 5


def test_create_cv_splitter_stratified() -> None:
    cv = create_cv_splitter("stratified", 5)
    assert isinstance(cv, StratifiedGroupKFold)


def test_merge_rare_classes_noop() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=int)
    result = _merge_rare_classes(labels, min_class_size=2)
    assert np.array_equal(result, labels)


def test_merge_rare_classes_merges_small() -> None:
    labels = np.array([0, 0, 1, 1, 2, 0, 0, 0], dtype=int)
    # class 2 has count=1 (<2), should merge into nearest (class 1)
    result = _merge_rare_classes(labels, min_class_size=2)
    assert 2 not in np.unique(result)


def test_strat_labels_produces_integers() -> None:
    """Synthetic test: build strat labels and verify they're usable."""
    n_wells = 100
    np.random.seed(0)
    labels, meta = _build_synthetic_labels(n_wells)
    assert len(labels) == n_wells
    assert labels.dtype in (np.int32, np.int64)
    assert len(np.unique(labels)) >= 2


def test_strat_labels_no_nan() -> None:
    """Strat labels must have no NaN values."""
    labels, _ = _build_synthetic_labels(100)
    assert np.isfinite(labels).all()
    assert not np.isnan(labels).any()


def test_strat_labels_no_group_overlap() -> None:
    """Strat labels can be used with StratifiedGroupKFold without overlap."""
    labels, _ = _build_synthetic_labels(60)
    X = np.random.randn(60, 3)
    groups = np.arange(60)

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, labels, groups))
    assert len(splits) == 3


def test_stratified_group_kfold_no_group_overlap() -> None:
    """StratifiedGroupKFold must not split the same group across folds."""
    labels, _ = _build_synthetic_labels(60)
    X = np.random.randn(60, 3)
    groups = np.arange(60)  # one group per sample

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(sgkf.split(X, labels, groups))
    assert len(splits) == 5

    all_val_sets = []
    for train_idx, val_idx in splits:
        assert len(set(groups[train_idx]) & set(groups[val_idx])) == 0
        all_val_sets.append(set(groups[val_idx]))

    for i in range(5):
        for j in range(i + 1, 5):
            assert len(all_val_sets[i] & all_val_sets[j]) == 0


def _build_synthetic_labels(n_wells: int) -> tuple[np.ndarray, dict]:
    """Build synthetic strat labels without real data files."""
    azimuths = np.random.uniform(-1, 1, n_wells)
    median_tvts = np.random.uniform(10000, 12000, n_wells)
    median_xs = np.random.uniform(0, 100, n_wells)
    median_ys = np.random.uniform(0, 100, n_wells)

    import pandas as pd
    from sklearn.cluster import KMeans

    az_bins = (azimuths >= 0).astype(int)
    n_tvt_bins = 3
    n_spatial_clusters = 3

    tvt_labels = pd.qcut(median_tvts, q=n_tvt_bins, labels=False)
    if hasattr(tvt_labels, "to_numpy"):
        tvt_bins = tvt_labels.to_numpy(dtype=int)
    else:
        tvt_bins = np.array(tvt_labels, dtype=int)

    coords = np.column_stack([median_xs, median_ys])
    km = KMeans(n_clusters=n_spatial_clusters, random_state=42, n_init=10)
    spatial_bins = km.fit_predict(coords)

    combined = az_bins * (n_tvt_bins * n_spatial_clusters) + tvt_bins * n_spatial_clusters + spatial_bins
    combined = _merge_rare_classes(combined, min_class_size=5)

    meta = {"n_tvt_bins": n_tvt_bins, "n_spatial_clusters": n_spatial_clusters}
    return combined, meta

