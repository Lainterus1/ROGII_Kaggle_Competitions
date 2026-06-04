import pytest

from rogii.validation import validate_no_group_overlap


def test_validate_no_group_overlap_allows_disjoint_groups() -> None:
    validate_no_group_overlap({"well_a"}, {"well_b"})


def test_validate_no_group_overlap_rejects_overlap() -> None:
    with pytest.raises(ValueError):
        validate_no_group_overlap({"well_a"}, {"well_a"})
