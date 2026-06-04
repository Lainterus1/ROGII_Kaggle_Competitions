"""Validation split contracts."""


def validate_no_group_overlap(train_groups: set[object], valid_groups: set[object]) -> None:
    """Raise if a group appears in both train and validation partitions."""
    overlap = train_groups & valid_groups
    if overlap:
        raise ValueError(f"Group leakage detected: {len(overlap)} overlapping groups")
