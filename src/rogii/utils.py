"""General utilities."""


def finite_or_raise(value: float, name: str = "value") -> float:
    """Validate that a numeric value is finite."""
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    return value
