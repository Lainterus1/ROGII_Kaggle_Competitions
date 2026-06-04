"""Model contracts for naive and classical ML baselines."""


def baseline_model_order() -> list[str]:
    """Return the planned initial model order."""
    return ["naive", "lightgbm", "catboost", "xgboost"]
