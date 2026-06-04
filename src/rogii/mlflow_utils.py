"""MLflow helper contracts.

MLflow logging is required once model baselines are implemented.
"""


def default_experiment_name() -> str:
    """Return the default MLflow experiment name."""
    return "rogii-wellbore-baseline"
