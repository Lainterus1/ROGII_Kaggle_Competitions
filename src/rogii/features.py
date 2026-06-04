"""Feature engineering contracts.

Feature construction must avoid target leakage and use only columns confirmed to be
available in both train and test data.
"""


def get_feature_set_name() -> str:
    """Return the placeholder feature set name used before data inspection."""
    return "pending_data_inspection"
