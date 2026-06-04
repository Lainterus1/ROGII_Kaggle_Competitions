from rogii.metrics import official_metric_name


def test_official_metric_is_explicitly_pending() -> None:
    assert official_metric_name() == "TBD_after_kaggle_evaluation_check"
