import pytest

from rogii.metrics import official_metric_name, rmse


def test_official_metric_is_rmse() -> None:
    assert official_metric_name() == "RMSE"


def test_rmse_computes_root_mean_squared_error() -> None:
    assert rmse([1.0, 3.0], [1.0, 1.0]) == pytest.approx(2**0.5)


def test_rmse_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        rmse([], [])
