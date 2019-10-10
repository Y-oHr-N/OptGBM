import numpy as np

from sklearn.datasets import load_boston

from optgbm.utils import check_fit_params
from optgbm.utils import check_X


def test_check_X() -> None:
    X, _ = load_boston(return_X_y=True)
    X = check_X(X)

    assert isinstance(X, np.ndarray)


def test_check_fit_params() -> None:
    X, y = load_boston(return_X_y=True)
    X, y, sample_weight = check_fit_params(X, y)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(sample_weight, np.ndarray)
