from typing import Optional

import numpy as np
import optuna
import pytest

from sklearn.datasets import load_boston
from sklearn.utils.estimator_checks import check_estimator

from optgbm import OGBMClassifier
from optgbm import OGBMRegressor


def test_ogbm_classifier() -> None:
    check_estimator(OGBMClassifier)


def test_ogbm_regressor() -> None:
    check_estimator(OGBMRegressor)


@pytest.mark.parametrize('storage', [None, 'sqlite:///:memory:'])
def test_fit_twice_with_study(storage: Optional[str]) -> None:
    X, y = load_boston(return_X_y=True)
    n_trials = 5
    study = optuna.create_study(storage=storage)
    reg = OGBMRegressor(n_trials=n_trials, study=study)

    reg.fit(X, y)

    assert len(study.trials) == n_trials

    reg.fit(X, y)

    assert len(study.trials) == 2 * n_trials


@pytest.mark.parametrize('n_jobs', [-1, 1])
def test_feature_importances(n_jobs: int) -> None:
    X, y = load_boston(return_X_y=True)
    reg = OGBMRegressor(n_jobs=n_jobs)

    reg.fit(X, y)

    assert isinstance(reg.feature_importances_, np.ndarray)
