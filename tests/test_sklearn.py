from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna
import pytest

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.utils.estimator_checks import check_estimator

from optgbm.sklearn import OGBMClassifier
from optgbm.sklearn import OGBMRegressor


def zero_one_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[str, float, bool]:
    return 'zero_one_loss', np.mean(y_true != y_pred), False


def test_ogbm_classifier() -> None:
    check_estimator(OGBMClassifier)


def test_ogbm_regressor() -> None:
    check_estimator(OGBMRegressor)


@pytest.mark.parametrize('storage', [None, 'sqlite:///:memory:'])
def test_fit_twice_with_study(storage: Optional[str]) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    n_trials = 5
    study = optuna.create_study(storage=storage)
    reg = OGBMClassifier(n_trials=n_trials, study=study)

    reg.fit(X, y)

    assert len(study.trials) == n_trials

    reg.fit(X, y)

    assert len(study.trials) == 2 * n_trials


@pytest.mark.parametrize('eval_metric', ['auc', zero_one_loss])
def test_fit_with_eval_metric(eval_metric: Union[str, Callable]) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    reg = OGBMClassifier()

    reg.fit(X, y, eval_metric=eval_metric)


@pytest.mark.parametrize('refit', [False, True])
def test_score(refit: bool) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    reg = lgb.LGBMClassifier(random_state=0)

    reg.fit(X_train, y_train)

    score = reg.score(X_test, y_test)

    reg = OGBMClassifier(n_trials=50, random_state=0, refit=refit)

    reg.fit(X_train, y_train)

    assert score < reg.score(X_test, y_test)


@pytest.mark.parametrize('n_jobs', [-1, 1])
def test_feature_importances(n_jobs: int) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    reg = OGBMClassifier(n_jobs=n_jobs)

    reg.fit(X, y)

    assert isinstance(reg.feature_importances_, np.ndarray)
