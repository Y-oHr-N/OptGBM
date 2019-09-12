"""scikit-learn compatible models."""

from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

DATA_TYPE = Union[np.ndarray, pd.DataFrame]
RANDOM_STATE_TYPE = Optional[Union[int, np.random.RandomState]]
TARGET_TYPE = Union[np.ndarray, pd.Series]


class LightGBMCallbackEnv(NamedTuple):
    """Callback environment used by callbacks."""

    model: lgb.engine._CVBooster
    params: Dict[str, Any]
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: List


class ExtractionCallback(object):
    """Callback that extracts trained boosters."""

    @property
    def boosters_(self) -> List[lgb.Booster]:
        """Trained boosters."""
        return self._env.model.boosters

    def __call__(self, env: LightGBMCallbackEnv) -> None:
        """Extract a callback environment."""
        self._env = env


class BaseOGBMModel(BaseEstimator):
    """Base class for models in OptGBM."""

    def __init__(self, random_state: RANDOM_STATE_TYPE = None) -> None:
        self.random_state = random_state

    def fit(self, X: DATA_TYPE, y: TARGET_TYPE) -> 'BaseOGBMModel':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        Returns
        -------
        self
            Return self.
        """
        return self


class OGBMClassifier(BaseOGBMModel, ClassifierMixin):
    """OptGBM classifier.

    Examples
    --------
    >>> from optgbm import OGBMClassifier
    >>> from sklearn.datasets import load_iris
    >>> clf = OGBMClassifier(random_state=0)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    OGBMClassifier(...)
    """


class OGBMRegressor(BaseOGBMModel, RegressorMixin):
    """OptGBM regressor.

    Examples
    --------
    >>> from optgbm import OGBMRegressor
    >>> from sklearn.datasets import load_boston
    >>> reg = OGBMRegressor(random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    OGBMRegressor(...)
    """
