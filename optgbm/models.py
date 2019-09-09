"""scikit-learn compatible models."""

from typing import Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

DATA_TYPE = Union[np.ndarray, pd.DataFrame]
RANDOM_STATE_TYPE = Union[int, np.random.RandomState]
TARGET_TYPE = Union[np.ndarray, pd.Series]


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
