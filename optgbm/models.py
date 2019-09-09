"""scikit-learn compatible models."""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin


class BaseOGBMModel(BaseEstimator):
    """Base class for models in OptGBM."""

    def __init__(self, random_state: int = None) -> None:
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseOGBMModel':
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
