"""Utilities."""

from typing import Any
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import is_classifier
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv as sklearn_check_cv
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _assert_all_finite
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import column_or_1d

from .typing import CVType
from .typing import OneDimArrayLikeType
from .typing import TwoDimArrayLikeType


def check_cv(
    cv: CVType = 5,
    y: Optional[OneDimArrayLikeType] = None,
    classifier: bool = False,
) -> BaseCrossValidator:
    """Check `cv`.

    Parameters
    ----------
    cv
        Cross-validation strategy.

    y
        Target.

    classifier
        If the task is a classification task, `StratifiedKFold` will be used.

    Returns
    -------
    cv
        Converted cross-validation strategy.
    """
    if classifier and isinstance(cv, int):
        _, counts = np.unique(y, return_counts=True)
        cv = max(2, min(cv, *counts))

    return sklearn_check_cv(cv, y, classifier)


def check_X(
    X: TwoDimArrayLikeType,
    estimator: Optional[BaseEstimator] = None,
    **kwargs: Any
) -> TwoDimArrayLikeType:
    """Check `X`.

    Parameters
    ----------
    X
        Data.

    estimator
        Object to use to fit the data.

    **kwargs
        Other keywords passed to `sklearn.utils.check_array`.

    Returns
    -------
    X
        Converted and validated data.
    """
    if not isinstance(X, pd.DataFrame):
        X = check_array(X, estimator=estimator, **kwargs)

    _, actual_n_features = X.shape
    expected_n_features = getattr(estimator, "n_features_", actual_n_features)

    if actual_n_features != expected_n_features:
        raise ValueError(
            "`n_features` must be {} but was {}.".format(
                expected_n_features, actual_n_features
            )
        )

    return X


def check_fit_params(
    X: TwoDimArrayLikeType,
    y: OneDimArrayLikeType,
    sample_weight: Optional[OneDimArrayLikeType] = None,
    estimator: Optional[BaseEstimator] = None,
    **kwargs: Any
) -> Tuple[TwoDimArrayLikeType, OneDimArrayLikeType, OneDimArrayLikeType]:
    """Check `X`, `y` and `sample_weight`.

    Parameters
    ----------
    X
        Data.

    y
        Target.

    sample_weight
        Weights of data.

    estimator
        Object to use to fit the data.

    **kwargs
        Other keywords passed to `sklearn.utils.check_array`.

    Returns
    -------
    X
        Converted and validated data.

    y
        Converted and validated target.

    sample_weight
        Converted and validated weights of data.
    """
    X = check_X(X, estimator=estimator, **kwargs)

    if not isinstance(y, pd.Series):
        y = column_or_1d(y, warn=True)

    _assert_all_finite(y)

    if is_classifier(estimator):
        check_classification_targets(y)

    if sample_weight is None:
        n_samples = _num_samples(X)
        sample_weight = np.ones(n_samples)

    sample_weight = np.asarray(sample_weight)

    class_weight = getattr(estimator, "class_weight", None)

    if class_weight is not None:
        sample_weight *= compute_sample_weight(class_weight, y)

    check_consistent_length(X, y, sample_weight)

    return X, y, sample_weight
