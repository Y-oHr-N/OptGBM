"""Utilities."""

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import is_classifier
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _assert_all_finite
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import column_or_1d

ONE_DIM_ARRAYLIKE_TYPE = Union[np.ndarray, pd.Series]
TWO_DIM_ARRAYLIKE_TYPE = Union[np.ndarray, pd.DataFrame]


def check_X(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    estimator: BaseEstimator = None,
    **kwargs: Any
) -> TWO_DIM_ARRAYLIKE_TYPE:
    """Check `X`."""
    if not isinstance(X, pd.DataFrame):
        X = check_array(X, **kwargs)

    _, actual_n_features = X.shape
    expected_n_features = getattr(estimator, 'n_features_', actual_n_features)

    if actual_n_features != expected_n_features:
        raise ValueError(
            '`n_features` must be {} but was {}.'.format(
                expected_n_features,
                actual_n_features
            )
        )

    return X


def check_fit_params(
    X: TWO_DIM_ARRAYLIKE_TYPE,
    y: ONE_DIM_ARRAYLIKE_TYPE,
    sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None,
    estimator: Optional[BaseEstimator] = None,
    **kwargs: Any
) -> Tuple[
    TWO_DIM_ARRAYLIKE_TYPE,
    ONE_DIM_ARRAYLIKE_TYPE,
    ONE_DIM_ARRAYLIKE_TYPE
]:
    """Check `X`, `y` and `sample_weight`."""
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

    class_weight = getattr(estimator, 'class_weight', None)

    if class_weight is not None:
        sample_weight *= compute_sample_weight(class_weight, y)

    check_consistent_length(X, y, sample_weight)

    return X, y, sample_weight
