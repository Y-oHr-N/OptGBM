"""Config."""

import numpy as np

from pretools.estimators import Astype
from pretools.estimators import CalendarFeatures
from pretools.estimators import ClippedFeatures
from pretools.estimators import CombinedFeatures
from pretools.estimators import DropCollinearFeatures
from pretools.estimators import ModifiedCatBoostRegressor
from pretools.estimators import ModifiedColumnTransformer
from pretools.estimators import ModifiedSelectFromModel
from pretools.estimators import NUniqueThreshold
from pretools.estimators import Profiler
from pretools.estimators import RowStatistics
from pretools.estimators import SortSamples
from scipy.stats import uniform
from sklearn.compose import make_column_selector
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

label_col = "revenue"

cv = TimeSeriesSplit(5)
encode = True
dtype = "float32"
has_time = True
include_unixtime = True
method = "spearman"
n_jobs = -1
param_distributions = {
    "bagging_temperature": uniform(0.0, 10.0),
    "max_depth": np.arange(1, 7),
    "reg_lambda": uniform(1e-06, 10.0),
}
random_state = 0
shuffle = False
threshold = 1e-06
verbose = 0

c = get_config()  # noqa

c.Recipe.data_path = "examples/restaurant-revenue-prediction/train.csv.gz"
c.Recipe.label_col = label_col
c.Recipe.read_params = {"index_col": "Id", "parse_dates": ["Open Date"]}

c.Recipe.model_instance = TransformedTargetRegressor(
    regressor=make_pipeline(
        Profiler(label_col=label_col),
        Astype(),
        SortSamples(),
        NUniqueThreshold(max_freq=None),
        ModifiedColumnTransformer(
            [
                (
                    "categorical_featrues",
                    NUniqueThreshold(),
                    make_column_selector(dtype_include="category"),
                ),
                (
                    "numerical_features",
                    make_pipeline(
                        DropCollinearFeatures(
                            method=method,
                            random_state=random_state,
                            shuffle=shuffle,
                        ),
                        ClippedFeatures(),
                    ),
                    make_column_selector(dtype_include="number"),
                ),
                (
                    "time_features",
                    CalendarFeatures(
                        dtype=dtype,
                        encode=encode,
                        include_unixtime=include_unixtime,
                    ),
                    make_column_selector(dtype_include="datetime64"),
                ),
            ]
        ),
        ModifiedSelectFromModel(
            ModifiedCatBoostRegressor(
                has_time=has_time, random_state=random_state, verbose=verbose
            ),
            random_state=random_state,
            shuffle=shuffle,
            threshold=threshold,
        ),
        ModifiedColumnTransformer(
            [
                ("original_features", "passthrough", make_column_selector()),
                (
                    "combined_features",
                    CombinedFeatures(),
                    make_column_selector(pattern=r"^.*(?<!_(sin|cos))$"),
                ),
                (
                    "row_statistics",
                    RowStatistics(dtype="float32"),
                    make_column_selector(),
                ),
            ]
        ),
        ModifiedSelectFromModel(
            ModifiedCatBoostRegressor(
                has_time=has_time, random_state=random_state, verbose=verbose
            ),
            random_state=random_state,
            shuffle=shuffle,
            threshold=threshold,
        ),
        Profiler(label_col=label_col),
        RandomizedSearchCV(
            ModifiedCatBoostRegressor(
                has_time=has_time, random_state=random_state, verbose=verbose
            ),
            param_distributions=param_distributions,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
    ),
    func=np.log1p,
    inverse_func=np.expm1,
)
c.Recipe.model_path = "examples/restaurant-revenue-prediction/model.pkl"
