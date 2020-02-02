"""Config."""

import os

import lightgbm as lgb
import numpy as np

from optgbm.sklearn import OGBMRegressor
from pretools.estimators import Astype
from pretools.estimators import CalendarFeatures
from pretools.estimators import ClippedFeatures
from pretools.estimators import CombinedFeatures
from pretools.estimators import DropCollinearFeatures
from pretools.estimators import ModifiedColumnTransformer
from pretools.estimators import ModifiedSelectFromModel
from pretools.estimators import ModifiedStandardScaler
from pretools.estimators import NUniqueThreshold
from pretools.estimators import Profiler
from pretools.estimators import RowStatistics
from pretools.estimators import SortSamples
from sklearn.compose import make_column_selector
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

root_dir_path = "examples/nyc-taxi-trip-duration"
label_col = "trip_duration"

cv = TimeSeriesSplit(5)
dtype = "float32"
enable_pruning = True
encode = True
importance_type = "gain"
include_unixtime = True
method = "spearman"
n_jobs = -1
n_estimators = 100_000
n_trials = 100
random_state = 0
subsample = 0.5
shuffle = True
threshold = 1e-06

early_stopping_rounds = 30

c = get_config()  # noqa

c.Recipe.data_path = os.path.join(root_dir_path, "train.csv.gz")
c.Recipe.label_col = label_col
c.Recipe.read_params = {
    "dtype": {"vendor_id": "category"},
    "index_col": "id",
    "parse_dates": ["pickup_datetime"],
    "usecols": [
        "id",
        "vendor_id",
        "pickup_datetime",
        "passenger_count",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "store_and_fwd_flag",
        label_col,
    ],
}

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
                        ModifiedStandardScaler(),
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
            lgb.LGBMRegressor(
                importance_type=importance_type,
                n_jobs=n_jobs,
                random_state=random_state,
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
            lgb.LGBMRegressor(
                importance_type=importance_type,
                n_jobs=n_jobs,
                random_state=random_state,
            ),
            random_state=random_state,
            shuffle=shuffle,
            threshold=threshold,
        ),
        OGBMRegressor(
            cv=cv,
            enable_pruning=enable_pruning,
            importance_type=importance_type,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            n_trials=n_trials,
            random_state=random_state,
        ),
    ),
    func=np.log1p,
    inverse_func=np.expm1,
)
c.Recipe.fit_params = {
    "ogbmregressor__early_stopping_rounds": early_stopping_rounds
}
c.Recipe.model_path = os.path.join(root_dir_path, "model.pkl")
