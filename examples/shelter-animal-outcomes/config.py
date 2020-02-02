"""Config."""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMClassifier
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

root_dir_path = "examples/shelter-animal-outcomes"
label_col = "OutcomeType"

cv = TimeSeriesSplit(5)
dtype = "float32"
enable_pruning = False
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


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    data["HasName"] = ~data["Name"].isnull()

    data["IsIntact"] = (
        data["SexuponOutcome"].str.contains("Intact").astype("category")
    )
    data["IsMale"] = (
        data["SexuponOutcome"].str.contains("Male").astype("category")
    )

    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(
        r"years?", "* 365.0"
    )
    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(
        r"months?", "* 30.417"
    )
    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(
        r"weeks?", "* 7.0"
    )
    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(r"days?", "")
    data["AgeuponOutcome"] = data["AgeuponOutcome"].apply(
        lambda x: np.nan if pd.isnull(x) else eval(x)
    )

    return data.drop(columns=["Name"])


c = get_config()  # noqa

c.Recipe.data_path = os.path.join(root_dir_path, "train.csv.gz")
c.Recipe.label_col = label_col
c.Recipe.read_params = {
    "index_col": 0,
    "na_values": {"SexuponOutcome": ["Unknown"]},
    "parse_dates": ["DateTime"],
    "usecols": lambda col: col not in ["OutcomeSubtype"],
}
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = make_pipeline(
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
                        subsample=subsample,
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
        lgb.LGBMClassifier(
            importance_type=importance_type,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        random_state=random_state,
        shuffle=shuffle,
        subsample=subsample,
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
                RowStatistics(dtype=dtype),
                make_column_selector(),
            ),
        ]
    ),
    ModifiedSelectFromModel(
        lgb.LGBMClassifier(
            importance_type=importance_type,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        random_state=random_state,
        shuffle=shuffle,
        subsample=subsample,
        threshold=threshold,
    ),
    OGBMClassifier(
        cv=cv,
        enable_pruning=enable_pruning,
        importance_type=importance_type,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        n_trials=n_trials,
        random_state=random_state,
    ),
)
c.Recipe.model_path = os.path.join(root_dir_path, "model.pkl")
