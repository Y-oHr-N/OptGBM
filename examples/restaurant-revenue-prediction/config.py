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

label_col = 'revenue'


c = get_config()  # noqa

c.Recipe.data_path = 'examples/restaurant-revenue-prediction/train.csv.gz'
c.Recipe.label_col = label_col
c.Recipe.read_params = {
    'index_col': 'Id',
    'parse_dates': ['Open Date']
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
                    'categorical_featrues',
                    NUniqueThreshold(),
                    make_column_selector(dtype_include='category')
                ),
                (
                    'numerical_features',
                    make_pipeline(
                        DropCollinearFeatures(
                            method='spearman',
                            shuffle=False
                        ),
                        ClippedFeatures()
                    ),
                    make_column_selector(dtype_include='number')
                ),
                (
                    'time_features',
                    CalendarFeatures(
                        dtype='float32',
                        encode=True,
                        include_unixtime=True
                    ),
                    make_column_selector(dtype_include='datetime64')
                )
            ]
        ),
        ModifiedSelectFromModel(
            ModifiedCatBoostRegressor(
                has_time=True,
                random_state=0,
                verbose=0
            ),
            shuffle=False,
            threshold=1e-06
        ),
        ModifiedColumnTransformer(
            [
                (
                    'original_features',
                    'passthrough',
                    make_column_selector()
                ),
                (
                    'combined_features',
                    CombinedFeatures(),
                    make_column_selector(pattern=r'^.*(?<!_(sin|cos))$')
                ),
                (
                    'row_statistics',
                    RowStatistics(dtype='float32'),
                    make_column_selector()
                )
            ]
        ),
        ModifiedSelectFromModel(
            ModifiedCatBoostRegressor(
                has_time=True,
                random_state=0,
                verbose=0
            ),
            shuffle=False,
            threshold=1e-06
        ),
        Profiler(label_col=label_col),
        RandomizedSearchCV(
            ModifiedCatBoostRegressor(
                has_time=True,
                random_state=0,
                verbose=0
            ),
            param_distributions={
                'bagging_temperature': uniform(0.0, 10.0),
                'max_depth': np.arange(1, 7),
                'reg_lambda': uniform(1e-06, 10.0)
            },
            cv=TimeSeriesSplit(5),
            n_jobs=-1,
            random_state=0
        )
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
c.Recipe.model_path = 'examples/restaurant-revenue-prediction/model.pkl'
