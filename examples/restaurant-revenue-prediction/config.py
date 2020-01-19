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

c.Recipe.model_instance = make_pipeline(
    Profiler(label_col=label_col),
    Astype(),
    SortSamples(),
    NUniqueThreshold(max_freq=None),
    ModifiedColumnTransformer(
        [
            (
                'categoricaltransformer',
                NUniqueThreshold(),
                make_column_selector(dtype_include='category')
            ),
            (
                'numericaltransformer',
                make_pipeline(
                    DropCollinearFeatures(method='spearman', random_state=0),
                    ClippedFeatures()
                ),
                make_column_selector(dtype_include='number')
            ),
            (
                'timetransformer',
                CalendarFeatures(dtype='float32', include_unixtime=True),
                make_column_selector(dtype_include='datetime64')
            ),
            (
                'othertransformer',
                RowStatistics(dtype='float32'),
                make_column_selector()
            )
        ]
    ),
    CombinedFeatures(include_data=True),
    ModifiedSelectFromModel(
        ModifiedCatBoostRegressor(has_time=True, random_state=0, verbose=0),
        random_state=0,
        # threshold=1e-06
    ),
    RandomizedSearchCV(
        ModifiedCatBoostRegressor(has_time=True, random_state=0, verbose=0),
        param_distributions={
            'bagging_temperature': uniform(0.0, 10.0),
            'max_depth': np.arange(1, 7),
            'reg_lambda': uniform(1e-06, 10.0)
        },
        cv=TimeSeriesSplit(5),
        n_jobs=-1,
        random_state=0
    )
)
c.Recipe.model_path = 'examples/restaurant-revenue-prediction/model.pkl'
