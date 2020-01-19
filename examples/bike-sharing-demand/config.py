"""Config."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from pretools.estimators import Astype
from pretools.estimators import CalendarFeatures
from pretools.estimators import ClippedFeatures
from pretools.estimators import CombinedFeatures
from pretools.estimators import DropCollinearFeatures
from pretools.estimators import ModifiedColumnTransformer
from pretools.estimators import ModifiedSelectFromModel
from pretools.estimators import NUniqueThreshold
from pretools.estimators import Profiler
from pretools.estimators import RowStatistics
from sklearn.compose import make_column_selector
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

label_col = 'count'


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    data = data.copy()

    if train:
        data = data.sort_index()

    data['datetime'] = data.index

    return data


c = get_config()  # noqa

c.Recipe.data_path = 'examples/bike-sharing-demand/train.csv.gz'
c.Recipe.label_col = label_col
c.Recipe.read_params = {
    'dtype': {
        'holiday': 'category',
        'season': 'category',
        'weather': 'category',
        'workingday': 'category'
    },
    'index_col': 'datetime',
    'na_values': {'windspeed': [0.0]},
    'parse_dates': ['datetime'],
    'usecols': [
        'datetime',
        'atemp',
        'holiday',
        'humidity',
        'season',
        'temp',
        'weather',
        'windspeed',
        'workingday',
        'count'
    ]
}
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = TransformedTargetRegressor(
    regressor=make_pipeline(
        Profiler(label_col=label_col),
        Astype(),
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
                        DropCollinearFeatures(
                            method='spearman',
                            random_state=0
                        ),
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
            lgb.LGBMRegressor(
                importance_type='gain',
                n_jobs=-1,
                random_state=0
            ),
            random_state=0,
            # threshold=1e-06
        ),
        OGBMRegressor(
            cv=TimeSeriesSplit(5),
            n_estimators=100_000,
            n_jobs=-1,
            n_trials=100,
            random_state=0
        )
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
c.Recipe.fit_params = {'ogbmregressor__early_stopping_rounds': 30}
c.Recipe.model_path = 'examples/bike-sharing-demand/model.pkl'
