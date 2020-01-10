"""Config."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from pretools.estimators import CalendarFeatures
# from pretools.estimators import ClippedFeatures
from pretools.estimators import CombinedFeatures
# from pretools.estimators import DiffFeatures
from pretools.estimators import ModifiedSelectFromModel
from pretools.estimators import Profiler
from pretools.estimators import RowStatistics
from pretools.utils import get_numerical_cols
from pretools.utils import get_time_cols
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

label_col = 'count'


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    X = data.copy()

    if train:
        X = X.sort_index()
        y = X.pop(label_col)
    else:
        y = None

    calendar_features = CalendarFeatures(
        dtype='float32',
        include_unixtime=True
    )
    # clipped_features = ClippedFeatures()
    # diff_features = DiffFeatures()
    row_statistics = RowStatistics(dtype='float32')

    X['datetime'] = X.index

    numerical_cols = get_numerical_cols(X)
    time_cols = get_time_cols(X)

    X.loc[:, numerical_cols] = X.loc[:, numerical_cols].astype('float32')

    # X.loc[:, numerical_cols] = \
    #     clipped_features.fit_transform(X.loc[:, numerical_cols])

    return pd.concat(
        [
            data,
            calendar_features.fit_transform(X.loc[:, time_cols]),
            # diff_features.fit_transform(X.loc[:, numerical_cols]),
            row_statistics.fit_transform(X)
        ],
        axis=1
    )


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
        CombinedFeatures(include_data=True),
        ModifiedSelectFromModel(
            lgb.LGBMRegressor(importance_type='gain', random_state=0),
            # threshold=1e-06
        ),
        OGBMRegressor(
            cv=TimeSeriesSplit(5),
            n_estimators=100_000,
            n_trials=100,
            random_state=0
        )
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
c.Recipe.model_path = 'examples/bike-sharing-demand/model.pkl'
