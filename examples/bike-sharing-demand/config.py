"""Config."""

import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined proprocessing."""
    if train:
        data = data.sort_index()

        # y = data['count']
        # q25, q75 = np.quantile(y, [0.25, 0.75])
        # iqr = q75 - q25
        # is_inlier = (q25 - 1.5 * iqr <= y) & (y <= q75 + 1.5 * iqr)
        # data = data[is_inlier]

    s = data.index.to_series()

    data['unixtime'] = 1e-09 * s.astype('int64')

    attrs = [
        # 'year',
        # 'weekofyear',
        'dayofyear',
        'quarter',
        'month',
        'day',
        'weekday',
        'hour',
        # 'minute',
        # 'second'
    ]

    for attr in attrs:
        if attr == 'dayofyear':
            period = np.where(s.dt.is_leap_year, 366.0, 365.0)
        elif attr == 'quarter':
            period = 4.0
        elif attr == 'month':
            period = 12.0
        elif attr == 'day':
            period = s.dt.daysinmonth
        elif attr == 'weekday':
            period = 7.0
        elif attr == 'hour':
            period = 24.0
        elif attr in ['minute', 'second']:
            period = 60.0

        theta = 2.0 * np.pi * getattr(s.dt, attr) / period

        data['{}_sin'.format(attr)] = np.sin(theta)
        data['{}_cos'.format(attr)] = np.cos(theta)

    return data


c = get_config()  # noqa

c.Recipe.data_path = 'examples/bike-sharing-demand/train.csv.gz'
c.Recipe.label_col = 'count'
c.Recipe.read_params = {
    'dtype': {'season': 'category', 'weather': 'category'},
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
    regressor=OGBMRegressor(
        cv=TimeSeriesSplit(5),
        n_estimators=100_000,
        n_trials=100,
        random_state=0
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
c.Recipe.model_path = 'examples/bike-sharing-demand/model.pkl'
