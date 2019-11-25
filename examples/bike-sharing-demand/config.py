"""Config."""

import numpy as np

from optgbm.sklearn import OGBMRegressor
from sklearn.compose import TransformedTargetRegressor


def transform_batch(X):
    """User-defined proprocessing."""
    s = X.index.to_series()

    X['unixtime'] = 1e-09 * s.astype('int64')

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

        X['{}_sin'.format(attr)] = np.sin(theta)
        X['{}_cos'.format(attr)] = np.cos(theta)

    return X


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
        n_estimators=100_000,
        n_trials=100,
        random_state=0
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
c.Recipe.model_path = 'examples/bike-sharing-demand/model.pkl'
