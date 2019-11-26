"""Config."""

import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from sklearn.model_selection import TimeSeriesSplit


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined proprocessing."""
    if train:
        data = data.sort_values('Open Date')

        # label = data['count']
        # q25, q75 = np.quantile(label, [0.25, 0.75])
        # iqr = q75 - q25
        # is_inlier = (q25 - 1.5 * iqr <= label) & (label <= q75 + 1.5 * iqr)
        # data = data[is_inlier]

    s = data['Open Date']

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

    return data.drop(columns='Open Date')


c = get_config()  # noqa

c.Recipe.data_path = 'examples/restaurant-revenue-prediction/train.csv.gz'
c.Recipe.label_col = 'revenue'
c.Recipe.read_params = {
    'index_col': 'Id',
    'parse_dates': ['Open Date']
}
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = OGBMRegressor(
    cv=TimeSeriesSplit(5),
    n_estimators=100_000,
    n_trials=100,
    random_state=0
)
c.Recipe.model_path = 'examples/restaurant-revenue-prediction/model.pkl'
