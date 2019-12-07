"""Config."""

import itertools
import lightgbm as lgb
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import make_pipeline

label_col = 'count'


def create_arithmetical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create arithmetical features."""
    Xt = pd.DataFrame(index=X.index)

    operands = [
        'add',
        'subtract',
        'multiply',
        'divide'
    ]

    for col1, col2 in itertools.combinations(X.columns, 2):
        for operand in operands:
            func = getattr(np, operand)
            Xt['{}_{}_{}'.format(operand, col1, col2)] = func(X[col1], X[col2])

    return Xt


def create_calendar_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create calendar features."""
    Xt = pd.DataFrame(index=X.index)

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

    for col in X:
        s = X[col]
        Xt[col] = 1e-09 * s.astype('int64')

        for attr in attrs:
            x = getattr(s.dt, attr)

            if x.nunique() == 1:
                continue

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
                x += s.dt.minute / 60.0 + s.dt.second / 60.0
                period = 24.0
            elif attr in ['minute', 'second']:
                period = 60.0

            theta = 2.0 * np.pi * x / period

            Xt['{}_{}_sin'.format(s.name, attr)] = np.sin(theta)
            Xt['{}_{}_cos'.format(s.name, attr)] = np.cos(theta)

    return Xt


def create_diff_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create diff features."""
    Xt = X.diff()

    return Xt.rename(columns='{}_diff'.format)


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined proprocessing."""
    if train:
        data = data.sort_index()

        label = data[label_col]
        q25, q75 = np.quantile(label, [0.25, 0.75])
        iqr = q75 - q25

        # is_inlier = (q25 - 1.5 * iqr <= label) & (label <= q75 + 1.5 * iqr)
        # data = data[is_inlier]

        data[label_col] = label.clip(q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        X = data.drop(columns=label_col)

    else:
        X = data.copy()

    X['datetime'] = X.index
    
    numerical_cols = X.dtypes == np.number
    time_cols = X.dtypes == 'datetime64[ns]'

    arithmetical_features = \
        create_arithmetical_features(X.loc[:, numerical_cols])
    calendar_features = create_calendar_features(X.loc[:, time_cols])
    diff_features = create_diff_features(X.loc[:, numerical_cols])

    return pd.concat(
        [
            data,
            arithmetical_features,
            calendar_features,
            diff_features
        ],
        axis=1
    )


c = get_config()  # noqa

c.Recipe.data_path = 'examples/bike-sharing-demand/train.csv.gz'
c.Recipe.label_col = label_col
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
    regressor=make_pipeline(
        # SelectFromModel(
        #     lgb.LGBMRegressor(importance_type='gain', random_state=0),
        #     threshold=1e-06
        # ),  # lightgbm>=2.3.0, scikit-learn>=0.22
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
