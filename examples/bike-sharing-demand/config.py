"""Config."""

import builtins
import itertools

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMRegressor
from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.base import TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

try:  # scikit-learn<=0.21
    from sklearn.feature_selection.from_model import _calculate_threshold
    from sklearn.feature_selection.from_model import _get_feature_importances
except ImportError:
    from sklearn.feature_selection._from_model import _calculate_threshold
    from sklearn.feature_selection._from_model import _get_feature_importances

label_col = 'count'


def get_categorical_cols(X: pd.DataFrame):
    """Get categorical columns."""
    X = pd.DataFrame(X)

    return X.dtypes == 'category'


def get_numerical_cols(X: pd.DataFrame):
    """Get numerical columns."""
    X = pd.DataFrame(X)

    return X.dtypes.apply(lambda x: issubclass(x.type, np.number))


def get_time_cols(X: pd.DataFrame):
    """Get time columns."""
    X = pd.DataFrame(X)

    return X.dtypes.apply(lambda x: issubclass(x.type, np.datetime64))


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    if train:
        data = data.sort_index()

        # label = data[label_col]
        # q25, q75 = np.quantile(label, [0.25, 0.75])
        # iqr = q75 - q25

        # data[label_col] = label.clip(q25 - 1.5 * iqr, q75 + 1.5 * iqr)

        # is_inlier = (q25 - 1.5 * iqr <= label) & (label <= q75 + 1.5 * iqr)
        # data = data[is_inlier]

        X = data.drop(columns=label_col)

    else:
        X = data.copy()

    X['datetime'] = X.index

    numerical_cols = get_numerical_cols(X)
    time_cols = get_time_cols(X)

    calendar_features = CalendarFeatures(dtype='float32')
    # clipped_features = ClippedFeatures()
    combined_features = CombinedFeatures()
    diff_features = DiffFeatures()
    row_statistics = RowStatistics(dtype='float32')

    X.loc[:, numerical_cols] = X.loc[:, numerical_cols].astype('float32')

    # X.loc[:, numerical_cols] = \
    #     clipped_features.fit_transform(X.loc[:, numerical_cols])

    return pd.concat(
        [
            data,
            calendar_features.fit_transform(X.loc[:, time_cols]),
            combined_features.fit_transform(X),
            diff_features.fit_transform(X.loc[:, numerical_cols]),
            row_statistics.fit_transform(X)
        ],
        axis=1
    )


class CombinedFeatures(BaseEstimator, TransformerMixin):
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'CombinedFeatures':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xt = pd.DataFrame()
        categorical_cols = get_categorical_cols(X)
        numerical_cols = get_numerical_cols(X)

        for col1, col2 in itertools.combinations(X.columns, 2):
            if np.all(categorical_cols[[col1, col2]]):
                func = np.vectorize(lambda x1, x2: '{}+{}'.format(x1, x2))

                feature = func(X[col1], X[col2])
                feature = pd.Series(feature, index=X.index)
                Xt['add_{}_{}'.format(col1, col2)] = feature.astype('category')

            if np.all(numerical_cols[[col1, col2]]):
                operands = [
                    'add',
                    'subtract',
                    'multiply',
                    'divide',
                    'equal'
                ]

                for operand in operands:
                    func = getattr(np, operand)

                    Xt['{}_{}_{}'.format(operand, col1, col2)] = \
                        func(X[col2], X[col2])

        return Xt


class CalendarFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, dtype: str = 'float32', encode: bool = False) -> None:
        self.dtype = dtype
        self.encode = encode

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'CalendarFeatures':
        X = pd.DataFrame(X)

        secondsinminute = 60.0
        secondsinhour = 60.0 * secondsinminute
        secondsinday = 24.0 * secondsinhour
        secondsinweekday = 7.0 * secondsinday
        secondsinmonth = 30.4167 * secondsinday
        secondsinyear = 12.0 * secondsinmonth

        self.attributes_: Dict[str, List[str]] = {}

        for col in X:
            s = X[col]
            duration = s.max() - s.min()
            duration = duration.total_seconds()
            attrs = []

            if duration >= 2.0 * secondsinyear:
                if s.dt.dayofyear.nunique() > 1:
                    attrs.append('dayofyear')
                if s.dt.weekofyear.nunique() > 1:
                    attrs.append('weekofyear')
                if s.dt.quarter.nunique() > 1:
                    attrs.append('quarter')
                if s.dt.month.nunique() > 1:
                    attrs.append('month')
            if duration >= 2.0 * secondsinmonth \
                    and s.dt.day.nunique() > 1:
                attrs.append('day')
            if duration >= 2.0 * secondsinweekday \
                    and s.dt.weekday.nunique() > 1:
                attrs.append('weekday')
            if duration >= 2.0 * secondsinday \
                    and s.dt.hour.nunique() > 1:
                attrs.append('hour')
            # if duration >= 2.0 * secondsinhour \
            #         and s.dt.minute.nunique() > 1:
            #     attrs.append('minute')
            # if duration >= 2.0 * secondsinminute \
            #         and s.dt.second.nunique() > 1:
            #     attrs.append('second')

            self.attributes_[col] = attrs

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        for col in X:
            s = X[col]

            unixtime = 1e-09 * s.astype('int64')
            unixtime = unixtime.astype(self.dtype)

            Xt[col] = unixtime

            for attr in self.attributes_[col]:
                x = getattr(s.dt, attr)

                if not self.encode:
                    x = x.astype('category')

                    Xt['{}_{}'.format(col, attr)] = x

                    continue

                if attr == 'dayofyear':
                    period = np.where(s.dt.is_leap_year, 366.0, 365.0)
                elif attr == 'weekofyear':
                    period = 52.1429
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
                sin_theta = np.sin(theta)
                sin_theta = sin_theta.astype(self.dtype)
                cos_theta = np.cos(theta)
                cos_theta = cos_theta.astype(self.dtype)

                Xt['{}_{}_sin'.format(col, attr)] = sin_theta
                Xt['{}_{}_cos'.format(col, attr)] = cos_theta

        return Xt


class ClippedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, high: float = 99.0, low: float = 1.0) -> None:
        self.high = high
        self.low = low

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'ClippedFeatures':
        self.data_min_, self.data_max_ = np.nanpercentile(
            X,
            [self.low, self.high],
            axis=0
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)

        return X.clip(self.data_min_, self.data_max_, axis=1)


class DiffFeatures(BaseEstimator, TransformerMixin):
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'DiffFeatures':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xt = X.diff()

        return Xt.rename(columns='{}_diff'.format)


class ModifiedSelectFromModel(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: BaseEstimator,
        threshold: Optional[Union[float, str]] = None
    ) -> None:
        self.estimator = estimator
        self.threshold = threshold

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        **fit_params: Any
    ) -> 'ModifiedSelectFromModel':
        self.estimator_ = clone(self.estimator)

        self.estimator_.fit(X, y, **fit_params)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)

        feature_importances = _get_feature_importances(self.estimator_)
        threshold = _calculate_threshold(
            self.estimator_,
            feature_importances,
            self.threshold
        )
        cols = feature_importances >= threshold

        return X.loc[:, cols]


class RowStatistics(BaseEstimator, TransformerMixin):
    def __init__(self, dtype: str = 'float32'):
        self.dtype = dtype

    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> 'RowStatistics':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        Xt = pd.DataFrame()

        is_null = X.isnull()

        Xt['#na_values'] = is_null.sum(axis=1)

        return Xt.astype(self.dtype)


builtins.ModifiedSelectFromModel = ModifiedSelectFromModel

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
        ModifiedSelectFromModel(
            lgb.LGBMRegressor(importance_type='gain', random_state=0),
            threshold=1e-06
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
