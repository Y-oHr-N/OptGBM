"""Config."""

import pandas as pd

from optgbm.sklearn import OGBMRegressor
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
from sklearn.compose import make_column_selector
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline

label_col = 'revenue'


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    if train:
        data = data.sort_values('Open Date')

    return data


c = get_config()  # noqa

c.Recipe.data_path = 'examples/restaurant-revenue-prediction/train.csv.gz'
c.Recipe.label_col = label_col
c.Recipe.read_params = {
    'index_col': 'Id',
    'parse_dates': ['Open Date']
}
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = make_pipeline(
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
        ModifiedCatBoostRegressor(random_state=0),
        random_state=0,
        # threshold=1e-06
    ),
    ModifiedCatBoostRegressor(random_state=0)
)
c.Recipe.model_path = 'examples/restaurant-revenue-prediction/model.pkl'
