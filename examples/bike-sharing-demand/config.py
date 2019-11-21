"""Config."""

import numpy as np

from optgbm.sklearn import OGBMRegressor
from sklearn.compose import TransformedTargetRegressor

c = get_config()  # noqa

c.Recipe.data_path = 'examples/bike-sharing-demand/train.csv.gz'
c.Recipe.label_col = 'count'
c.Recipe.dataset_kwargs = {
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
