"""Config."""

from optgbm.sklearn import OGBMRegressor

c = get_config()  # noqa

c.Recipe.data_path = \
    'examples/mercedes-benz-greener-manufacturing/train.csv.gz'
c.Recipe.label_col = 'y'
c.Recipe.dataset_kwargs = {'index_col': 'ID'}

c.Recipe.model_instance = OGBMRegressor(
    n_estimators=100_000,
    n_trials=100,
    random_state=0
)
c.Recipe.model_path = 'examples/mercedes-benz-greener-manufacturing/model.pkl'
