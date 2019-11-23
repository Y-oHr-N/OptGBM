"""Config."""

from optgbm.sklearn import OGBMRegressor

c = get_config()  # noqa

c.Recipe.data_path = \
    'examples/house-prices-advanced-regression-techniques/train.csv.gz'
c.Recipe.label_col = 'SalePrice'
c.Recipe.read_params = {'index_col': 'Id'}

c.Recipe.model_instance = OGBMRegressor(
    n_estimators=100_000,
    n_trials=100,
    random_state=0
)
c.Recipe.model_path = \
    'examples/house-prices-advanced-regression-techniques/model.pkl'
