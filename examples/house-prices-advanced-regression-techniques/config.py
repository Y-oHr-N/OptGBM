"""Config."""

from optgbm.sklearn import OGBMRegressor

c = get_config()  # noqa

c.Recipe.data_path = \
    'examples/house-prices-advanced-regression-techniques/train.csv.gz'
c.Recipe.label_col = 'SalePrice'
c.Recipe.dataset_kwargs = {'index_col': 'Id'}

c.Recipe.model_instance = OGBMRegressor()
c.Recipe.params = {
    'n_estimators': 100_000,
    'n_jobs': -1,
    'n_trials': 100,
    'random_state': 0
}
c.Recipe.model_path = \
    'examples/house-prices-advanced-regression-techniques/model.pkl'
