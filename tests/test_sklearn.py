from sklearn.utils.estimator_checks import check_estimator

from optgbm import OGBMRegressor


def test_ogbm_regressor() -> None:
    check_estimator(OGBMRegressor)
