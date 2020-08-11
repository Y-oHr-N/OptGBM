# OptGBM

[![Python package](https://github.com/Y-oHr-N/OptGBM/workflows/Python%20package/badge.svg?branch=master)](https://github.com/Y-oHr-N/OptGBM/actions?query=workflow%3A%22Python+package%22)
[![codecov](https://codecov.io/gh/Y-oHr-N/OptGBM/branch/master/graph/badge.svg)](https://codecov.io/gh/Y-oHr-N/OptGBM)
[![PyPI](https://img.shields.io/pypi/v/OptGBM)](https://pypi.org/project/OptGBM/)
[![PyPI - License](https://img.shields.io/pypi/l/OptGBM)](https://pypi.org/project/OptGBM/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Y-oHr-N/OptGBM/master)

OptGBM (= [Optuna](https://optuna.org/) + [LightGBM](http://github.com/microsoft/LightGBM)) provides a scikit-learn compatible estimator that tunes hyperparameters in LightGBM with Optuna.

## Examples

```python
import optgbm as lgb
from sklearn.datasets import load_boston

reg = lgb.LGBMRegressor(random_state=0)
X, y = load_boston(return_X_y=True)

reg.fit(X, y)

y_pred = reg.predict(X, y)
```

By default, the following hyperparameters will be searched.

- `bagging_fraction`
- `bagging_freq`
- `feature_fractrion`
- `lambda_l1`
- `lambda_l2`
- `max_depth`
- `min_data_in_leaf`
- `num_leaves`

## Installation

```
pip install optgbm
```

## Testing

```
tox
```
