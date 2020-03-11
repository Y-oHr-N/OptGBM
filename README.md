# OptGBM

[![Python package](https://github.com/Y-oHr-N/OptGBM/workflows/Python%20package/badge.svg?branch=master)](https://github.com/Y-oHr-N/OptGBM/actions?query=workflow%3A%22Python+package%22)
[![PyPI](https://img.shields.io/pypi/v/OptGBM)](https://pypi.org/project/OptGBM/)
[![PyPI - License](https://img.shields.io/pypi/l/OptGBM)](https://pypi.org/project/OptGBM/)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Y-oHr-N/OptGBM/master)

OptGBM (= Optuna + LightGBM) provides a scikit-learn compatible estimator that tunes hyperparameters in LightGBM with Optuna.

## Examples

```python
import optgbm as lgb
from sklearn.datasets import load_boston

reg = lgb.LGBMRegressor(random_state=0)
X, y = load_boston(return_X_y=True)

reg.fit(X, y)

score = reg.score(X, y)
```

## Installation

```
pip install optgbm
```

## Testing

```
python setup.py test
```
