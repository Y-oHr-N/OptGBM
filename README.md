# OptGBM

![CircleCI](https://img.shields.io/circleci/build/github/Y-oHr-N/OptGBM)
![PyPI](https://img.shields.io/pypi/v/OptGBM)
![PyPI - License](https://img.shields.io/pypi/l/OptGBM)
[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/Y-oHr-N/OptGBM/master)

Optuna + LightGBM = OptGBM

## Examples

```python
from optgbm.sklearn import OGBMRegressor
from sklearn.datasets import load_boston

reg = OGBMRegressor(random_state=0)
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
