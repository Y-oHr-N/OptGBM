# OptGBM

[![CircleCI](https://circleci.com/gh/Y-oHr-N/OptGBM.svg?style=svg)](https://circleci.com/gh/Y-oHr-N/OptGBM)

Optuna + LightGBM = OptGBM

## Examples

```python
from optgbm import OGBMClassifier
from sklearn.datasets import load_iris

clf = OGBMClassifier(random_state=0)
X, y = load_iris(return_X_y=True)

clf.fit(X, y)
```

## Installation

```
pip install optgbm
```

## Testing

```
pytest
```
