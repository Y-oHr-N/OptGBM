"""Config."""

import builtins
import os

import numpy as np

from optgbm.sklearn import OGBMRegressor
from sklearn.compose import TransformedTargetRegressor

root_dir_path = "examples/prudential-life-insurance-assessment"
label_col = "Response"

cv = 5
enable_pruning = True
importance_type = "gain"
n_estimators = 100_000
n_jobs = -1
n_trials = 100
random_state = 0


def inverse_func(X: np.ndarray) -> np.ndarray:
    """Apply to the prediction of the regressor."""
    X = np.round(X)
    X = np.clip(X, 1.0, 8.0)

    return X.astype("int32")


builtins.inverse_func = inverse_func

c = get_config()  # noqa

c.Recipe.data_path = os.path.join(root_dir_path, "train.csv.gz")
c.Recipe.label_col = label_col
c.Recipe.read_params = {"index_col": "Id"}

c.Recipe.model_instance = TransformedTargetRegressor(
    regressor=OGBMRegressor(
        cv=cv,
        enable_pruning=enable_pruning,
        importance_type=importance_type,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        n_trials=n_trials,
        random_state=random_state,
    ),
    inverse_func=inverse_func,
)
c.Recipe.model_path = os.path.join(root_dir_path, "model.pkl")
