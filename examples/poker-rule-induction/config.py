"""Config."""

import os

import pandas as pd

from optgbm.sklearn import OGBMClassifier

root_dir_path = "examples/poker-rule-induction"
label_col = "hand"

cv = 5
enable_pruning = True
importance_type = "gain"
n_jobs = -1
n_estimators = 1_000
n_trials = 100
random_state = 0


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    data.index.rename("id", inplace=True)

    data.index += 1

    return data


c = get_config()  # noqa

c.Recipe.data_path = os.path.join(root_dir_path, "train.csv.gz")
c.Recipe.label_col = label_col
c.Recipe.read_params = {
    "usecols": [
        "S1",
        "C1",
        "S2",
        "C2",
        "S3",
        "C3",
        "S4",
        "C4",
        "S5",
        "C5",
        label_col,
    ]
}
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = OGBMClassifier(
    cv=cv,
    enable_pruning=enable_pruning,
    importance_type=importance_type,
    n_estimators=n_estimators,
    n_jobs=n_jobs,
    n_trials=n_trials,
    random_state=random_state,
)
c.Recipe.model_path = os.path.join(root_dir_path, "model.pkl")
