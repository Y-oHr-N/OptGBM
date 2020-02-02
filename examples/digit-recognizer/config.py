"""Config."""

import os

import lightgbm as lgb
import pandas as pd

root_dir_path = "examples/digit-recognizer"

n_jobs = -1
random_state = 0


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    data.index.rename("ImageId", inplace=True)

    data.index += 1

    return data


c = get_config()  # noqa

c.Recipe.data_path = os.path.join(root_dir_path, "train.csv.gz")
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = lgb.LGBMClassifier(
    n_jobs=n_jobs, random_state=random_state
)
c.Recipe.model_path = os.path.join(root_dir_path, "model.pkl")
