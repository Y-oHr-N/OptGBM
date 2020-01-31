"""Config."""

import lightgbm as lgb
import pandas as pd


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    data.index.rename("ImageId", inplace=True)

    data.index += 1

    return data


c = get_config()  # noqa

c.Recipe.data_path = "examples/digit-recognizer/train.csv.gz"
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = lgb.LGBMClassifier(n_jobs=-1, random_state=0)
c.Recipe.model_path = "examples/digit-recognizer/model.pkl"
