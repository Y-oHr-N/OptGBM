"""Config."""

import lightgbm as lgb
import pandas as pd

label_col = "hand"


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    data.index.rename("id", inplace=True)

    data.index += 1

    return data


c = get_config()  # noqa

c.Recipe.data_path = "examples/poker-rule-induction/train.csv.gz"
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
        "hand",
    ]
}
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = lgb.LGBMClassifier(n_jobs=-1, random_state=0)
c.Recipe.model_path = "examples/poker-rule-induction/model.pkl"
