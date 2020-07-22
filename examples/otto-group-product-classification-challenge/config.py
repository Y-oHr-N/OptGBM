"""Config."""

import os

import lightgbm as lgb

root_dir_path = "examples/otto-group-product-classification-challenge"
label_col = "target"

n_jobs = -1
random_state = 0

c = get_config()  # noqa

c.Recipe.data_path = os.path.join(root_dir_path, "train.csv.gz")
c.Recipe.label_col = label_col
c.Recipe.read_params = {"index_col": "id"}

c.Recipe.model_instance = lgb.LGBMClassifier(
    n_jobs=n_jobs, random_state=random_state
)
c.Recipe.model_path = os.path.join(root_dir_path, "model.pkl")
