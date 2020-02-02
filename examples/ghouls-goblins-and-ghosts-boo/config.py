"""Config."""

import os

from optgbm.sklearn import OGBMClassifier

root_dir_path = "examples/ghouls-goblins-and-ghosts-boo"
label_col = "type"

cv = 5
enable_pruning = False
importance_type = "gain"
n_estimators = 100_000
n_jobs = -1
n_trials = 100
random_state = 0

c = get_config()  # noqa

c.Recipe.data_path = os.path.join(root_dir_path, "train.csv.gz")
c.Recipe.label_col = label_col
c.Recipe.read_params = {"index_col": "id"}

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
