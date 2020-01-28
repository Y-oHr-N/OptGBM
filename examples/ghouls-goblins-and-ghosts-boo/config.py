"""Config."""

from optgbm.sklearn import OGBMClassifier

c = get_config()  # noqa

c.Recipe.data_path = "examples/ghouls-goblins-and-ghosts-boo/train.csv.gz"
c.Recipe.label_col = "type"
c.Recipe.read_params = {"index_col": "id"}

c.Recipe.model_instance = OGBMClassifier(
    n_estimators=100_000, n_trials=100, random_state=0
)
c.Recipe.model_path = "examples/ghouls-goblins-and-ghosts-boo/model.pkl"
