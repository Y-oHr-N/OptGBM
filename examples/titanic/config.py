"""Config."""

from optgbm.sklearn import OGBMClassifier

c = get_config()  # noqa

c.Recipe.data_path = "examples/titanic/train.csv.gz"
c.Recipe.label_col = "Survived"
c.Recipe.read_params = {
    "dtype": {"Pclass": "category"},
    "index_col": "PassengerId",
    "usecols": [
        "PassengerId",
        "Age",
        "Embarked",
        "Fare",
        "Pclass",
        "Parch",
        "Sex",
        "SibSp",
        "Survived",
    ],
}

c.Recipe.model_instance = OGBMClassifier(
    n_estimators=100_000, n_trials=100, random_state=0
)
c.Recipe.model_path = "examples/titanic/model.pkl"