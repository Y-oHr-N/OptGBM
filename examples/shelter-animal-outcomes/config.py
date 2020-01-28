"""Config."""

import numpy as np
import pandas as pd

from optgbm.sklearn import OGBMClassifier
from sklearn.model_selection import TimeSeriesSplit


def transform_batch(data: pd.DataFrame, train: bool = True) -> pd.DataFrame:
    """User-defined preprocessing."""
    if train:
        data = data.sort_values("DateTime")

    s = data["DateTime"]

    data["{}_unixtime".format(s.name)] = 1e-09 * s.astype("int64")

    attrs = [
        # 'year',
        # 'weekofyear',
        "dayofyear",
        "quarter",
        "month",
        "day",
        "weekday",
        "hour",
        "minute",
        "second",
    ]

    for attr in attrs:
        if attr == "dayofyear":
            period = np.where(s.dt.is_leap_year, 366.0, 365.0)
        elif attr == "quarter":
            period = 4.0
        elif attr == "month":
            period = 12.0
        elif attr == "day":
            period = s.dt.daysinmonth
        elif attr == "weekday":
            period = 7.0
        elif attr == "hour":
            period = 24.0
        elif attr in ["minute", "second"]:
            period = 60.0

        theta = 2.0 * np.pi * getattr(s.dt, attr) / period

        data["{}_{}_sin".format(s.name, attr)] = np.sin(theta)
        data["{}_{}_cos".format(s.name, attr)] = np.cos(theta)

    data["HasName"] = ~data["Name"].isnull()

    data["IsIntact"] = (
        data["SexuponOutcome"].str.contains("Intact").astype("category")
    )
    data["IsMale"] = (
        data["SexuponOutcome"].str.contains("Male").astype("category")
    )

    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(
        r"years?", "* 365.0"
    )
    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(
        r"months?", "* 30.417"
    )
    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(
        r"weeks?", "* 7.0"
    )
    data["AgeuponOutcome"] = data["AgeuponOutcome"].str.replace(r"days?", "")
    data["AgeuponOutcome"] = data["AgeuponOutcome"].apply(
        lambda x: np.nan if pd.isnull(x) else eval(x)
    )

    return data.drop(columns=["Name", "DateTime"])


c = get_config()  # noqa

c.Recipe.data_path = "examples/shelter-animal-outcomes/train.csv.gz"
c.Recipe.label_col = "OutcomeType"
c.Recipe.read_params = {
    "index_col": 0,
    "na_values": {"SexuponOutcome": ["Unknown"]},
    "parse_dates": ["DateTime"],
    "usecols": lambda col: col not in ["OutcomeSubtype"],
}
c.Recipe.transform_batch = transform_batch

c.Recipe.model_instance = OGBMClassifier(
    cv=TimeSeriesSplit(5), n_estimators=100_000, n_trials=100, random_state=0
)
c.Recipe.model_path = "examples/shelter-animal-outcomes/model.pkl"
