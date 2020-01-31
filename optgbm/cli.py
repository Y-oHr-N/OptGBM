"""CLI."""

import inspect
import logging
import sys

from typing import Any
from typing import Callable
from typing import Optional

import click
import numpy as np
import pandas as pd
import traitlets
import traitlets.config

from joblib import dump
from joblib import load
from sklearn.base import BaseEstimator
from sklearn.base import clone

logger = logging.getLogger(__name__)


@click.group()
def optgbm() -> None:
    """Run optgbm."""


@optgbm.command()
@click.argument("config-path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    """Train the model with a recipe."""
    trainer = Trainer(config_path)

    trainer.train()


@optgbm.command()
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("input-path", type=click.Path(exists=True))
@click.option("--output-path", "-o", default=None, type=click.Path())
@click.option("--label-col", "-l", default=None)
def predict(
    config_path: str, input_path: str, output_path: str, label_col: str
) -> None:
    """Predict using the fitted model."""
    predictor = Predictor(config_path)

    y_pred = predictor.predict(input_path, label_col=label_col)

    if output_path is None:
        output_path = sys.stdout

    logger.info("Write the result to a csv file.")

    y_pred.to_csv(output_path, header=True)


@optgbm.command()
@click.argument("config-path", type=click.Path(exists=True))
@click.argument("input-path", type=click.Path(exists=True))
@click.option("--output-path", "-o", default=None, type=click.Path())
def predict_proba(config_path: str, input_path: str, output_path: str) -> None:
    """Predict class probabilities for data."""
    predictor = Predictor(config_path)

    probas = predictor.predict_proba(input_path)

    if output_path is None:
        output_path = sys.stdout

    logger.info("Write the result to a csv file.")

    probas.to_csv(output_path, header=True)


class Recipe(traitlets.config.Configurable):
    """Recipe."""

    data_path = traitlets.Unicode(
        default_value="/path/to/data.csv", help="Path to the dataset."
    ).tag(config=True)

    label_col = traitlets.Unicode(
        default_value="label", help="Label of the data."
    ).tag(config=True)

    read_params = traitlets.Dict(
        help="Parameters passed to `pd.read_csv`."
    ).tag(config=True)

    transform_batch = traitlets.Any(
        help="Callable that transforms the data."
    ).tag(config=True)

    model_instance = traitlets.Instance(
        help="Model to be fit.", klass=BaseEstimator, kw={}
    ).tag(config=True)

    fit_params = traitlets.Dict(
        help="Parameters passed to `fit` of the estimator."
    ).tag(config=True)

    model_path = traitlets.Unicode(
        default_value="/path/to/model.pkl", help="Path to the model."
    ).tag(config=True)


class Dataset(object):
    """Dataset."""

    def __init__(
        self,
        data: str,
        label: Optional[str] = None,
        train: bool = True,
        transform_batch: Optional[Callable] = None,
        **read_params: Any
    ) -> None:
        self.data = data
        self.label = label
        self.train = train
        self.transform_batch = transform_batch

        usecols = read_params.get("usecols")

        if not train and isinstance(usecols, list) and label in usecols:
            read_params["usecols"].remove(label)

        self._data = pd.read_csv(data, **read_params)

        categorical_cols = self._data.dtypes == object

        if np.sum(categorical_cols) > 0:
            self._data.loc[:, categorical_cols] = self._data.loc[
                :, categorical_cols
            ].astype("category")

        if transform_batch is not None:
            kwargs = {}
            signature = inspect.signature(transform_batch)

            if "train" in signature.parameters:
                kwargs["train"] = self.train

            self._data = transform_batch(self._data, **kwargs)

    def get_data(self) -> pd.DataFrame:
        """Get the data of the dataset."""
        if not self.train:
            return self._data

        return self._data.drop(columns=self.label)

    def get_label(self) -> Optional[pd.Series]:
        """Get the label of the dataset."""
        if not self.train:
            return None

        return self._data[self.label]


class Trainer(object):
    """Trainer."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

        loader = traitlets.config.loader.PyFileConfigLoader(self.config_path)
        config = loader.load_config()

        self._recipe = Recipe(config=config)

    def train(self) -> None:
        """Train the model with a recipe."""
        logger.info("Load the dataset.")

        dataset = Dataset(
            self._recipe.data_path,
            label=self._recipe.label_col,
            transform_batch=self._recipe.transform_batch,
            **self._recipe.read_params
        )
        data = dataset.get_data()
        label = dataset.get_label()

        logger.info("Fit the model according to the given training data.")

        model = clone(self._recipe.model_instance)

        model.fit(data, label, **self._recipe.fit_params)

        logger.info("Dump the model.")

        dump(model, self._recipe.model_path)


class Predictor(object):
    """Predictor."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

        loader = traitlets.config.loader.PyFileConfigLoader(self.config_path)
        config = loader.load_config()

        self._recipe = Recipe(config=config)

    def predict(
        self, input_path: str, label_col: Optional[str] = None
    ) -> pd.Series:
        """Predict using the fitted model."""
        logger.info("Load the dataset.")

        dataset = Dataset(
            input_path,
            label=self._recipe.label_col,
            train=False,
            transform_batch=self._recipe.transform_batch,
            **self._recipe.read_params
        )
        data = dataset.get_data()

        logger.info("Load the model.")

        model = load(self._recipe.model_path)

        logger.info("Predict using the fitted model.")

        y_pred = model.predict(data)

        if label_col is None:
            label_col = self._recipe.label_col

        return pd.Series(y_pred, index=data.index, name=label_col)

    def predict_proba(self, input_path: str) -> pd.DataFrame:
        """Predict class probabilities for data."""
        logger.info("Load the dataset.")

        dataset = Dataset(
            input_path,
            label=self._recipe.label_col,
            train=False,
            transform_batch=self._recipe.transform_batch,
            **self._recipe.read_params
        )
        data = dataset.get_data()

        logger.info("Load the model.")

        model = load(self._recipe.model_path)

        logger.info("Predict class probabilities for data.")

        probas = model.predict_proba(data)

        return pd.DataFrame(probas, columns=model.classes_, index=data.index)
