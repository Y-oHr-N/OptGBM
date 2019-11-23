"""CLI."""

import logging
import sys

from typing import Any
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
@click.argument('config-path', type=click.Path(exists=True))
def train(config_path: str) -> None:
    """Train the model with a recipe."""
    trainer = Trainer(config_path)

    trainer.train()


@optgbm.command()
@click.argument('config-path', type=click.Path(exists=True))
@click.argument('input-path', type=click.Path(exists=True))
@click.option('--output-path', '-o', default=None, type=click.Path())
def predict(config_path: str, input_path: str, output_path: str) -> None:
    """Predict using the fitted model."""
    predictor = Predictor(config_path)

    y_pred = predictor.predict(input_path)

    if output_path is None:
        output_path = sys.stdout

    logger.info('Write the result to a csv file.')

    y_pred.to_csv(output_path, header=True)


class Recipe(traitlets.config.Configurable):
    """Recipe."""

    data_path = traitlets.Unicode(
        default_value='/path/to/data.csv',
        help='Path to the dataset.'
    ).tag(config=True)

    label_col = traitlets.Unicode(
        default_value='label',
        help='Label of the data.'
    ).tag(config=True)

    dataset_kwargs = traitlets.Dict(
        help='Parameters passes to `pd.read_csv`.'
    ).tag(config=True)

    model_instance = traitlets.Instance(
        help='Model to be fit.',
        klass=BaseEstimator
    ).tag(config=True)

    fit_params = traitlets.Dict(
        help='Parameters passed to `fit` of the estimator.'
    ).tag(config=True)

    model_path = traitlets.Unicode(
        default_value='/path/to/model.pkl',
        help='Path to the model.'
    ).tag(config=True)


class Dataset(object):
    """Dataset."""

    def __init__(
        self,
        data: str,
        label: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self.data = data
        self.label = label

        self._data = pd.read_csv(data, **kwargs)

        categorical_cols = self._data.dtypes == object

        if np.sum(categorical_cols) > 0:
            self._data.loc[:, categorical_cols] = \
                self._data.loc[:, categorical_cols].astype('category')

    def get_data(self) -> pd.DataFrame:
        """Get the data of the dataset."""
        if self.label is None:
            return self._data

        return self._data.drop(columns=self.label)

    def get_label(self) -> Optional[pd.Series]:
        """Get the label of the dataset."""
        if self.label is None:
            return None

        return self._data[self.label]


class Trainer(object):
    """Trainer."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def train(self) -> None:
        """Train the model with a recipe."""
        logger.info('Load the recipe.')

        loader = traitlets.config.loader.PyFileConfigLoader(self.config_path)
        config = loader.load_config()
        recipe = Recipe(config=config)

        logger.info('Load the dataset.')

        dataset = Dataset(
            recipe.data_path,
            label=recipe.label_col,
            **recipe.dataset_kwargs
        )
        data = dataset.get_data()
        label = dataset.get_label()

        logger.info('Fit the model according to the given training data.')

        model = clone(recipe.model_instance)

        model.fit(data, label, **recipe.fit_params)

        logger.info('Dump the model.')

        dump(model, recipe.model_path)


class Predictor(object):
    """Predictor."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def predict(self, input_path: str) -> pd.Series:
        """Predict using the fitted model."""
        logger.info('Load the recipe.')

        loader = traitlets.config.loader.PyFileConfigLoader(self.config_path)
        config = loader.load_config()
        recipe = Recipe(config=config)

        dataset_kwargs = recipe.dataset_kwargs.copy()

        if recipe.label_col in dataset_kwargs.get('usecols', {}):
            dataset_kwargs['usecols'].remove(recipe.label_col)

        logger.info('Load the dataset.')

        dataset = Dataset(input_path, **dataset_kwargs)
        data = dataset.get_data()

        logger.info('Load the model.')

        model = load(recipe.model_path)

        logger.info('Predict using the fitted model.')

        y_pred = model.predict(data)

        return pd.Series(y_pred, index=data.index, name=recipe.label_col)
