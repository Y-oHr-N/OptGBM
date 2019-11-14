"""CLI."""

import importlib

from typing import Any
from typing import Optional

import click
import pandas as pd
import yaml

from joblib import dump


@click.group()
def optgbm() -> None:
    """Run optgbm."""


@optgbm.command()
@click.argument('recipe_path')
def train(recipe_path: str) -> None:
    """Train the model with a recipe."""
    trainer = Trainer()

    trainer.train(recipe_path)


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

    def train(self, recipe_path: str) -> None:
        """Train the model with a recipe."""
        with open(recipe_path, 'r') as f:
            content = yaml.load(f)

        data_kwargs = content.get('data_kwargs', {})
        params = content.get('params', {})
        fit_params = content.get('fit_params', {})

        dataset = Dataset(
            content['data_path'],
            label=content['label_col'],
            **data_kwargs
        )
        data = dataset.get_data()
        label = dataset.get_label()

        module_name, class_name = content['model_source'].rsplit(
            '.',
            maxsplit=1
        )
        module = importlib.import_module(module_name)
        klass = getattr(module, class_name)
        model = klass(**params)

        model.fit(data, label, **fit_params)

        dump(model, content['model_path'])
