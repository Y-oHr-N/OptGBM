"""CLI."""

import importlib

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


class Trainer(object):
    """Trainer."""

    def train(self, recipe_path: str) -> None:
        """Train the model with a recipe."""
        with open(recipe_path, 'r') as f:
            content = yaml.load(f)

        dtype = content.get('dtype')
        index_col = content.get('index_col')
        params = content.get('params', {})
        fit_params = content.get('fit_params', {})

        data = pd.read_csv(
            content['data_path'],
            dtype=dtype,
            index_col=index_col
        )
        label = data.pop(content['label_col'])

        module_name, class_name = content['model_source'].rsplit(
            '.',
            maxsplit=1
        )
        module = importlib.import_module(module_name)
        klass = getattr(module, class_name)
        model = klass(**params)

        model.fit(data, label, **fit_params)

        dump(model, content['model_path'])
