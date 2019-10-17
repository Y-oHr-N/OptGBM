"""CLI."""

import pandas as pd
import yaml

from joblib import dump

from .sklearn import OGBMClassifier
from .sklearn import OGBMRegressor


class Trainer(object):
    """Trainer."""

    def train(self, recipe_path: str) -> None:
        """Train the model."""
        with open(recipe_path, 'r') as f:
            recipe = yaml.load(f)

        if recipe['model_name'].lower() == 'ogbmclassifier':
            model = OGBMClassifier()
        elif recipe['model_name'].lower() == 'ogbmregressor':
            model = OGBMRegressor()
        else:
            raise ValueError(
                'Unknown `model_name`: {}.'.format(recipe['model_name'])
            )

        if recipe['params'] is not None:
            model.set_params(**recipe['params'])

        data = pd.read_csv(recipe['data_path'], dtype=recipe['dtype'])
        target = pd.read_csv(recipe['target_path'], squeeze=True)

        model.fit(data, target)

        dump(model, recipe['model_path'])
