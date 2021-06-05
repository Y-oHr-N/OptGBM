"""Proxies."""

import logging
import pathlib
import pickle
from typing import Any
from typing import List
from typing import Optional

import numpy as np
from natsort import natsorted

from .typing import TwoDimArrayLikeType


def _natsorted(x: List) -> List:
    return natsorted(x, key=lambda elm: str(elm))


class _VotingBooster(object):
    @property
    def feature_name(self) -> List[str]:
        return self._boosters[0].feature_name

    def __init__(
        self, model_dir: pathlib.Path, weights: Optional[np.ndarray] = None
    ) -> None:
        self.model_dir = model_dir
        self.weights = weights

        self._boosters = []

        booster_paths = _natsorted(
            [booster_path for booster_path in model_dir.glob("**/fold_*.pkl")]
        )

        for booster_path in booster_paths:
            with booster_path.open("rb") as f:
                b = pickle.load(f)

            self._boosters.append(b)

    def feature_importance(self, **kwargs: Any) -> np.ndarray:
        results = [b.feature_importance(**kwargs) for b in self._boosters]

        return np.average(results, axis=0, weights=self.weights)

    def predict(
        self,
        X: TwoDimArrayLikeType,
        num_iteration: Optional[int] = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **predict_params: Any
    ) -> np.ndarray:
        logger = logging.getLogger(__name__)

        if raw_score:
            raise ValueError("_VotingBooster cannot return raw scores.")

        if pred_leaf:
            raise ValueError("_VotingBooster cannot return leaf indices.")

        if pred_contrib:
            raise ValueError(
                "_VotingBooster cannot return feature contributions."
            )

        for key, value in predict_params.items():
            logger.warning("{}={} will be ignored.".format(key, value))

        results = [
            b.predict(X, num_iteration=num_iteration) for b in self._boosters
        ]

        return np.average(results, axis=0, weights=self.weights)
