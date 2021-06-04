"""Proxies."""

import logging
from typing import Any
from typing import List
from typing import Optional

import lightgbm as lgb
import numpy as np

from .typing import TwoDimArrayLikeType


class _VotingBooster(object):
    @property
    def feature_name(self) -> List[str]:
        return self.boosters[0].feature_name

    def __init__(
        self, boosters: List[lgb.Booster], weights: Optional[np.ndarray] = None
    ) -> None:
        if not boosters:
            raise ValueError("boosters must be non-empty array.")

        self.boosters = boosters
        self.weights = weights

    def feature_importance(self, **kwargs: Any) -> np.ndarray:
        results = [b.feature_importance(**kwargs) for b in self.boosters]

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
            b.predict(X, num_iteration=num_iteration) for b in self.boosters
        ]

        return np.average(results, axis=0, weights=self.weights)
