"""scikit-learn compatible models."""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import _num_samples

RANDOM_STATE_TYPE = Optional[Union[int, np.random.RandomState]]
ONE_DIM_ARRAYLIKE_TYPE = Optional[Union[np.ndarray, pd.Series]]
TWO_DIM_ARRAYLIKE_TYPE = Union[np.ndarray, pd.DataFrame]

MAX_INT = np.iinfo(np.int32).max

CLASSIFICATION_METRICS = {
    'binary': 'binary_logloss',
    'multiclass': 'multi_logloss',
    'softmax': 'multi_logloss',
    'multiclassova': 'multi_logloss',
    'multiclass_ova': 'multi_logloss',
    'ova': 'multi_logloss',
    'ovr': 'multi_logloss'
}
REGRESSION_METRICS = {
    'mean_absoluter_error': 'l1',
    'mae': 'l1',
    'regression_l1': 'l1',
    'l2_root': 'l2',
    'mean_squared_error': 'l2',
    'mse': 'l2',
    'regression': 'l2',
    'regression_l2': 'l2',
    'root_mean_squared_error': 'l2',
    'rmse': 'l2',
    'huber': 'huber',
    'fair': 'fair',
    'poisson': 'poisson',
    'quantile': 'quantile',
    'mean_absolute_percentage_error': 'mape',
    'mape': 'mape',
    'gamma': 'gamma',
    'tweedie': 'tweedie'
}
METRICS = {**CLASSIFICATION_METRICS, **REGRESSION_METRICS}

PARAM_DISTRIBUTIONS = {
    'colsample_bytree':
        optuna.distributions.DiscreteUniformDistribution(0.5, 0.9, 0.05),
    'min_child_samples':
        optuna.distributions.IntUniformDistribution(1, 100),
    'min_child_weight':
        optuna.distributions.LogUniformDistribution(1e-03, 10.0),
    'num_leaves':
        optuna.distributions.IntUniformDistribution(2, 127),
    'reg_alpha':
        optuna.distributions.LogUniformDistribution(1e-06, 10.0),
    'reg_lambda':
        optuna.distributions.LogUniformDistribution(1e-6, 10.0),
    'subsample':
        optuna.distributions.DiscreteUniformDistribution(0.5, 0.9, 0.05),
    'subsample_freq':
        optuna.distributions.IntUniformDistribution(1, 10)
}


class LightGBMCallbackEnv(NamedTuple):
    """Callback environment used by callbacks."""

    model: lgb.engine._CVBooster
    params: Dict[str, Any]
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: List


class ExtractionCallback(object):
    """Callback that extracts trained boosters."""

    @property
    def boosters_(self) -> List[lgb.Booster]:
        """Trained boosters."""
        return self._env.model.boosters

    def __call__(self, env: LightGBMCallbackEnv) -> None:
        """Extract a callback environment."""
        self._env = env


class Objective(object):
    """Objective function."""

    def __init__(
        self,
        params: Dict[str, Any],
        param_distributions: Dict[str, optuna.distributions.BaseDistribution],
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        categorical_feature: Union[List[Union[int, str]], str] = 'auto',
        cv: Optional[BaseCrossValidator] = None,
        early_stopping_rounds: Optional[int] = None,
        enable_pruning: bool = False,
        n_estimators: int = 100,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> None:
        self.categorical_feature = categorical_feature
        self.cv = cv
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_pruning = enable_pruning
        self.n_estimators = n_estimators
        self.params = params
        self.param_distributions = param_distributions
        self.sample_weight = sample_weight
        self.X = X
        self.y = y

    def __call__(self, trial: optuna.trial.Trial) -> float:
        """Return the CV score for a trial."""
        params: Dict[str, Any] = self._get_params(trial)
        callbacks: List[Callable] = self._get_callbacks(trial)
        dataset: lgb.Dataset = lgb.Dataset(
            self.X,
            label=self.y,
            weight=self.sample_weight
        )
        eval_hist: Dict[str, List[float]] = lgb.cv(
            params,
            dataset,
            callbacks=callbacks,
            categorical_feature=self.categorical_feature,
            early_stopping_rounds=self.early_stopping_rounds,
            folds=self.cv,
            num_boost_round=self.n_estimators
        )
        value: float = eval_hist[f'{self.params["metric"]}-mean'][-1]

        if self._is_best_trial(trial, value):
            boosters: List[lgb.Booster] = \
                callbacks[0].boosters_  # type: ignore

            for b in boosters:
                b.free_dataset()

            trial.study.set_user_attr('boosters', boosters)

        return value

    def _get_callbacks(self, trial: optuna.trial.Trial) -> List[Callable]:
        extraction_callback: ExtractionCallback = ExtractionCallback()
        callbacks: List[Callable] = [extraction_callback]

        if self.enable_pruning:
            pruning_callback: optuna.integration.LightGBMPruningCallback = \
                optuna.integration.LightGBMPruningCallback(
                    trial,
                    self.params['metric']
                )

            callbacks.append(pruning_callback)

        return callbacks

    def _get_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            name: trial._suggest(
                name, distribution
            ) for name, distribution in self.param_distributions.items()
        }

        params.update(self.params)

        return params

    def _is_best_trial(self, trial: optuna.trial.Trial, value: float) -> bool:
        try:
            return value < trial.study.best_value
        except ValueError:
            return True


class BaseOGBMModel(BaseEstimator):
    """Base class for models in OptGBM."""

    def __init__(
        self,
        categorical_features: Union[List[Union[int, str]], str] = 'auto',
        class_weight: Optional[Union[str, Dict[str, float]]] = None,
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = True,
        learning_rate: float = 0.1,
        max_iter: int = 100,
        n_iter_no_change: Optional[int] = None,
        n_jobs: int = 1,
        n_trials: int = 10,
        objective: Optional[str] = None,
        param_distributions:
            Optional[Dict[str, optuna.distributions.BaseDistribution]] = None,
        random_state: RANDOM_STATE_TYPE = None,
        study: optuna.study.Study = None,
        timeout: float = None,
    ) -> None:
        self.categorical_features = categorical_features
        self.class_weight = class_weight
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_no_change = n_iter_no_change
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.objective = objective
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.study = study
        self.timeout = timeout

    def fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> 'BaseOGBMModel':
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        sample_weight
            Weights of training data.

        Returns
        -------
        self
            Return self.
        """
        if not isinstance(X, pd.DataFrame):
            X, y = check_X_y(
                X,
                y,
                accept_sparse=True,
                ensure_min_samples=2,
                estimator=self,
                force_all_finite=False
            )

        is_classifier = self._estimator_type == 'classifier'
        cv = check_cv(self.cv, y, is_classifier)
        random_state = check_random_state(self.random_state)
        seed = random_state.randint(0, MAX_INT)
        params = {
            'learning_rate': self.learning_rate,
            'n_jobs': 1,
            'seed': seed,
            'verbose': -1
        }

        if is_classifier:
            self.encoder_ = LabelEncoder()

            y = self.encoder_.fit_transform(y)
            n_classes = len(self.encoder_.classes_)

            if n_classes > 2:
                params['num_classes'] = n_classes
                params['objective'] = 'multiclass'
            else:
                params['objective'] = 'binary'

        else:
            params['objective'] = 'regression'

        if self.objective is not None:
            params['objective'] = self.objective

        params['metric'] = METRICS[params['objective']]

        if self.param_distributions is None:
            param_distributions = PARAM_DISTRIBUTIONS
        else:
            param_distributions = self.param_distributions

        if sample_weight is None:
            n_samples = _num_samples(X)
            sample_weight = np.ones(n_samples)

        if self.class_weight is not None:
            sample_weight *= compute_sample_weight(self.class_weight, y)

        if self.study is None:
            sampler = optuna.samplers.TPESampler(seed=seed)

            self.study_ = optuna.create_study(sampler=sampler)

        else:
            self.study_ = self.study

        objective = Objective(
            params,
            param_distributions,
            X,
            y,
            categorical_feature=self.categorical_features,
            cv=cv,
            early_stopping_rounds=self.n_iter_no_change,
            enable_pruning=self.enable_pruning,
            n_estimators=self.max_iter,
            sample_weight=sample_weight,
        )

        self.weights_ = np.array([
            np.sum(sample_weight[train]) for train, _ in cv.split(X, y)
        ])

        self.study_.optimize(
            objective,
            n_jobs=self.n_jobs,
            n_trials=self.n_trials,
            timeout=self.timeout
        )

        return self


class OGBMClassifier(BaseOGBMModel, ClassifierMixin):
    """OptGBM classifier.

    Examples
    --------
    >>> from optgbm import OGBMClassifier
    >>> from sklearn.datasets import load_iris
    >>> clf = OGBMClassifier(random_state=0)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    OGBMClassifier(...)
    """


class OGBMRegressor(BaseOGBMModel, RegressorMixin):
    """OptGBM regressor.

    Examples
    --------
    >>> from optgbm import OGBMRegressor
    >>> from sklearn.datasets import load_boston
    >>> reg = OGBMRegressor(random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    OGBMRegressor(...)
    """
