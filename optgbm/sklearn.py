"""scikit-learn compatible models."""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from joblib import delayed
from joblib import effective_n_jobs
from joblib import Parallel
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import check_cv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _assert_all_finite
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import column_or_1d

RANDOM_STATE_TYPE = Optional[Union[int, np.random.RandomState]]
ONE_DIM_ARRAYLIKE_TYPE = Union[np.ndarray, pd.Series]
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

DEFAULT_PARAM_DISTRIBUTIONS = {
    'colsample_bytree':
        optuna.distributions.DiscreteUniformDistribution(0.5, 1.0, 0.05),
    'min_child_samples':
        optuna.distributions.IntUniformDistribution(1, 100),
    'min_child_weight':
        optuna.distributions.LogUniformDistribution(1e-03, 10.0),
    'num_leaves':
        optuna.distributions.IntUniformDistribution(2, 127),
    'reg_alpha':
        optuna.distributions.LogUniformDistribution(1e-06, 10.0),
    'reg_lambda':
        optuna.distributions.LogUniformDistribution(1e-06, 10.0),
    'subsample':
        optuna.distributions.DiscreteUniformDistribution(0.5, 1.0, 0.05),
    'subsample_freq':
        optuna.distributions.IntUniformDistribution(1, 10)
}


class _LightGBMCallbackEnv(NamedTuple):
    model: lgb.engine._CVBooster
    params: Dict[str, Any]
    iteration: int
    begin_iteration: int
    end_iteration: int
    evaluation_result_list: List


class _ModelExtractionCallback(object):
    def __init__(self) -> None:
        self._best_iteration: Optional[int] = None
        self._model: Optional[lgb.engine._CVBooster] = None

    def __call__(self, env: _LightGBMCallbackEnv) -> None:
        self._best_iteration = env.iteration + 1
        self._model = env.model


class _Objective(object):
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

        self._best_iteration: Optional[int] = None
        self._model: Optional[lgb.engine._CVBooster] = None

    def __call__(self, trial: optuna.trial.Trial) -> float:
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

        if self._model is None or value < trial.study.best_value:
            self._best_iteration = callbacks[0]._best_iteration  # type: ignore
            self._model = callbacks[0]._model  # type: ignore

        return value

    def _get_callbacks(self, trial: optuna.trial.Trial) -> List[Callable]:
        extraction_callback: _ModelExtractionCallback = \
            _ModelExtractionCallback()
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


class _BaseOGBMModel(BaseEstimator):
    @property
    def feature_importances_(self) -> np.ndarray:
        self._check_is_fitted()

        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.feature_importance)
        results = parallel(
            func(b, self.importance_type) for b in self.boosters_
        )

        return np.average(results, axis=0, weights=self.weights_)

    def __init__(
        self,
        categorical_features: Union[List[Union[int, str]], str] = 'auto',
        class_weight: Optional[Union[str, Dict[str, float]]] = None,
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = True,
        importance_type: str = 'split',
        learning_rate: float = 0.1,
        max_iter: int = 1_000,
        n_iter_no_change: Optional[int] = 10,
        n_jobs: int = 1,
        n_trials: int = 10,
        objective: Optional[str] = None,
        param_distributions:
            Optional[Dict[str, optuna.distributions.BaseDistribution]] = None,
        random_state: RANDOM_STATE_TYPE = None,
        study: Optional[optuna.study.Study] = None,
        timeout: Optional[float] = None
    ) -> None:
        self.categorical_features = categorical_features
        self.class_weight = class_weight
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.importance_type = importance_type
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

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, 'n_features_')

    def _check_X(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> TWO_DIM_ARRAYLIKE_TYPE:
        if not isinstance(X, pd.DataFrame):
            X = check_array(
                X,
                accept_sparse=True,
                estimator=self,
                force_all_finite=False
            )

        _, n_features = X.shape

        if n_features != getattr(self, 'n_features_', n_features):
            raise ValueError(f'Invalid data: X.shape={X.shape}.')

        return X

    def _check_y(self, y: ONE_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        if not isinstance(y, pd.Series):
            y = column_or_1d(y, warn=True)

        _assert_all_finite(y)

        if self._estimator_type == 'classifier':
            check_classification_targets(y)

        return y

    def _check_X_y(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE
    ) -> Tuple[TWO_DIM_ARRAYLIKE_TYPE, ONE_DIM_ARRAYLIKE_TYPE]:
        X = self._check_X(X)
        y = self._check_y(y)

        check_consistent_length(X, y)

        return X, y

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'non_deterministic': True}

    def fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: ONE_DIM_ARRAYLIKE_TYPE = None
    ) -> '_BaseOGBMModel':
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
        X, y = self._check_X_y(X, y)

        _, self.n_features_ = X.shape

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
            param_distributions = DEFAULT_PARAM_DISTRIBUTIONS
        else:
            param_distributions = self.param_distributions

        if sample_weight is None:
            n_samples = _num_samples(X)
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)

        if self.class_weight is not None:
            sample_weight *= compute_sample_weight(self.class_weight, y)

        if self.study is None:
            sampler = optuna.samplers.TPESampler(seed=seed)

            self.study_ = optuna.create_study(sampler=sampler)

        else:
            self.study_ = self.study

        objective = _Objective(
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

        model: lgb.engine._CVBooster = objective._model

        model.free_dataset()

        self.boosters_ = model.boosters
        self.n_iter_ = objective._best_iteration

        return self


class OGBMClassifier(_BaseOGBMModel, ClassifierMixin):
    """OptGBM classifier.

    Parameters
    ----------
    categorical_features
        Categorical features.

    class_weight
        Weights associated with classes.

    cv
        Cross-validation strategy.

    enable_pruning
        Used to activate pruning.

    importance_type
        Type of feature importances.

    learning_rate
        Learning rate.

    max_iter
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boosting_round`.

    n_iter_no_change
        Used to activate early stopping. a.k.a. `early_stopping_rounds`.

    n_jobs
        Number of parallel jobs.

    n_trials
        Number of trials.

    objective
        Learning objective.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    random_state
        Seed of the pseudo random number generator.

    study
        Study that corresponds to the optimization task.

    timeout
        Time limit in seconds for the search of appropriate models.

    Attributes
    ----------
    boosters_
        Trained boosters of CV.

    encoder_
        Label encoder.

    n_features_
        Number of features of fitted model.

    n_iter_
        Number of iterations as selected by early stopping. a.k.a.
        `best_iteration_`.

    study_
        Actual study.

    weights_
        Weights to weight the occurrences of predicted values before averaging.

    Examples
    --------
    >>> from optgbm import OGBMClassifier
    >>> from sklearn.datasets import load_iris
    >>> clf = OGBMClassifier(random_state=0)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    OGBMClassifier(...)
    >>> clf.score(X, y)
    0.9...
    """

    @property
    def classes_(self) -> np.ndarray:
        """Class labels."""
        self._check_is_fitted()

        return self.encoder_.classes_

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the Fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """
        probas = self.predict_proba(X)
        class_index = np.argmax(probas, axis=1)

        return self.encoder_.inverse_transform(class_index)

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        """Predict class probabilities for data.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        p
            Class probabilities of data.
        """
        self._check_is_fitted()

        X = self._check_X(X)
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.predict)
        results = parallel(func(b, X) for b in self.boosters_)
        result = np.average(results, axis=0, weights=self.weights_)
        n_classes = len(self.encoder_.classes_)

        if n_classes > 2:
            return result

        else:
            result = result.reshape(-1, 1)

            return np.concatenate([1.0 - result, result], axis=1)


class OGBMRegressor(_BaseOGBMModel, RegressorMixin):
    """OptGBM regressor.

    Parameters
    ----------
    categorical_features
        Categorical features.

    class_weight
        Weights associated with classes.

    cv
        Cross-validation strategy.

    enable_pruning
        Used to activate pruning.

    importance_type
        Type of feature importances.

    learning_rate
        Learning rate.

    max_iter
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boosting_round`.

    n_iter_no_change
        Used to activate early stopping. a.k.a. `early_stopping_rounds`.

    n_jobs
        Number of parallel jobs.

    n_trials
        Number of trials.

    objective
        Learning objective.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    random_state
        Seed of the pseudo random number generator.

    study
        Study that corresponds to the optimization task.

    timeout
        Time limit in seconds for the search of appropriate models.

    Attributes
    ----------
    boosters_
        Trained boosters of CV.

    n_features_
        Number of features of fitted model.

    n_iter_
        Number of iterations as selected by early stopping. a.k.a.
        `best_iteration_`.

    study_
        Actual study.

    weights_
        Weights to weight the occurrences of predicted values before averaging.

    Examples
    --------
    >>> from optgbm import OGBMRegressor
    >>> from sklearn.datasets import load_boston
    >>> reg = OGBMRegressor(random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    OGBMRegressor(...)
    >>> reg.score(X, y)
    0.9...
    """

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the Fitted model.

        Parameters
        ----------
        X
            Data.

        Returns
        -------
        y_pred
            Predicted values.
        """
        self._check_is_fitted()

        X = self._check_X(X)
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.predict)
        results = parallel(func(b, X) for b in self.boosters_)

        return np.average(results, axis=0, weights=self.weights_)
