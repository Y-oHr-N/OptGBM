"""scikit-learn compatible models."""

import copy

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna

from joblib import delayed
from joblib import effective_n_jobs
from joblib import Parallel
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from .utils import check_cv
from .utils import check_fit_params
from .utils import check_X
from .utils import LightGBMCallbackEnv
from .utils import ONE_DIM_ARRAYLIKE_TYPE
from .utils import RANDOM_STATE_TYPE
from .utils import TWO_DIM_ARRAYLIKE_TYPE

MAX_INT = np.iinfo(np.int32).max

OBJECTIVE2METRIC = {
    # classification
    'binary': 'binary_logloss',
    'multiclass': 'multi_logloss',
    'softmax': 'multi_logloss',
    'multiclassova': 'multi_logloss',
    'multiclass_ova': 'multi_logloss',
    'ova': 'multi_logloss',
    'ovr': 'multi_logloss',
    # regression
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


def _is_higher_better(metric: str) -> bool:
    return metric in ['auc']


class _LightGBMExtractionCallback(object):
    def __init__(self) -> None:
        self._best_iteration: Optional[int] = None
        self._boosters: Optional[List[lgb.Booster]] = None

    def __call__(self, env: LightGBMCallbackEnv) -> None:
        self._best_iteration = env.iteration + 1
        self._boosters = env.model.boosters


class _Objective(object):
    def __init__(
        self,
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        param_distributions: Dict[str, optuna.distributions.BaseDistribution],
        cv: Optional[BaseCrossValidator] = None,
        early_stopping_rounds: Optional[int] = None,
        enable_pruning: bool = False,
        n_estimators: int = 100
    ) -> None:
        self.cv = cv
        self.dataset = dataset
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_pruning = enable_pruning
        self.n_estimators = n_estimators
        self.params = params
        self.param_distributions = param_distributions

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params: Dict[str, Any] = self._get_params(trial)
        dataset = copy.copy(self.dataset)
        callbacks: List[Callable] = self._get_callbacks(trial)
        eval_hist: Dict[str, List[float]] = lgb.cv(
            params,
            dataset,
            callbacks=callbacks,
            early_stopping_rounds=self.early_stopping_rounds,
            folds=self.cv,
            num_boost_round=self.n_estimators
        )
        value: float = eval_hist[f'{params["metric"]}-mean'][-1]
        is_best_trial: bool = True

        try:
            is_best_trial = (value < trial.study.best_value) \
                ^ _is_higher_better(params['metric'])
        except ValueError:
            pass

        if is_best_trial:
            best_iteration: int = callbacks[0]._best_iteration  # type: ignore
            boosters: List[lgb.Booster] = \
                callbacks[0]._boosters  # type: ignore
            representations: List[str] = []

            for b in boosters:
                b.free_dataset()
                representations.append(b.model_to_string())

            trial.study.set_user_attr('best_iteration', best_iteration)
            trial.study.set_user_attr('representations', representations)

        return value

    def _get_callbacks(self, trial: optuna.trial.Trial) -> List[Callable]:
        extraction_callback: _LightGBMExtractionCallback = \
            _LightGBMExtractionCallback()
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
    def _param_distributions(
        self
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        if self.param_distributions is None:
            return DEFAULT_PARAM_DISTRIBUTIONS

        return self.param_distributions

    @property
    def _random_state(self) -> Optional[int]:
        if self.random_state is None or isinstance(self.random_state, int):
            return self.random_state

        random_state = check_random_state(self.random_state)

        return random_state.randint(0, MAX_INT)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances."""
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
        class_weight: Optional[Union[str, Dict[str, float]]] = None,
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = False,
        importance_type: str = 'split',
        learning_rate: float = 0.1,
        n_estimators: int = 1_000,
        n_jobs: int = 1,
        n_trials: int = 25,
        objective: Optional[str] = None,
        param_distributions:
            Optional[Dict[str, optuna.distributions.BaseDistribution]] = None,
        random_state: Optional[RANDOM_STATE_TYPE] = None,
        refit: bool = False,
        study: Optional[optuna.study.Study] = None,
        timeout: Optional[float] = None
    ) -> None:
        self.class_weight = class_weight
        self.cv = cv
        self.enable_pruning = enable_pruning
        self.importance_type = importance_type
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.objective = objective
        self.param_distributions = param_distributions
        self.random_state = random_state
        self.refit = refit
        self.study = study
        self.timeout = timeout

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, 'n_features_')

    def _more_tags(self) -> Dict[str, Any]:
        return {'allow_nan': True, 'non_deterministic': True}

    def fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: Optional[ONE_DIM_ARRAYLIKE_TYPE] = None,
        eval_metric: Optional[str] = None,
        early_stopping_rounds: Optional[int] = 10,
        categorical_feature: Union[List[Union[int, str]], str] = 'auto'
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

        eval_metric
            Evaluation metric.

        early_stopping_rounds
            Used to activate early stopping.

        categorical_feature
            Categorical features.

        Returns
        -------
        self
            Return self.
        """
        X, y, sample_weight = check_fit_params(
            X,
            y,
            sample_weight=sample_weight,
            accept_sparse=True,
            estimator=self,
            force_all_finite=False
        )

        _, self.n_features_ = X.shape

        is_classifier = self._estimator_type == 'classifier'
        cv = check_cv(self.cv, y, is_classifier)

        seed = self._random_state

        params: Dict[str, Any] = {
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

        if eval_metric is None:
            params['metric'] = OBJECTIVE2METRIC[params['objective']]
        else:
            params['metric'] = eval_metric

        if _is_higher_better(params['metric']):
            direction = 'maximize'
        else:
            direction = 'minimize'

        if self.study is None:
            sampler = optuna.samplers.TPESampler(seed=seed)

            self.study_ = optuna.create_study(
                direction=direction,
                sampler=sampler
            )

        else:
            self.study_ = self.study

        dataset = lgb.Dataset(
            X,
            categorical_feature=categorical_feature,
            label=y,
            weight=sample_weight
        )

        objective = _Objective(
            params,
            dataset,
            self._param_distributions,
            cv=cv,
            early_stopping_rounds=early_stopping_rounds,
            enable_pruning=self.enable_pruning,
            n_estimators=self.n_estimators
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

        self.n_iter_ = self.study_.user_attrs['best_iteration']

        if self.refit:
            params.update(self.study_.best_params)

            booster = lgb.train(params, dataset, num_boost_round=self.n_iter_)

            booster.free_dataset()

            self.boosters_ = [booster]

        else:
            try:  # lightgbm<=2.2.3
                self.boosters_ = [
                    lgb.Booster(
                        params={'model_str': model_str}
                    ) for model_str
                    in self.study_.user_attrs['representations']
                ]
            except TypeError:
                self.boosters_ = [
                    lgb.Booster(
                        model_str=model_str,
                        silent=True
                    ) for model_str
                    in self.study_.user_attrs['representations']
                ]

        return self


class OGBMClassifier(_BaseOGBMModel, ClassifierMixin):
    """OptGBM classifier.

    Parameters
    ----------
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

    n_estimators
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boost_round`.

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

    refit
        If True, refit the estimator with the best found hyperparameters.

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
    >>> from optgbm.sklearn import OGBMClassifier
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

        X = check_X(
            X,
            accept_sparse=True,
            estimator=self,
            force_all_finite=False
        )
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
    cv
        Cross-validation strategy.

    enable_pruning
        Used to activate pruning.

    importance_type
        Type of feature importances.

    learning_rate
        Learning rate.

    n_estimators
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boost_round`.

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

    refit
        If True, refit the estimator with the best found hyperparameters.

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
    >>> from optgbm.sklearn import OGBMRegressor
    >>> from sklearn.datasets import load_boston
    >>> reg = OGBMRegressor(random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    OGBMRegressor(...)
    >>> reg.score(X, y)
    0.9...
    """

    def __init__(
        self,
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = False,
        importance_type: str = 'split',
        learning_rate: float = 0.1,
        n_estimators: int = 1_000,
        n_jobs: int = 1,
        n_trials: int = 25,
        objective: Optional[str] = None,
        param_distributions:
            Optional[Dict[str, optuna.distributions.BaseDistribution]] = None,
        random_state: Optional[RANDOM_STATE_TYPE] = None,
        refit: bool = False,
        study: Optional[optuna.study.Study] = None,
        timeout: Optional[float] = None
    ) -> None:
        super().__init__(
            cv=cv,
            enable_pruning=enable_pruning,
            importance_type=importance_type,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            n_trials=n_trials,
            objective=objective,
            param_distributions=param_distributions,
            random_state=random_state,
            refit=refit,
            study=study,
            timeout=timeout
        )

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

        X = check_X(
            X,
            accept_sparse=True,
            estimator=self,
            force_all_finite=False
        )
        n_jobs = effective_n_jobs(self.n_jobs)
        parallel = Parallel(n_jobs=n_jobs)
        func = delayed(lgb.Booster.predict)
        results = parallel(func(b, X) for b in self.boosters_)

        return np.average(results, axis=0, weights=self.weights_)
