"""scikit-learn compatible models."""

import copy
import logging

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing
from sklearn.utils.validation import check_is_fitted

try:  # lightgbm<=2.2.3
    from lightgbm.sklearn import _eval_function_wrapper as _EvalFunctionWrapper
    from lightgbm.sklearn import (
        _objective_function_wrapper as _ObjectiveFunctionWrapper,
    )
except ImportError:
    from lightgbm.sklearn import _EvalFunctionWrapper
    from lightgbm.sklearn import _ObjectiveFunctionWrapper

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
    "binary": "binary_logloss",
    "multiclass": "multi_logloss",
    "softmax": "multi_logloss",
    "multiclassova": "multi_logloss",
    "multiclass_ova": "multi_logloss",
    "ova": "multi_logloss",
    "ovr": "multi_logloss",
    # regression
    "mean_absoluter_error": "l1",
    "mae": "l1",
    "regression_l1": "l1",
    "l2_root": "l2",
    "mean_squared_error": "l2",
    "mse": "l2",
    "regression": "l2",
    "regression_l2": "l2",
    "root_mean_squared_error": "l2",
    "rmse": "l2",
    "huber": "huber",
    "fair": "fair",
    "poisson": "poisson",
    "quantile": "quantile",
    "mean_absolute_percentage_error": "mape",
    "mape": "mape",
    "gamma": "gamma",
    "tweedie": "tweedie",
}

DEFAULT_PARAM_DISTRIBUTIONS = {
    # "boosting_type": optuna.distributions.CategoricalDistribution(
    #     ["gbdt", "rf"]
    # ),
    "colsample_bytree": optuna.distributions.DiscreteUniformDistribution(
        0.1, 1.0, 0.05
    ),
    "min_child_samples": optuna.distributions.IntUniformDistribution(1, 100),
    # "min_child_weight": optuna.distributions.LogUniformDistribution(
    #     1e-03, 10.0
    # ),
    "num_leaves": optuna.distributions.IntUniformDistribution(2, 127),
    "reg_alpha": optuna.distributions.LogUniformDistribution(1e-09, 10.0),
    "reg_lambda": optuna.distributions.LogUniformDistribution(1e-09, 10.0),
    "subsample": optuna.distributions.DiscreteUniformDistribution(
        0.5, 0.95, 0.05
    ),
    "subsample_freq": optuna.distributions.IntUniformDistribution(1, 10),
}


def _is_higher_better(metric: str) -> bool:
    return metric in ["auc"]


class _LightGBMExtractionCallback(object):
    def __init__(self) -> None:
        self._best_iteration: Optional[int] = None
        self._boosters: Optional[List[lgb.Booster]] = None

    def __call__(self, env: LightGBMCallbackEnv) -> None:
        self._best_iteration = env.iteration + 1
        self._boosters = env.model.boosters


class _Objective(object):
    @property
    def _param_distributions(
        self,
    ) -> Dict[str, optuna.distributions.BaseDistribution]:
        if self.param_distributions is None:
            return DEFAULT_PARAM_DISTRIBUTIONS

        return self.param_distributions

    def __init__(
        self,
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        eval_name: str,
        is_higher_better: bool,
        callbacks: Optional[List[Callable]] = None,
        categorical_feature: Union[List[int], List[str], str] = "auto",
        cv: Optional[BaseCrossValidator] = None,
        early_stopping_rounds: Optional[int] = None,
        enable_pruning: bool = False,
        feature_name: Union[List[str], str] = "auto",
        feval: Optional[Callable] = None,
        fobj: Optional[Callable] = None,
        n_estimators: int = 100,
        param_distributions: Optional[
            Dict[str, optuna.distributions.BaseDistribution]
        ] = None,
    ) -> None:
        self.callbacks = callbacks
        self.categorical_feature = categorical_feature
        self.cv = cv
        self.dataset = dataset
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_pruning = enable_pruning
        self.eval_name = eval_name
        self.feature_name = feature_name
        self.feval = feval
        self.fobj = fobj
        self.is_higher_better = is_higher_better
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
            categorical_feature=self.categorical_feature,
            early_stopping_rounds=self.early_stopping_rounds,
            feature_name=self.feature_name,
            feval=self.feval,
            fobj=self.fobj,
            folds=self.cv,
            num_boost_round=self.n_estimators,
        )
        value: float = eval_hist[f"{self.eval_name}-mean"][-1]
        is_best_trial: bool = True

        try:
            is_best_trial = (
                value < trial.study.best_value
            ) ^ self.is_higher_better
        except ValueError:
            pass

        if is_best_trial:
            best_iteration: int = callbacks[0]._best_iteration  # type: ignore
            boosters: List[lgb.Booster] = (
                callbacks[0]._boosters  # type: ignore
            )
            representations: List[str] = []

            for b in boosters:
                b.free_dataset()
                representations.append(b.model_to_string())

            trial.study.set_user_attr("best_iteration", best_iteration)
            trial.study.set_user_attr("representations", representations)

        return value

    def _get_callbacks(self, trial: optuna.trial.Trial) -> List[Callable]:
        extraction_callback: _LightGBMExtractionCallback = (
            _LightGBMExtractionCallback()
        )
        callbacks: List[Callable] = [extraction_callback]

        if self.enable_pruning:
            pruning_callback: optuna.integration.LightGBMPruningCallback = (
                optuna.integration.LightGBMPruningCallback(
                    trial, self.eval_name
                )
            )

            callbacks.append(pruning_callback)

        if self.callbacks is not None:
            callbacks += self.callbacks

        return callbacks

    def _get_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        params: Dict[str, Any] = self.params.copy()

        for name, distribution in self._param_distributions.items():
            params[name] = trial._suggest(name, distribution)

        return params


class _VotingBooster(object):
    @property
    def feature_name(self) -> List[str]:
        return self.boosters[0].feature_name

    def __init__(
        self, boosters: List[lgb.Booster], weights: Optional[np.ndarray] = None
    ) -> None:
        self.boosters = boosters
        self.weights = weights

    @classmethod
    def from_representations(
        cls, representations: List[str], weights: Optional[np.ndarray] = None
    ) -> "_VotingBooster":
        try:  # lightgbm<=2.2.3
            boosters = [
                lgb.Booster(params={"model_str": model_str})
                for model_str in representations
            ]
        except TypeError:
            boosters = [
                lgb.Booster(model_str=model_str, silent=True)
                for model_str in representations
            ]

        return cls(boosters, weights=weights)

    def feature_importance(self, **kwargs: Any) -> np.ndarray:
        results = [b.feature_importance(**kwargs) for b in self.boosters]

        return np.average(results, axis=0, weights=self.weights)

    def predict(
        self, X: TWO_DIM_ARRAYLIKE_TYPE, **kwargs: Any
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        results = [b.predict(X, **kwargs) for b in self.boosters]

        return np.average(results, axis=0, weights=self.weights)


class _BaseOGBMModel(lgb.LGBMModel):
    @property
    def best_index_(self) -> int:
        """Get the best trial's number."""
        return self.study_.best_trial.number

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 1_000,
        subsample_for_bin: int = 200_000,
        objective: Optional[Union[Callable, str]] = None,
        class_weight: Optional[Union[Dict[str, float], str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-03,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        refit: bool = False,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[RANDOM_STATE_TYPE] = None,
        n_jobs: int = 1,
        importance_type: str = "split",
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = False,
        n_trials: int = 10,
        param_distributions: Optional[
            Dict[str, optuna.distributions.BaseDistribution]
        ] = None,
        study: Optional[optuna.study.Study] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            boosting_type=boosting_type,
            class_weight=class_weight,
            colsample_bytree=colsample_bytree,
            importance_type=importance_type,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            objective=objective,
            random_state=random_state,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            subsample_for_bin=subsample_for_bin,
            subsample_freq=subsample_freq,
            **kwargs,
        )

        self.cv = cv
        self.enable_pruning = enable_pruning
        self.n_trials = n_trials
        self.param_distributions = param_distributions
        self.refit = refit
        self.study = study
        self.timeout = timeout

    def _check_is_fitted(self) -> None:
        check_is_fitted(self, "n_features_")

    def _get_objective(self) -> str:
        if isinstance(self.objective, str):
            return self.objective

        if self._n_classes is None:
            return "regression"
        elif self._n_classes > 2:
            return "multiclass"
        else:
            return "binary"

    def _get_random_state(self) -> Optional[int]:
        if self.random_state is None or isinstance(self.random_state, int):
            return self.random_state

        random_state = check_random_state(self.random_state)

        return random_state.randint(0, MAX_INT)

    def _train_booster(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: Optional[ONE_DIM_ARRAYLIKE_TYPE] = None,
        callbacks: Optional[List[Callable]] = None,
        categorical_feature: Union[List[int], List[str], str] = "auto",
        feature_name: Union[List[str], str] = "auto",
    ) -> lgb.Booster:
        """Refit the estimator with the best found hyperparameters.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        sample_weight
            Weights of training data.

        callbacks
            List of callback functions that are applied at each iteration.

        categorical_feature
            Categorical features.

        feature_name
            Feature names.

        Returns
        -------
        booster
            Trained booster.
        """
        self._check_is_fitted()

        params = self.best_params_.copy()
        dataset = lgb.Dataset(X, label=y, weight=sample_weight)
        booster = lgb.train(
            params, dataset, num_boost_round=self._best_iteration
        )

        booster.free_dataset()

        return booster

    def fit(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        y: ONE_DIM_ARRAYLIKE_TYPE,
        sample_weight: Optional[ONE_DIM_ARRAYLIKE_TYPE] = None,
        groups: Optional[ONE_DIM_ARRAYLIKE_TYPE] = None,
        callbacks: Optional[List[Callable]] = None,
        categorical_feature: Union[List[int], List[str], str] = "auto",
        early_stopping_rounds: Optional[int] = 10,
        eval_metric: Optional[Union[Callable, str]] = None,
        feature_name: Union[List[str], str] = "auto",
    ) -> "_BaseOGBMModel":
        """Fit the model according to the given training data.

        Parameters
        ----------
        X
            Training data.

        y
            Target.

        sample_weight
            Weights of training data.

        groups
            Group labels for the samples used while splitting the dataset into
            train/test set.

        callbacks
            List of callback functions that are applied at each iteration.

        categorical_feature
            Categorical features.

        early_stopping_rounds
            Used to activate early stopping.

        eval_metric
            Evaluation metric.

        feature_name
            Feature names.

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
            ensure_min_samples=2,
            estimator=self,
            force_all_finite=False,
        )

        _, self._n_features = X.shape

        is_classifier = self._estimator_type == "classifier"
        cv = check_cv(self.cv, y, is_classifier)

        seed = self._get_random_state()

        params = self.get_params()

        params.pop("class_weight", None)
        params.pop("cv")
        params.pop("enable_pruning")
        params.pop("importance_type")
        params.pop("n_estimators")
        params.pop("n_trials")
        params.pop("param_distributions")
        params.pop("study")
        params.pop("timeout")

        params["random_state"] = seed
        params["verbose"] = -1

        if is_classifier:
            self.encoder_ = LabelEncoder()

            y = self.encoder_.fit_transform(y)

            self._classes = self.encoder_.classes_
            self._n_classes = len(self.encoder_.classes_)

            if self._n_classes > 2:
                params["num_classes"] = self._n_classes

        if callable(self.objective):
            fobj = _ObjectiveFunctionWrapper(self.objective)
        else:
            fobj = None

        self._objective = self._get_objective()

        params["objective"] = self._objective

        if callable(eval_metric):
            params["metric"] = "None"
            feval = _EvalFunctionWrapper(eval_metric)
            eval_name, _, is_higher_better = eval_metric(y, y)

        else:
            if eval_metric is None:
                params["metric"] = OBJECTIVE2METRIC[params["objective"]]
            else:
                params["metric"] = eval_metric

            feval = None
            eval_name = params["metric"]
            is_higher_better = _is_higher_better(params["metric"])

        if self.study is None:
            sampler = optuna.samplers.TPESampler(seed=seed)

            self.study_ = optuna.create_study(
                direction="maximize" if is_higher_better else "minimize",
                sampler=sampler,
            )

        else:
            self.study_ = self.study

        if groups is None:
            group = None
        else:
            indices = np.argsort(groups)
            X = safe_indexing(X, indices)
            y = safe_indexing(y, indices)
            sample_weight = safe_indexing(sample_weight, indices)
            groups = safe_indexing(groups, indices)
            _, group = np.unique(groups, return_counts=True)

        dataset = lgb.Dataset(X, label=y, group=group, weight=sample_weight)

        objective = _Objective(
            params,
            dataset,
            eval_name,
            is_higher_better,
            callbacks=callbacks,
            categorical_feature=categorical_feature,
            cv=cv,
            early_stopping_rounds=early_stopping_rounds,
            enable_pruning=self.enable_pruning,
            feature_name=feature_name,
            feval=feval,
            fobj=fobj,
            n_estimators=self.n_estimators,
            param_distributions=self.param_distributions,
        )

        self.study_.optimize(
            objective, catch=(), n_trials=self.n_trials, timeout=self.timeout
        )

        self.best_params_ = {**params, **self.study_.best_params}
        self._best_iteration = self.study_.user_attrs["best_iteration"]
        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)

        logger = logging.getLogger(__name__)

        logger.info(f"The best_iteration is {self._best_iteration}.")

        weights = np.array(
            [
                np.sum(sample_weight[train])
                for train, _ in cv.split(X, y, groups=groups)
            ]
        )

        self._Booster: Union[lgb.Booster, _VotingBooster]

        if self.refit:
            self._Booster = self._train_booster(
                X,
                y,
                sample_weight=sample_weight,
                callbacks=callbacks,
                categorical_feature=categorical_feature,
                feature_name=feature_name,
            )
        else:
            self._Booster = _VotingBooster.from_representations(
                self.study_.user_attrs["representations"], weights=weights
            )

        return self


class OGBMClassifier(_BaseOGBMModel, ClassifierMixin):
    """OptGBM classifier.

    Parameters
    ----------
    boosting_type
        Boosting type.

    num_leaves
        Maximum tree leaves for base learners.

    max_depth
        Maximum depth of each tree.

    learning_rate
        Learning rate.

    n_estimators
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boost_round`.

    subsample_for_bin
        Number of samples for constructing bins.

    objective
        Learning objective.

    class_weight
        Weights associated with classes.

    min_split_gain
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree.

    min_child_weight
        Minimum sum of instance weight (hessian) needed in a child (leaf).

    min_child_samples
        Minimum number of data needed in a child (leaf).

    subsample
        Subsample ratio of the training instance.

    subsample_freq
        Frequence of subsample.

    colsample_bytree
        Subsample ratio of columns when constructing each tree.

    reg_alpha
        L1 regularization term on weights.

    reg_lambda
        L2 regularization term on weights.

    random_state
        Seed of the pseudo random number generator.

    n_jobs
        Number of parallel jobs.

    importance_type
        Type of feature importances.

    cv
        Cross-validation strategy.

    enable_pruning
        Used to activate pruning.

    n_trials
        Number of trials.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    refit
        If True, refit the estimator with the best found hyperparameters.

    study
        Study that corresponds to the optimization task.

    timeout
        Time limit in seconds for the search of appropriate models.

    **kwargs
        Other parameters for the model.

    Attributes
    ----------
    best_iteration_
        Number of iterations as selected by early stopping.

    best_params_
        Parameters of the best trial in the `Study`.

    booster_
        Trained booster.

    encoder_
        Label encoder.

    n_features_
        Number of features of fitted model.

    n_splits_:
        Number of cross-validation splits.

    study_
        Actual study.

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
        """Get the class labels."""
        self._check_is_fitted()

        return self._classes

    @property
    def n_classes_(self) -> int:
        """Get the number of classes."""
        return self._n_classes

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the fitted model.

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
        self, X: TWO_DIM_ARRAYLIKE_TYPE
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
            X, accept_sparse=True, estimator=self, force_all_finite=False
        )
        preds = self._Booster.predict(X)

        if self._n_classes > 2:
            return preds

        else:
            preds = preds.reshape(-1, 1)

            return np.concatenate([1.0 - preds, preds], axis=1)


class OGBMRegressor(_BaseOGBMModel, RegressorMixin):
    """OptGBM regressor.

    Parameters
    ----------
    boosting_type
        Boosting type.

    num_leaves
        Maximum tree leaves for base learners.

    max_depth
        Maximum depth of each tree.

    learning_rate
        Learning rate.

    n_estimators
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boost_round`.

    subsample_for_bin
        Number of samples for constructing bins.

    objective
        Learning objective.

    min_split_gain
        Minimum loss reduction required to make a further partition on a leaf
        node of the tree.

    min_child_weight
        Minimum sum of instance weight (hessian) needed in a child (leaf).

    min_child_samples
        Minimum number of data needed in a child (leaf).

    subsample
        Subsample ratio of the training instance.

    subsample_freq
        Frequence of subsample.

    colsample_bytree
        Subsample ratio of columns when constructing each tree.

    reg_alpha
        L1 regularization term on weights.

    reg_lambda
        L2 regularization term on weights.

    random_state
        Seed of the pseudo random number generator.

    n_jobs
        Number of parallel jobs.

    importance_type
        Type of feature importances.

    cv
        Cross-validation strategy.

    enable_pruning
        Used to activate pruning.

    n_trials
        Number of trials.

    param_distributions
        Dictionary where keys are parameters and values are distributions.

    refit
        If True, refit the estimator with the best found hyperparameters.

    study
        Study that corresponds to the optimization task.

    timeout
        Time limit in seconds for the search of appropriate models.

    **kwargs
        Other parameters for the model.

    Attributes
    ----------
    best_iteration_
        Number of iterations as selected by early stopping.

    best_params_
        Parameters of the best trial in the `Study`.

    booster_
        Trained booster.

    n_features_
        Number of features of fitted model.

    n_splits_:
        Number of cross-validation splits.

    study_
        Actual study.

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
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 1_000,
        subsample_for_bin: int = 200_000,
        objective: Optional[Union[Callable, str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-03,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[RANDOM_STATE_TYPE] = None,
        n_jobs: int = 1,
        importance_type: str = "split",
        cv: Union[BaseCrossValidator, int] = 5,
        enable_pruning: bool = False,
        n_trials: int = 10,
        param_distributions: Optional[
            Dict[str, optuna.distributions.BaseDistribution]
        ] = None,
        refit: bool = False,
        study: Optional[optuna.study.Study] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            boosting_type=boosting_type,
            colsample_bytree=colsample_bytree,
            cv=cv,
            enable_pruning=enable_pruning,
            importance_type=importance_type,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            min_child_weight=min_child_weight,
            min_split_gain=min_split_gain,
            num_leaves=num_leaves,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            n_trials=n_trials,
            objective=objective,
            param_distributions=param_distributions,
            random_state=random_state,
            refit=refit,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            study=study,
            subsample=subsample,
            subsample_freq=subsample_freq,
            subsample_for_bin=subsample_for_bin,
            timeout=timeout,
            **kwargs,
        )

    def predict(self, X: TWO_DIM_ARRAYLIKE_TYPE) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the fitted model.

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
            X, accept_sparse=True, estimator=self, force_all_finite=False
        )

        return self._Booster.predict(X)
