"""scikit-learn compatible models."""

import copy
import logging
import time

from pkg_resources import parse_version
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import lightgbm as lgb
import numpy as np
import optuna
import sklearn

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

if parse_version(lgb.__version__) >= parse_version("2.3"):
    from lightgbm.sklearn import _EvalFunctionWrapper
    from lightgbm.sklearn import _ObjectiveFunctionWrapper
else:
    from lightgbm.sklearn import _eval_function_wrapper as _EvalFunctionWrapper
    from lightgbm.sklearn import (
        _objective_function_wrapper as _ObjectiveFunctionWrapper,
    )

if parse_version(sklearn.__version__) >= parse_version("0.22"):
    from sklearn.utils import _safe_indexing as safe_indexing
else:
    from sklearn.utils import safe_indexing

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


def _is_higher_better(metric: str) -> bool:
    return metric in ["auc"]


class _LightGBMExtractionCallback(object):
    def __init__(self) -> None:
        self._best_iteration = None  # type: Optional[int]
        self._boosters = None  # type: Optional[List[lgb.Booster]]

    def __call__(self, env: LightGBMCallbackEnv) -> None:
        self._best_iteration = env.iteration + 1
        self._boosters = env.model.boosters


class _Objective(object):
    def __init__(
        self,
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        eval_name: str,
        is_higher_better: bool,
        n_samples: int,
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
        self.n_samples = n_samples
        self.params = params
        self.param_distributions = param_distributions

    def __call__(self, trial: optuna.trial.Trial) -> float:
        params = self._get_params(trial)  # type: Dict[str, Any]
        dataset = copy.copy(self.dataset)
        callbacks = self._get_callbacks(trial)  # type: List[Callable]
        eval_hist = lgb.cv(
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
        )  # Dict[str, List[float]]
        value = eval_hist["{}-mean".format(self.eval_name)][-1]  # type: float
        is_best_trial = True  # type: bool

        try:
            is_best_trial = (
                value < trial.study.best_value
            ) ^ self.is_higher_better
        except ValueError:
            pass

        if is_best_trial:
            best_iteration = callbacks[0]._best_iteration  # type: ignore
            boosters = (
                callbacks[0]._boosters  # type: ignore
            )  # type: List[lgb.Booster]
            representations = []  # type: List[str]

            for b in boosters:
                b.free_dataset()
                representations.append(b.model_to_string())

            trial.study.set_user_attr("best_iteration", best_iteration)
            trial.study.set_user_attr("representations", representations)

        return value

    def _get_callbacks(self, trial: optuna.trial.Trial) -> List[Callable]:
        extraction_callback = (
            _LightGBMExtractionCallback()
        )  # type: _LightGBMExtractionCallback
        callbacks = [extraction_callback]  # type: List[Callable]

        if self.enable_pruning:
            pruning_callback = (
                optuna.integration.LightGBMPruningCallback(
                    trial, self.eval_name
                )
            )  # type: optuna.integration.LightGBMPruningCallback

            callbacks.append(pruning_callback)

        if self.callbacks is not None:
            callbacks += self.callbacks

        return callbacks

    def _get_params(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        params = self.params.copy()  # type: Dict[str, Any]

        if self.param_distributions is None:
            params["colsample_bytree"] = trial.suggest_discrete_uniform(
                "colsample_bytree", 0.1, 1.0, 0.05
            )
            params["max_depth"] = trial.suggest_int("max_depth", 1, 7)
            params["num_leaves"] = trial.suggest_int(
                "num_leaves", 2, 2 ** params["max_depth"]
            )
            params["min_child_samples"] = trial.suggest_int(
                "min_child_samples",
                1,
                max(1, int(self.n_samples / params["num_leaves"])),
            )
            params["reg_alpha"] = trial.suggest_loguniform(
                "reg_alpha", 1e-09, 10.0
            )
            params["reg_lambda"] = trial.suggest_loguniform(
                "reg_lambda", 1e-09, 10.0
            )

            if params["boosting_type"] != "goss":
                params["subsample"] = trial.suggest_discrete_uniform(
                    "subsample", 0.5, 0.95, 0.05
                )
                params["subsample_freq"] = trial.suggest_int(
                    "subsample_freq", 1, 10
                )

            return params

        for name, distribution in self.param_distributions.items():
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
        if parse_version(lgb.__version__) >= parse_version("2.3"):
            boosters = [
                lgb.Booster(model_str=model_str, silent=True)
                for model_str in representations
            ]
        else:
            boosters = [
                lgb.Booster(params={"model_str": model_str})
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
        n_estimators: int = 1000,
        subsample_for_bin: int = 200000,
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
        n_trials: int = 20,
        param_distributions: Optional[
            Dict[str, optuna.distributions.BaseDistribution]
        ] = None,
        study: Optional[optuna.study.Study] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
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
            **kwargs
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
        group: Optional[ONE_DIM_ARRAYLIKE_TYPE] = None,
        eval_metric: Optional[Union[Callable, str]] = None,
        early_stopping_rounds: Optional[int] = 10,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        groups: Optional[ONE_DIM_ARRAYLIKE_TYPE] = None,
        **fit_params: Any
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

        group
            Group data of training data.

        eval_metric
            Evaluation metric.

        early_stopping_rounds
            Used to activate early stopping.

        feature_name
            Feature names.

        categorical_feature
            Categorical features.

        callbacks
            List of callback functions that are applied at each iteration.

        groups
            Group labels for the samples used while splitting the dataset into
            train/test set. If `group` is not None, this parameter is ignored.

        **fit_params
            Always ignored, exists for compatibility.

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

        n_samples, self._n_features = X.shape

        is_classifier = self._estimator_type == "classifier"
        cv = check_cv(self.cv, y, is_classifier)

        seed = self._get_random_state()

        params = self.get_params()

        for attr in (
            "class_weight",
            "cv",
            "enable_pruning",
            "importance_type",
            "n_estimators",
            "n_trials",
            "param_distributions",
            "refit",
            "study",
            "timeout",
        ):
            params.pop(attr, None)

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

        # https://github.com/microsoft/LightGBM/issues/2319
        if group is None and groups is not None:
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
            n_samples,
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

        logger = logging.getLogger(__name__)

        logger.info("Searching the best hyperparameters...")

        self.study_.optimize(
            objective, catch=(), n_trials=self.n_trials, timeout=self.timeout
        )

        logger.info("Finished hyperparemeter search!")

        self.best_params_ = {**params, **self.study_.best_params}
        self._best_iteration = self.study_.user_attrs["best_iteration"]
        self._best_score = self.study_.best_value
        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)

        logger.info("The best_iteration is {}.".format(self._best_iteration))

        if self.refit:
            logger.info("Refitting the estimator...")

            start_time = time.perf_counter()

            self._Booster = self._train_booster(
                X,
                y,
                sample_weight=sample_weight,
                callbacks=callbacks,
                categorical_feature=categorical_feature,
                feature_name=feature_name,
            )
            self.refit_time_ = time.perf_counter() - start_time

            logger.info(
                "Finished refitting! "
                "(elapsed time: {:.3f} sec.)".format(self.refit_time_)
            )

        else:
            weights = np.array(
                [
                    np.sum(sample_weight[train])
                    for train, _ in cv.split(X, y, groups=groups)
                ]
            )

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

    best_score_
        Mean cross-validated score of the best estimator.

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

    refit_time_
        Time for refitting the best estimator.

    Examples
    --------
    >>> from optgbm.sklearn import OGBMClassifier
    >>> from sklearn.datasets import load_iris
    >>> clf = OGBMClassifier(random_state=0)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    OGBMClassifier(...)
    >>> y_pred = clf.predict(X)
    """

    @property
    def classes_(self) -> np.ndarray:
        """Get the class labels."""
        self._check_is_fitted()

        return self._classes

    @property
    def n_classes_(self) -> int:
        """Get the number of classes."""
        self._check_is_fitted()

        return self._n_classes

    def predict(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        num_iteration: Optional[int] = None,
        **predict_params: Any
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the fitted model.

        Parameters
        ----------
        X
            Data.

        num_iteration
            Limit number of iterations in the prediction.

        **predict_params
            Always ignored, exists for compatibility.

        Returns
        -------
        y_pred
            Predicted values.
        """
        probas = self.predict_proba(
            X,
            num_iteration=num_iteration,
            **predict_params
        )
        class_index = np.argmax(probas, axis=1)

        return self.encoder_.inverse_transform(class_index)

    def predict_proba(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        num_iteration: Optional[int] = None,
        **predict_params: Any
    ) -> TWO_DIM_ARRAYLIKE_TYPE:
        """Predict class probabilities for data.

        Parameters
        ----------
        X
            Data.

        num_iteration
            Limit number of iterations in the prediction.

        **predict_params
            Always ignored, exists for compatibility.

        Returns
        -------
        p
            Class probabilities of data.
        """
        self._check_is_fitted()

        X = check_X(
            X, accept_sparse=True, estimator=self, force_all_finite=False
        )
        preds = self._Booster.predict(X, num_iteration=num_iteration)

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

    best_score_
        Mean cross-validated score of the best estimator.

    booster_
        Trained booster.

    n_features_
        Number of features of fitted model.

    n_splits_:
        Number of cross-validation splits.

    study_
        Actual study.

    refit_time_
        Time for refitting the best estimator.

    Examples
    --------
    >>> from optgbm.sklearn import OGBMRegressor
    >>> from sklearn.datasets import load_boston
    >>> reg = OGBMRegressor(random_state=0)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    OGBMRegressor(...)
    >>> y_pred = reg.predict(X)
    """

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 1000,
        subsample_for_bin: int = 200000,
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
        n_trials: int = 20,
        param_distributions: Optional[
            Dict[str, optuna.distributions.BaseDistribution]
        ] = None,
        refit: bool = False,
        study: Optional[optuna.study.Study] = None,
        timeout: Optional[float] = None,
        **kwargs: Any
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
            **kwargs
        )

    def predict(
        self,
        X: TWO_DIM_ARRAYLIKE_TYPE,
        num_iteration: Optional[int] = None,
        **predict_params: Any
    ) -> ONE_DIM_ARRAYLIKE_TYPE:
        """Predict using the fitted model.

        Parameters
        ----------
        X
            Data.

        num_iteration
            Limit number of iterations in the prediction.

        **predict_params
            Always ignored, exists for compatibility.

        Returns
        -------
        y_pred
            Predicted values.
        """
        self._check_is_fitted()

        X = check_X(
            X, accept_sparse=True, estimator=self, force_all_finite=False
        )

        return self._Booster.predict(X, num_iteration=num_iteration)
