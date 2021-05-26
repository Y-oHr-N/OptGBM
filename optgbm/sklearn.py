"""scikit-learn compatible models."""

import copy
import logging
import pathlib
import pickle
import time
from inspect import signature

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pandas as pd

from optuna import distributions
from optuna import integration
from optuna.integration._lightgbm_tuner import alias
from optuna import samplers
from optuna import study as study_module
from optuna import structs
from optuna import trial as trial_module
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state

from .compat import _EvalFunctionWrapper
from .compat import _ObjectiveFunctionWrapper
from .compat import _safe_indexing
from .typing import CVType
from .typing import LightGBMCallbackEnvType
from .typing import OneDimArrayLikeType
from .typing import RandomStateType
from .typing import TwoDimArrayLikeType
from .utils import check_cv
from .utils import check_fit_params

__all__ = [
    "LGBMModel",
    "LGBMClassifier",
    "LGBMRegressor",
    "OGBMClassifier",
    "OGBMRegressor",
]

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
    @property
    def boosters_(self) -> List[lgb.Booster]:
        return self.env_.model.boosters

    def __call__(self, env: LightGBMCallbackEnvType) -> None:
        self.env_ = env


class _Objective(object):
    def __init__(
        self,
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        eval_name: str,
        is_higher_better: bool,
        n_samples: int,
        model_dir: pathlib.Path,
        weights: np.ndarray,
        callbacks: Optional[List[Callable]] = None,
        cv: Optional[CVType] = None,
        early_stopping_rounds: Optional[int] = None,
        enable_pruning: bool = False,
        feval: Optional[Callable] = None,
        fobj: Optional[Callable] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        n_estimators: int = 100,
        param_distributions: Optional[
            Dict[str, distributions.BaseDistribution]
        ] = None,
    ) -> None:
        self.callbacks = callbacks
        self.cv = cv
        self.dataset = dataset
        self.early_stopping_rounds = early_stopping_rounds
        self.enable_pruning = enable_pruning
        self.eval_name = eval_name
        self.feval = feval
        self.fobj = fobj
        self.init_model = init_model
        self.is_higher_better = is_higher_better
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.params = params
        self.param_distributions = param_distributions
        self.model_dir = model_dir
        self.weights = weights

    def __call__(self, trial: trial_module.Trial) -> float:
        params = self._get_params(trial)  # type: Dict[str, Any]
        dataset = copy.copy(self.dataset)
        callbacks = self._get_callbacks(trial)  # type: List[Callable]
        eval_hist = lgb.cv(
            params,
            dataset,
            callbacks=callbacks,
            early_stopping_rounds=self.early_stopping_rounds,
            feval=self.feval,
            fobj=self.fobj,
            folds=self.cv,
            init_model=self.init_model,
            num_boost_round=self.n_estimators,
        )  # Dict[str, List[float]]
        values = eval_hist[
            "{}-mean".format(self.eval_name)
        ]  # type: List[float]
        best_iteration = len(values)  # type: int

        trial.set_user_attr("best_iteration", best_iteration)

        booster_path = self.model_dir / "trial_{}.pkl".format(
            trial.number
        )  # type: pathlib.Path

        boosters = callbacks[0].boosters_  # type: ignore

        for b in boosters:
            b.best_iteration = best_iteration

            b.free_dataset()

        booster = _VotingBooster(
            boosters, weights=self.weights
        )  # type: _VotingBooster

        with booster_path.open("wb") as f:
            pickle.dump(booster, f)

        return values[-1]

    def _get_callbacks(self, trial: trial_module.Trial) -> List[Callable]:
        extraction_callback = (
            _LightGBMExtractionCallback()
        )  # type: _LightGBMExtractionCallback
        callbacks = [extraction_callback]  # type: List[Callable]

        if self.enable_pruning:
            pruning_callback = integration.LightGBMPruningCallback(
                trial, self.eval_name
            )  # type: integration.LightGBMPruningCallback

            callbacks.append(pruning_callback)

        if self.callbacks is not None:
            callbacks += self.callbacks

        return callbacks

    def _get_params(self, trial: trial_module.Trial) -> Dict[str, Any]:
        params = self.params.copy()  # type: Dict[str, Any]

        if self.param_distributions is None:
            params["feature_fraction"] = trial.suggest_discrete_uniform(
                "feature_fraction", 0.1, 1.0, 0.05
            )
            params["max_depth"] = trial.suggest_int("max_depth", 1, 7)
            params["num_leaves"] = trial.suggest_int(
                "num_leaves", 2, 2 ** params["max_depth"]
            )
            # See https://github.com/Microsoft/LightGBM/issues/907
            params["min_data_in_leaf"] = trial.suggest_int(
                "min_data_in_leaf",
                1,
                max(1, int(self.n_samples / params["num_leaves"])),
            )
            params["lambda_l1"] = trial.suggest_loguniform(
                "lambda_l1", 1e-09, 10.0
            )
            params["lambda_l2"] = trial.suggest_loguniform(
                "lambda_l2", 1e-09, 10.0
            )

            if params["boosting_type"] != "goss":
                params["bagging_fraction"] = trial.suggest_discrete_uniform(
                    "bagging_fraction", 0.5, 0.95, 0.05
                )
                params["bagging_freq"] = trial.suggest_int(
                    "bagging_freq", 1, 10
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


class LGBMModel(lgb.LGBMModel):
    """Base class for models in OptGBM."""

    @property
    def best_index_(self) -> int:
        """Get the best trial's number."""
        self._check_is_fitted()

        return self.study_.best_trial.number

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Optional[Union[Callable, str]] = None,
        class_weight: Optional[Union[Dict[str, float], str]] = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 1e-03,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Optional[RandomStateType] = None,
        n_jobs: int = -1,
        silent: bool = True,
        importance_type: str = "split",
        cv: CVType = 5,
        enable_pruning: bool = False,
        n_trials: int = 40,
        param_distributions: Optional[
            Dict[str, distributions.BaseDistribution]
        ] = None,
        refit: bool = True,
        study: Optional[study_module.Study] = None,
        timeout: Optional[float] = None,
        model_dir: Union[pathlib.Path, str] = "optgbm_info",
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
            silent=silent,
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
        self.model_dir = model_dir

    def _check_is_fitted(self) -> None:
        getattr(self, "n_features_")

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

    def _get_model_dir(self) -> pathlib.Path:
        model_dir = str(self.model_dir)

        return pathlib.Path(model_dir)

    def _make_booster(
        self,
        params: Dict[str, Any],
        dataset: lgb.Dataset,
        num_boost_round: int,
        booster_path: pathlib.Path,
        fobj: Optional[Callable] = None,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
    ) -> Union[_VotingBooster, lgb.Booster]:
        if self.refit:
            booster = lgb.train(
                params,
                dataset,
                num_boost_round=num_boost_round,
                fobj=fobj,
                feature_name=feature_name,
                categorical_feature=categorical_feature,
                callbacks=callbacks,
                init_model=init_model,
            )

            booster.free_dataset()

            return booster

        with booster_path.open("rb") as f:
            booster = pickle.load(f)

        return booster

    def _make_study(self, is_higher_better: bool) -> study_module.Study:
        direction = "maximize" if is_higher_better else "minimize"

        if self.study is None:
            seed = self._get_random_state()
            sampler = samplers.TPESampler(seed=seed)

            return study_module.create_study(
                direction=direction, sampler=sampler
            )

        _direction = (
            structs.StudyDirection.MAXIMIZE
            if is_higher_better
            else structs.StudyDirection.MINIMIZE
        )

        if self.study.direction != _direction:
            raise ValueError(
                "direction of study must be '{}'.".format(direction)
            )

        return self.study

    def fit(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType,
        sample_weight: Optional[OneDimArrayLikeType] = None,
        group: Optional[OneDimArrayLikeType] = None,
        eval_metric: Optional[Union[Callable, List[str], str]] = None,
        early_stopping_rounds: Optional[int] = 10,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        groups: Optional[OneDimArrayLikeType] = None,
        optuna_callbacks: Optional[List[Callable]] = None,
        **fit_params: Any
    ) -> "LGBMModel":
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
            Evaluation metric. See
            https://lightgbm.readthedocs.io/en/latest/Parameters.html#metric.

        early_stopping_rounds
            Used to activate early stopping. The model will train until the
            validation score stops improving.

        feature_name
            Feature names. If 'auto' and data is pandas DataFrame, data columns
            names are used.

        categorical_feature
            Categorical features. If list of int, interpreted as indices. If
            list of strings, interpreted as feature names. If 'auto' and data
            is pandas DataFrame, pandas categorical columns are used. All
            values in categorical features should be less than int32 max value
            (2147483647). Large values could be memory consuming. Consider
            using consecutive integers starting from zero. All negative values
            in categorical features will be treated as missing values.

        callbacks
            List of callback functions that are applied at each iteration.

        init_model
            Filename of LightGBM model, Booster instance or LGBMModel instance
            used for continue training.

        groups
            Group labels for the samples used while splitting the dataset into
            train/test set. If `group` is not None, this parameter is ignored.

        optuna_callbacks
            List of Optuna callback functions that are invoked at the end of
            each trial.

        **fit_params
            Always ignored. This parameter exists for compatibility.

        Returns
        -------
        self
            Return self.
        """
        logger = logging.getLogger(__name__)

        X, y, sample_weight = check_fit_params(
            X,
            y,
            sample_weight=sample_weight,
            accept_sparse=True,
            ensure_min_samples=2,
            estimator=self,
            force_all_finite=False,
        )

        # See https://github.com/microsoft/LightGBM/issues/2319
        if group is None and groups is not None:
            groups, _ = pd.factorize(groups)
            indices = np.argsort(groups)
            X = _safe_indexing(X, indices)
            y = _safe_indexing(y, indices)
            sample_weight = _safe_indexing(sample_weight, indices)
            groups = _safe_indexing(groups, indices)
            _, group = np.unique(groups, return_counts=True)

        n_samples, self._n_features = X.shape  # type: Tuple[int, int]

        self._n_features_in = self._n_features

        is_classifier = self._estimator_type == "classifier"
        cv = check_cv(self.cv, y, is_classifier)

        seed = self._get_random_state()

        for key, value in fit_params.items():
            logger.warning("{}={} will be ignored.".format(key, value))

        params = self.get_params()

        alias._handling_alias_parameters(params)

        if (
            not any(
                verbose_alias in params
                for verbose_alias in ("verbose", "verbosity")
            )
            and self.silent
        ):
            params["verbose"] = -1

        for attr in (
            "class_weight",
            "cv",
            "enable_pruning",
            "importance_type",
            "n_estimators",
            "n_trials",
            "param_distributions",
            "refit",
            "silent",
            "study",
            "timeout",
            "model_dir",
        ):
            params.pop(attr, None)

        params["objective"] = self._get_objective()
        params["random_state"] = seed

        if self._n_classes is not None and self._n_classes > 2:
            params["num_classes"] = self._n_classes

        if callable(eval_metric):
            params["metric"] = "None"
            feval = _EvalFunctionWrapper(eval_metric)

            args = [p.name for p in signature(eval_metric).parameters.values()]

            if len(args) > 3:
                eval_name, _, is_higher_better = eval_metric(
                    y, y, sample_weight, group
                )
            elif len(args) > 2:
                eval_name, _, is_higher_better = eval_metric(
                    y, y, sample_weight
                )
            else:
                eval_name, _, is_higher_better = eval_metric(y, y)

        elif isinstance(eval_metric, list):
            raise ValueError("eval_metric is not allowed to be a list.")

        else:
            if eval_metric is None:
                params["metric"] = OBJECTIVE2METRIC[params["objective"]]
            else:
                params["metric"] = eval_metric

            feval = None
            eval_name = params["metric"]
            is_higher_better = _is_higher_better(params["metric"])

        fobj = (
            _ObjectiveFunctionWrapper(self.objective)
            if callable(self.objective)
            else None
        )

        init_model = (
            init_model.booster_
            if isinstance(init_model, lgb.LGBMModel)
            else init_model
        )

        self.study_ = self._make_study(is_higher_better)

        dataset = lgb.Dataset(
            X,
            label=y,
            group=group,
            weight=sample_weight,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
        )

        model_dir = self._get_model_dir()
        weights = np.array(
            [
                np.sum(sample_weight[train])
                for train, _ in cv.split(X, y, groups=groups)
            ]
        )

        model_dir.mkdir(exist_ok=True, parents=True)

        objective = _Objective(
            params,
            dataset,
            eval_name,
            is_higher_better,
            n_samples,
            model_dir,
            weights,
            callbacks=callbacks,
            cv=cv,
            early_stopping_rounds=early_stopping_rounds,
            enable_pruning=self.enable_pruning,
            feval=feval,
            fobj=fobj,
            init_model=init_model,
            n_estimators=self.n_estimators,
            param_distributions=self.param_distributions,
        )

        logger.info("Searching the best hyperparameters...")

        start_time = time.perf_counter()

        self.study_.optimize(
            objective,
            callbacks=optuna_callbacks,
            catch=(),
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

        elapsed_time = time.perf_counter() - start_time

        best_iteration = self.study_.best_trial.user_attrs["best_iteration"]

        self._best_iteration = (
            None if early_stopping_rounds is None else best_iteration
        )
        self._best_score = self.study_.best_value
        self._objective = params["objective"]
        self.best_params_ = {**params, **self.study_.best_params}
        self.n_splits_ = cv.get_n_splits(X, y, groups=groups)

        logger.info(
            "Finished hyperparemeter search! "
            "(elapsed time: {:.3f} sec.) "
            "The best_iteration is {}.".format(elapsed_time, best_iteration)
        )

        booster_path = model_dir / "trial_{}.pkl".format(
            self.study_.best_trial.number
        )

        logger.info("Making booster(s)...")

        start_time = time.perf_counter()

        self._Booster = self._make_booster(
            self.best_params_,
            dataset,
            best_iteration,
            booster_path,
            fobj=fobj,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
        )

        elapsed_time = time.perf_counter() - start_time

        logger.info(
            "Finished making booster(s)! "
            "(elapsed time: {:.3f} sec.)".format(elapsed_time)
        )

        if self.refit:
            self.refit_time_ = elapsed_time

        return self


class LGBMClassifier(LGBMModel, ClassifierMixin):
    """LightGBM classifier using Optuna.

    Parameters
    ----------
    boosting_type
        Boosting type.

        - 'dart', Dropouts meet Multiple Additive Regression Trees,
        - 'gbdt', traditional Gradient Boosting Decision Tree,
        - 'goss', Gradient-based One-Side Sampling,
        - 'rf', Random Forest.

    num_leaves
        Maximum tree leaves for base learners.

    max_depth
        Maximum depth of each tree. -1 means no limit.

    learning_rate
        Learning rate. You can use `callbacks` parameter of `fit` method to
        shrink/adapt learning rate in training using `reset_parameter`
        callback. Note, that this will ignore the `learning_rate` argument in
        training.

    n_estimators
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boost_round`.

    subsample_for_bin
        Number of samples for constructing bins.

    objective
        Objective function. See
        https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective.

    class_weight
        Weights associated with classes in the form `{class_label: weight}`.
        This parameter is used only for multi-class classification task. For
        binary classification task you may use `is_unbalance` or
        `scale_pos_weight` parameters. The 'balanced' mode uses the values of y
        to automatically adjust weights inversely proportional to class
        frequencies in the input data as
        `n_samples / (n_classes * np.bincount(y))`. If None, all classes are
        supposed to have weight one. Note, that these weights will be
        multiplied with `sample_weight` if `sample_weight` is specified.

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
        Frequence of subsample. <=0 means no enable.

    colsample_bytree
        Subsample ratio of columns when constructing each tree.

    reg_alpha
        L1 regularization term on weights.

    reg_lambda
        L2 regularization term on weights.

    random_state
        Seed of the pseudo random number generator. If int, this is the
        seed used by the random number generator. If `numpy.random.RandomState`
        object, this is the random number generator. If None, the global random
        state from `numpy.random` is used.

    n_jobs
        Number of parallel jobs. -1 means using all processors.

    silent
        If False, print messages while running boosting.

    importance_type
        Type of feature importances. If 'split', result contains numbers of
        times the feature is used in a model. If 'gain', result contains total
        gains of splits which use the feature.

    cv
        Cross-validation strategy. Possible inputs for cv are:

        - integer to specify the number of folds in a CV splitter,
        - a CV splitter,
        - an iterable yielding (train, test) splits as arrays of indices.

        If int, `sklearn.model_selection.StratifiedKFold` is used.

    enable_pruning
        If True, pruning is performed.

    n_trials
        Number of trials. If None, there is no limitation on the number of
        trials. If `timeout` is also set to None, the study continues to create
        trials until it receives a termination signal such as Ctrl+C or
        SIGTERM. This trades off runtime vs quality of the solution.

    param_distributions
        Dictionary where keys are parameters and values are distributions.
        Distributions are assumed to implement the optuna distribution
        interface. If None, `num_leaves`, `max_depth`, `min_child_samples`,
        `subsample`, `subsample_freq`, `colsample_bytree`, `reg_alpha` and
        `reg_lambda` are searched.

    refit
        If True, refit the estimator with the best found hyperparameters.

    study
        Study corresponds to the optimization task. If None, a new study is
        created.

    timeout
        Time limit in seconds for the search of appropriate models. If None,
        the study is executed without time limitation. If `n_trials` is also
        set to None, the study continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM. This trades off runtime
        vs quality of the solution.

    model_dir
        Directory for storing the files generated during training.

    **kwargs
        Other parameters for the model. See
        http://lightgbm.readthedocs.io/en/latest/Parameters.html for more
        parameters. Note, that `**kwargs` is not supported in sklearn, so it
        may cause unexpected issues.

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

    n_splits_
        Number of cross-validation splits.

    objective_
        Concrete objective used while fitting this model.

    study_
        Actual study.

    refit_time_
        Time for refitting the best estimator. This is present only if `refit`
        is set to True.

    Examples
    --------
    >>> import optgbm as lgb
    >>> from sklearn.datasets import load_iris
    >>> tmp_path = getfixture("tmp_path")  # noqa
    >>> clf = lgb.LGBMClassifier(random_state=0, model_dir=tmp_path)
    >>> X, y = load_iris(return_X_y=True)
    >>> clf.fit(X, y)
    LGBMClassifier(...)
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

    def fit(
        self,
        X: TwoDimArrayLikeType,
        y: OneDimArrayLikeType,
        sample_weight: Optional[OneDimArrayLikeType] = None,
        group: Optional[OneDimArrayLikeType] = None,
        eval_metric: Optional[Union[Callable, List[str], str]] = None,
        early_stopping_rounds: Optional[int] = 10,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[int], List[str], str] = "auto",
        callbacks: Optional[List[Callable]] = None,
        init_model: Optional[Union[lgb.Booster, lgb.LGBMModel, str]] = None,
        groups: Optional[OneDimArrayLikeType] = None,
        optuna_callbacks: Optional[List[Callable]] = None,
        **fit_params: Any
    ) -> "LGBMClassifier":
        """Docstring is inherited from the LGBMModel."""
        self.encoder_ = LabelEncoder()

        y = self.encoder_.fit_transform(y)

        self._classes = self.encoder_.classes_
        self._n_classes = len(self.encoder_.classes_)

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
            group=group,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            feature_name=feature_name,
            categorical_feature=categorical_feature,
            callbacks=callbacks,
            init_model=init_model,
            groups=groups,
            optuna_callbacks=optuna_callbacks,
            **fit_params
        )

    fit.__doc__ = LGBMModel.fit.__doc__

    def predict(
        self,
        X: TwoDimArrayLikeType,
        raw_score: bool = False,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **predict_params: Any
    ) -> np.ndarray:
        """Docstring is inherited from the LGBMModel."""
        result = self.predict_proba(
            X,
            raw_score=raw_score,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **predict_params
        )

        if raw_score or pred_leaf or pred_contrib:
            return result

        class_index = np.argmax(result, axis=1)

        return self.encoder_.inverse_transform(class_index)

    predict.__doc__ = LGBMModel.predict.__doc__

    def predict_proba(
        self,
        X: TwoDimArrayLikeType,
        raw_score: bool = False,
        num_iteration: Optional[int] = None,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        **predict_params: Any
    ) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X
            Data.

        raw_score
            If True, return raw scores.

        num_iteration
            Limit number of iterations in the prediction. If None, if the best
            iteration exists, it is used; otherwise, all trees are used. If
            <=0, all trees are used (no limits).

        pred_leaf
            If True, return leaf indices.

        pred_contrib
            If True, return feature contributions.

        **predict_params
            Ignored if refit is set to False.

        Returns
        -------
        p
            Class probabilities, raw scores, leaf indices or feature
            contributions.
        """
        result = super().predict(
            X,
            raw_score=raw_score,
            num_iteration=num_iteration,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            **predict_params
        )

        if self._n_classes > 2 or raw_score or pred_leaf or pred_contrib:
            return result

        preds = result.reshape(-1, 1)

        return np.concatenate([1.0 - preds, preds], axis=1)


class LGBMRegressor(LGBMModel, RegressorMixin):
    """LightGBM regressor using Optuna.

    Parameters
    ----------
    boosting_type
        Boosting type.

        - 'dart', Dropouts meet Multiple Additive Regression Trees,
        - 'gbdt', traditional Gradient Boosting Decision Tree,
        - 'goss', Gradient-based One-Side Sampling,
        - 'rf', Random Forest.

    num_leaves
        Maximum tree leaves for base learners.

    max_depth
        Maximum depth of each tree. -1 means no limit.

    learning_rate
        Learning rate. You can use `callbacks` parameter of `fit` method to
        shrink/adapt learning rate in training using `reset_parameter`
        callback. Note, that this will ignore the `learning_rate` argument in
        training.

    n_estimators
        Maximum number of iterations of the boosting process. a.k.a.
        `num_boost_round`.

    subsample_for_bin
        Number of samples for constructing bins.

    objective
        Objective function. See
        https://lightgbm.readthedocs.io/en/latest/Parameters.html#objective.

    class_weight
        Weights associated with classes in the form `{class_label: weight}`.
        This parameter is used only for multi-class classification task. For
        binary classification task you may use `is_unbalance` or
        `scale_pos_weight` parameters. The 'balanced' mode uses the values of y
        to automatically adjust weights inversely proportional to class
        frequencies in the input data as
        `n_samples / (n_classes * np.bincount(y))`. If None, all classes are
        supposed to have weight one. Note, that these weights will be
        multiplied with `sample_weight` if `sample_weight` is specified.

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
        Frequence of subsample. <=0 means no enable.

    colsample_bytree
        Subsample ratio of columns when constructing each tree.

    reg_alpha
        L1 regularization term on weights.

    reg_lambda
        L2 regularization term on weights.

    random_state
        Seed of the pseudo random number generator. If int, this is the
        seed used by the random number generator. If `numpy.random.RandomState`
        object, this is the random number generator. If None, the global random
        state from `numpy.random` is used.

    n_jobs
        Number of parallel jobs. -1 means using all processors.

    silent
        If False, print messages while running boosting.

    importance_type
        Type of feature importances. If 'split', result contains numbers of
        times the feature is used in a model. If 'gain', result contains total
        gains of splits which use the feature.

    cv
        Cross-validation strategy. Possible inputs for cv are:

        - integer to specify the number of folds in a CV splitter,
        - a CV splitter,
        - an iterable yielding (train, test) splits as arrays of indices.

        If int, `sklearn.model_selection.StratifiedKFold` is used.

    enable_pruning
        If True, pruning is performed.

    n_trials
        Number of trials. If None, there is no limitation on the number of
        trials. If `timeout` is also set to None, the study continues to create
        trials until it receives a termination signal such as Ctrl+C or
        SIGTERM. This trades off runtime vs quality of the solution.

    param_distributions
        Dictionary where keys are parameters and values are distributions.
        Distributions are assumed to implement the optuna distribution
        interface. If None, `num_leaves`, `max_depth`, `min_child_samples`,
        `subsample`, `subsample_freq`, `colsample_bytree`, `reg_alpha` and
        `reg_lambda` are searched.

    refit
        If True, refit the estimator with the best found hyperparameters.

    study
        Study corresponds to the optimization task. If None, a new study is
        created.

    timeout
        Time limit in seconds for the search of appropriate models. If None,
        the study is executed without time limitation. If `n_trials` is also
        set to None, the study continues to create trials until it receives a
        termination signal such as Ctrl+C or SIGTERM. This trades off runtime
        vs quality of the solution.

    model_dir
        Directory for storing the files generated during training.

    **kwargs
        Other parameters for the model. See
        http://lightgbm.readthedocs.io/en/latest/Parameters.html for more
        parameters. Note, that `**kwargs` is not supported in sklearn, so it
        may cause unexpected issues.

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

    n_splits_
        Number of cross-validation splits.

    objective_
        Concrete objective used while fitting this model.

    study_
        Actual study.

    refit_time_
        Time for refitting the best estimator. This is present only if `refit`
        is set to True.

    Examples
    --------
    >>> import optgbm as lgb
    >>> from sklearn.datasets import load_boston
    >>> tmp_path = getfixture("tmp_path")  # noqa
    >>> reg = lgb.LGBMRegressor(random_state=0, model_dir=tmp_path)
    >>> X, y = load_boston(return_X_y=True)
    >>> reg.fit(X, y)
    LGBMRegressor(...)
    >>> y_pred = reg.predict(X)
    """


# alias classes
OGBMClassifier = LGBMClassifier
OGBMRegressor = LGBMRegressor
