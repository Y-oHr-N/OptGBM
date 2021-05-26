import pathlib

from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import lightgbm as lgb
import numpy as np
import pytest

from optuna import structs
from optuna import study as study_module
from optuna import trial as trial_module
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from optgbm.sklearn import OGBMClassifier
from optgbm.sklearn import OGBMRegressor
from optgbm.sklearn import _VotingBooster

n_estimators = 10
n_trials = 5
random_state = 0
callback = lgb.reset_parameter(
    learning_rate=lambda iteration: 0.05 * (0.99 ** iteration)
)
early_stopping_rounds = 3


def log_likelihood(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))

    return y_pred - y_true, y_pred * (1.0 - y_pred)


def optuna_callback(
    study: study_module.Study, trial: trial_module.Trial
) -> None:
    pass


def zero_one_loss(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[str, np.number, bool]:
    return "zero_one_loss", np.mean(y_true != y_pred), False


def zero_one_loss_with_sample_weight(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray
) -> Tuple[str, np.number, bool]:
    return "zero_one_loss", np.mean(y_true != y_pred), False


def zero_one_loss_with_sample_weight_and_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray,
    group: np.ndarray,
) -> Tuple[str, np.number, bool]:
    return "zero_one_loss", np.mean(y_true != y_pred), False


def test_ogbm_classifier(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("sklearn", minversion="0.20.0")

    # from sklearn.utils.estimator_checks import check_estimator
    from sklearn.utils.estimator_checks import check_estimators_pickle
    from sklearn.utils.estimator_checks import check_set_params

    clf = OGBMClassifier(model_dir=tmp_path)
    name = clf.__class__.__name__

    # check_estimator(clf)

    check_estimators_pickle(name, clf)
    check_set_params(name, clf)


def test_ogbm_regressor(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("sklearn", minversion="0.20.0")

    # from sklearn.utils.estimator_checks import check_estimator
    from sklearn.utils.estimator_checks import check_estimators_pickle
    from sklearn.utils.estimator_checks import check_set_params

    reg = OGBMRegressor(model_dir=tmp_path)
    name = reg.__class__.__name__

    # check_estimator(reg)

    check_estimators_pickle(name, reg)
    check_set_params(name, reg)


@pytest.mark.parametrize("refit", [False, True])
@pytest.mark.parametrize(
    "early_stopping_rounds", [None, early_stopping_rounds]
)
def test_hasattr(
    tmp_path: pathlib.Path, refit: bool, early_stopping_rounds: int
) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators,
        n_trials=n_trials,
        refit=refit,
        model_dir=tmp_path,
    )

    attrs = {
        "classes_": np.ndarray,
        "best_index_": int,
        "best_iteration_": (int, type(None)),
        "best_params_": dict,
        "best_score_": float,
        "booster_": (lgb.Booster, _VotingBooster),
        "encoder_": LabelEncoder,
        "feature_importances_": np.ndarray,
        "n_classes_": int,
        "n_features_": int,
        "n_splits_": int,
        "study_": study_module.Study,
    }

    for attr in attrs:
        with pytest.raises(AttributeError):
            getattr(clf, attr)

    clf.fit(X, y, early_stopping_rounds=early_stopping_rounds)

    for attr, klass in attrs.items():
        assert isinstance(getattr(clf, attr), klass)

    if refit:
        assert hasattr(clf, "refit_time_")

        assert clf.booster_.best_iteration == 0

    else:
        assert not hasattr(clf, "refit_time_")

        boosters = clf.booster_.boosters

        for b in boosters:
            if early_stopping_rounds is None:
                assert b.best_iteration == clf.n_estimators
            else:
                assert b.best_iteration == clf.best_iteration_

    if early_stopping_rounds is None:
        assert clf.best_iteration_ is None
    else:
        assert clf.best_iteration_ > 0


@pytest.mark.parametrize("boosting_type", ["dart", "gbdt", "goss", "rf"])
@pytest.mark.parametrize("objective", [None, "binary", log_likelihood])
def test_fit_with_params(
    tmp_path: pathlib.Path,
    boosting_type: str,
    objective: Optional[Union[Callable, str]],
) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        boosting_type=boosting_type,
        n_estimators=n_estimators,
        n_trials=n_trials,
        objective=objective,
        model_dir=tmp_path,
    )

    # See https://github.com/microsoft/LightGBM/issues/2328
    if boosting_type == "rf" and callable(objective):
        with pytest.raises(lgb.basic.LightGBMError):
            clf.fit(X, y)
    else:
        clf.fit(X, y)


def test_fit_with_empty_param_distributions(tmp_path: pathlib.Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        colsample_bytree=0.1,
        n_estimators=n_estimators,
        n_trials=n_trials,
        param_distributions={},
        model_dir=tmp_path,
    )

    clf.fit(X, y)

    df = clf.study_.trials_dataframe()
    values = df["value"]

    assert values.nunique() == 1


def test_fit_with_invalid_study(tmp_path: pathlib.Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    study = study_module.create_study(direction="maximize")
    clf = OGBMClassifier(study=study, model_dir=tmp_path)

    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_fit_with_pruning(tmp_path: pathlib.Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(enable_pruning=True, model_dir=tmp_path)

    clf.fit(X, y)

    if hasattr(clf.study_, "get_trials"):
        trials = clf.study_.get_trials()
    else:
        trials = clf.study_.trials

    pruned_trials = [t for t in trials if t.state == structs.TrialState.PRUNED]

    assert len(pruned_trials) > 0


@pytest.mark.parametrize("callbacks", [None, [callback]])
@pytest.mark.parametrize(
    "eval_metric",
    [
        None,
        "auc",
        zero_one_loss,
        zero_one_loss_with_sample_weight,
        zero_one_loss_with_sample_weight_and_group,
    ],
)
@pytest.mark.parametrize("optuna_callbacks", [None, [optuna_callback]])
def test_fit_with_fit_params(
    tmp_path: pathlib.Path,
    callbacks: Optional[List[Callable]],
    eval_metric: Union[Callable, str],
    optuna_callbacks: Optional[List[Callable]],
) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators, n_trials=n_trials, model_dir=tmp_path
    )

    clf.fit(
        X,
        y,
        callbacks=callbacks,
        eval_metric=eval_metric,
        optuna_callbacks=optuna_callbacks,
    )


def test_fit_with_unused_fit_params(tmp_path: pathlib.Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators, n_trials=n_trials, model_dir=tmp_path
    )

    clf.fit(X, y, eval_set=None)


def test_fit_with_group_k_fold(tmp_path: pathlib.Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        cv=GroupKFold(5),
        n_estimators=n_estimators,
        n_trials=n_trials,
        model_dir=tmp_path,
    )

    n_samples, _ = X.shape
    groups = np.random.choice(10, size=n_samples)

    clf.fit(X, y, groups=groups)


@pytest.mark.parametrize("n_jobs", [-1, 1])
def test_fit_twice_without_study(tmp_path: pathlib.Path, n_jobs: int) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        n_trials=n_trials,
        random_state=random_state,
        model_dir=tmp_path,
    )

    clf.fit(X, y)

    df = clf.study_.trials_dataframe()
    values = df["value"]

    clf = OGBMClassifier(
        bagging_fraction=1.0,
        bagging_freq=0,
        feature_fraction=1.0,
        lambda_l1=0.0,
        lambda_l2=0.0,
        min_data_in_leaf=20,
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        n_trials=n_trials,
        random_state=random_state,
        model_dir=tmp_path,
    )

    clf.fit(X, y)

    df = clf.study_.trials_dataframe()

    np.testing.assert_array_equal(values, df["value"])


@pytest.mark.parametrize("storage", [None, "sqlite:///:memory:"])
def test_fit_twice_with_study(
    tmp_path: pathlib.Path, storage: Optional[str]
) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    study = study_module.create_study(storage=storage)
    clf = OGBMClassifier(
        n_estimators=n_estimators,
        n_trials=n_trials,
        study=study,
        model_dir=tmp_path,
    )

    clf.fit(X, y)

    assert len(study.trials) == n_trials

    clf.fit(X, y)

    assert len(study.trials) == 2 * n_trials


@pytest.mark.parametrize("num_iteration", [None, 3])
def test_predict_with_predict_params(
    tmp_path: pathlib.Path, num_iteration: Optional[int]
) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators, n_trials=n_trials, model_dir=tmp_path
    )

    clf.fit(X, y)

    y_pred = clf.predict(X, num_iteration=num_iteration)

    assert isinstance(y_pred, np.ndarray)
    assert y.shape == y_pred.shape


def test_predict_with_unused_predict_params(tmp_path: pathlib.Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators, n_trials=n_trials, model_dir=tmp_path
    )

    clf.fit(X, y)

    y_pred = clf.predict(X, pred_leaf=False)

    assert isinstance(y_pred, np.ndarray)
    assert y.shape == y_pred.shape


@pytest.mark.parametrize(
    "early_stopping_rounds", [None, early_stopping_rounds]
)
def test_refit(
    tmp_path: pathlib.Path, early_stopping_rounds: Optional[int]
) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators,
        n_trials=n_trials,
        random_state=random_state,
        refit=True,
        model_dir=tmp_path,
    )

    clf.fit(X, y, early_stopping_rounds=early_stopping_rounds)

    y_pred = clf.predict(X)

    if early_stopping_rounds is None:
        _n_estimators = n_estimators
    else:
        _n_estimators = clf.best_iteration_

    clf = lgb.LGBMClassifier(n_estimators=_n_estimators, **clf.best_params_)

    clf.fit(X, y)

    np.testing.assert_array_equal(y_pred, clf.predict(X))


def test_score(tmp_path: pathlib.Path) -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state
    )

    clf = lgb.LGBMClassifier(random_state=random_state)

    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)

    clf = OGBMClassifier(random_state=random_state, model_dir=tmp_path)

    clf.fit(X_train, y_train)

    assert score <= clf.score(X_test, y_test)


@pytest.mark.parametrize("n_jobs", [-1, 1])
def test_plot_importance(tmp_path: pathlib.Path, n_jobs: int) -> None:
    X, y = load_breast_cancer(return_X_y=True)

    clf = OGBMClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        n_trials=n_trials,
        refit=False,
        model_dir=tmp_path,
    )

    clf.fit(X, y)

    lgb.plot_importance(clf)
