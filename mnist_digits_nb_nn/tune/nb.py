from typing import Any, Final

import joblib
import numpy as np
import optuna
import sklearn.metrics
from optuna import TrialPruned
from optuna.pruners import ThresholdPruner
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline

from mnist_digits_nb_nn.models.nb import NaiveBayesSk

DEFAULT_NB_STUDY_JOURNAL_PATH: Final[str] = "nb_journal.log"
DEFAULT_NB_STUDY_NAME: Final[str] = "nb-study"


def suggest_params(trial, X):
    trial.suggest_int("pca_n_components", 10, X.shape[1] * X.shape[2])
    trial.suggest_categorical("grouping", ["none", "one-vs-one", "one-vs-rest"])
    trial.suggest_float("smoothing_factor", 0.001, 10.0)


def reshape_X(X):
    return X.reshape((X.shape[0], -1))


def realise_params(params: dict[str, Any], random_state) -> Pipeline:
    nb = NaiveBayesSk(params["smoothing_factor"])
    clf = (
        OneVsOneClassifier(estimator=nb)
        if params["grouping"] == "one-vs-one"
        else OneVsRestClassifier(estimator=nb)
        if params["grouping"] == "one-vs-rest"
        else nb
    )

    return Pipeline(
        [
            (
                "flattening",
                FunctionTransformer(reshape_X),
            ),
            (
                "feature_extraction",
                PCA(n_components=params["pca_n_components"], random_state=random_state),
            ),
            (
                "classification",
                clf,
            ),
        ]
    )


def make_objective(study, X, y, n_folds, random_state):
    def objective(trial):
        trials = study.get_trials()

        suggest_params(trial, X)
        print(f"Trial {trial.number} parameters: {trial.params}")

        train_accuracies = []
        val_accuracies = []
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for idx, (train_indices, val_indices) in enumerate(cv.split(X, y)):
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            pipeline = realise_params(trial.params, random_state)
            pipeline.fit(X_train, y_train)

            y_train_pred = pipeline.predict(X_train)
            train_accuracy = sklearn.metrics.accuracy_score(y_train, y_train_pred)
            train_accuracies.append(train_accuracy)

            print(f"Trial {trial.number} fold {idx} train accuracy: {train_accuracy}")

            y_val_pred = pipeline.predict(X_val)
            val_accuracy = sklearn.metrics.accuracy_score(y_val, y_val_pred)
            val_accuracies.append(val_accuracy)

            print(
                f"Trial {trial.number} fold {idx} validation accuracy: {val_accuracy}"
            )

            # We only prune trials after we have at least one completed trial.
            if len(trials) > 1:
                trial.report(val_accuracy, idx)
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned due to poor accuracy")
                    raise TrialPruned

        mean_train_accuracy = np.mean(train_accuracies)
        print(f"Trial {trial.number} mean train accuracy: {mean_train_accuracy}")

        mean_val_accuracy = np.mean(val_accuracies)
        print(f"Trial {trial.number} mean validation accuracy: {mean_val_accuracy}")

        return mean_val_accuracy

    return objective


def tune_nb(
    X,
    y,
    n_trials,
    n_folds,
    n_jobs,
    random_state,
    anonymous_study: bool = False,
    journal_path=DEFAULT_NB_STUDY_JOURNAL_PATH,
    study_name=DEFAULT_NB_STUDY_NAME,
):
    direction = "maximize"
    sampler = TPESampler(seed=random_state)
    pruner = ThresholdPruner(lower=0.8)
    study = (
        optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner,
        )
        if anonymous_study
        else optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(journal_path),
            ),
            load_if_exists=True,
        )
    )

    with joblib.parallel_backend("multiprocessing"):
        study.optimize(
            make_objective(study, X, y, n_folds, random_state),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

    return study.best_trial.params
