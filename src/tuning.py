import argparse
from typing import Final

import numpy as np
import sys
import optuna
import skimage.filters
import sklearn.metrics
from optuna.samplers import TPESampler
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import Binarizer

from data_loaders import DIGIT_DATA_PATH, NBDataLoader
from naive_bayes import BernoulliNaiveBayesSkLearn

RAND_STATE: Final[int] = 0
NB_STUDY_NAME: Final[str] = "nb-study"
NB_STUDY_JOURNAL_PATH: Final[str] = "nb_study_journal.log"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--classifier",
        help='the classifier for which to tune or retrieve hyperparameters ("nb" or "nn")',
        choices=["nb", "nn"],
        required=True,
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        help='the data set folder path [default: "data/digitdata"]',
        type=str,
        default=DIGIT_DATA_PATH,
    )

    parser.add_argument(
        "-m",
        "--mode",
        help='the mode to run in ("tune" or "retrieve") [default: "retrieve"]',
        type=str,
        default="retrieve",
    )

    return parser.parse_args()


def tune_nb(X_train, y_train, X_val, y_val):
    # Since we will be performing K-fold cross-validation,
    # we can combine the train and validation sets.
    # In theory, using K-fold cross-validation should
    # reduce the variance of our performance estimates.
    X = np.concatenate((X_train, X_val), axis=0)
    assert X.shape[0] == (X_train.shape[0] + X_val.shape[0])

    y = np.concatenate((y_train, y_val), axis=0)
    assert y.shape == (X.shape[0],)

    def objective_nb(trial):
        smoothing_factor = trial.suggest_float("smoothing_factor", 0.001, 10.0)

        binarise_threshold_func_name = trial.suggest_categorical(
            "binarise_threshold",
            [
                "isodata",
                "li",
                "mean",
                "minimum",
                "otsu",
                "triangle",
                "yen",
            ],
        )

        binarise_threshold_func = getattr(
            skimage.filters, "threshold_{}".format(binarise_threshold_func_name)
        )

        variance_threshold_value = trial.suggest_float(
            "variance_threshold_value", 0, 10000
        )

        grouping = trial.suggest_categorical(
            "grouping", ["none", "bagging", "one-vs-one", "one-vs-rest"]
        )

        if grouping == "bagging":
            bag_n_estimators = trial.suggest_int("bag_n_estimators", 3, 10)
            bag_max_samples = trial.suggest_float("bag_max_samples", 0.1, 1.0)
            bag_max_features = trial.suggest_float("bag_max_features", 0.1, 1.0)

        scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAND_STATE)
        for train_indices, val_indices in cv.split(X, y):
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            nb = BernoulliNaiveBayesSkLearn(smoothing_factor)
            clf = (
                BaggingClassifier(
                    estimator=nb,
                    n_estimators=bag_n_estimators,
                    max_samples=bag_max_samples,
                    max_features=bag_max_features,
                    bootstrap_features=True,
                    random_state=RAND_STATE,
                )
                if grouping == "bagging"
                else OneVsOneClassifier(estimator=nb)
                if grouping == "one-vs-one"
                else OneVsRestClassifier(estimator=nb)
                if grouping == "one-vs-rest"
                else nb
            )

            pipeline = Pipeline(
                [
                    (
                        "flattening",
                        FunctionTransformer(lambda X: X.reshape((X.shape[0], -1))),
                    ),
                    ("feature_selection", VarianceThreshold(variance_threshold_value)),
                    (
                        "binarisation",
                        Binarizer(threshold=binarise_threshold_func(X_train)),
                    ),
                    (
                        "classification",
                        clf,
                    ),
                ]
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            accuracy = sklearn.metrics.accuracy_score(y_val, y_pred)
            scores.append(accuracy)
        return np.mean(scores)

    study = optuna.create_study(
        study_name=NB_STUDY_NAME,
        direction="maximize",
        sampler=TPESampler(seed=RAND_STATE),
        storage=optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(NB_STUDY_JOURNAL_PATH),
        ),
        load_if_exists=True,
    )
    study.optimize(objective_nb, n_trials=25)


def main():
    args = parse_args()

    # Load the train and validation sets.
    X_train, y_train = NBDataLoader(args.data_dir, "train").load()
    X_val, y_val = NBDataLoader(args.data_dir, "val").load()

    assert X_train.shape[1] == X_val.shape[1]

    if args.classifier == "nb":
        if args.mode == "tune":
            tune_nb(X_train, y_train, X_val, y_val)
            return 0

        assert args.mode == "retrieve"
        try:
            study = optuna.load_study(
                study_name=NB_STUDY_NAME,
                storage=optuna.storages.JournalStorage(
                    optuna.storages.JournalFileStorage(NB_STUDY_JOURNAL_PATH),
                ),
            )
        except KeyError:
            print("error: failed to retrieve tuned hyperparameters (did you perform tuning?)", file=sys.stderr)
            return 1

        print("Best trial:")
        print("  Value:", study.best_value)
        print("  Params:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")

        return 0

    print("Not yet implemented for the neural network model.")


if __name__ == "__main__":
    exit(main())
