import argparse
from typing import Final

import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline

import tuning
from data_loaders import DIGIT_DATA_PATH, NBDataLoader
from naive_bayes import NaiveBayesSk

RAND_STATE: Final[int] = 0


def cross_validate_nb(data_dir, n_outer_folds, n_inner_folds, n_trials, n_jobs):
    # Load the train and validation sets.
    X_train, y_train = NBDataLoader(data_dir, "train").load()
    X_val, y_val = NBDataLoader(data_dir, "val").load()
    assert X_train.shape[1] == X_val.shape[1]

    # Since we will be performing K-fold cross-validation,
    # we can combine the train and validation sets.
    # In theory, using K-fold cross-validation should
    # reduce the variance of our performance estimates.
    X = np.concatenate((X_train, X_val), axis=0)
    assert X.shape[0] == (X_train.shape[0] + X_val.shape[0])

    y = np.concatenate((y_train, y_val), axis=0)
    assert y.shape == (X.shape[0],)

    # To reduce the computation time, we'll use a subset (20%) of the data.
    # This should still be enough data to get a good estimate of a naive Bayes classifier.
    X, _, y, _ = train_test_split(
        X, y, train_size=0.2, stratify=y, random_state=RAND_STATE
    )

    train_accuracies = []
    test_accuracies = []
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=RAND_STATE
    )
    for idx, (train_indices, test_indices) in enumerate(outer_cv.split(X, y)):
        print(f"Outer fold: {idx}")

        X_train = X[train_indices]
        y_train = y[train_indices]

        # To have more diverse results, we pass the index as
        # the random state for hyperparameter tuning.
        best_params = tuning.tune_nb(
            X_train,
            y_train,
            n_trials=n_trials,
            n_folds=n_inner_folds,
            anonymous_study=True,
            rand_state=idx,
            n_jobs=n_jobs,
        )
        print(f"Outer fold {idx} best parameters: {best_params}")

        # Create the classifier with the current best parameters.
        nb = NaiveBayesSk(smoothing_factor=best_params["smoothing_factor"])
        clf = (
            OneVsOneClassifier(estimator=nb)
            if best_params["grouping"] == "one-vs-one"
            else OneVsRestClassifier(estimator=nb)
            if best_params["grouping"] == "one-vs-rest"
            else nb
        )

        pipeline = Pipeline(
            [
                (
                    "flattening",
                    FunctionTransformer(lambda X: X.reshape((X.shape[0], -1))),
                ),
                (
                    "feature_extraction",
                    PCA(n_components=best_params["pca_n_components"]),
                ),
                (
                    "classification",
                    clf,
                ),
            ]
        )

        pipeline.fit(X_train, y_train)

        # Evaluate the classifier on the outer fold test set.
        y_train_pred = pipeline.predict(X_train)
        train_accuracy = sklearn.metrics.accuracy_score(y_train_pred, y_train)
        print(f"Outer fold {idx} train accuracy: {train_accuracy}")
        train_accuracies.append(train_accuracy)

        X_test = X[test_indices]
        y_test = y[test_indices]

        y_test_pred = pipeline.predict(X_test)
        test_accuracy = sklearn.metrics.accuracy_score(y_test_pred, y_test)
        print(f"Outer fold {idx} test accuracy: {test_accuracy}")
        test_accuracies.append(test_accuracy)

    # Get the means of the accuracies as our final performance estimate.
    print(f"Average train accuracy: {np.mean(train_accuracies)}")
    print(f"Average test accuracy: {np.mean(test_accuracies)}")


def cross_validate_nn(data_dir):
    pass


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--classifier",
        help='the classifier to cross-validate ("nb" or "nn")',
        choices=["nb", "nn"],
        required=True,
    )

    parser.add_argument(
        "-d",
        "--data-dir",
        help=f'the data set folder path [default: "{DIGIT_DATA_PATH}"]',
        type=str,
        default=DIGIT_DATA_PATH,
    )

    parser.add_argument(
        "-j",
        "--jobs",
        help="the number of parallel jobs [default: 1]",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-o",
        "--outer-folds",
        help="the number of outer folds [default: 5]",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-i",
        "--inner-folds",
        help="the number of inner folds [default: 5]",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-t",
        "--trials",
        help="the number of trials for hyperparameter tuning [default: 10]",
        type=int,
        default=10,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.classifier == "nb":
        return cross_validate_nb(
            args.data_dir,
            n_outer_folds=args.outer_folds,
            n_inner_folds=args.inner_folds,
            n_trials=args.trials,
            n_jobs=args.jobs,
        )

    assert args.classifier == "nn"
    return cross_validate_nn(args.data_dir)


if __name__ == "__main__":
    exit(main())
