import argparse
from typing import Final

import numpy as np
import optuna
import sklearn.metrics
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline
from torch.utils.data import DataLoader

import tuning
from data_loaders import DIGIT_DATA_PATH, NBDataLoader, NumpyMnistDataset
from naive_bayes import NaiveBayesSk

RAND_STATE: Final[int] = 0


def cross_validate_nb(X, y, n_outer_folds, n_inner_folds, n_trials, n_jobs):
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
            n_jobs=n_jobs,
            anonymous_study=True,
            rand_state=idx,
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


def cross_validate_nn(X, y, n_outer_folds, n_inner_folds, n_epochs, n_trials, n_jobs):
    torch.manual_seed(RAND_STATE)

    # Use CUDA and MPS if available.
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=RAND_STATE
    )
    for idx, (train_indices, test_indices) in enumerate(outer_cv.split(X, y)):
        print(f"Outer fold: {idx}")

        X_train = X[train_indices]
        y_train = y[train_indices]

        best_params = tuning.tune_nn(
            X_train,
            y_train,
            n_trials=n_trials,
            n_folds=n_inner_folds,
            n_epochs=n_epochs,
            n_jobs=n_jobs,
            device=device,
            anonymous_study=True,
            rand_state=idx,
        )
        print(f"Outer fold {idx} best parameters: {best_params}")

        X_test, y_test = X[test_indices], y[test_indices]

        train_dataset = NumpyMnistDataset(X_train, y_train)
        test_dataset = NumpyMnistDataset(X_test, y_test)

        batch_size = best_params["batch_size"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params["batch_size"],
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=best_params["batch_size"],
        )

        model, optimizer, scheduler = tuning.realise_nn_params(best_params)
        model.to(device)

        # Training loop.
        model.train()
        for epoch in range(n_epochs):
            train_loss_sum = 0
            images_count = 0
            for batch_idx, (images, target) in enumerate(train_loader):
                images, target = images.to(device), target.to(device)
                optimizer.zero_grad()

                output = model(images)

                loss = nn.functional.cross_entropy(output, target)
                train_loss_sum += loss.item() * len(images)
                images_count += len(images)
                loss.backward()

                optimizer.step()

                if (batch_idx + 1) % 10 == 0:
                    print(
                        f"Outer fold {idx} epoch {epoch}",
                        f"[{(batch_idx + 1) * batch_size}/{len(train_loader.dataset)}]",
                        f"train loss: {loss.item()}",
                    )

            # The `ReduceLROnPlateau` scheduler needs to see the mean train loss.
            train_loss = train_loss_sum / images_count
            scheduler.step() if best_params[
                "scheduler"
            ] == "exponential" else scheduler.step(train_loss)

            print(
                f"Outer fold {idx} epoch {epoch}",
                f"mean train loss: {loss.item()}",
            )

        model.eval()

        # Evaluation loop on the train set.
        train_loss_sum = 0
        train_correct = 0
        with torch.no_grad():
            for images, target in train_loader:
                images, target = images.to(device), target.to(device)
                output = model(images)
                train_loss_sum += nn.functional.cross_entropy(
                    output, target, reduction="sum"
                ).item()
                train_correct += (
                    (output.argmax(1) == target).type(torch.float).sum().item()
                )

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_accuracy = train_correct / len(train_loader.dataset)
        print(f"Outer fold {idx} train loss: {train_loss}")
        print(f"Outer fold {idx} train accuracy: {train_accuracy}")

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Evaluation loop on the test set.
        test_loss_sum = 0
        test_correct = 0
        with torch.no_grad():
            for images, target in test_loader:
                images, target = images.to(device), target.to(device)
                output = model(images)
                test_loss_sum += nn.functional.cross_entropy(
                    output, target, reduction="sum"
                ).item()
                test_correct += (
                    (output.argmax(1) == target).type(torch.float).sum().item()
                )

        test_loss = test_loss_sum / len(test_loader.dataset)
        test_accuracy = test_correct / len(test_loader.dataset)
        print(f"Outer fold {idx} test loss: {test_loss}")
        print(f"Outer fold {idx} test accuracy: {test_accuracy}")

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    print(f"Mean train loss: {np.mean(train_losses)}")
    print(f"Mean train accuracy: {np.mean(train_accuracies)}")

    print(f"Mean test loss: {np.mean(test_losses)}")
    print(f"Mean test accuracy: {np.mean(test_accuracies)}")


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
        "-e",
        "--epochs",
        help="the number of epochs for training the neural network"
        + "(ignored for the naive Bayes classifier) [default: 5]",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-t",
        "--trials",
        help="the number of trials for hyperparameter tuning [default: 25]",
        type=int,
        default=25,
    )

    parser.add_argument(
        "-j",
        "--jobs",
        help="the number of parallel jobs [default: 1]",
        type=int,
        default=1,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load the train and validation sets.
    X_train, y_train = NBDataLoader(args.data_dir, "train").load()
    X_val, y_val = NBDataLoader(args.data_dir, "val").load()
    assert X_train.shape[1] == X_val.shape[1]

    # Since we will be performing K-fold cross-validation,
    # we can combine the train and validation sets.
    # In theory, using K-fold cross-validation should
    # reduce the variance of our performance estimates.
    X = np.concatenate((X_train, X_val), axis=0)
    assert X.shape[0] == (X_train.shape[0] + X_val.shape[0])

    y = np.concatenate((y_train, y_val), axis=0)
    assert y.shape == (X.shape[0],)

    # To reduce the computation time, we'll use a subset of the data.
    # This should still be enough data to get a good estimate of a naive Bayes classifier.
    X, _, y, _ = train_test_split(
        X, y, train_size=0.5, stratify=y, random_state=RAND_STATE
    )

    if args.classifier == "nb":
        return cross_validate_nb(
            X,
            y,
            n_outer_folds=args.outer_folds,
            n_inner_folds=args.inner_folds,
            n_trials=args.trials,
            n_jobs=args.jobs,
        )

    assert args.classifier == "nn"
    return cross_validate_nn(
        X,
        y,
        n_outer_folds=args.outer_folds,
        n_inner_folds=args.inner_folds,
        n_epochs=args.epochs,
        n_trials=args.trials,
        n_jobs=args.jobs,
    )


if __name__ == "__main__":
    exit(main())
