import argparse
import sys
from typing import Final

import numpy as np
import optuna
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim
import torch.optim.lr_scheduler
from optuna import TrialPruned
from optuna.pruners import ThresholdPruner
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import FunctionTransformer, Pipeline
from torch.utils.data import DataLoader

from data_loaders import DIGIT_DATA_PATH, CustomTensorDataset, NBDataLoader
from naive_bayes import NaiveBayesSk

RAND_STATE: Final[int] = 0
NB_STUDY_NAME: Final[str] = "nb-study"
NB_STUDY_JOURNAL_PATH: Final[str] = "nb_study_journal.log"

NN_STUDY_NAME: Final[str] = "nn-study"
NN_STUDY_JOURNAL_PATH: Final[str] = "nn_study_journal.log"


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

    # To reduce the computation time, we'll use a subset (20%) of the data.
    X, _, y, _ = train_test_split(
        X, y, train_size=0.2, stratify=y, random_state=RAND_STATE
    )

    def objective_nb(trial):
        pca_n_components = trial.suggest_int(
            "pca_n_components_count", 10, X.shape[1] * X.shape[2]
        )

        grouping = trial.suggest_categorical(
            "grouping", ["none", "one-vs-one", "one-vs-rest"]
        )

        smoothing_factor = trial.suggest_float("smoothing_factor", 0.001, 10.0)

        scores = []
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RAND_STATE)
        for train_indices, val_indices in cv.split(X, y):
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            nb = NaiveBayesSk(smoothing_factor)
            clf = (
                OneVsOneClassifier(estimator=nb)
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
                    (
                        "feature_extraction",
                        PCA(n_components=pca_n_components),
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


class TuneableNeuralNetwork(nn.Module):
    def __init__(self, trial, input_dims):
        super().__init__()

        # Choose an activation function for the inner layers.
        activation_suggestion = trial.suggest_categorical(
            "activation", ["relu", "sigmoid", "softplus", "selu", "leaky_relu"]
        )
        activation_class = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "softplus": nn.Softplus,
            "selu": nn.SELU,
            "leaky_relu": nn.LeakyReLU,
        }[activation_suggestion]
        self.activation = (
            activation_class(
                negative_slope=trial.suggest_float(
                    "leaky_relu_negative_slope", 1e-3, 1e-1
                )
            )
            if activation_suggestion == "leaky_relu"
            else activation_class()
        )

        # Keeps track of the dimensions of the image after each convolution and pool.
        first_linear_dims = input_dims

        # Set up the convolution layers.
        self.conv_layers = nn.ModuleList()
        num_conv_layers = trial.suggest_int("num_conv_layers", 1, 3)
        for idx in range(num_conv_layers):
            # The first convolution layer will see the original images, which only have 1 channel.
            # Each subsequent convolution layer will see the output channels of the previous layer.
            in_channels = 1 if idx == 0 else self.conv_layers[idx - 1].out_channels

            # Choose between 32 and 128 for each convoluion layer's number of output channels.
            out_channels = trial.suggest_int(f"conv{idx}_out_channels", 32, 128)

            # Choose between 3 and 5 for each convoluion layer's kernel size.
            kernel_size = trial.suggest_int(f"conv{idx}_kernel_size", 3, 5)

            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,  # Move the kernel filter by one each time.
            )
            self.conv_layers.append(conv)

            # After the layer has been applied, the dimensions of the images get reduced.
            first_linear_dims -= kernel_size - 1

        # Choose between 2 and 3 for the pooling kernel size.
        # Usually, pooling is done between convolution layers,
        # but the dimensions of the input images are only 28 by 28,
        # which is small compared to typical images. If we pool after
        # every layer, we reduce the dimensions of the images too much
        # and either lose too much information or end up with a 0-dimension image.
        pool_kernel_size = trial.suggest_int("pool_kernel_size", 2, 3)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size)

        # After pooling, each dimension of the image gets divided by the pooling kernel size.
        first_linear_dims //= pool_kernel_size

        # The number of inputs the first linear layer will see is the
        # number of output channels of the last convolutional layer
        # multiplied by the number of pixels of the output image.
        first_linear_in_features = (
            self.conv_layers[-1].out_channels * first_linear_dims.prod()
        )

        # This shouldn't happen since we're careful with the convolutional and pooling
        # kernel sizes, but this is kept as a sanity check.
        if first_linear_in_features == 0:
            print("Encountered 0 input features for the first linear layer")
            raise TrialPruned

        # Use dropout for regularisation.
        self.conv_dropout = nn.Dropout(trial.suggest_float("conv_dropout_p", 0.0, 0.5))
        self.linear_dropout = nn.Dropout(
            trial.suggest_float("linear_dropout_p", 0.0, 0.5)
        )

        # Set up the linear layers.
        # Similar to setting up the convolutional layers.
        # However, the last layer needs to be handled separately
        # since the number of outputs must be 10.
        self.linear_layers = nn.ModuleList()
        num_linear_layers = trial.suggest_int("num_linear_layers", 2, 4)
        for idx in range(num_linear_layers - 1):
            in_features = (
                first_linear_in_features
                if idx == 0
                else self.linear_layers[idx - 1].out_features
            )

            out_features = trial.suggest_categorical(
                f"linear{idx}_out_features", [48, 64, 80, 96, 112, 128]
            )

            linear = nn.Linear(in_features=in_features, out_features=out_features)
            self.linear_layers.append(linear)

        # The last layer needs to have 10 output features (corresponding to the digits).
        self.linear_layers.append(
            nn.Linear(in_features=self.linear_layers[-1].out_features, out_features=10)
        )

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.activation(x)
            x = self.conv_dropout(x)

        x = self.pool(x)

        # The convolution layers spit out 3-dimensional data
        # (4-dimensional if you count the rows),
        # which need to be flattened before passing to the linear layers.
        x = torch.flatten(x, 1)

        for linear_layer in self.linear_layers[:-1]:
            x = linear_layer(x)
            x = self.activation(x)
            x = self.linear_dropout(x)

        x = self.linear_layers[-1](x)

        return x


def tune_nn(train_dataset, val_dataset):
    torch.manual_seed(RAND_STATE)

    # Use CUDA and MPS if available.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Use a subset of the data to save time.
    train_subset = torch.utils.data.Subset(
        train_dataset, range(int(len(train_dataset) * 0.2))
    )
    val_subset = torch.utils.data.Subset(
        val_dataset, range(int(len(val_dataset) * 0.2))
    )

    def objective_nn(trial):
        # Set up the data loaders.
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=True,
        )

        # Create model instance
        model = TuneableNeuralNetwork(trial, torch.tensor([28, 28]))

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1.0)

        # Choose an optimiser.
        optimizer_suggestion = trial.suggest_categorical(
            "optimizer", ["adam", "adadelta", "sgd", "asgd"]
        )
        optimizer_class = (
            torch.optim.Adam
            if optimizer_suggestion == "adam"
            else torch.optim.Adadelta
            if optimizer_suggestion == "adadelta"
            else torch.optim.SGD
            if optimizer_suggestion == "sgd"
            else torch.optim.ASGD
        )
        optimizer = optimizer_class(model.parameters(), lr=learning_rate)

        # Choose a scheduler to decay the learning rate.
        scheduler_suggestion = trial.suggest_categorical(
            "scheduler", ["exponential", "reduce_lr_on_plateau"]
        )
        scheduler = (
            torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=trial.suggest_float("exponential_lr_gamma", 0.01, 0.9)
            )
            if scheduler_suggestion == "exponential"
            else torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=trial.suggest_float("reduce_lr_on_plateau_factor", 0.1, 0.5),
                patience=0,
                threshold=1e-3,
            )
        )

        print(f"Trial {trial.number} params: {trial.params}")

        # Training loop.
        num_epochs = 3
        step = 0
        for epoch in range(num_epochs):
            train_loss_sum = 0
            images_count = 0
            model.train()
            for batch_idx, (images, target) in enumerate(train_loader):
                images, target = images.to(device), target.to(device)
                optimizer.zero_grad()

                output = model(images)

                loss = nn.functional.cross_entropy(output, target)
                train_loss_sum += loss.item() * len(images)
                images_count += len(images)
                loss.backward()

                optimizer.step()

                if batch_idx % 10 == 0:
                    print(
                        f"Trial {trial.number} epoch {epoch}:",
                        f"[{batch_idx * batch_size}/{len(train_loader.dataset)}]",
                        f"loss: {loss.item():.6f}",
                    )

                # The Optuna study uses a threshold pruner. If the loss doesn't
                # go below 1.0 after 50 steps, then we are probably wasting tune,
                # so we prune the trial.
                if step == 50:
                    trial.report(loss.item(), step)
                    if trial.should_prune():
                        raise TrialPruned
                step += 1

            # The `ReduceLROnPlateau` scheduler needs to see the mean train loss.
            train_loss = train_loss_sum / images_count
            scheduler.step() if scheduler_suggestion == "exponential" else scheduler.step(
                train_loss
            )

        # Evaluation loop.
        model.eval()
        val_loss_sum = 0
        correct = 0
        with torch.no_grad():
            for images, target in val_loader:
                images, target = images.to(device), target.to(device)
                output = model(images)
                val_loss_sum += nn.functional.cross_entropy(
                    output, target, reduction="sum"
                ).item()
                correct += (output.argmax(1) == target).type(torch.float).sum().item()

        val_loss = val_loss_sum / len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)

        print(
            f"Trial {trial.number} validation summary:\n ",
            f"average loss: {val_loss:.4f}\n ",
            f"accuracy: {accuracy}\n",
        )

        return val_loss

    study = optuna.create_study(
        study_name=NN_STUDY_NAME,
        sampler=TPESampler(seed=RAND_STATE),
        pruner=ThresholdPruner(upper=1.0),
        storage=optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(NN_STUDY_JOURNAL_PATH),
        ),
        load_if_exists=True,
    )
    study.optimize(objective_nn, n_trials=25)


def retrieve_best_trial(study_name, study_journal_path):
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=optuna.storages.JournalStorage(
                optuna.storages.JournalFileStorage(study_journal_path),
            ),
        )
    except KeyError:
        print(
            "error: failed to retrieve tuned hyperparameters (did you perform tuning?)",
            file=sys.stderr,
        )
        return 1

    print("Best trial:")
    print("  Value:", study.best_value)
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    return 0


def main():
    args = parse_args()

    if args.classifier == "nb":
        # Load the train and validation sets.
        X_train, y_train = NBDataLoader(args.data_dir, "train").load()
        X_val, y_val = NBDataLoader(args.data_dir, "val").load()
        assert X_train.shape[1] == X_val.shape[1]

        if args.mode == "tune":
            tune_nb(X_train, y_train, X_val, y_val)
            return 0

        assert args.mode == "retrieve"
        return retrieve_best_trial(NN_STUDY_NAME, NN_STUDY_JOURNAL_PATH)

    # Neural network model.
    assert args.classifier == "nn"

    # Load the train and validation sets.
    train_dataset = CustomTensorDataset(args.data_dir, "train")
    val_dataset = CustomTensorDataset(args.data_dir, "val")
    if args.mode == "tune":
        tune_nn(train_dataset, val_dataset)
        return 0

    assert args.mode == "retrieve"
    return retrieve_best_trial(NN_STUDY_NAME, NN_STUDY_JOURNAL_PATH)


if __name__ == "__main__":
    exit(main())
