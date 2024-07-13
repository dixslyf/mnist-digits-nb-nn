import math
from typing import Any, Final

import joblib
import numpy as np
import optuna
import torch
from optuna import TrialPruned
from optuna.pruners import ThresholdPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from assignment.data_loaders import NumpyMnistDataset
from assignment.models.nn import MnistNN

DEFAULT_NN_STUDY_JOURNAL_PATH: Final[str] = "nn_journal.log"
DEFAULT_NN_STUDY_NAME: Final[str] = "nn-study"


def suggest_params(trial):
    trial.suggest_categorical("batch_size", [32, 64, 128])
    trial.suggest_categorical("epochs", [5, 10, 15])
    trial.suggest_float("learning_rate", 1e-4, 1.0)

    # Choose an optimizer.
    trial.suggest_categorical("optimizer", ["adam", "adadelta", "sgd", "asgd"])

    # Choose a scheduler to decay the learning rate.
    scheduler_suggestion = trial.suggest_categorical(
        "scheduler", ["exponential", "reduce_lr_on_plateau"]
    )

    if scheduler_suggestion == "exponential":
        trial.suggest_float("exponential_lr_gamma", 0.01, 0.9),
    else:
        assert scheduler_suggestion == "reduce_lr_on_plateau"
        trial.suggest_float("reduce_lr_on_plateau_factor", 0.1, 0.5),

    # Choose an activation function for the inner layers.
    activation = trial.suggest_categorical(
        "activation", ["relu", "sigmoid", "softplus", "selu", "leaky_relu"]
    )
    if activation == "leaky_relu":
        trial.suggest_float("leaky_relu_negative_slope", 1e-3, 1e-1)

    for idx in range(trial.suggest_int("num_conv_layers", 1, 3)):
        # Choose between 32 and 128 for each convoluion layer's number of output channels.
        trial.suggest_int(f"conv{idx}_out_channels", 32, 128)

        # Choose between 3 and 5 for each convoluion layer's kernel size.
        trial.suggest_int(f"conv{idx}_kernel_size", 3, 5)

    trial.suggest_int("pool_kernel_size", 2, 3)
    trial.suggest_float("conv_dropout_p", 0.0, 0.5)
    trial.suggest_float("linear_dropout_p", 0.0, 0.5)

    # Skip the last linear layer since the number of output features
    # for the last layer is fixed to 10.
    for idx in range(trial.suggest_int("num_linear_layers", 2, 4) - 1):
        trial.suggest_int(f"linear{idx}_out_features", 48, 128)


def realise_params(
    params: dict[str, Any]
) -> tuple[
    torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler
]:
    activation_suggestion = params["activation"]
    activation_class = {
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
        "softplus": torch.nn.Softplus,
        "selu": torch.nn.SELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }[activation_suggestion]
    activation = (
        activation_class(negative_slope=params["leaky_relu_negative_slope"])
        if activation_suggestion == "leaky_relu"
        else activation_class()
    )

    conv_params = []
    for idx in range(params["num_conv_layers"]):
        out_channels = params[f"conv{idx}_out_channels"]
        kernel_size = params[f"conv{idx}_kernel_size"]
        conv_params.append((out_channels, kernel_size))

    pool_kernel_size = params["pool_kernel_size"]

    conv_dropout_p = params["conv_dropout_p"]
    linear_dropout_p = params["linear_dropout_p"]

    linear_out_features = []
    for idx in range(params["num_linear_layers"] - 1):
        out_features = params[f"linear{idx}_out_features"]
        linear_out_features.append(out_features)

    model = MnistNN(
        activation,
        conv_params,
        pool_kernel_size,
        conv_dropout_p,
        linear_dropout_p,
        linear_out_features,
    )

    optimizer_class = (
        torch.optim.Adam
        if params["optimizer"] == "adam"
        else torch.optim.Adadelta
        if params["optimizer"] == "adadelta"
        else torch.optim.SGD
        if params["optimizer"] == "sgd"
        else torch.optim.ASGD
    )
    optimizer = optimizer_class(model.parameters(), lr=params["learning_rate"])

    scheduler = (
        torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params["exponential_lr_gamma"],
        )
        if params["scheduler"] == "exponential"
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params["reduce_lr_on_plateau_factor"],
            patience=0,
            threshold=1e-3,
        )
    )

    return model, optimizer, scheduler


def train(
    study, trial, fold_idx, train_loader, model, optimizer, scheduler, n_epochs, device
):
    trials = study.get_trials()

    batch_size = trial.params["batch_size"]

    # Training loop.
    step = fold_idx * n_epochs * len(train_loader)
    model.train()
    for epoch in range(n_epochs):
        train_loss_sum = 0
        images_count = 0
        for batch_idx, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(images)

            loss = torch.nn.functional.cross_entropy(output, target)
            if math.isnan(loss.item()):
                print(f"Trial {trial.number} pruned due to NaN loss")
                raise TrialPruned

            train_loss_sum += loss.item() * len(images)
            images_count += len(images)
            loss.backward()

            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Trial {trial.number} fold {fold_idx} epoch {epoch}",
                    f"[{(batch_idx + 1) * batch_size}/{len(train_loader.dataset)}]",
                    f"train loss: {loss.item()}",
                )

            # The Optuna study uses a threshold pruner. If the loss doesn't
            # go below 1.0 after 50 steps, then we are probably wasting tune,
            # so we prune the trial. However, we want to have at least one
            # completed trial, so we only do this after we have at least one
            # trial completed.
            if len(trials) > 1 and step == 50:
                trial.report(loss.item(), step)
                if trial.should_prune():
                    print(f"Trial {trial.number} pruned due to non-convergence of loss")
                    raise TrialPruned
            step += 1

        # The `ReduceLROnPlateau` scheduler needs to see the mean train loss.
        train_loss = train_loss_sum / images_count
        scheduler.step() if trial.params[
            "scheduler"
        ] == "exponential" else scheduler.step(train_loss)

        print(
            f"Trial {trial.number} fold {fold_idx} epoch {epoch}",
            f"mean train loss: {train_loss}",
        )


def evaluate(trial, fold_idx, val_loader, model, device):
    model.eval()
    val_loss_sum = 0
    correct = 0
    with torch.no_grad():
        for images, target in val_loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            val_loss_sum += torch.nn.functional.cross_entropy(
                output, target, reduction="sum"
            ).item()
            correct += (output.argmax(1) == target).type(torch.float).sum().item()

    val_loss = val_loss_sum / len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print(
        f"Trial {trial.number} fold {fold_idx} validation loss: {val_loss}",
    )

    print(
        f"Trial {trial.number} fold {fold_idx} validation accuracy: {accuracy}",
    )

    return val_loss, accuracy


def make_objective(study, X, y, n_folds, device, rand_state):
    def objective(trial):
        suggest_params(trial)
        print(f"Trial {trial.number} parameters: {trial.params}")

        # Set up K-fold cross-validation.
        val_losses = []
        val_accuracies = []
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
        for fold_idx, (train_indices, val_indices) in enumerate(cv.split(X, y)):
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            train_dataset = NumpyMnistDataset(X_train, y_train)
            val_dataset = NumpyMnistDataset(X_val, y_val)

            batch_size = trial.params["batch_size"]
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

            model, optimizer, scheduler = realise_params(trial.params)
            model.to(device)

            train(
                study=study,
                trial=trial,
                fold_idx=fold_idx,
                train_loader=train_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                n_epochs=trial.params["epochs"],
                device=device,
            )

            val_loss, accuracy = evaluate(trial, fold_idx, val_loader, model, device)

            val_losses.append(val_loss)
            val_accuracies.append(accuracy)

        mean_val_loss = np.mean(val_losses)
        mean_val_accuracy = np.mean(val_accuracies)
        print(f"Trial {trial.number} mean validation loss: {mean_val_loss}")
        print(f"Trial {trial.number} mean validation accuracy: {mean_val_accuracy}")

        # Minimize loss.
        return val_loss

    return objective


def tune_nn(
    X,
    y,
    n_trials,
    n_folds,
    n_jobs,
    device,
    random_state,
    anonymous_study: bool = False,
    journal_path=DEFAULT_NN_STUDY_JOURNAL_PATH,
    study_name=DEFAULT_NN_STUDY_NAME,
):
    torch.manual_seed(random_state)

    sampler = TPESampler(seed=random_state)
    pruner = ThresholdPruner(upper=1.0)
    study = (
        optuna.create_study(sampler=sampler, pruner=pruner)
        if anonymous_study
        else optuna.create_study(
            study_name=study_name,
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
            make_objective(study, X, y, n_folds, device, random_state),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

    return study.best_trial.params
