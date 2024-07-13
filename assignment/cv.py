import numpy as np
import optuna
import sklearn.metrics
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

import assignment.tune.nb
import assignment.tune.nn
from assignment.data_loaders import NBDataLoader, NumpyMnistDataset
from assignment.models import check_model


def cross_validate_nb(
    X, y, n_outer_folds, n_inner_folds, n_trials, n_jobs, random_state
):
    train_accuracies = []
    val_accuracies = []
    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=random_state
    )
    for idx, (train_indices, val_indices) in enumerate(outer_cv.split(X, y)):
        print(f"OUTER FOLD {idx}")
        print("=" * 15)

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

        # To have more diverse results, we add the index to the random state.
        best_params = assignment.tune.nb.tune_nb(
            X_train,
            y_train,
            n_trials=n_trials,
            n_folds=n_inner_folds,
            n_jobs=n_jobs,
            random_state=random_state + idx,
            anonymous_study=True,
        )
        print(f"Outer fold {idx} best parameters: {best_params}")

        # Create a classifier with the current best parameters.
        pipeline = assignment.tune.nb.realise_params(best_params, random_state)

        pipeline.fit(X_train, y_train)

        # Evaluate the classifier on the outer fold train and validation set.
        y_train_pred = pipeline.predict(X_train)
        train_accuracy = sklearn.metrics.accuracy_score(y_train_pred, y_train)
        print(f"Outer fold {idx} train accuracy: {train_accuracy}")
        train_accuracies.append(train_accuracy)

        y_val_pred = pipeline.predict(X_val)
        val_accuracy = sklearn.metrics.accuracy_score(y_val_pred, y_val)
        print(f"Outer fold {idx} validation accuracy: {val_accuracy}")
        val_accuracies.append(val_accuracy)

        print()

    # Get the means of the accuracies as our final performance estimate.
    print(f"Mean train accuracy: {np.mean(train_accuracies)}")
    print(f"Mean validation accuracy: {np.mean(val_accuracies)}")


def train_nn(
    outer_fold_idx,
    model,
    optimizer,
    scheduler,
    train_loader,
    n_epochs,
    batch_size,
    device,
):
    model.train()
    for epoch in range(n_epochs):
        train_loss_sum = 0
        images_count = 0
        for batch_idx, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(images)

            loss = torch.nn.functional.cross_entropy(output, target)
            train_loss_sum += loss.item() * len(images)
            images_count += len(images)
            loss.backward()

            optimizer.step()

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Outer fold {outer_fold_idx} epoch {epoch}",
                    f"[{(batch_idx + 1) * batch_size}/{len(train_loader.dataset)}]",
                    f"train loss: {loss.item()}",
                )

        # The `ReduceLROnPlateau` scheduler needs to see the mean train loss.
        mean_train_loss = train_loss_sum / images_count
        scheduler.step() if type(
            scheduler
        ).__name__ == "ExponentialLR" else scheduler.step(mean_train_loss)

        print(
            f"Outer fold {outer_fold_idx} epoch {epoch}",
            f"mean train loss: {loss.item()}",
        )


def evaluate_nn(model, loader, device):
    model.eval()
    loss_sum = 0
    correct = 0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss_sum += torch.nn.functional.cross_entropy(
                output, target, reduction="sum"
            ).item()
            correct += (output.argmax(1) == target).type(torch.float).sum().item()
    loss = loss_sum / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return loss, accuracy


def cross_validate_nn(
    X,
    y,
    n_outer_folds,
    n_inner_folds,
    n_trials,
    n_jobs,
    random_state,
    device,
):
    torch.manual_seed(random_state)

    train_losses = []
    train_accuracies = []

    val_losses = []
    val_accuracies = []

    outer_cv = StratifiedKFold(
        n_splits=n_outer_folds, shuffle=True, random_state=random_state
    )
    for idx, (train_indices, val_indices) in enumerate(outer_cv.split(X, y)):
        print(f"OUTER FOLD {idx}")
        print("=" * 15)

        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        train_dataset = NumpyMnistDataset(X_train, y_train)
        val_dataset = NumpyMnistDataset(X_val, y_val)

        best_params = assignment.tune.nn.tune_nn(
            X_train,
            y_train,
            n_trials=n_trials,
            n_folds=n_inner_folds,
            n_jobs=n_jobs,
            device=device,
            random_state=random_state + idx,
            anonymous_study=True,
        )
        print(f"Outer fold {idx} best parameters: {best_params}")

        batch_size = best_params["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        model, optimizer, scheduler = assignment.tune.nn.realise_params(best_params)
        model.to(device)

        # Training loop.
        train_nn(
            idx,
            model,
            optimizer,
            scheduler,
            train_loader,
            best_params["epochs"],
            batch_size,
            device,
        )

        # Evaluation loop on the train set.
        train_loss, train_accuracy = evaluate_nn(model, train_loader, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        print(f"Outer fold {idx} train loss: {train_loss}")
        print(f"Outer fold {idx} train accuracy: {train_accuracy}")

        # Evaluation loop on the validation set.
        val_loss, val_accuracy = evaluate_nn(model, val_loader, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Outer fold {idx} validation loss: {val_loss}")
        print(f"Outer fold {idx} validation accuracy: {val_accuracy}")

        print()

    print(f"Mean train loss: {np.mean(train_losses)}")
    print(f"Mean train accuracy: {np.mean(train_accuracies)}")

    print(f"Mean validation loss: {np.mean(val_losses)}")
    print(f"Mean validation accuracy: {np.mean(val_accuracies)}")


def cli_entry(
    model,
    data_dir,
    trials,
    outer_folds,
    inner_folds,
    jobs,
    random_state,
    device,
) -> int:
    check_model(model)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

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

    if model == "nb":
        return cross_validate_nb(
            X,
            y,
            n_outer_folds=outer_folds,
            n_inner_folds=inner_folds,
            n_trials=trials,
            n_jobs=jobs,
            random_state=random_state,
        )

    assert model == "nn"
    return cross_validate_nn(
        X,
        y,
        n_outer_folds=outer_folds,
        n_inner_folds=inner_folds,
        n_trials=trials,
        n_jobs=jobs,
        random_state=random_state,
        device=device,
    )
