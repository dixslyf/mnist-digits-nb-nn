import pickle
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

import assignment.tune.nb
import assignment.tune.nn
from assignment.data_loaders import NBDataLoader, NumpyMnistDataset
from assignment.models import check_model

NB_PARAMS: dict[str, Any] = {
    "pca_n_components": 67,
    "grouping": "none",
    "smoothing_factor": 0.23584673798488343,
}

NN_PARAMS: dict[str, Any] = {
    "batch_size": 64,
    "epochs": 15,
    "learning_rate": 0.845073619862434,
    "optimizer": "adadelta",
    "scheduler": "exponential",
    "exponential_lr_gamma": 0.7041331729249224,
    "activation": "relu",
    "num_conv_layers": 3,
    "conv0_out_channels": 41,
    "conv0_kernel_size": 3,
    "conv1_out_channels": 123,
    "conv1_kernel_size": 3,
    "conv2_out_channels": 119,
    "conv2_kernel_size": 4,
    "pool_kernel_size": 3,
    "conv_dropout_p": 0.27438816607290456,
    "linear_dropout_p": 0.23955112992546726,
    "num_linear_layers": 2,
    "linear0_out_features": 126,
}


def create_and_train_nb(X_train, y_train, random_state):
    print(f"Parameters: {NB_PARAMS}")
    clf = assignment.tune.nb.realise_params(NB_PARAMS, random_state)

    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    duration = end - start
    print(f"Training time: {duration:.2f} seconds")

    return clf


def create_and_train_nn(
    X_train,
    y_train,
    random_state,
    device,
    batch_size_override,
    epochs_override,
    lr_override,
):
    params = NN_PARAMS.copy()

    if batch_size_override is not None:
        params["batch_size"] = batch_size_override

    if epochs_override is not None:
        params["epochs"] = epochs_override

    if lr_override is not None:
        params["learning_rate"] = lr_override

    print(f"Parameters: {params}")

    model, optimizer, scheduler = assignment.tune.nn.realise_params(params)
    model.to(device)

    train_dataset = NumpyMnistDataset(X_train, y_train)
    batch_size = params["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    start = time.time()
    model.train()
    for epoch in range(params["epochs"]):
        train_loss_sum = 0
        images_count = 0
        for batch_idx, (images, target) in enumerate(train_loader):
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()

            output = model(images)

            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()

            optimizer.step()

            train_loss_sum += loss.item() * len(images)
            images_count += len(images)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch}",
                    f"[{(batch_idx + 1) * batch_size}/{len(train_loader.dataset)}]",
                    f"train loss: {loss.item()}",
                )

        scheduler.step()

        train_loss = train_loss_sum / images_count
        print(
            f"Epoch {epoch} mean train loss: {train_loss}",
        )

    end = time.time()
    duration = end - start
    print(f"Training time: {duration:.2f} seconds")

    return model


def save_nb(nb, output_path):
    with open(output_path, "wb") as output_file:
        pickle.dump(nb, output_file)
    print(f'Saved naive Bayes classifier to: "{output_path}"')


def save_nn(nn, output_path):
    torch.save(nn.state_dict(), output_path)
    print(f'Saved neural network model to: "{output_path}"')


def cli_entry(
    model,
    data_dir,
    output_path,
    random_state,
    device,
    batch_size_override,
    epochs_override,
    lr_override,
) -> int:
    check_model(model)

    print(f'Training model: {"naive Bayes" if model == "nb" else "neural network"}')

    # Load the train set.
    X_train, y_train = NBDataLoader(data_dir, "train").load()

    if model == "nb":
        clf = create_and_train_nb(X_train, y_train, random_state)
        save_nb(clf, output_path)
    else:
        torch.manual_seed(random_state)
        model = create_and_train_nn(
            X_train,
            y_train,
            random_state,
            device,
            batch_size_override,
            epochs_override,
            lr_override,
        )
        save_nn(model, output_path)

    return 0
