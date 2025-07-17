import json
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
from torch.utils.data import DataLoader

import mnist_digits_nb_nn.tune.nb
import mnist_digits_nb_nn.tune.nn
from mnist_digits_nb_nn.data_loaders import NBDataLoader, NumpyMnistDataset
from mnist_digits_nb_nn.models import check_model
from mnist_digits_nb_nn.train import NN_PARAMS


def load_nb(input_path):
    with open(input_path, "rb") as input_file:
        clf = pickle.load(input_file)
    print(f'Loaded naive Bayes classifier from: "{input_path}"')
    return clf


def load_nn(input_path, device):
    state_dict = torch.load(input_path, map_location=device)
    model, _, _ = mnist_digits_nb_nn.tune.nn.realise_params(NN_PARAMS)
    model.load_state_dict(state_dict)
    model.to(device)
    print(f'Loaded neural network model from: "{input_path}"')
    return model


def display_metrics(y_true, y_pred):
    report = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)
    report_json = json.dumps(report, indent=4)
    print(report_json)
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()


def test_nb(clf, X, y):
    start = time.time()
    y_pred = clf.predict(X)
    end = time.time()
    duration = end - start
    print(f"Prediction time: {duration:.2f} seconds")

    display_metrics(y, y_pred)


def test_nn(model, X, y, random_state, device):
    model.to(device)

    dataset = NumpyMnistDataset(X, y)
    batch_size = NN_PARAMS["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    start = time.time()
    model.eval()
    loss_sum = 0
    y_pred_list = []
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss_sum += torch.nn.functional.cross_entropy(
                output, target, reduction="sum"
            ).item()
            y_pred_list.extend(output.argmax(1).tolist())
    end = time.time()
    duration = end - start
    print(f"Prediction time: {duration:.2f} seconds")

    mean_loss = loss_sum / len(loader.dataset)
    print(f"Mean loss: {mean_loss}")

    y_pred = np.array(y_pred_list)
    display_metrics(y, y_pred)


def cli_entry(model, data_dir, input_path, random_state, device) -> int:
    check_model(model)

    # Load the test set.
    X, y = NBDataLoader(data_dir, "test").load()

    if model == "nb":
        clf = load_nb(input_path)
        test_nb(clf, X, y)
    else:
        torch.manual_seed(random_state)
        model = load_nn(input_path, device)
        test_nn(model, X, y, random_state, device)

    return 0
