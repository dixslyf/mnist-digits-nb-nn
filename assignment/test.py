import pickle
import time

import sklearn.metrics
import torch
from torch.utils.data import DataLoader

import assignment.tune.nb
import assignment.tune.nn
from assignment.data_loaders import NBDataLoader, NumpyMnistDataset
from assignment.models import check_model
from assignment.train import NN_PARAMS


def load_nb(input_path):
    with open(input_path, "rb") as input_file:
        clf = pickle.load(input_file)
    print(f'Loaded naive Bayes classifier from: "{input_path}"')
    return clf


def load_nn(input_path):
    state_dict = torch.load(input_path)
    model, _, _ = assignment.tune.nn.realise_params(NN_PARAMS)
    model.load_state_dict(state_dict)
    print(f'Loaded neural network model from: "{input_path}"')
    return model


def test_nb(clf, X, y):
    start = time.time()
    y_pred = clf.predict(X)
    end = time.time()
    duration = end - start
    print(f"Prediction time: {duration:.2f} seconds")

    print(f"Accuracy: {sklearn.metrics.accuracy_score(y_pred, y)}")


def test_nn(model, X, y, random_state, device):
    model.to(device)

    dataset = NumpyMnistDataset(X, y)
    batch_size = NN_PARAMS["batch_size"]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start = time.time()
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
    end = time.time()
    duration = end - start
    print(f"Prediction time: {duration:.2f} seconds")

    loss = loss_sum / len(loader.dataset)
    accuracy = correct / len(loader.dataset)

    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")


def cli_entry(model, data_dir, input_path, random_state, device) -> int:
    check_model(model)

    # Load the test set.
    X, y = NBDataLoader(data_dir, "test").load()

    if model == "nb":
        clf = load_nb(input_path)
        test_nb(clf, X, y)
    else:
        torch.manual_seed(random_state)
        model = load_nn(input_path)
        test_nn(model, X, y, random_state, device)

    return 0
