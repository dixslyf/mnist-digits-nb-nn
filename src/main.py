# main.py
# Copyright (c) 2024 Sirui Li (sirui.li@murdoch.edu.au), Kevin Wong (K.Wong@murdoch.edu.au),
#                    and Dixon Sean Low Yan Feng (35170945@student.murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import argparse
import time
from functools import wraps
from typing import Callable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from data_loaders import DIGIT_DATA_PATH, NBDataLoader
from naive_bayes import NaiveBayesSk

DESCRIPTION_STRING = """
examples:
    (1) To train the naive Bayes classifier on the digit dataset:
            python main.py --c nb -d data/digitdata -m train
    (2) To train the neural network model on the digit dataset:
            python main.py -c nn -d data/digitdata -m train -b 64 -e 5 -l 0.0001
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=DESCRIPTION_STRING,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--classifier",
        help='the classifier to use ("nb" or "nn")',
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
        "-m",
        "--mode",
        help='the mode to run under ("train", "test" or "val") [default: "test"]',
        choices=["train", "test", "val"],
        default="test",
        type=str,
    )

    parser.add_argument(
        "-b",
        "--batch-size",
        help="the batch size for the neural network model",
        type=int,
    )

    parser.add_argument(
        "-e",
        "--epoch",
        help="the number of epochs for the neural network model",
        type=int,
    )

    parser.add_argument(
        "-l",
        "--learning-rate",
        help="the learning rate of the neural network model",
        type=float,
    )

    return parser.parse_args()


def timed(duration_consumer: Callable[[float], None]):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kw):
            start = time.time()
            result = f(*args, **kw)
            end = time.time()
            duration_consumer(end - start)
            return result

        return wrapper

    return decorator


@timed(lambda duration: print(f"Training time: {duration:.2f} seconds"))
def train(train_f, x_train, y_train):
    train_f(x_train, y_train)


@timed(lambda duration: print(f"Prediction time: {duration:.2f} seconds"))
def predict(predict_f, x):
    return predict_f(x)


def main():
    args = parse_args()

    print("Classifier: " + args.classifier)
    print("=" * 30)

    if args.classifier == "nb":
        x_train, y_train = NBDataLoader(args.data_dir, mode="train").load()

        pipeline = Pipeline(
            [
                (
                    "flattening",
                    FunctionTransformer(lambda X: X.reshape((X.shape[0], -1))),
                ),
                (
                    "feature_extraction",
                    PCA(n_components=80),
                ),
                (
                    "classification",
                    OneVsRestClassifier(
                        NaiveBayesSk(smoothing_factor=0.17077024697134568)
                    ),
                ),
            ]
        )

        train(lambda x_train, y_train: pipeline.fit(x_train, y_train), x_train, y_train)

        if args.mode == "test":
            x_test, y_test = NBDataLoader(args.data_dir, mode="test").load()

            print()

            y_train_pred = predict(lambda x_train: pipeline.predict(x_train), x_train)
            accuracy = np.mean(y_train_pred == y_train) * 100
            print(f"Accuracy on train set: {accuracy:.2f}%")
            print()

            y_test_pred = predict(lambda x_test: pipeline.predict(x_test), x_test)
            accuracy = np.mean(y_test_pred == y_test) * 100
            print(f"Accuracy on test set: {accuracy:.2f}%")

        return

    print("Neural network model has not yet been implemented.")


if __name__ == "__main__":
    main()
