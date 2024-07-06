# main.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import argparse
import time

import numpy as np

import preprocess as pp
from naive_bayes import BernoulliNaiveBayes
from nb_data_loader import NBDataLoader

DESCRIPTION_STRING = """
examples:
    (1) To train the naive Bayes classifier on the digit dataset:
            python main.py --c nb -d digitdata -m train
    (2) To train the neural network model on the digit dataset:
            python main.py -c nn -d digitdata -m train -b 64 -e 5 -l 0.0001
"""


def load_train(data_dir):
    data_loader = NBDataLoader(data_dir)
    x_train, y_train = data_loader.get_train_data()
    return x_train, y_train


def load_test(data_dir):
    data_loader = NBDataLoader(data_dir)
    x_test, y_test = data_loader.get_test_data()
    return x_test, y_test


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
        "-d", "--data_dir", help="the dataset folder path", type=str, required=True
    )

    parser.add_argument(
        "-m",
        "--mode",
        help='the mode to run under ("train", "test" or "val")',
        choices=["train", "test", "val"],
        type=str,
        required=True,
    )

    parser.add_argument(
        "-b",
        "--batch_size",
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
        "--learning_rate",
        help="the learning rate of the neural network model",
        type=float,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("Doing classification")
    print("--------------------")
    print("classifier:\t" + args.classifier)

    if args.classifier == "nb":
        x_train, y_train = load_train(args.data_dir)
        x_train = pp.preprocess(x_train)

        start_time = time.time()
        nb = BernoulliNaiveBayes()
        nb.train(x_train, y_train)
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")

        if args.mode == "test":
            x_test, y_test = load_test(args.data_dir)
            x_test = pp.preprocess(x_test)

            y_pred = nb.predict(x_train)
            accuracy = np.mean(y_pred == y_train) * 100
            print(f"Accuracy on train set: {accuracy:.2f}%")

            y_pred = nb.predict(x_test)
            accuracy = np.mean(y_pred == y_test) * 100
            print(f"Accuracy on test set: {accuracy:.2f}%")

        return

    print("Neural network model has not yet been implemented.")


if __name__ == "__main__":
    main()
