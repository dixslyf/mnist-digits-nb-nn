# naive_bayes.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import numpy as np


class NaiveBayes:
    def __init__(self, smoothing_factor=1.0):
        """
        Args:
        1. smoothing_factor: Laplace smoothing factor to handle zero probabilities.
        2. class_probs: the prior probabilities for each class.
        3. feature_probs: the conditional probabilities for each feature given the class.
        """
        self.class_probs = None
        self.feature_probs = None
        self.smoothing_factor = smoothing_factor

    def calculate_class_probs(self, y_train):
        """
        Calculate the prior probabilities for each class.

        Args:
        y_train: Training labels.

        Returns:
        class_probs: Array of prior probabilities for each class.
        """
        num_classes = len(np.unique(y_train))
        class_probs = np.zeros(num_classes)
        "*** YOUR CODE HERE ***"

        return class_probs

    def calculate_feature_probs(self, x_train, y_train):
        """
        Calculate the conditional probabilities for each feature given the class.

        Args:
        x_train: Training features.
        y_train: Training labels.

        Returns:
        feature_probs: Array of conditional probabilities for each feature and class.
        """
        num_classes = len(np.unique(y_train))
        _, num_features = x_train.shape
        feature_probs = None  # need to change the intialize
        "*** YOUR CODE HERE ***"

        return feature_probs

    def train(self, x_train, y_train):
        """
        Train the NaiveBayes classifier. Do not modify this method.

        Args:
        x_train: Training features.
        y_train: Training labels.
        """
        self.class_probs = self.calculate_class_probs(y_train)
        self.feature_probs = self.calculate_feature_probs(x_train, y_train)

    def predict(self, x_test):
        """
        Predict the class labels for test sample.

        Args:
        x_test: Test features.

        Returns:
        predictions: Predicted class labels for test features.
        """
        num_samples, num_features = x_test.shape
        num_classes = len(self.class_probs)
        predictions = np.zeros(num_samples)
        "*** YOUR CODE HERE ***"

        return predictions
