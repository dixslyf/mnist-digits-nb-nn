# naive_bayes.py
# Copyright (c) 2024 Sirui Li (sirui.li@murdoch.edu.au), Kevin Wong (K.Wong@murdoch.edu.au)
#                    and Dixon Sean Low Yan Feng (35170945@student.murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import numpy as np


class BernoulliNaiveBayes:
    def __init__(self, smoothing_factor=1.0):
        """
        Args:
        smoothing_factor: Laplace smoothing factor to handle zero probabilities.
        """
        self.smoothing_factor = smoothing_factor

        # Prior probabilities for each class.
        self.class_probs = None

        # Conditional probabilities for each feature given the class.
        self.feature_probs = None

    def calculate_class_probs(self, y_train):
        """
        Calculate the prior probabilities for each class.

        Laplace smoothing is applied to handle classes that do not appear in the training data.

        Args:
        y_train: Training labels.

        Returns:
        class_probs: Array of prior probabilities for each class.
        """
        classes, class_counts = np.unique(y_train, return_counts=True)
        num_classes = len(classes)

        # Calculate class probabilities with Laplace smoothing.
        sample_count = len(y_train)
        class_probs = np.zeros(num_classes)
        class_probs = (class_counts + self.smoothing_factor) / (
            sample_count + num_classes * self.smoothing_factor
        )

        return class_probs

    def calculate_feature_probs(self, x_train, y_train):
        """
        Calculate the conditional probabilities for each feature given the class.

        Laplace smoothing is applied to avoid zero probabilities.

        Args:
        x_train: Training features (binary).
        y_train: Training labels.

        Returns:
        feature_probs: Two-dimensional array of conditional probabilities for
                       each feature and class. Each row corresponds to a class
                       and each column corresponds to a feature.
        """
        num_classes = len(np.unique(y_train))
        num_samples, num_features = x_train.shape
        feature_probs = np.zeros((num_classes, num_features))

        for c in range(num_classes):
            # Select samples belonging to class `c`.
            class_samples = x_train[y_train == c]
            class_sample_count = class_samples.shape[0]

            # Calculate the probability of each feature being 1 given class `c`,
            # with Laplace smoothing. Note that there is an assumption that each
            # feature is binary (0 or 1) since this classifier is based on the
            # Bernoulli distribution.
            #
            # Since there are only two possible values, in the denominator,
            # the Laplace smoothing factor is multiplied by 2.
            feature_probs[c, :] = (
                np.sum(class_samples, axis=0) + self.smoothing_factor
            ) / (class_sample_count + 2 * self.smoothing_factor)

        return feature_probs

    def train(self, x_train, y_train):
        """
        Train the naive Bayes classifier.

        Args:
        x_train: Training features.
        y_train: Training labels.
        """
        self.class_probs = self.calculate_class_probs(y_train)
        self.feature_probs = self.calculate_feature_probs(x_train, y_train)

    def predict(self, x_test):
        """
        Predict the class labels for test samples.

        Args:
        x_test: Test features (binary).

        Returns:
        predictions: Predicted class labels for test samples.
        """
        num_samples, num_features = x_test.shape
        num_classes = len(self.class_probs)
        predictions = np.zeros(num_samples)

        # To avoid underflows with multiplication and small probabilities,
        # we calculate the log probabilities so that multiplication operations
        # are transformed into addition.

        # Calculate the log probability for each sample and class combination.
        log_probs = np.zeros((num_samples, num_classes))
        for c in range(num_classes):
            # Corresponds to the product of P(x_i | c).
            log_probs[:, c] = np.sum(
                # Here is where the Bernoulli distribution comes in.
                np.log(self.feature_probs[c]) * x_test
                + np.log(1 - self.feature_probs[c]) * (1 - x_test),
                axis=1,
            )

            # Corresponds to multiplying by P(c).
            log_probs[:, c] += np.log(self.class_probs[c])

        # Predict the class with the highest log probability.
        # We don't need to "unlog" the log probabilities since
        # logarithms are monotonic and preserve the order of the
        # probabilities.
        predictions = np.argmax(log_probs, axis=1)

        return predictions
