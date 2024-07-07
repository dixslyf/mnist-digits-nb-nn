# naive_bayes.py
# Copyright (c) 2024 Sirui Li (sirui.li@murdoch.edu.au), Kevin Wong (K.Wong@murdoch.edu.au)
#                    and Dixon Sean Low Yan Feng (35170945@student.murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class BernoulliNaiveBayes:
    def __init__(self, smoothing_factor=1.0):
        """
        Args:
        smoothing_factor: Laplace smoothing factor to handle zero probabilities.
        """
        self._smoothing_factor = smoothing_factor

        # Prior probabilities for each class.
        self._class_probs = None

        # Conditional probabilities for each feature given the class.
        self._feature_probs = None

    def train(self, x_train, y_train):
        """
        Train the naive Bayes classifier.

        Args:
        x_train: Training features.
        y_train: Training labels.
        """
        self._classes = np.unique(y_train)

        self._class_probs = self._calculate_class_probs(y_train)
        self._feature_probs = self._calculate_feature_probs(x_train, y_train)

    def predict(self, x_test):
        """
        Predict the class labels for test samples.

        Args:
        x_test: Test features (binary).

        Returns:
        predictions: Predicted class labels for test samples.
        """
        # Predict the class with the highest log probability.
        # We don't need to "unlog" the log probabilities since
        # logarithms are monotonic and preserve the order of the
        # probabilities.
        prediction_indices = np.argmax(self._predict_log_probabilities(x_test), axis=1)
        predictions = self._classes[prediction_indices]
        return predictions

    def predict_probabilities(self, x_test):
        return np.exp(self._predict_log_probabilities(x_test))

    def _predict_log_probabilities(self, x_test):
        num_samples, num_features = x_test.shape
        num_classes = len(self._classes)

        # To avoid underflows with multiplication and small probabilities,
        # we calculate the log probabilities so that multiplication operations
        # are transformed into addition.

        # Calculate the log probability for each sample and class combination.
        log_probs = np.zeros((num_samples, num_classes))
        for c_idx in range(num_classes):
            # Corresponds to the product of P(x_i | c).
            log_probs[:, c_idx] = np.sum(
                # Here is where the Bernoulli distribution comes in.
                np.log(self._feature_probs[c_idx]) * x_test
                + np.log(1 - self._feature_probs[c_idx]) * (1 - x_test),
                axis=1,
            )

            # Corresponds to multiplying by P(c).
            log_probs[:, c_idx] += np.log(self._class_probs[c_idx])

        return log_probs

    def _calculate_class_probs(self, y_train):
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
        class_probs = (class_counts + self._smoothing_factor) / (
            sample_count + num_classes * self._smoothing_factor
        )

        return class_probs

    def _calculate_feature_probs(self, x_train, y_train):
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
        num_classes = len(self._classes)
        num_samples, num_features = x_train.shape
        feature_probs = np.zeros((num_classes, num_features))

        for c_idx in range(num_classes):
            # Select samples belonging to class `c`.
            class_samples = x_train[y_train == self._classes[c_idx]]
            class_sample_count = class_samples.shape[0]

            # Calculate the probability of each feature being 1 given class `c`,
            # with Laplace smoothing. Note that there is an assumption that each
            # feature is binary (0 or 1) since this classifier is based on the
            # Bernoulli distribution.
            #
            # Since there are only two possible values, in the denominator,
            # the Laplace smoothing factor is multiplied by 2.
            feature_probs[c_idx, :] = (
                np.sum(class_samples, axis=0) + self._smoothing_factor
            ) / (class_sample_count + 2 * self._smoothing_factor)

        return feature_probs


class BernoulliNaiveBayesSkLearn(BaseEstimator, ClassifierMixin):
    """
    An adapter for BernoulliNaiveBayes to fit scikit-learn's API.
    """

    def __init__(self, smoothing_factor=1.0):
        self.smoothing_factor = smoothing_factor

    def fit(self, X, y):
        # `_validate_data` is provided by `BaseEstimator`.
        X, y = self._validate_data(X, y, accept_sparse=False)
        check_classification_targets(y)

        self.nb = BernoulliNaiveBayes(self.smoothing_factor)
        self.nb.train(X, y)

        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        return self.nb.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._validate_data(X, reset=False)
        return self.nb.predict_probabilities(X)

    def get_params(self, deep=True):
        return {"smoothing_factor": self.smoothing_factor}

    def set_params(self, **parameters):
        for key, value in parameters.items():
            if key not in ("smoothing_factor"):
                valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key} for estimator {self}. "
                    f"Valid parameters are: {valid_params}."
                )
            setattr(self, key, value)
        return self

    def _get_param_names(self):
        return ["smoothing_factor"]
