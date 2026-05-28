import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7

        n = len(y_true)
        loss = 0
        for i in range(0, n):
            loss += (y_true[i]*np.log(y_pred[i] + epsilon) + (1-y_true[i])*np.log(1-y_pred[i]));

        loss /= -n

        return np.round(loss, 4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        epsilon = 1e-7

        n_samples = len(y_true)
        n_classes = len(y_true[0])

        loss = 0
        for i in range(0, n_samples):
            for j in range(0, n_classes):
                loss += y_true[i][j]*np.log(y_pred[i][j]);

        loss /= -n_samples;

        return np.round(loss, 4)

