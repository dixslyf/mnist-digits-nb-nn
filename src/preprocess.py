import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu, try_all_threshold


def preprocess(X):
    return binarize(X).reshape((X.shape[0], 28 * 28))


def binarize(X):
    threshold = threshold_otsu(X)
    return X > threshold


if __name__ == "__main__":
    # Try out each threshold algorithm.
    x_train = np.load(os.path.join("./data", "digitdata", "x_train.npy"))
    sample_idx = 400
    fig, ax = try_all_threshold(x_train[sample_idx, :], figsize=(10, 8), verbose=False)
    plt.show()
