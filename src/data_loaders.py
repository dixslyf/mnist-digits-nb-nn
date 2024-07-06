# data_loaders.py
# Copyright (c) 2024 Sirui Li (sirui.li@murdoch.edu.au), Kevin Wong (K.Wong@murdoch.edu.au),
#                    and Dixon Sean Low Yan Feng (35170945@student.murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import os
from typing import Final

import numpy as np
import torch
from torch.utils.data import Dataset

DATA_PATH: Final[str] = "data"
DIGIT_DATA_PATH: Final[str] = os.path.join(DATA_PATH, "digitdata")


class NBDataLoader:
    """
    A data loader that loads a dataset into Numpy arrays.
    """

    def __init__(self, data_path: str, mode: str):
        if mode not in ("train", "val", "test"):
            raise ValueError('mode must be one of "train", "val" and "test"')

        self._data_path = data_path
        self._mode = mode

    def load(self):
        """
        Returns the loaded data set as Numpy arrays.

        Returns:
            A tuple whose first element is a Numpy array of the input features (x)
            and whose second element is a Numpy array of the target feature (y).
        """
        x = np.load(os.path.join(self._data_path, "x_{}.npy".format(self._mode)))
        y = np.load(os.path.join(self._data_path, "y_{}.npy".format(self._mode)))
        return x, y


class TensorDataLoader(Dataset):
    """
    A data loader that loads a dataset into tensors.
    """

    def __init__(self, data_path: str, mode: str):
        """
        Args:
            data_path: Directory containing data files.
            mode: "train", "val", or "test" mode for loading corresponding data.
        """
        if mode not in ("train", "val", "test"):
            raise ValueError('mode must be one of "train", "val" and "test"')

        nb_loader = NBDataLoader(data_path, mode)
        self.x, self.y = nb_loader.load()

    def __len__(self):
        """
        Get the length of the dataset.
        """
        return len(self.x)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            x (tensor): Input tensor.
            y (tensor): Target tensor.
        """
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
