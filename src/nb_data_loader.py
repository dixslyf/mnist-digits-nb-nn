# nb_data_loader.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import os
import numpy as np


class NBDataLoader:
    """
    Do not modify this file.
    Please note that this dataloader is NOT based on "torch.utils.data import Dataset"
    """

    def __init__(self, data_dir):
        self.x_train = np.load(os.path.join("./data", data_dir, "x_train.npy"))
        self.y_train = np.load(os.path.join("./data", data_dir, "y_train.npy"))
        self.x_val = np.load(os.path.join("./data", data_dir, "x_val.npy"))
        self.y_val = np.load(os.path.join("./data", data_dir, "y_val.npy"))
        self.x_test = np.load(os.path.join("./data", data_dir, "x_test.npy"))
        self.y_test = np.load(os.path.join("./data", data_dir, "y_test.npy"))

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_val_data(self):
        return self.x_val, self.y_val

    def get_test_data(self):
        return self.x_test, self.y_test
