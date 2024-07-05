# alt_data_loader.py
# Copyright (c) 2023 Sirui Li (sirui.li@murdoch.edu.au) and Kevin Wong (K.Wong@murdoch.edu.au)
# ICT203 - Artificial Intelligence and Intelligent Agents
# Murdoch University

import torch
from torch.utils.data import Dataset
import os
import numpy as np

class ALTDataLoader(Dataset):
    """
    Do not modify this file.
    Please note that this dataloader is based on "torch.utils.data import Dataset"
    """
    def __init__(self, data_dir, mode):
        """
        Args:
        data_dir: Directory containing data files.
        mode: 'train', 'valid', or 'test' mode for loading corresponding data.
        """
        self.x = np.load(os.path.join("./data", data_dir, 'x_{}.npy'.format(mode)))
        self.y = np.load(os.path.join("./data", data_dir, 'y_{}.npy'.format(mode)))
        
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
        y (tensor): Label tensor.
        """
        return torch.tensor(self.x[idx].reshape(784), dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)