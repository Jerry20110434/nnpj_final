import torch
from torch.utils.data import Dataset
import os
import numpy as np


class dataset_gat_ts(Dataset):
    """
    custom dataset for loading the stock data (could be varying in batch_size) for GAT_ts model
    """

    def __init__(self, data_features, data_labels):
        """

        :param data:
        """
        self.data_features = data_features
        self.data_labels = data_labels


    def __getitem__(self, index: int):
        """given index, returns tuple of (data, label)"""

        return img, target


    def __len__(self):
        """sample size"""
        return len(self.data)