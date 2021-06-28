"""
custom dataset. finished.
"""


import torch
from torch.utils.data import Dataset
import os
import numpy as np


class dataset_gat_ts(Dataset):
    """
    custom dataset for loading the stock data (could be varying in batch_size) for GAT_ts model
    the model accepts batches of shape [n_samples, step_len, d_feat+1], where the +1 comes from label (returns)
    """

    def __init__(self, data_features, data_labels, step_len=20, valid_threshold=30):
        """

        :param data_features:  shape [n_samples, step_len, d_feat]
        :param data_labels:  shape [n_samples, step_len]
        :param step_len:
        :param valid_threshold: number of stocks that have data in all step_len days and all features.
        """
        self.step_len = step_len
        self.valid_threshold = valid_threshold
        self.data = np.concatenate([data_features, data_labels[:, :, np.newaxis]], axis=2)
        self.valid_indices = []  # we calculate the indices with enough training samples in days [T, T+step_len]
        print('initialising dataset...')
        for i in range(self.data.shape[1] - step_len + 1):
            # we only consider batches with at least valid_threshold valid stocks
            if (np.isnan(self.data[:, i: i + step_len, :]).sum((1, 2)) == 0).sum() >= valid_threshold:
                self.valid_indices.append(i)


    def __getitem__(self, index: int):
        """
        given index, returns data batch of shape [n_samples, step_len, d_feat+1]
        given an index, return data
        """
        i = self.valid_indices[index]
        ret = self.data[:, i: i + self.step_len, :]
        mask_valid = (np.isnan(ret).sum((1, 2)) == 0)
        ret = ret[mask_valid, :, :] # remove stocks with nan data
        return ret


    def __len__(self):
        """sample size"""
        return len(self.valid_indices)
