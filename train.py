"""
code for training.

this file should be under data/../ (i.e. parent folder of data)
run example:
python3 train.py
"""

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import pdb
import argparse
from model import *
from features import *


def load_data(mode):
    """
    loads data.
    :param mode: 'train' or 'valid ' or 'test'.
    :return: ndarray
    """
    # find all files with name data_5min_train_pt_xxx
    part = 1
    while True:
        file_name_candidate = 'data/processed_data/data_5min_{}_pt_{}.npy'.format(mode, part)
        if os.path.isfile(file_name_candidate):
            # load
            print('loading {}...'.format(file_name_candidate))
            with open(file_name_candidate, 'rb') as f:
                if part == 1:
                    data = np.load(f)
                else:
                    data = np.concatenate([data, np.load(f)], axis=0)
            part += 1
        else:  # no more files to load.
            break
    return data


def train(model, epochs):
    pass


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--start_year', type=int, required=True, help='')
    # parser.add_argument('--end_year', type=int, required=True, help='')
    # parser.add_argument('--name', type=str, required=True, help='')  # e.g. train
    # args = parser.parse_args()
    data_train = load_data('train')
    features_train = alpha360(data_train)
    labels_train = ret1d(data_train)


    train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=num_worker, pin_memory=pin_memory)
    for train_features, train_labels in train_dataloader:
        break

    pdb.set_trace() # debug
