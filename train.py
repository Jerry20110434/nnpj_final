"""
code for training. unfinished.

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
from dataset import *


def load_data(mode):
    """
    loads data.
    :param mode: 'train' or 'test'.
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


def train(model, epochs, dataloader_train, dataloader_valid, device):
    """train and validation"""
    for batch_train in dataloader_train:
        model(batch_train)



if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--start_year', type=int, required=True, help='')
    # parser.add_argument('--end_year', type=int, required=True, help='')
    # parser.add_argument('--name', type=str, required=True, help='')  # e.g. train
    # args = parser.parse_args()

    step_len = 20
    epochs = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.seterr(divide='ignore')  # disable division by zero warnings

    data_train = load_data('train')
    features_train = alpha360(data_train)
    labels_train = ret1d(data_train)
    features_valid = features_train[-244:]; features_train = features_train[:-244] # create year 2019 as validation set
    labels_valid = labels_train[-244:]; labels_train = labels_train[:-244]

    dataset_train = dataset_gat_ts(features_train, labels_train, step_len=step_len)

    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=32)
    model = GATModel()
    for batch_train in dataloader_train:
        pdb.set_trace()
        train(model, epochs, dataloader_train, dataloader_train, device)
