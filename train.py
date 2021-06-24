"""
code for training.

this file should be under data/../ (i.e. parent folder of data)
run example:
python3 train.py
"""

import numpy as np
import os
import torch
import pdb
import argparse
from model import GATModel


def load_training_data():
    """load all training data"""
    # find all files with name data_5min_train_pt_xxx
    part = 1
    while True:
        file_name_candidate = 'data/processed_data/data_5min_train_pt_{}.npy'.format(part)
        if os.path.isfile(file_name_candidate):
            # load
            print('loading {}...'.format(file_name_candidate))
            with open(file_name_candidate, 'rb') as f:
                if part == 1:
                    data_train = np.load(f)
                else:
                    data_train = np.concatenate([data_train, np.load(f)], axis=0)
            part += 1
        else:  # no more files to load.
            break
    return data_train


def train(model, epochs):
    pass


if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--start_year', type=int, required=True, help='')
    # parser.add_argument('--end_year', type=int, required=True, help='')
    # parser.add_argument('--name', type=str, required=True, help='')  # e.g. train
    # args = parser.parse_args()
    train_data = load_training_data()
    pdb.set_trace() # debug
