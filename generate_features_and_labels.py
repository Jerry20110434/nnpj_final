"""
generate features and save them. testing.
"""

import numpy as np
import torch
import ufuncs as f
import pdb
import pandas as pd
import argparse
from features import *
from train import load_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, required=True, help='')
    args = parser.parse_args()
    data = load_data('train_and_test')
    with open('data/processed_data/data_5min_000300.npy', 'rb') as f:
        data_index = np.load(f)
    features = alpha158(data, data_index, args.interval)
    labels = ret1d(data)
    pdb.set_trace()
    # reshape to [n_samples, step_len, d_feat+1]
    ret = np.concatenate([np.moveaxis(features, -1, 0), np.swapaxes(labels, 0, 1)[:, :, np.newaxis]], axis=2)
    with open('data/processed_data/data_features_and_labels_interval_{}.npy'.format(args.interval), 'wb') as f:
        np.save(f, ret)
