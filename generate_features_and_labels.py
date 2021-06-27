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
    data = load_data('train_and_test', use_all_samples=True)
    with open('data/processed_data/data_5min_000300.npy', 'rb') as f:
        data_index = np.load(f)
    pdb.set_trace()
    features = alpha158(data, data_index, args.interval)
    labels = ret1d(data)
