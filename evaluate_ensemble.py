"""
evaluate ensemble. this script is a bit hard-coded, please do not use.
"""

import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch import optim as optim
from torch import nn as nn
import pdb
import argparse
from model import *
from features import *
from dataset import *
import pickle
from train import *
from ic import *
import ufuncs as f
import pandas as pd


if __name__ == "__main__":

    torch.backends.cudnn.enabled = False  # causes bug, may be due to server cudnn version
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.seterr(divide='ignore')  # disable division by zero warnings

    # hyperparamters
    step_len = 20
    epochs = 1000  # we use early stopping

    dataloader_train, dataloader_valid, dataloader_test = prepare_data(158, 48, step_len, "False")
    models = []
    models.append(prepare_model("GAT", 158, 1, 1, 10, device, "./pth/GAT_1_alpha158"))  # hard-code the best val loss epoch
    models.append(prepare_model("GAT", 158, 1, 1, 6, device, "./pth/GAT_6_alpha158"))
    models.append(prepare_model("GAT", 158, 1, 1, 12, device, "./pth/GAT_48_alpha158"))
    models.append(prepare_model("GAT", 158, 1, 1, 25, device, "./pth/GAT_240_alpha158"))

    # other hyperparameters
    criterion = nn.MSELoss()

    # evaluate and calculate IC
    pdb.set_trace()
    print('validation set performance:')
    evaluate_model(models, "", dataloader_valid, criterion, device, save=False)
    print('test set performance:')
    ic_total = evaluate_model(models, "", dataloader_test, criterion, device, save=False)
    pdb.set_trace()
    pd.DataFrame(ic_total).to_csv('./pth/ICs_by_month_ensemble.csv')
