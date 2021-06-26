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
from torch import optim as optim
from torch import nn as nn
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


def train(model, epochs, dataloader_train, dataloader_valid, device, optimizer, criterion):
    """train and validation"""

    for epoch in range(epochs):
        pdb.set_trace()
        print("Epoch {}:".format(epoch), end='')
        train_loss = 0.0
        train_corrects = 0
        train_samples = 0
        print("training...\t", end='')
        model.train()

        for inputs in dataloader_train:  # e.g. torch.Size([1, 1761, 20, 359])
            inputs = inputs.squeeze()  # e.g. torch.Size([1761, 20, 359])
            features = inputs[:, :, :-1].to(device)
            labels = inputs[:, -1, -1].to(device)

            train_samples += inputs.shape[0]
            preds = model(features.float())
            # loss
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 3.0) # gradient clipping
            optimizer.step()

        print("evaluating...\t", end='')
        model.eval()
        scores = []
        losses = []
        for inputs in dataloader_valid:
            inputs = inputs.squeeze()
            features = inputs[:, :, :-1].to(device)
            labels = inputs[:, -1, -1].to(device)
            # feature[torch.isnan(feature)] = 0  # WE NEED TO EVALUTE ALL SAMPLES!??
            pred = model(features.float())
            loss = criterion(pred, labels)
            losses.append(loss.item())
        print(np.mean(losses))


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
    del data_train  # save RAM
    features_valid = features_train[-244:]; features_train = features_train[:-244] # create year 2019 as validation set
    labels_valid = labels_train[-244:]; labels_train = labels_train[:-244]

    dataset_train = dataset_gat_ts(features_train, labels_train, step_len=step_len)
    dataset_valid = dataset_gat_ts(features_valid, labels_valid, step_len=step_len)

    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=32)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=32)

    model = GATModel(d_feat=358)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    train(model, epochs, dataloader_train, dataloader_valid, device, optimizer, criterion)
