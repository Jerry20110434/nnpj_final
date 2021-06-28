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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, required=True, default=0, help='')
    args = parser.parse_args()

    data = load_features_and_labels(interval=48)[:2000, ...]
    features_all, labels_all = data[..., :-1], data[..., -1]
    valid_length = int(data.shape[1] / 10)
    features_train = features_all[:, :-242][:, :-valid_length]
    labels_train = labels_all[:, :-242][:, :-valid_length]
    # extra step_len days for validation and test set because the first step_len days are discarded in the dateset
    features_valid = features_all[:, :-242][:, -valid_length - step_len:]
    labels_valid = labels_all[:, :-242][:, -valid_length - step_len:]
    features_test = features_all[:, -242 - step_len:]
    labels_test = labels_all[:, -242 - step_len:]

    dataset_test = dataset_gat_ts(features_test, labels_test, step_len=step_len, valid_threshold=30)
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=32)

    model = GATModel(d_feat=158, dropout=0.7)
    load_pth(model, "./pth", epoch=args.epoch)
    model.evaluate()
    losses = []
    for inputs in dataloader_test:  # e.g. torch.Size([1, 1761, 20, 359])
        inputs = inputs.squeeze()  # e.g. torch.Size([1761, 20, 359])
        features = inputs[:, :, :-1].to(device)
        labels = inputs[:, -1, -1].to(device)

        preds = model(features.float())
        # loss
        loss = criterion(preds.double(), labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), 3.0)  # gradient clipping
        optimizer.step()
        losses.append(loss.item())

    print(np.mean(losses), end='\t')
