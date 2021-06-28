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
import pickle


torch.backends.cudnn.enabled = False  # causes bug, may be due to server cudnn version


def save_pth(model,path,epoch):
    pickle.dump(model.state_dict(),open(os.path.join(path,"%d.pth"%epoch),"wb"))


def load_pth(model,path,epoch=-1):
    if epoch==0:
        return
    if epoch==-1:
        pth_index=[int(i[:-4]) for i in os.listdir(path) if i[-4:]==".pth"]
        epoch=max(pth_index)
    print("Loading %d.pth..."%epoch)
    model.load_state_dict(pickle.load(open(os.path.join(path,"%d.pth"%epoch),"rb")))


def sav_log(epoch,trainloss,evalloss,file="log.csv"):
    with open(file,"a+",encoding="utf8") as f:
        f.write(",".join([str(epoch),str(trainloss),str(evalloss)]))


def load_data(mode):
    """
    loads original data (6 fields).
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


def load_features_and_labels(interval):
    """
    loads features and labels.
    :param interval:
    :return: ndarray of shape [n_samples, step_len, d_feat+1]
    """
    file_name_candidate = 'data/processed_data/data_features_and_labels_interval_{}.npy'.format(interval)
    if os.path.isfile(file_name_candidate):
        print('loading features and labels...')
        with open(file_name_candidate, 'rb') as f:
            data = np.load(f)
        print('load finished.')
    return data



def train(model, epochs, dataloader_train, dataloader_valid, device, optimizer, criterion):
    """train and validation"""

    for epoch in range(epochs):
        print("Epoch {}:".format(epoch), end='')
        train_loss = 0.0
        train_corrects = 0
        train_samples = 0
        print("training...\t", end='')
        model.train()

        losses = []
        for inputs in dataloader_train:  # e.g. torch.Size([1, 1761, 20, 359])
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
        train_loss=np.mean(losses)

        print("evaluating...\t", end='')
        model.eval()
        scores = []
        losses = []
        for inputs in dataloader_valid:
            inputs = inputs.squeeze()
            features = inputs[:, :, :-1].to(device)
            labels = inputs[:, -1, -1].to(device)
            # feature[torch.isnan(feature)] = 0  # WE NEED TO EVALUTE ALL SAMPLES!??
            preds = model(features.float())
            loss = criterion(preds.double(), labels)
            losses.append(loss.item())
        print(np.mean(losses))
        eval_loss=np.mean(losses)
        sav_log(epoch,train_loss,eval_loss)
        save_pth(model,"./pth",epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_epoch', type=int, required=True, default=-1, help='-1 to load latest model, 0 to start new model, or other number to order a model')
    args = parser.parse_args()


    step_len = 20
    epochs = 200

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.seterr(divide='ignore')  # disable division by zero warnings

    # data_train = load_data('train_and_test')
    # features_train = alpha360(data_train)
    # labels_train = ret1d(data_train)
    # valid_length = int(len(data_train) / 10)
    # del data_train  # save RAM
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

    dataset_train = dataset_gat_ts(features_train, labels_train, step_len=step_len, valid_threshold=30)
    dataset_valid = dataset_gat_ts(features_valid, labels_valid, step_len=step_len, valid_threshold=30)
    dataset_test = dataset_gat_ts(features_test, labels_test, step_len=step_len, valid_threshold=30)
    print('available samples/days: training set {}, validation set {}, test set {}'.format(
        dataset_train.__len__(), dataset_valid.__len__(), dataset_test.__len__()
    ))

    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=32)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=32)
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=32)

    model = GATModel(d_feat=158, dropout=0.7)
    load_pth(model, "./pth", epoch=args.start_epoch)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    pdb.set_trace()

    train(model, epochs, dataloader_train, dataloader_valid, device, optimizer, criterion)
