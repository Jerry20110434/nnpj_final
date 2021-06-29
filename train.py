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
import sys
import shutil
from evaluate import *


def save_pth(model, path, epoch):
    """saves model"""
    pickle.dump(model.state_dict(), open(os.path.join(path, "%d.pth" % epoch), "wb"))


def load_pth(model, path, epoch):
    """loads model"""
    print("Loading %d.pth..." % epoch)
    model.load_state_dict(pickle.load(open(os.path.join(path, "%d.pth" % epoch), "rb")))


def sav_log(epoch, trainloss, evalloss, path, file="log.csv"):
    """records log"""
    with open(os.path.join(path, file), "a+", encoding="utf8") as f:
        f.write(",".join([str(epoch), str(trainloss), str(evalloss)])+'\n')


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


def load_features_and_labels_360():
    """
    loads features and labels for alpha360
    """
    file_name_candidate = 'data/processed_data/alpha360.npy'
    if os.path.isfile(file_name_candidate):
        print('loading features and labels...')
        with open(file_name_candidate, 'rb') as f:
            data = np.load(f)
        print('load finished.')
    return data


def train(model, epochs, dataloader_train, dataloader_valid, device, optimizer, criterion, path, patience=10):
    """train and validation"""

    lowest_val_loss = 1e9
    best_epoch = 0
    patience_left = patience

    for epoch in range(epochs):
        print("Epoch {}:".format(epoch), end='')

        # train
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
        train_loss = np.mean(losses)
        print(train_loss, end='\t')

        # validate
        print("evaluating...\t", end='')
        model.eval()
        losses = []
        with torch.no_grad():
            for inputs in dataloader_valid:
                inputs = inputs.squeeze()
                features = inputs[:, :, :-1].to(device)
                labels = inputs[:, -1, -1].to(device)
                preds = model(features.float())
                loss = criterion(preds.double(), labels)
                losses.append(loss.item())
            eval_loss = np.mean(losses)
            print(eval_loss)
            if not os.path.exists(path):
                os.mkdir(path)
            sav_log(epoch, train_loss, eval_loss, path)
            save_pth(model, path, epoch)

        # check for early stopping
        if eval_loss < lowest_val_loss:
            lowest_val_loss = eval_loss
            patience_left = patience
            best_epoch = epoch
            print("patience restored!")
        else:
            patience_left -= 1
            if patience_left == 0:
                load_pth(model, path, best_epoch)  # load best model
                print("early stop. load model from epoch {}.".format(best_epoch))
                return


def prepare_data(alpha, interval, step_len, top):
    if alpha == 158:
        data = load_features_and_labels(interval=interval)
    elif alpha == 358:
        data = load_features_and_labels_360()
    else:
        print('invalid features!')
        sys.exit()

    features_test_full = data[..., :-1][:, -242 - step_len:]
    labels_test_full = data[..., -1][:, -242 - step_len:]
    data = data[:2000, ...]  # [:2000] because smol CUDA memory

    features_all, labels_all = data[..., :-1], data[..., -1]
    valid_length = int(data.shape[1] / 10)

    features_train = features_all[:, :-242][:, :-valid_length]
    labels_train = labels_all[:, :-242][:, :-valid_length]

    # extra step_len days for validation and test set because the first step_len days are discarded in the dateset
    features_valid = features_all[:, :-242][:, -valid_length - step_len:]
    labels_valid = labels_all[:, :-242][:, -valid_length - step_len:]

    features_test = features_all[:, -242 - step_len:]
    labels_test = labels_all[:, -242 - step_len:]

    # check if use TOP universe
    if top == "True":
        top_stocks = pickle.load(open('top500bool.pickle', 'rb')).T  # (4185, 1706)
        top_stocks_train = top_stocks[:, :-242][:, :-valid_length]
        top_stocks_valid = top_stocks[:, :-242][:, -valid_length - step_len:]
        top_stocks_test = top_stocks[:, -242 - step_len:]
    else:
        top_stocks_train = None
        top_stocks_valid = None
        top_stocks_test = None

    # create datasets and dataloader
    dataset_train = dataset_gat_ts(features_train, labels_train, step_len=step_len, valid_threshold=30, top_stocks=top_stocks_train)
    dataset_valid = dataset_gat_ts(features_valid, labels_valid, step_len=step_len, valid_threshold=30, top_stocks=top_stocks_valid)
    dataset_test = dataset_gat_ts(features_test, labels_test, step_len=step_len, valid_threshold=30, top_stocks=top_stocks_test)
    print('available samples/days: training set {}, validation set {}, test set {}'.format(
        dataset_train.__len__(), dataset_valid.__len__(), dataset_test.__len__()
    ))
    dataloader_train = DataLoader(dataset_train, batch_size=1, num_workers=32)
    dataloader_valid = DataLoader(dataset_valid, batch_size=1, num_workers=32)
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=32)

    return dataloader_train, dataloader_valid, dataloader_test


def prepare_model(model, alpha, GAThead, GATlayers, start_epoch, device, path):
    if model == "GAT":
        model = GATModel(d_feat=alpha, dropout=0.7, num_head=GAThead, num_layers_gat=GATlayers)
        print('using {} GAT heads and {}-hop neighbors'.format(GAThead, GATlayers))
        for gl in range(len(model.gat_layers)):
            for a in range(len(model.gat_layers[gl].a_list)):
                model.gat_layers[gl].a_list[a] = model.gat_layers[gl].a_list[a].to(device)
    elif model == "LSTM":
        model = LSTMModel(d_feat=alpha, dropout=0.7)
    elif model == "GRU":
        model = GRUModel(d_feat=alpha, dropout=0.7)
    if start_epoch >= 0:
        load_pth(model, path, start_epoch)
    print("Model:")
    print(model)
    model = model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default="GAT", help='')
    parser.add_argument('--interval', type=int, required=True, default=48, help='')
    parser.add_argument('--folder_name', type=str, required=True, default='temp', help='')
    parser.add_argument('--start_epoch', type=int, required=False, default=-1, help='')
    parser.add_argument('--alpha', type=int, required=False, default=158, help='')
    parser.add_argument('--GAThead', type=int, required=False, default=1, help='')
    parser.add_argument('--GATlayers', type=int, required=False, default=1, help='')
    parser.add_argument('--top', type=str, required=False, default="False", help='')  # not in use.
    args = parser.parse_args()

    # check for directory
    path = "./pth/%s" % args.folder_name
    if os.path.exists(path):
        clear_folder = input("A previous model with the same name already exists. Do you wish to clear the directory? [Y/N] ")
        if clear_folder == "Y":
            shutil.rmtree(path)
        else:
            sys.exit()

    torch.backends.cudnn.enabled = False  # causes bug, may be due to server cudnn version
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.seterr(divide='ignore')  # disable division by zero warnings

    # hyperparamters
    step_len = 20
    epochs = 1000  # we use early stopping

    dataloader_train, dataloader_valid, dataloader_test = prepare_data(args.alpha, args.interval, step_len, args.top)
    model = prepare_model(args.model, args.alpha, args.GAThead, args.GATlayers, args.start_epoch, device, path)

    # other hyperparameters
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # train and validation
    train(model, epochs, dataloader_train, dataloader_valid, device, optimizer, criterion, path=path, patience=10)

    # evaluate and calculate IC
    print('validation set performance:')
    evaluate_model(model, path, dataloader_valid, criterion, device)
    print('test set performance:')
    evaluate_model(model, path, dataloader_test, criterion, device)

    # dataset_test_full = dataset_gat_ts(
    #     features_test_full, labels_test_full, step_len=step_len, valid_threshold=30, top_stocks=top_stocks_test)
    # dataloader_test_full = DataLoader(dataset_test_full, batch_size=1, num_workers=32)
    # evaluate_model(model, path, dataloader_test_full, criterion, device)
