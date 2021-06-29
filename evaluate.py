"""
evaluation is usually done after training in train.py. we could also perform it separately with this file.
args are the exact same with train.py, but pass start_epoch as the model epoch to laod.
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


def evaluate_model(models, path, dataloader_test, criterion, device, save=True):
    """
    evaluate model(s).
    :param model: could be single model of list of models. if list, equal weight ensemble of rank of pred is calculated.
    """
    if not isinstance(models, list):
        models = [models]
    for model in models:
        model.eval()
    losses = []
    ics = []
    pred = 0
    pred_rank = 0
    with torch.no_grad():
        for inputs in dataloader_test:  # e.g. torch.Size([1, 1761, 20, 359])
            inputs = inputs.squeeze()  # e.g. torch.Size([1761, 20, 359])
            features = inputs[:, :, :-1].to(device)
            labels = inputs[:, -1, -1].to(device)
            for model in models:
                pred += model(features.float())
                pred_rank += f.rank(pred.cpu().detach().numpy(), axis=0)
            pred /= len(models)
            pred_rank /= len(models)
            ics.append(calPearsonR(pred_rank, f.rank(labels.cpu().detach().numpy(), axis=0), axis=0))
            loss = criterion(pred.double(), labels)
            losses.append(loss.item())

    print('rank ic', np.mean(ics), 'rank icir', np.mean(ics) / np.std(ics))
    ics_by_month = from_ics_calc_ic_ir(ics, mode='month')
    print('Rank IC by month\n', ics_by_month)
    print('loss:', np.mean(losses), end='\t')
    ic_total = np.concatenate([ics_by_month, np.array((np.mean(ics), np.mean(ics) / np.std(ics)))[np.newaxis, :]], axis=0)
    if save:
        pd.DataFrame(ic_total).to_csv(os.path.join(path, 'ICs_by_month.csv'))
    return ic_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default="GAT", help='')
    parser.add_argument('--interval', type=int, required=True, default=48, help='')
    parser.add_argument('--folder_name', type=str, required=True, default='temp', help='')
    parser.add_argument('--start_epoch', type=int, required=False, default=-1, help='')
    parser.add_argument('--dataset', type=str, required=False, default="alpha158", help='')
    parser.add_argument('--GAThead', type=int, required=False, default=1, help='')
    parser.add_argument('--GATlayers', type=int, required=False, default=1, help='')
    parser.add_argument('--top', type=str, required=False, default="False", help='')
    args = parser.parse_args()

    # check for directory
    path = "./pth/%s" % args.folder_name
    if not os.path.exists(path):
        print("folder not found!")
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
    criterion = nn.MSELoss()

    # evaluate and calculate IC
    pdb.set_trace()
    print('validation set performance:')
    evaluate_model(model, path, dataloader_valid, criterion, device)
    print('test set performance:')
    evaluate_model(model, path, dataloader_test, criterion, device)
    pdb.set_trace()
