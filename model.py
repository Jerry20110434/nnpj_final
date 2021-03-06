"""
code for the models used. testing.

this file should be under data/../ (i.e. parent folder of data)
run example: None

"""


import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pdb


class GATModel(nn.Module):

    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, num_layers_gat=1, num_head=1, dropout=0.0, base_model="LSTM"):
        super().__init__()
        if base_model == "GRU":
            self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif base_model == "LSTM":
            self.rnn = nn.LSTM(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("unknown base model name `%s`" % base_model)
        self.hidden_size = hidden_size
        self.d_feat = d_feat
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.num_layers_gat = num_layers_gat
        self.gat_layers = nn.ModuleList()
        self.num_head = num_head
        for l in range(self.num_layers_gat):
            self.gat_layers.append(GATLayer(self.hidden_size, self.num_head))

    def forward(self, x):
        """

        :param x: input data  of shape [n_samples, step_len, d_feat+1]
        :return:
        """
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        for l in range(self.num_layers_gat):
            hidden = self.gat_layers[l](hidden)
            hidden = hidden + out[:, -1, :]  # residual connection from initial feature (layer k=0)
        hidden = self.fc(hidden)
        hidden = self.leaky_relu(hidden)
        return self.fc_out(hidden).squeeze()


class GATLayer(nn.Module):

    def __init__(self, hidden_size=64, num_head=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.num_head = num_head
        self.a_list = []
        for i in range(self.num_head):
            a = nn.Parameter(torch.randn(1, self.hidden_size * 2), requires_grad=True)
            a.requires_grad = True
            self.a_list.append(a)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y, a):
        x = self.transformation(x)
        y = self.transformation(y)
        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        attention_out = a.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        att_weight = 0
        for ai in self.a_list:
            att_weight += self.cal_attention(x, x, ai)
        att_weight /= len(self.a_list)  # mean of i.i.d. heads
        hidden = att_weight.mm(x)
        return hidden


class LSTMModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        """

        :param x: input data  of shape [n_samples, step_len, d_feat+1]
        :return:
        """
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()


class GRUModel(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x):
        """

        :param x: input data  of shape [n_samples, step_len, d_feat+1]
        :return:
        """
        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, :]).squeeze()
