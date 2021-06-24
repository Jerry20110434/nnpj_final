"""
functions that calculate features

this file should be under data/../ (i.e. parent folder of data)
run example: None

"""

import numpy as np
import torch
import ufuncs as f


def alpha360(data):
    """
    calculates alpha360 features
    :param data: input ndarray of shape (~1300, 48, ~3700, 6)
    :return: ndarray
    """
    # first, we only take daily stats, i.e. the last intraday data, to make thing easy
    data = data[:, -1, :, :]
    return data


def ret1d(data):
    """
    calculates 1-day ahead returns, as labels for training
    :param data: input ndarray of shape (~1300, 48, ~3700, 6). data is arranged in order [open, close, high, low, volume, money].
    :return: ndarray, return of (T+1 day close, T+2 day close). This is the common 'delay 1' practice, since we can only trade
     after knowing the features for day T.
    """
    close = data[:, -1, :, 0]
    return f.ts_delay(close, close, -2) / f.ts_delay(close, close, -1) - 1