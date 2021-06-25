"""
functions that calculate features. testing.

this file should be under data/../ (i.e. parent folder of data)
run example: None

"""

import numpy as np
import torch
import ufuncs as f
import pdb


def alpha360(data):
    """
    calculates alpha360 features
    :param data: input ndarray of shape (~1300, 48, ~3700, 6). data is arranged in order [open, close, high, low, volume, money].
    :return: ndarray of shape (~1300, ~3700, 358)
    """

    # first, we only take daily stats, i.e. the last intraday data, to make thing easy
    np.seterr(divide='ignore')  # disable division by zero warnings

    data = data[:, -1, :, :]
    open = data[..., 0]
    close = data[..., 1]
    high = data[..., 2]
    low = data[..., 3]
    volume = data[..., 4]
    vwap = f.remove_inf(data[..., 5] / volume)  # vwap = money / volume
    signals = []
    for d in range(60):
        signals.append(f.ts_delay(open, d) / close)
        signals.append(f.ts_delay(high, d) / close)
        signals.append(f.ts_delay(low, d) / close)
        signals.append(f.ts_delay(vwap, d) / close)
        if d > 0:
            signals.append(f.ts_delay(close, d) / close)
            signals.append(f.ts_delay(volume, d) / volume)
        if (d + 1) % 10 == 0:
            print('calculating alpha 360, progress {}/{}'.format(d + 1, 60))
    return np.stack(signals, axis=2)


def ret1d(data):
    """
    calculates 1-day ahead returns, as labels for training
    :param data: input ndarray of shape (~1300, 48, ~3700, 6). data is arranged in order [open, close, high, low, volume, money].
    :return: ndarray, return of (T+1 day close, T+2 day close). This is the common 'delay 1' practice, since we can only trade
     after knowing the features for day T.
    """
    close = data[:, -1, :, 0]
    ret = f.rank(f.remove_inf(f.ts_delay(close, -2) / f.ts_delay(close, -1) - 1), axis=1)
    return ret - np.nanmean(ret, axis=1, keepdims=True)