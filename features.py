"""
functions that calculate features. testing.

this file should be under data/../ (i.e. parent folder of data)
run example: None

"""

import numpy as np
import torch
import ufuncs as f
import pdb
import pandas as pd


def Greater(a, b):
    return np.maximum(a, b)


def Less(a, b):
    return np.minimum(a, b)


def alpha360(data):
    """
    calculates alpha360 features. This is no longer being used.
    :param data: input ndarray of shape (~1300, 48, ~3700, 6). fields are arranged in order [open, close, high, low, volume, money].
    :return: ndarray of shape (~1300, ~3700, 358)
    """

    # first, we only take daily stats, i.e. the last intraday data, to make thing easy
    np.seterr(divide='ignore')  # disable division by zero warnings

    open = data[:, 0, :, 0]
    close = data[:, -1, :, 1]
    high = np.max(data[:, :, :, 2], 1)
    low = np.min(data[:, :, :, 3], 1)
    volume = np.sum(data[:, :, :, 4], 1)
    vwap = f.remove_inf(np.sum(data[:, :, :, 5], 1) / volume)  # vwap = money / volume
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
    return f.remove_inf(np.log(np.stack(signals, axis=2)))


def alpha158(data, interval):
    """
    calculates alpha158 features.
    :param data: input ndarray of shape (~1300, 48, ~3700, 6). fields are arranged in order [open, close, high, low,
    volume, money].
    :param interval: sample interval after data is reshaped into (~62400, ~3700, 6). e.g. interval = 48 means to
    calculate signals using data every 48 5mins. Used choices are 1 (5min pediod), 48 (1d period), 240 (1w/5d
    period). However, the signals are always calculated every 48 5mins (1d). e.g. MA5 for the 30th day uses
    the 30*48th, (30*48-interval)th, (30*48-interval*2)th, ..., (30*48-interval*4)th data.

    """
    data = data.reshape(-1, data.shape[2], data.shape[3])  # reshape into (~62400, ~3700, 6)

    open = data[:, 0, :, 0]
    close = data[:, -1, :, 1]
    high = np.max(data[:, :, :, 2], 1)
    low = np.min(data[:, :, :, 3], 1)
    volume = np.sum(data[:, :, :, 4], 1)
    vwap = f.remove_inf(np.sum(data[:, :, :, 5], 1) / volume)  # vwap = money / volume
    signals = []

    #kbar 9dim
    signals.append((close-open)/open)
    signals.append((high-low)/open)
    signals.append((close-open)/(high-low+1e-12))
    signals.append((high-Greater(open, close))/open)
    signals.append((high-Greater(open, close))/(high-low+1e-12))
    signals.append((Less(open, close)-low)/open)
    signals.append((Less(open, close)-low)/(high-low+1e-12))
    signals.append((2*close-high-low)/open)
    signals.append((2*close-high-low)/(high-low+1e-12))

    #price & volume 4dim
    for d in range(5):
        signals.append(f.ts_delay(open, d) / close)
        signals.append(f.ts_delay(high, d) / close)
        signals.append(f.ts_delay(low, d) / close)
        signals.append(f.ts_delay(vwap, d) / close)
        #if d > 0:
        #    signals.append(f.ts_delay(close, d) / close)
        #    signals.append(f.ts_delay(volume, d) / volume)

    #rolling
    windows=[5, 10, 20, 30, 60]
    for d in windows:
        signals.append(f.ts_delay(close, d) / close)
        signals.append(pd.DataFrame(close).rolling(window=d,axis=0,min_periods=d).mean().to_numpy() / close)
        signals.append(pd.DataFrame(close).rolling(window=d,axis=0,min_periods=d).std().to_numpy() / close)
        #signals.append()


def ret1d(data):
    """
    calculates 1-day ahead returns, as labels for training
    :param data: input ndarray of shape (~1300, 48, ~3700, 6). data is arranged in order [open, close, high, low, volume, money].
    :return: ndarray, return of (T+1 day close, T+2 day close). This is the common 'delay 1' practice, since we can only trade
     after knowing the features for day T.
    """
    close = data[:, -1, :, 1]
    ret = f.rank(f.remove_inf(f.ts_delay(close, -2) / f.ts_delay(close, -1) - 1), axis=1)
    return (ret - np.nanmean(ret, axis=1, keepdims=True)) / np.nanstd(ret, axis=1, keepdims=True)
