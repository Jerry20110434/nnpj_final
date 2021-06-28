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


def alpha360(data):
    """
    calculates alpha360 features. This is no longer being used.
    :param data: input ndarray of shape (~1300, 48, ~3700, 6). fields are arranged in order [open, close, high, low, volume, money].
    :return: ndarray of shape (~1300, ~3700, 358)
    """

    # first, we only take daily stats, i.e. the last intraday data, to make thing easy

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


def alpha158(data, data_index, interval):
    """
    calculates alpha158 features. this is an attempt to reimplement all 158 formulaic alphas
    from https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py, and may have mistakes.

    :param data: input ndarray of shape (~1300, 48, ~3700, 6). fields are arranged in order [open, close, high, low,
    volume, money].
    :param data_index: stock index data of shape (~1300, 48, 6). e.g. hs300.
    :param interval: sample interval after data is reshaped into (~62400, ~3700, 6). e.g. interval = 48 means to
    calculate signals using data every 48 5mins. Used choices are 1 (5min pediod), 48 (1d period), 240 (1w/5d
    period). However, the signals are always calculated every 48 5mins (1d). e.g. MA5 for the 30th day uses
    the 30*48th, (30*48-interval)th, (30*48-interval*2)th, ..., (30*48-interval*4)th data.

    :return: ndarray of shape (1706, 158, 4185)
    """

    def RELU(arr):
        return np.maximum(arr, 0)

    windows = [6, 11, 21, 31, 61]  # we add one to all windows because ts_delay causes the first value to be nan
    n_features = 13 + 29 * len(windows)  # 158

    data = data.reshape(-1, data.shape[2], data.shape[3])  # reshape into (81888, 4185, 6)

    # we change volume and money column into their expanding sum values(ignore nan), as it is otherwise difficult
    # to calculate the rolling sum of volumes over a specific period
    data[..., -2:] = np.nancumsum(data[..., -2:], axis=0)

    data_index = np.expand_dims(data_index.reshape(-1, data_index.shape[2]), 1)  # reshape into (81888, 6)
    data_index[-2:] = np.nancumsum(data_index[-2:], axis=0)
    signals = []
    for di in range(1, data.shape[0] // 48 + 1):
        print('calculating features for day {}/{}'.format(di, data.shape[0] // 48))
        signals_di = []
        ti = di * 48  # data up to data[ti - 1, ...] can be used
        if ti < np.max(windows) * interval: # we need all features to have values
            signals_di = np.full((n_features, data.shape[1]), np.nan)
            print('skipping this day. too few samples for rolling.')
        else:
            # kbar. calculated from the last available bar of interval 5mins
            OPEN = data[ti - interval, :, 0]
            CLOSE = data[ti - 1, :, 1]
            HIGH = np.nanmax(data[ti - interval: ti, :, 2], 0)  # high is max of highs
            LOW = np.nanmin(data[ti - interval: ti, :, 3], 0)
            VOLUME = np.nansum(data[ti - interval: ti, :, 4], 0)
            VWAP = np.nansum(data[ti - interval: ti, :, 5], 0) / VOLUME
            signals_di.append((CLOSE - OPEN) / OPEN)  # 1
            signals_di.append((HIGH - LOW) / OPEN)  # 2
            signals_di.append((CLOSE - OPEN) / (HIGH - LOW))  # 3
            signals_di.append((HIGH - np.maximum(OPEN, CLOSE)) / OPEN)  # 4
            signals_di.append((HIGH - np.maximum(OPEN, CLOSE)) / (HIGH - LOW))  # 5
            signals_di.append((np.minimum(OPEN, CLOSE) - LOW) / OPEN)  # 6
            signals_di.append((np.minimum(OPEN, CLOSE) - LOW) / (HIGH - LOW))  # 7
            signals_di.append((2 * CLOSE - HIGH - LOW) / OPEN)  # 8
            signals_di.append((2 * CLOSE - HIGH - LOW) / (HIGH - LOW))  # 9

            # price. same as kbar.
            signals_di.append(OPEN / CLOSE - 1)  # 10. modified the original feature to have -1.
            signals_di.append(HIGH / CLOSE - 1)  # 11
            signals_di.append(LOW / CLOSE - 1)  # 12
            signals_di.append(VWAP / CLOSE - 1)  # 13

            # rolling. calculate with various rolling windows.
            for window in windows:  # 14 - 158
                MASK = np.full(data.shape[0], False)
                for d in range(window):
                    MASK[ti - 1 - d * interval] = True
                data_rolling = data[MASK, :, :]
                data_rolling[..., -2:] = f.ts_delta(data_rolling[..., -2:], 1)

                # scalars (for individual stock)
                S_CLOSE = data_rolling[-1, :, 1]  # S stands for scalar
                S_HIGH = np.nanmax(data_rolling[:, :, 2], 0)
                S_LOW = np.nanmin(data_rolling[:, :, 3], 0)
                S_VOLUME = data_rolling[-1, :, 4]  # volume of last period, not sum of all
                # vectors (for individual stock)
                DELTA_CLOSE = f.ts_delta(data_rolling[:, :, 1], 1)
                RETURN = DELTA_CLOSE / f.ts_delay(data_rolling[:, :, 1], 1)
                DELTA_VOLUME = f.ts_delta(data_rolling[:, :, 4], 1)
                INDEXRETURN = np.repeat(f.ts_delta(data_index[MASK, :, 1], 1) / f.ts_delay(data_index[MASK, :, 1], 1), data.shape[1], axis=1)

                signals_di.append(data_rolling[0, :, 1] / S_CLOSE)  # ROC  Ref($close, %d)/$close
                signals_di.append(np.nanmean(data_rolling[:, :, 1], 0) / S_CLOSE)  # MA  Mean($close, %d)/$close
                signals_di.append(np.nanstd(data_rolling[:, :, 1], 0) / S_CLOSE)  # STD  Std($close, %d)/$close

                Y = RETURN - np.nanmean(RETURN, 0, keepdims=True)
                X = INDEXRETURN - np.nanmean(INDEXRETURN, 0, keepdims=True)
                nanmask = (np.isnan(X) | np.isnan(Y))
                X[nanmask] = np.nan
                Y[nanmask] = np.nan
                COV_CLOSE_INDEXCLOSE = np.nansum(X * Y, 0)  # actually cov of return
                VAR_CLOSE = np.nansum(X ** 2, 0)
                VAR_INDEXCLOSE = np.nansum(Y ** 2, 0)
                BETA = COV_CLOSE_INDEXCLOSE / VAR_INDEXCLOSE
                signals_di.append(BETA)  # BETA  Slope($close, %d)/$close
                signals_di.append(COV_CLOSE_INDEXCLOSE ** 2 / (VAR_CLOSE * VAR_INDEXCLOSE))  # RSQR  Rsquare($close, %d)
                signals_di.append((Y - X * BETA)[-1, ...])  # RESI  Resi($close, %d)/$close

                signals_di.append(S_HIGH / S_CLOSE)  # MAX  Max($high, %d)/$close
                signals_di.append(S_LOW / S_CLOSE)  # LOW  Min($low, %d)/$close
                k1 = int(0.2 * window)
                k2 = int(0.8 * window)

                signals_di.append(np.partition(data_rolling[:, :, 1], k2, axis=0)[k2, ...] / S_CLOSE)  # QTLU  Quantile($close, %d, 0.8)/$close
                signals_di.append(np.partition(data_rolling[:, :, 1], k1, axis=0)[k1, ...] / S_CLOSE)  # QTLD  Quantile($close, %d, 0.2)/$close
                signals_di.append(f.rank(data_rolling[:, :, 1], axis=0)[-1, ...])  # RANK  Rank($close, %d)
                signals_di.append((S_CLOSE - S_LOW) / (S_HIGH - S_LOW))  # RSV  ($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)
                IMAX = f.wraped_nanfuncs(np.nanargmax, data_rolling[:, :, 1], 0)
                IMIN = f.wraped_nanfuncs(np.nanargmin, data_rolling[:, :, 1], 0)
                signals_di.append(IMAX / window)  # IMAX  IdxMax($high, %d)/%d
                signals_di.append(IMIN / window)  # IMIN  IdxMin($low, %d)/%d
                signals_di.append((IMAX - IMIN) / window)  # IMXD  (IdxMax($high, %d)-IdxMin($low, %d))/%d
                signals_di.append(f.calPearsonR(data_rolling[:, :, 1], np.log(data_rolling[:, :, 4] + 1), axis=0))  # CORR  Corr($close, Log($volume+1), %d)
                signals_di.append(f.calPearsonR(RETURN + 1, np.log(data_rolling[:, :, 4] / f.ts_delay(data_rolling[:, :, 4], 1) + 1), axis=0))  # CORD  Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)

                signals_di.append(np.nanmean(DELTA_CLOSE > 0, 0))  # CNTP  Mean($close>Ref($close, 1), %d)
                signals_di.append(np.nanmean(DELTA_CLOSE < 0, 0))  # CNTN  Mean($close<Ref($close, 1), %d)
                signals_di.append(np.nanmean(((DELTA_CLOSE > 0) + 0) - ((DELTA_CLOSE < 0) + 0), 0))  # CNTD  Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)
                DELTA_PRICE_POS_SUM = np.nansum(RELU(DELTA_CLOSE), 0)
                DELTA_PRICE_NEG_SUM = np.nansum(RELU(-DELTA_CLOSE), 0)
                DELTA_PRICE_ABS_SUM = np.nansum(np.abs(DELTA_CLOSE), 0)
                signals_di.append(DELTA_PRICE_POS_SUM / DELTA_PRICE_ABS_SUM)  # SUMP  Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)
                signals_di.append(DELTA_PRICE_NEG_SUM / DELTA_PRICE_ABS_SUM)  # SUMN  Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)
                signals_di.append((DELTA_PRICE_POS_SUM - DELTA_PRICE_NEG_SUM) / DELTA_PRICE_ABS_SUM)  # SUMD  (Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d)) / (Sum(Abs($close-Ref($close, 1)), %d)+1e-12)

                signals_di.append(np.nanmean(data_rolling[:, :, 4], 0) / S_VOLUME)  # VMA  Mean($volume, %d)/($volume+1e-12)
                signals_di.append(np.nanstd(data_rolling[:, :, 4], 0) / S_VOLUME)  # VSTD  Std($volume, %d)/($volume+1e-12)
                DELTA_VOLUME_POS_SUM = np.nansum(RELU(DELTA_VOLUME), 0)
                DELTA_VOLUME_NEG_SUM = np.nansum(RELU(-DELTA_VOLUME), 0)
                DELTA_VOLUME_ABS_SUM = np.nansum(np.abs(DELTA_VOLUME), 0)
                signals_di.append(np.nanstd(np.abs(RETURN) * S_VOLUME, 0) / np.nanmean(np.abs(RETURN) * S_VOLUME, 0))  # WVMA  Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)
                signals_di.append(DELTA_VOLUME_POS_SUM / DELTA_VOLUME_ABS_SUM)  # VSUMP  Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)
                signals_di.append(DELTA_VOLUME_NEG_SUM / DELTA_VOLUME_ABS_SUM)  # VSUMN  Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)
                signals_di.append((DELTA_VOLUME_POS_SUM - DELTA_VOLUME_NEG_SUM) / DELTA_VOLUME_ABS_SUM)  # VSUMD  (Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d)) / (Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)

        signals.append(signals_di)
    return f.remove_inf(np.array(signals))


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
