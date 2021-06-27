"""
finished.

loads pkl data and concatenate/reshape them into a structured 4d array as .npy file

this file should be under data/../ (i.e. parent folder of data)
this script takes a lot of RAM.
data is arranged in order [open, close, high, low, volume, money]

run example: 
python3 data_preprocessing.py --start_year 2014 --end_year 2019 --name train
python3 data_preprocessing.py --start_year 2020 --end_year 2020 --name test
"""

import os
import pandas as pd
import numpy as np
import argparse
import datetime


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_year', type=int, required=True, help='')
    parser.add_argument('--end_year', type=int, required=True, help='')
    parser.add_argument('--name', type=str, required=True, help='')  # e.g. train
    args = parser.parse_args()

    try:
        os.mkdir('data/processed_data')
    except Exception: # folder already exists
        pass


    # load data
    print('loading data...')
    data = pd.DataFrame([])
    for year in range(args.start_year, args.end_year + 1):
        data = pd.concat([data, pd.read_pickle(r'data/jqdata_a_stocks_5min_{}_modified.pkl'.format(year))])
        print(year, '...')

    # generate indices
    dates_all = np.unique(data.date.iloc[range(0, len(data), 48)])
    times = data.time.iloc[:48]
    stocks = np.unique(data.code.iloc[range(0, len(data), 48)])

    print('processing...')
    T = 60
    part = 1
    for i in range(0, len(dates_all), T):
        day = dates_all[i]
        day_end_excluded = dates_all[min(i + T, len(dates_all) - 1)]
        print(day, day_end_excluded)
        start_index = np.searchsorted(data.date, day)
        end_index = np.searchsorted(data.date, day_end_excluded)
        if day_end_excluded == dates_all[-1]:
            end_index = len(data)
        dates = np.unique(data.date.iloc[range(start_index, end_index, 48)])
        print('constructing multiindex...')
        multiindex = pd.MultiIndex.from_tuples([(i, j, k) for i in dates for j in times for k in stocks])
        print('reindexing...')
        data_this_year = data.iloc[start_index: end_index].set_index(['date', 'time', 'code']).reindex(multiindex)
        print('reshaping...')
        arr_data_this_year = data_this_year.to_numpy().reshape(len(dates), len(times), len(stocks), len(data_this_year.columns))
        daily_trade_value = np.nansum(arr_data_this_year[..., np.where(data_this_year.columns == 'money')[0][0]], 1)
        print('setting zero trades to nan...')
        mask_no_trade = np.broadcast_to(daily_trade_value[:, np.newaxis, :, np.newaxis] == 0, arr_data_this_year.shape)
        arr_data_this_year[mask_no_trade] = np.nan
        print('concatenating...')
        print('saving...')
        with open(r'data/processed_data/data_5min_{}_pt_{}.npy'.format(args.name, part), 'wb') as f:
            np.save(f, arr_data_this_year.astype('float64'))
        part += 1
