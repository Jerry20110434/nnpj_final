"""
preprocess stock index data (e.g. hs300). does both create_columns and preprocess.
"""

import numpy as np
import pandas as pd
import argparse
import pdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=str, required=True, help='')  # e.g. 000001
    args = parser.parse_args()

    index_name = args.index
    data = pd.read_pickle(r'data/jqdata_a_stocks_5min_{}.pkl'.format(index_name))
    data.index = data.index.rename('time')
    data.reset_index(inplace=True)
    data.loc[:, 'date'] = data.time.apply(lambda x: x.date())
    data.loc[:, 'time'] = data.time.apply(lambda x: x.time())
    dates = np.unique(data.date.iloc[range(0, len(data), 48)])
    times = data.time.iloc[:48]
    multiindex = pd.MultiIndex.from_tuples([(i, j) for i in dates for j in times])
    data = data.set_index(['date', 'time']).reindex(multiindex)
    arr_data = data.to_numpy().reshape(len(dates), len(times), len(data.columns))
    daily_trade_value = np.nansum(arr_data[..., -1], 1)
    mask_no_trade = np.broadcast_to(daily_trade_value[:, np.newaxis, np.newaxis] == 0, arr_data.shape)  # should sum 0
    arr_data[mask_no_trade] = np.nan
    with open(r'data/processed_data/data_5min_{}.npy'.format(index_name), 'wb') as f:
        np.save(f, arr_data[:-1, ...].astype('float64'))  # [:-1, ...] because stock data lacks 2020-12-31 somehow
