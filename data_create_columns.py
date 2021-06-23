"""
this file should be under stock_data/
"""

import os
import pandas as pd
import numpy as np
import argparse


if __name__ == '__main__':
    for year in range(2014, 2021):
        print(year)
        data = pd.read_pickle(r'jqdata_a_stocks_5min_{}.pkl'.format(year))
        if 'date' in data.columns: # already processed
            continue
        # create a separate date column
        data.loc[:, 'date'] = data.time.apply(lambda x: x.date())
        print('...')
        data.loc[:, 'time'] = data.time.apply(lambda x: x.time())
        print('...')
        data.to_pickle(r'jqdata_a_stocks_5min_{}_modified.pkl'.format(year))
        print('saved!\n')
