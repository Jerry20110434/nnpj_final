# """
# generate market capitalization data
# """
# import pandas as pd
# import numpy as np
# from train import load_data
#
#
# if __name__ == "__main__":
#     data = pd.DataFrame([])
#     for year in range(2014, 2021):
#         file_name_candidate = 'data/jqdata_a_stocks_capitalization_{}.pkl'.format(year)
#         data = pd.concat([data, pd.read_pickle(file_name_candidate)])
#
#     data = data[data.date != '2020-12-31']
#     dates = np.unique(data.date)
#     stocks = np.unique(data.code)
#     multiindex = pd.MultiIndex.from_tuples([(i, j) for i in dates for j in stocks])
#     data = data.set_index(['date', 'code']).reindex(multiindex)
#     arr_data = data.to_numpy().reshape(len(dates), len(stocks), len(data.columns)).squeeze()
#     data_pv = load_data('train_and_test')
#
