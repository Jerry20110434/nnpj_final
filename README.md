# nnpj_final
final project

数据预处理有两个文件：

data_create_columns.py 这个脚本将time处理成date和time两列 保存为新的文件xxx_modified.pkl

运行：

python3 data_create_columns.py


data_preprocessing.py 这个脚本对已经处理过date和time的数据用统一的multiindex进行reindex 然后保存为numpy 4维数组

运行：

python3 data_preprocessing.py --start_year 2014 --end_year 2019 --name train

python3 data_preprocessing.py --start_year 2020 --end_year 2020 --name test

