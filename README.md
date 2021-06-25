This repository includes source code for our final project of DATA130011.01, Spring 2021. The results can be fully 
replicated by following the "run" commands is this readme file.

# Data preprocessing
Data preprocessing includes two files：  
\
data_create_columns.py This script processes ['time'] into 2 columns: date and time, and saves as a new file xxx_modified.pkl  
Run：  
python3 data_create_columns.py  
\
data_preprocessing.py This script concatenates xxx_modified.pkl files and reindexes them with a pandas multiindex, then save as 
several 4-d ndarrays (the data is too large to be processed simultaneously).  
Run：  
python3 data_preprocessing.py --start_year 2014 --end_year 2019 --name train  
python3 data_preprocessing.py --start_year 2020 --end_year 2020 --name test  

# Model
model.py includes a Temporal GAT with multihop neighbors (paramter num_layers_gat).