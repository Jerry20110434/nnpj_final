This repository includes source code for our final project of DATA130011.01, Spring 2021. The results can be fully 
replicated by running the following commands, after copying stock_data/ into project folder and renaming it as data/:

```
python3 data_create_columns.py  
python3 data_preprocessing.py --start_year 2014 --end_year 2020 --name train_and_test  
python3 data_preprocessing_stock_index.py --index 000300  
python3 generate_features_and_labels.py --interval 48
```

# Data preprocessing
Data preprocessing includes two files：  
data_create_columns.py This script processes ['time'] into 2 columns: date and time, and saves as a new file xxx_modified.pkl  

data_preprocessing.py This script concatenates xxx_modified.pkl files and reindexes them with a pandas multiindex, then save as 
several 4-d ndarrays (the data is too large to be processed simultaneously).  

# Model
model.py includes a Temporal GAT with multihop neighbors (parameter num_layers_gat).  