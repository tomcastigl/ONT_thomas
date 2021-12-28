import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
import numpy as np
from collections import Counter

files=['UNM_10_train.pkl','UNM_11_train.pkl','UNM_12_train.pkl','UNM_13_train.pkl',#'UNM_14_train.pkl','UNM_15_train.pkl',
        'm5C_8_train.pkl',#'m5C_9_train.pkl','m5C_1_train.pkl','m5C_2_train.pkl',
        'h5mC_1_train.pkl','m6A_11_train.pkl','m6A_13_train.pkl','m6A_14_train.pkl','m6A_16_train.pkl',#'m6A_17_train.pkl',
        'pU_1_train.pkl','pU_2_train.pkl']
data_paths=[os.path.join('/scratch/tcastigl/data/classification_dataset/train',file) for file in files]
print(data_paths)
classification_df=pd.DataFrame()
for i,data_path in enumerate(data_paths):
    print(f'df {i}, {files[i]}')
    df=pd.read_pickle(data_path)
    print(df.shape)
    classification_df=classification_df.append(df)
classification_df.dataset=classification_df.dataset.astype('category')
print('-------classification df ---------')
print(classification_df.info())
print(classification_df.shape)
print(classification_df.dataset.unique())
print(Counter(classification_df.dataset))
X_train, X_val = train_test_split(classification_df, test_size=0.1, random_state=42, stratify=classification_df.dataset)

X_train.to_pickle("../data/classification_dataset_conformer_chunks_curlk_500_big/train.pkl")
X_val.to_pickle("../data/classification_dataset_conformer_chunks_curlk_500_big/val.pkl")
#X_test.to_pickle("../data/classification_dataset_conformer_12keach/test.pkl")

