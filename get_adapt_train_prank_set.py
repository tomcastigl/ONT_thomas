import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split

tail_data_path='/scratch/tcastigl/data/tailfindr_res'
ONT_signal_path='/scratch/tcastigl/data/pkl_data'
tail_files=os.listdir(tail_data_path)
ONT_files=os.listdir(ONT_signal_path)
couples_adapt=[['/scratch/tcastigl/data/tailfindr_res/m6A_11.csv', '/scratch/tcastigl/data/pkl_data/m6A_11_df.pkl'],
               ['/scratch/tcastigl/data/tailfindr_res/m6A_13.csv', '/scratch/tcastigl/data/pkl_data/m6A_13_df.pkl'],
              ['/scratch/tcastigl/data/tailfindr_res/m5C_3.csv', '/scratch/tcastigl/data/pkl_data/m5C_3_df.pkl'],
              ['/scratch/tcastigl/data/tailfindr_res/UNM_10.csv', '/scratch/tcastigl/data/pkl_data/UNM_10_df.pkl'],
                ['/scratch/tcastigl/data/tailfindr_res/UNM_15.csv', '/scratch/tcastigl/data/pkl_data/UNM_15_df.pkl'],
              ['/scratch/tcastigl/data/tailfindr_res/pU_1.csv', '/scratch/tcastigl/data/pkl_data/pU_1_df.pkl'],
              ['/scratch/tcastigl/data/tailfindr_res/h5mC_2.csv', '/scratch/tcastigl/data/pkl_data/h5mC_2_df.pkl']]

adapt_df=pd.DataFrame()
for couple in couples_adapt:
    print(couple)
    tailfindres_df=pd.read_csv(couple[0])
    tailfindres_df['read_id'] ='read_'+tailfindres_df['read_id'].astype(str)
    ONT_df=pd.read_pickle(couple[1])
    #ONT_df=ONT_df[['read_id', 'type', 'signal']]
    #print(f'tailfindr shape {tailfindres_df.shape}')
    #print(f'ONT shape {ONT_df.shape}')
    if ONT_df.shape[0] != 0:
        merged_df=ONT_df.merge(tailfindres_df,how='inner',on='read_id').head(9000)
        print(merged_df.info())
        if merged_df.shape[0] != 0:
            print('---------------doing-------------')
            #print(ONT_df.info())
            #merged_df=ONT_df.merge(tailfindres_df,how='inner',on='read_id')
            merged_df=merged_df.dropna()
  
            merged_df.signal = merged_df.apply(lambda row: row.signal[:int(row.tail_start)-100], axis=1) #keeping only adapter
            merged_df['signal_length']=merged_df['signal'].str.len()
            merged_df=merged_df.drop(merged_df[merged_df['signal_length'] < 200].index)
            merged_df=merged_df.drop(merged_df[merged_df['signal_length'] > 2e4].index)

            adapt_df=adapt_df.append(merged_df[['read_id','signal','type']])
print(f'mean before {np.mean(adapt_df.signal.apply(np.mean))}')
print(f'std before {np.mean(adapt_df.signal.apply(np.std))}')

adapt_df.signal=[(x-np.mean(x))/np.std(x) for x in adapt_df.signal]
print(f'mean after {np.mean(adapt_df.signal.apply(np.mean))}')
print(f'std after {np.mean(adapt_df.signal.apply(np.std))}')

adapt_df=adapt_df.astype({'type':'category'})
adapt_df=adapt_df.rename(columns={'type':'dataset'})            
print(adapt_df.info())
X_train, X_test = train_test_split(adapt_df, test_size=0.1, random_state=42, stratify=adapt_df.dataset)
#X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=42, stratify=X_train.dataset)

X_train.to_pickle('/work/upnae/thomas_trna/data/adapt_prank_std/train.pkl')
X_test.to_pickle('/work/upnae/thomas_trna/data/adapt_prank_std/val.pkl')

print('\n ----------------done-------------')
