import numpy as np
import pandas as pd
import csv

df=pd.read_csv('/work/upnae/thomas_trna/data/sam_files/h5mC_simply.sam', sep='\t',skiprows=4)
dfU=pd.read_csv('/work/upnae/thomas_trna/data/sam_files/UNM_simply.sam', sep='\t',skiprows=4)
df=df.append(dfU)
df=df.dropna()
df.set_axis(['read_id','ref'],axis=1,inplace=True)
df=df.loc[df.ref != '*']
refs=df.ref.unique()
df_readstotrain=df.loc[(df.ref != refs[2]) & (df.ref != refs[3])]
df_readstoeval=df.loc[(df.ref != refs[0]) & (df.ref != refs[1])]
#df_readstotrain=df_readstotrain.append(df.loc[df.ref == refs[1]])
df_readstotrain=df_readstotrain['read_id']
df_readstoeval=df_readstoeval['read_id']
print(df_readstotrain.shape)
print(df_readstoeval.shape)
df_readstotrain.to_csv('/work/upnae/thomas_trna/data/read_ref_train.csv')
df_readstoeval.to_csv('/work/upnae/thomas_trna/data/read_ref_eval.csv')

df=pd.read_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_12K_readsandadapt/train.pkl')
val_df=pd.read_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_12K_readsandadapt/val.pkl')
reads_to_train=pd.read_csv('/work/upnae/thomas_trna/data/read_ref_train.csv')['read_id'].to_list()
reads_to_eval=pd.read_csv('/work/upnae/thomas_trna/data/read_ref_eval.csv')['read_id'].to_list()
df=df.append(val_df)
reads_to_train=['read_'+ read for read in reads_to_train]
reads_to_eval=['read_'+ read for read in reads_to_eval]
df_train=df.loc[df['read_id'].isin(reads_to_train)]
df_eval=df.loc[df['read_id'].isin(reads_to_eval)]
print(f'whole df before separating: \n {df.shape}')
print(f'\n train and val dfs after sep curlcakes: \n {df_train.shape} \n {df_eval.shape}')
print(df_train.dataset.unique())
df_train.to_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_sep_curlk/train.pkl')
df_eval.to_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_sep_curlk/val.pkl')
