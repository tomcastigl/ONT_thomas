import numpy as np
import pandas as pd

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
