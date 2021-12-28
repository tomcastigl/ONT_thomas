import pandas as pd
import os
from sklearn.model_selection import train_test_split
tailfind_path='/scratch/tcastigl/data/tail_findr_liver_res/0/out_liver_0.csv'
ONT_df_path='/scratch/tcastigl/data/ONT_liver_df/0/concat_liver_df.pkl'

tailfindres_df=pd.read_csv(tailfind_path) 
tailfindres_df['read_id'] ='read_'+tailfindres_df['read_id'].astype(str)
ONT_df=pd.read_pickle(ONT_df_path)
#ONT_df=ONT_df[['read_id', 'type', 'signal']]
#print(f'tailfindr shape {tailfindres_df.shape}')
#print(f'ONT shape {ONT_df.shape}')
print(tailfindres_df.head())
print(ONT_df.head())
print(tailfindres_df.loc[tailfindres_df.read_id.isin(ONT_df.read_id)])
merged_df=ONT_df.merge(tailfindres_df,how='inner',on='read_id')
print('merged_df:',merged_df.info())
print('---------------doing-------------')
#print(ONT_df.info())
merged_df=merged_df.dropna()
merged_df.signal = merged_df.apply(lambda row: row.signal[int(row.tail_end):], axis=1) #cutting adapter
merged_df['signal_length']=merged_df['signal'].str.len()
merged_df=merged_df.drop(merged_df[merged_df['signal_length'] < 4000].index)
merged_df=merged_df.drop(merged_df[merged_df['signal_length'] > 2e5].index)

class_df=merged_df[['read_id','signal']]
class_df.to_pickle('/scratch/tcastigl/data/ONT_liver_df/0/curlk_df.pkl')