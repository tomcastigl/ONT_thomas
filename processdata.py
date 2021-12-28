import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

tail_data_path='/scratch/tcastigl/data/tailfindr_res_liver_0'
ONT_signal_path='/scratch/tcastigl/data/ONT_liver_df/0'
tail_files=os.listdir(tail_data_path)
ONT_files=os.listdir(ONT_signal_path)

couples=[]
count=0
all=0
for tailfile in tail_files:
    tryone=tailfile[:-4]
    if tryone+'_df.pkl' in ONT_files:
        couples.append([os.path.join(tail_data_path,tailfile),os.path.join(ONT_signal_path,(tryone+'_df.pkl'))])
        count +=1
    all+=1
print(count/all)
print(len(couples))
count=0
for couple in couples:
    print(couple)
    tailfindres_df=pd.read_csv(couple[0]) 
    tailfindres_df['read_id'] ='read_'+tailfindres_df['read_id'].astype(str)
    ONT_df=pd.read_pickle(couple[1])
    #ONT_df=ONT_df[['read_id', 'type', 'signal']]
    #print(f'tailfindr shape {tailfindres_df.shape}')
    #print(f'ONT shape {ONT_df.shape}')

    if ONT_df.shape[0] != 0:
        merged_df=ONT_df.merge(tailfindres_df,how='inner',on='read_id')
        if merged_df.shape[0] != 0:
            print('---------------doing-------------')
            #print(ONT_df.info())
            merged_df=ONT_df.merge(tailfindres_df,how='inner',on='read_id')
            merged_df=merged_df.dropna()
            merged_df.signal = merged_df.apply(lambda row: row.signal[int(row.tail_end):], axis=1) #cutting adapter
            merged_df['signal_length']=merged_df['signal'].str.len()
            merged_df=merged_df.drop(merged_df[merged_df['signal_length'] < 4000].index)
            merged_df=merged_df.drop(merged_df[merged_df['signal_length'] > 2e5].index)

            class_df=merged_df[['read_id','signal','type']]
            class_df=class_df.astype({'type':'category'})
            class_df=class_df.rename(columns={'type':'dataset'})
            X_train, X_test = train_test_split(class_df, test_size=0.1, random_state=42, stratify=class_df.dataset)
            X_train, X_val = train_test_split(X_train, test_size=0.1, random_state=42, stratify=X_train.dataset)

            X_train.to_pickle(f'/scratch/tcastigl/data/classification_dataset/train/{couple[0][37:-4]}_train.pkl')
            X_val.to_pickle(f'/scratch/tcastigl/data/classification_dataset/val/{couple[0][37:-4]}_val.pkl')
            X_test.to_pickle(f'/scratch/tcastigl/data/classification_dataset/test/{couple[0][37:-4]}_test.pkl')
            count +=1
print(count)du 