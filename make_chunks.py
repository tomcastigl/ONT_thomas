import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
chunksize=500
traindf=pd.read_pickle('../data/classification_dataset_conformer_curlk_big/train.pkl')
valdf=pd.read_pickle('../data/classification_dataset_conformer_curlk_big/val.pkl')
traindf=shuffle(traindf)
valdf=shuffle(valdf)
traindf.reset_index(inplace=True)
valdf.reset_index(inplace=True)
print('loaded data \n\n')
j=0
chunks,labels,read_ids=[],[],[]
for i,row in traindf.iterrows():
    newlen=len(row.signal)-len(row.signal)%chunksize
    signal=row.signal[:newlen]
    new_chunks=[signal[x:x+chunksize] for x in range(0,len(signal),chunksize)]
    chunks.append(new_chunks)
    labels.append([row.dataset]*len(new_chunks))
    read_ids.append([row.read_id]*len(new_chunks))
    #train_chunks_df=train_chunks_df.append(pd.DataFrame(list(zip(read_id, chunks, label)),columns=['read_id','signal','dataset']))
    if j%1000==0: print(j)
    j+=1
    
chunks= [val for sublist in chunks for val in sublist]
labels= [val for sublist in labels for val in sublist]
read_ids= [val for sublist in read_ids for val in sublist]
print(len(chunks))

train_chunks_df=pd.DataFrame(data=list(zip(read_ids,chunks,labels)),columns=['read_id','signal','dataset'])
print('------train df------ \n\n')
train_chunks_df.head(-10)
train_chunks_df.dataset=train_chunks_df.dataset.astype('category')
print(f'shape: {train_chunks_df.shape}')
train_chunks_df.info()
print(f'chunk size = {train_chunks_df.signal.apply(len).mean()}')
print(f'modifs: {train_chunks_df.dataset.unique()}')
df_for_training_grouped = train_chunks_df.groupby('dataset')
df_for_training_grouped.groups.values()
frames_of_groups_train = [x.sample(df_for_training_grouped.size().min()) for y, x in df_for_training_grouped]
train_chunks_df = pd.concat(frames_of_groups_train)
print(f'count: {Counter(train_chunks_df.dataset)}')
print('\n\n')

j=0
chunks,labels,read_ids=[],[],[]
for i,row in valdf.iterrows():
    newlen=len(row.signal)-len(row.signal)%chunksize
    signal=row.signal[:newlen]
    new_chunks=[signal[x:x+chunksize] for x in range(0,len(signal),chunksize)]
    chunks.append(new_chunks)
    labels.append([row.dataset]*len(new_chunks))
    read_ids.append([row.read_id]*len(new_chunks))
    #train_chunks_df=train_chunks_df.append(pd.DataFrame(list(zip(read_id, chunks, label)),columns=['read_id','signal','dataset']))
    if j%1000==0: print(j)
    j+=1

chunks= [val for sublist in chunks for val in sublist]
labels= [val for sublist in labels for val in sublist]
read_ids= [val for sublist in read_ids for val in sublist]

val_chunks_df=pd.DataFrame(data=list(zip(read_ids,chunks,labels)),columns=['read_id','signal','dataset'])
print('------val df------ \n\n')
val_chunks_df.dataset=val_chunks_df.dataset.astype('category')
print(f'shape: {val_chunks_df.shape}')
val_chunks_df.info()
print(f'chunk size = {val_chunks_df.signal.apply(len).mean()}')
print(f'modifs: {val_chunks_df.dataset.unique()}')

df_for_val_grouped = val_chunks_df.groupby('dataset')
df_for_val_grouped.groups.values()
frames_of_groups_val = [x.sample(df_for_val_grouped.size().min()) for y, x in df_for_val_grouped]
val_chunks_df = pd.concat(frames_of_groups_val)
print(f'count: {Counter(val_chunks_df.dataset)}')

train_chunks_df.to_pickle('../data/classification_dataset_conformer_chunks_curlk_500_big/train.pkl')
val_chunks_df.to_pickle('../data/classification_dataset_conformer_chunks_curlk_500_big/val.pkl')





