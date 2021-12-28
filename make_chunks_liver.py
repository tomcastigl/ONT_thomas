import pandas as pd
import os
import numpy as np
from sklearn.utils import shuffle
from collections import Counter
chunksize=500
df=pd.read_pickle('/scratch/tcastigl/data/ONT_liver_df/0/concat_liver_df.pkl')
print('loaded data \n\n')
j=0
chunks,read_ids,chunks_idx=[],[],[]
for i,row in df.iterrows():
    newlen=len(row.signal)-len(row.signal)%chunksize
    signal=row.signal[:newlen]
    new_chunks=[signal[x:x+chunksize] for x in range(0,len(signal),chunksize)]
    chunks.append(new_chunks)
    chunks_idx.append(range(len(new_chunks)))
    read_ids.append([row.read_id]*len(new_chunks))
    if j%1000==0: print(j)
    j+=1
chunks= [val for sublist in chunks for val in sublist]
read_ids= [val for sublist in read_ids for val in sublist]
chunks_idx= [val for sublist in chunks_idx for val in sublist]
print(len(chunks))
print(len(read_ids))
print(len(chunks_idx))
chunks_df=pd.DataFrame(data=list(zip(read_ids,chunks_idx,chunks)),columns=['read_id','chunk_idx','signal'])
print(f'shape: {chunks_df.shape}')
print(f'chunk size = {chunks_df.signal.apply(len).mean()}')
print('\n\n')

chunks_df.to_pickle('/scratch/tcastigl/data/ONT_liver_df/0/chunks_500_curlk.pkl')




