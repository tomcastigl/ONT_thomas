import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from basecall_viz_utils import get_signal_basecall_for_plotting

def common_member(a, b):
    a_set = set(a)
    b_set = set(b)
    if (a_set & b_set):
        return True 
    else:
        return False
print('loading data')    
df=pd.read_csv('probas_df_liver_full.csv')
rid_tokens=pd.read_csv('readids_tokens.csv')
rid_tokens.set_axis(['read_id','read_id_token'],axis=1,inplace=True)
df.drop('Unnamed: 0',axis=1,inplace=True)
df['pred_proba']=df[['UNM','h5mC','pU','m5C','m6A']].max(axis=1)
mods=['UNM','h5mC','pU','m5C','m6A']
df['pred']=[mods[pred] for pred in df.pred.tolist()]
df_max=df
df_max.head()
df_max.shape
rids=rid_tokens.loc[rid_tokens.read_id_token.isin(df_max.read_id_token.unique()),'read_id']


signals=pd.read_pickle('/home/tcastigl/concat_liver_df.pkl')
signals=signals[signals.read_id.isin(rids)]
signals.reset_index(inplace=True)

mods_l=[]
signals_mods_idxs_l=[]
print('getting mods and mods_idxs')
for i,r in enumerate(df_max.read_id_token.unique()):
    rid=rid_tokens.loc[rid_tokens.read_id_token ==  r,'read_id'].item()
    if i%1000==0: print(f'read {i+1} over {len(df_max.read_id_token.unique())}')
    mods=df_max[df_max.read_id_token==r].pred.tolist()
    signal_chunks_idx=[list(range(500*df_max.chunk_idx.loc[i],500*(df_max.chunk_idx.loc[i]+1))) for i in df_max.loc[df_max.read_id_token == r].index]
    mods_l.extend([mods])
    signals_mods_idxs_l.extend([signal_chunks_idx])
    #signals.loc[signals.read_id == rid,'mods']=mods
    #print(signals.loc[signals.read_id == rid,'mods'])
    #signals.loc[signals.read_id == rid,'signal_mod_idxs']=signal_chunks_idx
signals['mods']=mods_l
signals['signal_mods_idxs']=signals_mods_idxs_l
signals.head()

mod_seqs=[]
chunk_idx_l=[]
print('getting sequences')
for i,row in signals.iterrows():
    if i%100==0: print(f'read {i} over {len(signals)}')
    _,basecall=get_signal_basecall_for_plotting(signals, row.read_id)
    for j,sign_mod in enumerate(row.signal_mods_idxs):
        mod_seq_bases=[]
        for base in basecall:
             if sign_mod[0]-1000 < base['start'] < sign_mod[-1]+1000:
                if common_member(list(range(base['start'],base['end'])),sign_mod):
                    mod_seq_bases.extend(base['base'])
        mod_seq=''.join(mod_seq_bases)
        mod_seqs.extend([mod_seq])
        chunk_idx_l.extend([[sign_mod[0],sign_mod[-1]]])

df['seq']=mod_seqs
df['chunk_idxs']=chunk_idx_l
df_map=df.merge(rid_tokens, on='read_id_token',how='inner')
df_map=df_map[['read_id','seq','chunk_idxs','UNM','h5mC','pU','m5C','m6A','pred','pred_proba']]
df_map

df_map.to_csv('mapping_df.csv')
print('-----done-----')