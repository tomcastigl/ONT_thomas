import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import glob
import h5py
import pickle
from ont_fast5_api import fast5_interface
import gc
def list_keys(f):
    return [key for key in f.keys()]

def retrieve_read_info_from_folder(fast5folder,type_mod):
    datalist=[]
    fast5files=os.listdir(fast5folder)
    print(f'folders are \n {fast5files}')
    for fast5file in fast5files:
        f=h5py.File(os.path.join(fast5folder,fast5file),'r')
        print('reading ',fast5file)
        reads=list(f.keys())
        for read in reads:
            signal=np.array(f[read]['Raw']['Signal'])
            #range_ONT = f[read]['channel_id'].attrs.get('range')
            #offset = f[read]['channel_id'].attrs.get('offset')
            #pA_val = range_ONT/digitisation * (signal + offset)
            move = np.array(f[read]['Analyses/Basecall_1D_001/BaseCalled_template/Move'])
            trace = np.array(f[read]['Analyses/Basecall_1D_001/BaseCalled_template/Trace'])
            fastq = f[read]['Analyses/Basecall_1D_001/BaseCalled_template']['Fastq']
            seq_fastq = fastq[()].decode('UTF-8').split("\n")[1]
            stride = f[read]['Analyses/Basecall_1D_001/Summary/basecall_1d_template/'].attrs.get('block_stride')
            first_sample_template =f[read]['Analyses/Segmentation_000/Summary/segmentation'].attrs.get('first_sample_template')
            sampling_freq = f[read]['channel_id'].attrs.get('sampling_rate')
            read_info=[read,type_mod,signal,move,trace,seq_fastq,stride, first_sample_template, sampling_freq,
                      fast5file]
            #read_info_pd=pd.Series(read_info,index=data.columns)
            #data=data.append(read_info_pd,ignore_index=True)
            datalist.append(read_info)
        f.close()
    return datalist
    

def main():
#make one pkl per folder
    data='/work/upnae/ONT_modif/data' #change?
    mods=os.listdir(data)
    mod_dirs={mod : os.path.join(data,mod) for mod in ['UNM','m5C','pU']}
    print(mod_dirs)

    columns=['read_id','type','signal','move','trace','seq_fastq',
                'stride','first_sample_template','sampling_freq','fast5_filename']
    for mod in mod_dirs:
        folders=list(os.listdir(mod_dirs[mod]))
        print(mod)
        for folder in folders:
            mod_list_batch=[]
            folder_path=os.path.join(mod_dirs[mod],folder,'workspace')
            print(f'reading{folder_path}')
            mod_list_batch.extend(retrieve_read_info_from_folder(folder_path,mod))
            mod_sample_batch_df=pd.DataFrame(data=mod_list_batch,columns=columns)                   
            mod_sample_batch_df.to_pickle(f'/scratch/tcastigl/data/{mod}_{folder}_df.pkl')
            print('done')

if __name__ == "__main__": main()

    
'''    for folder in folders:
        print('in ',folder)
        mod_list.extend(retrieve_read_info_from_folder(folder,mod))
        size_chunk += 1
        if size_chunk >= 5 or folder == folders[-1]:
            mod_df=pd.DataFrame(data=mod_list,columns=columns)
            print(mod_df.info())
            mod_df.to_pickle(f'/work/upnae/thomas_trna/data/{mod}_{chunk}_df.pkl')
            size_chunk=0
            chunk+=1
            mod_list.clear()


data='/scratch/ONT_samuel/Novoa/25062020/data'
mods=os.listdir(data)
mod_dirs={mod : os.path.join(data,mod) for mod in mods}
print(mod_dirs)

allreads_df=pd.DataFrame(columns=['read_id','type','signal','move','trace','seq_fastq',
                                  'stride','first_sample_template','sampling_freq','fast5_filename'])
for mod in mod_dirs:
    folders=list(os.path.join(mod_dirs[mod],folder,'workspace') for folder in os.listdir(mod_dirs[mod]))
    for folder in folders:
        print('in ',folder)
        allreads_df=allreads_df.append(retrieve_read_info_from_folder(folder,mod))
print(allreads_df.info())
allreads_df.to_pickle("/home/tcastiglione/data/all_reads_df.pkl")
'''
