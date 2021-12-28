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


def main():
#make one pkl per folder
    data='/scratch/tcastigl/ONT_liver/0' #change?
    fast5files_path=[os.path.join(data,file) for file in os.listdir(data)]
    for i,file in enumerate(fast5files_path):
        datalist=[]
        f=h5py.File(file,'r')
        print('reading ',file)
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
            first_sample_template =f[read]['Analyses/Segmentation_001/Summary/segmentation'].attrs.get('first_sample_template')
            read_info=[read,signal,move,trace,seq_fastq,stride,first_sample_template]
            #read_info_pd=pd.Series(read_info,index=data.columns)
            #data=data.append(read_info_pd,ignore_index=True)
            datalist.append(read_info)
        f.close()
        df=pd.DataFrame(data=datalist,columns=['read_id','signal','move','trace','seq','stride','first_sample_template'])                   
        df.to_pickle(f'/scratch/tcastigl/data/ONT_liver_df/0/ONT_liver_{i}.pkl')
    print('done')

if __name__ == "__main__": main()
