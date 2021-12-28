'''
utils to load signals reads
'''
import numpy as np
import os
import h5py
import sys
import glob
from ont_fast5_api.fast5_interface import get_fast5_file
from ont_fast5_api import fast5_interface

def get_filenames(folder):
    filenames = [file for file in glob.glob(os.path.join(folder,'*.fast5'))]
    return np.array(filenames)

def list_keys(f):
    return [key for key in f.keys()]

def get_ids(read_filename):
    """Return a list with Fast5File or Multi_Fast5file read ids """
    
    fast5_filepath = read_filename #This can be a single- or multi-read file
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        return f5.get_read_ids()

def get_signal(read_filename, read_id):
    """Return raw signal of the id translocation in the passed file"""
    
    with fast5_interface.get_fast5_file(read_filename, 'r') as f5file:
            read = f5file.get_read(read_id)
            sig = read.get_raw_data()
            return sig
        
def get_signal_pA(read_filename, read_id):
    """Return raw signal in pA of the id translocation in the passed file"""
    
    f = h5py.File(read_filename, 'r') 
    dset = f["read_" + read_id]
    
    signal = dset['Raw']['Signal'][:]
    digitisation = dset['channel_id'].attrs.get('digitisation')
    range_ONT = dset['channel_id'].attrs.get('range')
    offset = dset['channel_id'].attrs.get('offset')
    pA_val = range_ONT/digitisation * (signal + offset)
    return pA_val

def retrieve_all_info(read_filename):
        dset = h5py.File(read_filename, 'r')
        read = list_keys(dset['Raw']['Reads'])
        signal = dset['Raw']['Reads'][read[0]]['Signal'][:]
        
        digitisation = dset['UniqueGlobalKey/channel_id'].attrs.get('digitisation')
        range_ONT = dset['UniqueGlobalKey/channel_id'].attrs.get('range')
        offset = dset['UniqueGlobalKey/channel_id'].attrs.get('offset')
        pA_val = range_ONT/digitisation * (signal + offset)

        move = dset['Analyses/Basecall_1D_000/BaseCalled_template/Events']['move'][:]
        trace = dset['Analyses/Basecall_1D_000/BaseCalled_template']['trace']
        
        
        fastq = dset['Analyses/Basecall_1D_000/BaseCalled_template']['Fastq']
        seq_fastq = fastq[()].decode('UTF-8').split("\n")[1]
        stride = dset['Analyses/Basecall_1D_000/Summary/basecall_1d_template/'].attrs.get('block_stride')
        first_sample_template =dset['Analyses/Segmentation_000/Summary/segmentation'].attrs.get('first_sample_template')
        sampling_freq = dset['UniqueGlobalKey/channel_id'].attrs.get('sampling_rate')

        return pA_val, move, seq_fastq, first_sample_template, sampling_freq  