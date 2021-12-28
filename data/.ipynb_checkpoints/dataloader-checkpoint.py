import os
import torch
import pandas as pd

from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    def __init__(self, pkl_filename, data_dir, input_dim):
        dataset_df = pd.read_pickle(os.path.join(data_dir, pkl_filename))
        self.data_list = dataset_df.signal.apply(
            lambda x: (x[:-(len(x) % input_dim)] if len(x) % input_dim != 0 else x).reshape((-1, input_dim))).tolist()
        self.target = torch.tensor(dataset_df.dataset.cat.codes.to_numpy().astype(int))
        #self.rid_token=torch.tensor(dataset_df.token.to_numpy())
        #self.chunk_idx=torch.tensor(dataset_df.chunk_idx.to_numpy())
    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return torch.tensor(self.data_list[idx]).float(), self.target[idx]#, self.rid_token[idx], self.chunk_idx[idx] #added .float() 


def collate_fn(batch):
    lengths = torch.tensor([ t[0].shape[0] for t in batch ])
    #print('\n',batch[0][0].dtype, '\n')
    signal_batch = [ torch.Tensor(t[0]) for t in batch ]
    signal_batch = torch.nn.utils.rnn.pad_sequence(signal_batch, padding_value=-500, batch_first=True)
    #token=torch.tensor([t[2] for t in batch])
    #chunk_idx=torch.tensor([t[3] for t in batch])
    return signal_batch, lengths, torch.tensor([t[1] for t in batch])#,token,chunk_idx
