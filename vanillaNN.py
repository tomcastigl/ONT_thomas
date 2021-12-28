print('running...')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.optim as optim
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
#from data.dataloader import ClassificationDataset, collate_fn

def main():
    class customDataset:
        def __init__(self,data,targets):
            self.data=data
            self.targets=targets

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self,idx):
            sample=self.data[idx,:]
            target=self.targets[idx]
            return [torch.tensor(sample),
                    torch.tensor(target)]

    class  Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1=nn.Linear(10000,40)
            self.fc2=nn.Linear(40,40)
            self.fc3=nn.Linear(40,5)

            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.xavier_uniform_(self.fc3.weight)
        def forward(self, input):
            x=F.relu(self.fc1(input))
            x=F.relu(self.fc2(x))
            x=F.relu(self.fc3(x))
            return x

    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
    print('handling data ...')
    data_path='../data/classification_dataset'
    valdf=pd.read_pickle(os.path.join(data_path,'val.pkl'))
    traindf=pd.read_pickle(os.path.join(data_path,'train.pkl'))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    #traindf=traindf.head(500)
    traindf=traindf.drop('read_id', axis=1)
    traindf=traindf[traindf['chunk'].str.len()==10000]
    valdf=valdf.head(10000)
    valdf=valdf.drop('read_id', axis=1)
    valdf=valdf[valdf['chunk'].str.len()==10000]
    print(f'train df : \n {traindf.info()}')
    print(f'val df : \n {valdf.info()}')

    onehot=np.eye(5)
    train_labels=traindf['type'].values
    le = preprocessing.LabelEncoder()
    train_targets = onehot[le.fit_transform(train_labels)]
    #at the end inverse_transform it.
    train_data=np.array(traindf['chunk'].values.tolist(),dtype=float)
    val_labels=valdf['type'].values
    val_targets = onehot[le.fit_transform(val_labels)]
    #at the end inverse_transform it.
    val_data=np.array(valdf['chunk'].values.tolist(),dtype=float)

    train_dataset=customDataset(data=train_data, targets=train_targets)
    val_dataset=customDataset(data=val_data, targets=val_targets)
    train_loader= torch.utils.data.DataLoader(train_dataset,batch_size=10)
    val_loader= torch.utils.data.DataLoader(val_dataset,batch_size=10)
    print('training...')
    torch.manual_seed(42)
    model=Model()
    model.to(device)
    model.apply(weights_init)
    optimizer=optim.Adam(model.parameters(),lr=1e-3)
    EPOCHS=5
    for epoch in range(EPOCHS):
        print(f'epoch {epoch} on {EPOCHS}')
        for data in train_loader:
            X, y = data
            X=X.float().to(device)
            y=y.float().to(device)
            X=F.normalize(X)
            model.zero_grad()
            output=model(X)
            loss=F.binary_cross_entropy(output, y).to(device)
            loss.backward()
            optimizer.step()
        print(f'loss={loss}')

    correct=0
    total=0

    with torch.no_grad():
        for data in val_loader:
            X,y=data
            X=X.float().to(device)
            y=y.float().to(device)
            output=model(X)
            for idx, i in enumerate(output):
                if torch.argmax(i) == torch.argmax(y[idx]):
                    correct +=1
                total +=1
    print(f'evaluated accuracy: {100*correct/total}%')

if __name__ == "__main__":
    main()