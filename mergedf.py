import pandas as pd
print('mdrrr')
adapdf=pd.read_pickle('/work/upnae/thomas_trna/data/adapt_train.pkl')
curldf=pd.read_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_12keach/train.pkl')
adapvaldf=pd.read_pickle('/work/upnae/thomas_trna/data/adapt_val.pkl')
curlvaldf=pd.read_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_12keach/val.pkl')
print(adapdf.dataset.unique())
adapdf=adapdf.append(curldf)
adapvaldf=adapvaldf.append(curlvaldf)

#adapdf.to_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_12K_readsandadapt/train.pkl')
#adapvaldf.to_pickle('/work/upnae/thomas_trna/data/classification_dataset_conformer_12K_readsandadapt/test.pkl')
