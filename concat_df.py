import os
import pandas as pd

path='/scratch/tcastigl/data/ONT_liver_df/0'
files=[os.path.join(path,f) for f in os.listdir(path) if 'ONT_liver' in f]
small_df=pd.DataFrame()
for file in files:
    print(file)
    df=pd.read_pickle(file)
    small_df=small_df.append(df)
small_df.to_pickle(path+'/concat_liver_df.pkl')