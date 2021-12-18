import torch 
from torch import nn
from schnetpack import AtomsData, AtomsLoader
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from Utils import data_pre_preparation, noneq_dataset
from qml.representations import *

config_num=5
data=AtomsData('./QM7X_eq_4noneq_16_prop.db')
print('done')
data=data_pre_preparation(data,shuffle=False,normalize=True,save_to_file=True)
print('done')
df_eq=pd.read_json('./dataset{}.json'.format(int(data.__len__()/config_num)))
print('done')
df_noneq=pd.read_json('./dataset{}.json'.format(data.__len__()))
print('done')
df_noneq=noneq_dataset(df_eq,df_noneq,config_num=config_num,save_to_file=True)
print('done')
df_noneq=pd.read_json('./dataset{}.json'.format(data.__len__()))
print('done')
print(df)
