import torch 
from torch import nn
from schnetpack import AtomsData, AtomsLoader
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from Utils import data_pre_preparation
from qml.representations import *

data=AtomsData('./QM7X_eq_16_prop_kse.db')
data=data_pre_preparation(data,shuffle=False,normalize=True,save_to_file=True)
df=pd.read_json('./dataset{}.json'.format(data.__len__()))
print(df.columns)
