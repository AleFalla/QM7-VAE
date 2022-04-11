import torch 
from torch import nn
from schnetpack import AtomsData, AtomsLoader
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from Utils import data_pre_preparation
from qml.representations import *
fname='dataset_eq_atC6_unnorm'
data=AtomsData('./QM7X_eq_plus_atC6.db')
data=data_pre_preparation(data,shuffle=False,normalize=False,save_to_file=True,fname=fname)
df=pd.read_json('./{}.json'.format(fname))
print(df.columns)
