from Train_Test_utils import TRAIN_reco, TEST_reco
import pandas as pd
from Utils import split_train_test, pandas_to_dataset_mol_prop
from VAE import base_VAE
import torch
from matplotlib import pyplot as plt

seed=0
torch.manual_seed(seed)

df=pd.read_json('./dataset41537.json') #put the name of the json file with the right data
print('done')
config_num=int(len(df)/41537)

df_tr,df_test=split_train_test(df,config_num=config_num,save_to_file=True)
print(df_test.columns)
print(df_tr.columns)