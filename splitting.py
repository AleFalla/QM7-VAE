from Train_Test_utils import TRAIN_reco, TEST_reco
import pandas as pd
from Utils import split_train_test, pandas_to_dataset_mol_prop
from VAE import base_VAE
import torch
from matplotlib import pyplot as plt

seed=0
torch.manual_seed(seed)


df=pd.read_json('./dataset41537.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
""" prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df.columns[df.columns != 'BoB'])#[0:30] #comment this if you want all the properties
 """
df_tr,df_test=split_train_test(df,save_to_file=True)