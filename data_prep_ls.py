from Train_Test_utils import TRAIN_reco, TEST_reco
import pandas as pd
from Utils import data_preparation_latent_space
from VAE import base_VAE
import torch
from matplotlib import pyplot as plt


df=pd.read_json('./dataframe41537.json') #put the name of the json file with the right data
df_tr=pd.read_json('./dataset41537_training.json') #put the name of the json file with the right data
df_val=pd.read_json('./dataset41537_validation.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df_tr.columns[df.columns != 'BoB'])[8:53] #comment this if you want all the properties
latent_size=30#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
PATH='./VAE_prop_to_ls_NN/reco_prop_ls_NN_trained_models/reco_config_num_{}'.format(config_num) 
model = base_VAE(input_dim=input_dim, latent_size=latent_size)
model.load_state_dict(torch.load(PATH))

df_ls_tr=data_preparation_latent_space(model,df_tr,save_to_file=True,ID='training')
df_ls_val=data_preparation_latent_space(model,df_val,save_to_file=True,ID='validation')

