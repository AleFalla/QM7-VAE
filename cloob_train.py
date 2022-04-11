from Train_Test_utils import TEST_prop_mol, TRAIN_reco_prop_ls, CLIP_training, decoding_training
import pandas as pd
from separate_models import prop_ls_NN, mol_ls_NN, ls_mol_NN, prop_ls_NN_conv, prior, corrector, prop_ls_Transformer
from Utils import split_train_test, pandas_to_dataset_mol_prop, generate_hCHG_rep, generate_MCM_rep, generate_adj_rep, generate_distance, generate_pol_rep
from VAE import base_VAE, toy_VAE, multi_input_VAE, TRANSFORMER_VAE
import torch
from matplotlib import pyplot as plt
import numpy as np

seed=0
torch.manual_seed(seed)
device='cuda'
device_model=torch.device(device)

#properties
#prop_list_1=['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'DIP', 'HLgap', 'HOMO_0', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#list(df.columns[df.columns != 'BoB'])[92:104]

hchg=['hCHG_0','hCHG_1','hCHG_2','hCHG_3','hCHG_4','hCHG_5','hCHG_6','hCHG_7','hCHG_8','hCHG_9','hCHG_10','hCHG_11','hCHG_12','hCHG_13','hCHG_14','hCHG_15','hCHG_16','hCHG_17','hCHG_18','hCHG_19','hCHG_20','hCHG_21','hCHG_22']
pols=['atPOL_0','atPOL_1','atPOL_2','atPOL_3','atPOL_4','atPOL_5','atPOL_6','atPOL_7','atPOL_8','atPOL_9','atPOL_10','atPOL_11','atPOL_12','atPOL_13','atPOL_14','atPOL_15','atPOL_16','atPOL_17','atPOL_18','atPOL_19','atPOL_20','atPOL_21','atPOL_22']
prop_list_1=hchg+pols
#load the dataset
df=pd.read_json('./dataset41537.json')

#number of configurations per molecule
config_num=1

#generate the representation
el_list=['#S','#Cl','#O','#N','#C','#H']
at_nums=[16,17,8,7,6,1]
df=generate_MCM_rep(df,el_list,at_nums)
df['BoB']=df['CM']

#split train test
df_tr0,df_test0=split_train_test(df,config_num=config_num,save_to_file=False)
df_tr=df_tr0[prop_list_1+['BoB']]
df_test=df_test0[df_tr.columns]

#define models
latent_size=128#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=528#len(df['BoB'][0])
print(input_dim)
prop_embedder=prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list_1),extra_size=16)
mol_embedder=mol_ls_NN(latent_size=latent_size, input_dim=input_dim)

#male model_list
model_list=[mol_embedder, prop_embedder]#, prior_model]

#allocate models
for model in model_list:
    model.to(device_model)

#useless
property_type='global'

#define the datasets
dataset_1=pandas_to_dataset_mol_prop(df_tr,rep='BoB',property_list=prop_list_1)
dataset_2=pandas_to_dataset_mol_prop(df_test,rep='BoB',property_list=prop_list_1) 

model_list = CLIP_training(model_list , dataset_1, reset=True, device='cuda', batch_size=500,beta=.1,alpha=1,lr_conv=1e-7,epochs_max=10000, train_size=28000, learning_rate=1e-3, factor=.9, patience=15, config_num=1)

molecular_embedder = model_list[0]
prop_list = ['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'DIP', 'HLgap', 'HOMO_0', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#list(df.columns[df.columns != 'BoB'])[92:104]
df_tr=df_tr0[prop_list+['BoB']]
df_test=df_test0[df_tr.columns]

#define the datasets
dataset_1=pandas_to_dataset_mol_prop(df_tr,rep='BoB',property_list=prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,rep='BoB',property_list=prop_list) 

encoder = prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=16)
decoder = ls_mol_NN(latent_size=latent_size, input_dim=input_dim)
encoder.to(device_model)
decoder.to(device_model)

deco_enc, error = decoding_training(encoder,decoder , molecular_embedder , dataset_1, dataset_2, reset=True, device='cuda', batch_size=500,beta=.1,alpha=1,lr_conv=1e-7,epochs_max=10000, train_size=28000, learning_rate=1e-3, factor=.9, patience=15, config_num=1)

print('***',error.item(),'***')

