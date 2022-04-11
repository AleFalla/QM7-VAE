from Train_Test_utils import TEST_prop_mol, TRAIN_reco_prop_ls
import pandas as pd
from separate_models import prop_ls_NN, prop_ls_NN_conv, prior, corrector, prop_ls_Transformer
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
prop_list=['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'DIP', 'HLgap', 'HOMO_0', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#list(df.columns[df.columns != 'BoB'])[92:104]

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
df_tr,df_test=split_train_test(df,config_num=config_num,save_to_file=False)
df_tr=df_tr[prop_list+['BoB']]
df_test=df_test[df_tr.columns]

#define models
latent_size=32#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=528#len(df['BoB'][0])
print(input_dim)
model_prop_ls=prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
prior_model=prior(latent_size=latent_size,prop_size=len(prop_list))#,extra_size=32-len(prop_list))#,coeff=coeff)
VAE_model=base_VAE(input_dim=input_dim, latent_size=latent_size)#toy_VAE(imgChannels=2,zDim=latent_size)#input_dim=input_dim, latent_size=latent_size) #
corrector_model=corrector(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list),output_size=input_dim)

#male model_list
model_list=[VAE_model, model_prop_ls]#, prior_model]

#allocate models
for model in model_list:
    model.to(device_model)

#useless
property_type='global'

#define the datasets
dataset_1=pandas_to_dataset_mol_prop(df_tr,rep='BoB',property_list=prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,rep='BoB',property_list=prop_list) 

log, model_list, test_error_1 = TRAIN_reco_prop_ls(model_list, dataset_1, dataset_2, lr_conv=1e-7, device=device, patience=25, factor=0.8, beta=1, alpha=1, reset=True, config_num=config_num, train_size=10000, batch_size=500, folder_name='VAE_prop_to_ls_NN/checkpoints_reco_prop_ls_{}'.format(property_type))
log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_task_error","val_reco_error","lr"])
print('***',test_error_1.item(),'***')

