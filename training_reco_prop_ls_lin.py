from Train_Test_utils import TRAIN_reco_prop_ls, TEST_prop_mol
import pandas as pd
from Utils import pandas_to_dataset_mol_prop
from VAE import base_VAE
from separate_models import prop_ls_lin
from matplotlib import pyplot as plt
from torch import nn
import torch
seed=0
torch.manual_seed(seed)

df=pd.read_json('./dataframe41537.json') #put the name of the json file with the right data
df_tr=pd.read_json('./dataset30000_training.json') #put the name of the json file with the right data
df_test=pd.read_json('./dataset11537_test.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df.columns[df.columns != 'BoB'])[8:53] #comment this if you want all the properties
dataset_1=pandas_to_dataset_mol_prop(df_tr,prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,prop_list) 

latent_size=len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
model_prop_ls=prop_ls_lin(latent_size=latent_size,prop_size=len(prop_list))
model_reco=base_VAE(input_dim=input_dim, latent_size=latent_size)

train_losses, val_losses, val_error, last_model_reco, last_model_prop_ls, test_error_1 = TRAIN_reco_prop_ls(model_reco,model_prop_ls,dataset_1, dataset_2, config_num=config_num, folder_name='VAE_prop_to_ls_lin/checkpoints_reco_prop_ls')

print('first_check_test: {}%'.format(test_error_1))


l=1
a=1
plt.figure('train_and_val')
fig, ax = plt.subplots()
ax.plot(train_losses,linestyle='solid',linewidth=l,label=r'train loss',alpha=a)
ax.plot(val_losses,linestyle='solid',linewidth=l,label=r'validation loss',alpha=a)
leg = ax.legend(prop={'size': 10})
text=leg.get_texts()
for i in range(0,len(text)):
    text[i].set_color('black')
plt.xlabel('epochs/50')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig('./VAE_prop_to_ls_lin/plots/train_val_vs_epochs_reco_prop_ls_lin.pdf')

plt.figure('train_and_val')
fig, ax = plt.subplots()
ax.plot(val_error,linestyle='solid',linewidth=l,label=r'val reco error',alpha=a)
leg = ax.legend(prop={'size': 10})
text=leg.get_texts()
for i in range(0,len(text)):
    text[i].set_color('black')
plt.xlabel('epochs/10')
plt.ylabel('validation recostruction error in %')
plt.tight_layout()
plt.savefig('./VAE_prop_to_ls_lin/plots/val_reco_error_prop_ls_lin.pdf')

PATH='./VAE_prop_to_ls_lin/reco_prop_ls_lin_trained_models/prop_ls_config_num_{}'.format(config_num)
torch.save(last_model_prop_ls.state_dict(), PATH)
model_prop_ls = prop_ls_lin(latent_size=latent_size,prop_size=len(prop_list))
model_prop_ls.load_state_dict(torch.load(PATH))

PATH='./VAE_prop_to_ls_lin/reco_prop_ls_lin_trained_models/reco_config_num_{}'.format(config_num)
torch.save(last_model_reco.state_dict(), PATH)
model_reco = base_VAE(input_dim=input_dim, latent_size=latent_size)
model_reco.load_state_dict(torch.load(PATH))


test_error_2=TEST_prop_mol(model_prop_ls,model_reco, dataset_2)
print('second_check_test: {}%'.format(test_error_2))

file = open("VAE_prop_to_ls_lin/test_error_log.txt", "w")
file.write("test error torch seed {}: ".format(seed) + '{}%'.format(test_error_2))
file.close()

