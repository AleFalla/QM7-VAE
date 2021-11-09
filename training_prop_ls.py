from Train_Test_utils import TRAIN_prop_ls, TEST_prop_mol
import pandas as pd
from Utils import pandas_to_dataset_ls_prop, pandas_to_dataset_mol_prop
from VAE import base_VAE
from separate_models import prop_ls_lin, prop_ls_NN
import torch
from matplotlib import pyplot as plt

seed=0
torch.manual_seed(seed)

df=pd.read_json('./dataframe41537.json') #put the name of the json file with the right data
df_tr=pd.read_json('./dataset30000_ls_training.json') #put the name of the json file with the right data
df_val=pd.read_json('./dataset11537_ls_validation.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df.columns[df.columns != 'BoB'])[8:53] #comment this if you want all the properties
dataset_1=pandas_to_dataset_ls_prop(df_tr,prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_val,prop_list) 

latent_size=30 #len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
model=prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list))
PATH_0='./VAE_prop_to_ls_NN/reco_prop_ls_NN_trained_models/reco_config_num_{}'.format(config_num) #'./VAE_separate/onlyreco_trained_models/config_num_{}'.format(config_num) #
decoder = base_VAE(input_dim=input_dim, latent_size=latent_size)
decoder.load_state_dict(torch.load(PATH_0))

train_losses, val_losses, last_model, test_error_1 = TRAIN_prop_ls(model,decoder,dataset_1, dataset_2, config_num=config_num, folder_name='VAE_separate/checkpoints_prop_ls')

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
plt.savefig('./VAE_separate/plots/train_val_vs_epochs_prop_ls_NN.pdf')

PATH='./VAE_separate/prop_ls_NN_trained_models/config_num_{}'.format(config_num)
torch.save(last_model.state_dict(), PATH)
model = prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list))
model.load_state_dict(torch.load(PATH))

test_error_2=TEST_prop_mol(model,decoder, dataset_2)
print('second_check_test: {}%'.format(test_error_2))

file = open("VAE_separate/error_log.txt", "w")
file.write("error torch seed {}: ".format(seed) + '{}%'.format(test_error_2))
file.close()

