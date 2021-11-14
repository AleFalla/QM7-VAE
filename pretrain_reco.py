from Train_Test_utils import reco_pretrain
import pandas as pd
from Utils import split_train_validation, pandas_to_dataset_mol_prop
from VAE import base_VAE
import torch
from matplotlib import pyplot as plt

seed=0
torch.manual_seed(seed)


df=pd.read_json('./dataset41537.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df.columns[df.columns != 'BoB'])[47:93] #comment this if you want all the properties

latent_size=len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
model=base_VAE(input_dim=input_dim,latent_size=latent_size)
train_losses, val_losses, val_error, last_model, test_error_1 = reco_pretrain(model,config_num=config_num, folder_name='pretrained/checkpoints')
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
plt.savefig('./pretrained/plots/train_val_vs_epochs.pdf')

plt.figure('train_and_val')
fig, ax = plt.subplots()
ax.plot(val_error,linestyle='solid',linewidth=l,label=r'val reco error',alpha=a)
leg = ax.legend(prop={'size': 10})
text=leg.get_texts()
for i in range(0,len(text)):
    text[i].set_color('black')
plt.xlabel('epochs/50')
plt.ylabel('validation recostruction error in %')
plt.tight_layout()
plt.savefig('./pretrained/plots/val_error.pdf')

PATH='./pretrained/trained_models/config_num_{}'.format(config_num)
torch.save(last_model.state_dict(), PATH)
model = base_VAE(input_dim=input_dim, latent_size=latent_size)
model.load_state_dict(torch.load(PATH))


file = open("pretrained/test_error_log_reco.txt", "w")
file.write("test error torch seed {}: ".format(seed) + '{}%'.format(test_error_1))
file.close()

