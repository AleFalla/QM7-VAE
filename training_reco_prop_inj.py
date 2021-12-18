from Train_Test_utils import TRAIN_reco_prop_ls_extra, TEST_prop_mol
import pandas as pd
from separate_models import prop_ls_NN,prop_ls_lin
from Utils import split_train_test, pandas_to_dataset_mol_prop
from VAE import base_VAE
import torch
from matplotlib import pyplot as plt

seed=0
torch.manual_seed(seed)
device='cuda'
device_model=torch.device(device)

df=pd.read_json('./dataset41537.json') #put the name of the json file with the right data
df_test=pd.read_json('./dataset30000_training.json') #put the name of the json file with the right data
df_tr=pd.read_json('./dataset11537_test.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df.columns[df.columns != 'BoB'])[92:104] #comment this if you want all the properties
dataset_1=pandas_to_dataset_mol_prop(df_tr,prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,prop_list) 

latent_size=32#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
model_prop_ls=prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
model_prop_ls_2=prop_ls_lin(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
model_reco=base_VAE(input_dim=input_dim, latent_size=latent_size)

model_prop_ls.to(device_model)
model_prop_ls_2.to(device_model)
model_reco.to(device_model)

property_type='global'

log, last_model_reco, last_model_prop_ls, test_error_1 = TRAIN_reco_prop_ls_extra(model_reco,model_prop_ls,model_prop_ls_2,dataset_1, dataset_2,device='cuda',patience=10,beta=1, config_num=config_num,train_size=10000, batch_size=200, folder_name='VAE_prop_to_ls_NN/checkpoints_reco_prop_ls_{}'.format(property_type))
log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_task_error","val_reco_error","lr"])
log.to_json('VAE_prop_to_ls_NN/log_{}2.json'.format(property_type))

print('first_check_test JPLNN: {}%_{}'.format(test_error_1,property_type))


l=1
a=1
plt.figure('train_and_val')
fig, ax = plt.subplots()
ax.plot(log['train_loss'],linestyle='solid',linewidth=l,label=r'train loss',alpha=a)
ax.plot(log['val_loss'],linestyle='solid',linewidth=l,label=r'validation loss',alpha=a)
leg = ax.legend(prop={'size': 10})
text=leg.get_texts()
for i in range(0,len(text)):
    text[i].set_color('black')
plt.xlabel('epochs/50')
plt.ylabel('loss')
plt.tight_layout()
plt.savefig('./VAE_prop_to_ls_NN/plots/train_val_vs_epochs_reco_prop_ls_NN_{}2.pdf'.format(property_type))

plt.figure('train_and_val')
fig, ax = plt.subplots()
ax.plot(log['val_reco_error'],linestyle='solid',linewidth=l,label=r'val reco error',alpha=a)
ax.plot(log['val_task_error'],linestyle='solid',linewidth=l,label=r'val task error',alpha=a)
leg = ax.legend(prop={'size': 10})
text=leg.get_texts()
for i in range(0,len(text)):
    text[i].set_color('black')
plt.xlabel('epochs/50')
plt.ylabel('validation recostruction error in %')
plt.tight_layout()
plt.savefig('./VAE_prop_to_ls_NN/plots/val_reco_error_prop_ls_NN_{}2.pdf'.format(property_type))

PATH='./VAE_prop_to_ls_NN/reco_prop_ls_NN_trained_models/prop_ls_config_num_{}_{}2'.format(config_num,property_type)
torch.save(last_model_prop_ls.state_dict(), PATH)
model_prop_ls = prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
model_prop_ls.load_state_dict(torch.load(PATH))

PATH='./VAE_prop_to_ls_NN/reco_prop_ls_NN_trained_models/reco_config_num_{}_{}2'.format(config_num,property_type)
torch.save(last_model_reco.state_dict(), PATH)
model_reco = base_VAE(input_dim=input_dim, latent_size=latent_size)
model_reco.load_state_dict(torch.load(PATH))


test_error_2=TEST_prop_mol(model_prop_ls,model_reco, dataset_2,device='cpu')
print('second_check_test: {}%_{}2'.format(test_error_2,property_type))

file = open("VAE_prop_to_ls_NN/test_error_log_{}2.txt".format(property_type), "w")
file.write("test error torch seed {}2: ".format(seed) + '{}%'.format(test_error_2))
file.close()

