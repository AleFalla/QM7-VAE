from Train_Test_utils import TRAIN_reco_special, TEST_reco
import pandas as pd
from Utils import split_train_test, pandas_to_dataset_mol_prop
from VAE import special_VAE
import torch
from matplotlib import pyplot as plt

seed=0
torch.manual_seed(seed)


df=pd.read_json('./dataset41537.json') #put the name of the json file with the right data
df_tr=pd.read_json('./dataset30000_training.json') #put the name of the json file with the right data
df_test=pd.read_json('./dataset11537_test.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df.columns[df.columns != 'BoB'])[93:104] #comment this if you want all the properties
dataset_1=pandas_to_dataset_mol_prop(df_tr,prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,prop_list)

property_type=32

latent_size=32 #len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
model=special_VAE(input_dim=input_dim,latent_size=latent_size)
log, last_model, test_error_1 = TRAIN_reco_special(model,dataset_1, dataset_2,learning_rate=1e-3, config_num=config_num,batch_size=500,patience=10, folder_name='special_VAE/checkpoints_reco')
log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_error","lr"])
log.to_json('special_VAE/log_{}.json'.format(property_type))

print('first_check_test: {}%'.format(test_error_1))

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
plt.savefig('./special_VAE/plots/train_val_vs_epochs_reco_50.pdf')

plt.figure('train_and_val')
fig, ax = plt.subplots()
ax.plot(log['val_error'],linestyle='solid',linewidth=l,label=r'val reco error',alpha=a)
leg = ax.legend(prop={'size': 10})
text=leg.get_texts()
for i in range(0,len(text)):
    text[i].set_color('black')
plt.xlabel('epochs/50')
plt.ylabel('validation recostruction error in %')
plt.tight_layout()
plt.savefig('./special_VAE/plots/val_reco_error_50.pdf')

PATH='./special_VAE/onlyreco_trained_models/config_num_{}_50'.format(config_num)
torch.save(last_model.state_dict(), PATH)
model = special_VAE(input_dim=input_dim, latent_size=latent_size)
model.load_state_dict(torch.load(PATH))


test_error_2=TEST_reco(model, dataset_2)
print('second_check_test: {}%'.format(test_error_2))


file = open("special_VAE/test_error_log_reco_32.txt", "w")
file.write("test error torch seed {}: ".format(seed) + '{}%'.format(test_error_2))
file.close()

