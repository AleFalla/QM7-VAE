from Train_Test_utils import TRAIN_prop_mol, TEST_prop_mol_direct
import pandas as pd
from Utils import pandas_to_dataset_mol_prop
from separate_models import prop_mol_NN
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
prop_list=list(df.columns[df.columns != 'BoB'])[47:93] #comment this if you want all the properties
dataset_1=pandas_to_dataset_mol_prop(df_tr,prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,prop_list)

input_dim=len(df['BoB'][0])
model=prop_mol_NN(mol_size=input_dim,prop_size=len(prop_list))
train_losses, val_losses, val_error, last_model, test_error_1 = TRAIN_prop_mol(model,dataset_1, dataset_2, config_num=config_num, folder_name='direct_model/checkpoints')
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
plt.savefig('./direct_model/plots/train_val_vs_epochs.pdf')

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
plt.savefig('./direct_model/plots/val_error.pdf')

PATH='./direct_model/trained_models/config_num_{}'.format(config_num)
torch.save(last_model.state_dict(), PATH)
model = prop_mol_NN(mol_size=input_dim,prop_size=len(prop_list))
model.load_state_dict(torch.load(PATH))


test_error_2=TEST_prop_mol_direct(model, dataset_2)
print('second_check_test: {}%'.format(test_error_2))


file = open("direct_model/test_error_log_reco.txt", "w")
file.write("test_error torch seed {}: ".format(seed) + '{}%'.format(test_error_2))
file.close()

