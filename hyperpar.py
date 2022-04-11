from Train_Test_utils import TRAIN_reco_prop_ls_extra, TEST_prop_mol,TRAIN_reco_prop_ls
import pandas as pd
from separate_models import prop_ls_NN,prop_ls_lin,prop_ls_lin_coeff
from Utils import split_train_test, pandas_to_dataset_mol_prop, generate_hCHG_rep, generate_MCM_rep
from VAE import base_VAE, toy_VAE
import torch
from matplotlib import pyplot as plt
import numpy as np

seed=0
torch.manual_seed(seed)
device='cuda'
device_model=torch.device(device)

# df=pd.read_parquet('./dataset41537.parquet') #put the name of the json file with the right data
# df_test=pd.read_parquet('./dataset11537_test.parquet')  #put the name of the json file with the right data
# df_tr=pd.read_parquet('./dataset30000_training.parquet')#put the name of the json file with the right data
# config_num=int(len(df)/41537)
# prop_list=df.columns[df.columns != 'BoB']
# prop_list=list(prop_list)
# prop_list.remove('atom_numbers')
# prop_list.remove('positions') 
# prop_list=list(df.columns[df.columns != 'BoB'])[92:104] #comment this if you want all the properties
# prop_list=['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'DIP', 'HLgap', 'HOMO_0', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#list(df.columns[df.columns != 'BoB'])[92:104]

def get_image_layered(x,y):
    image=[]
    image.append(x)
    image.append(y)
    image=np.array(image)
    image=np.reshape(image,(2,len(x),len(x)))
    return image


df=pd.read_json('./dataset41537.json')#,usecols=fields)
el_list=['#S','#Cl','#O','#N','#C','#H']
at_nums=[16,17,8,7,6,1]
df=generate_MCM_rep(df,el_list,at_nums)
df=generate_hCHG_rep(df,el_list,at_nums)
df['BoB']=df['CM'].combine(df['hCHG_mat'],lambda x,y: get_image_layered(x,y))

config_num=1
# df=df[abs(df['HOMO_0'])<=5]
# df=df[abs(df['HOMO_1'])<=5]
# print(len(df))
# df['HOMO_0']=(df['HOMO_0']-df['HOMO_0'].mean())/df['HOMO_0'].std()
# df['HOMO_1']=(df['HOMO_1']-df['HOMO_1'].mean())/df['HOMO_1'].std()
#df=df.drop([0],axis=0)
df_tr,df_test=split_train_test(df,config_num=config_num,save_to_file=False)
prop_list=['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'HLgap', 'HOMO_0','DIP', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#['atPOL_0', 'atPOL_1', 'atPOL_2', 'atPOL_3', 'atPOL_4', 'atPOL_5', 'atPOL_6', 'atPOL_7', 'atPOL_8', 'atPOL_9', 'atPOL_10', 'atPOL_11', 'atPOL_12', 'atPOL_13', 'atPOL_14', 'atPOL_15']#, 'atPOL_16', 'atPOL_17', 'atPOL_18', 'atPOL_19', 'atPOL_20', 'atPOL_21', 'atPOL_22']#
#prop_list=['eXX','eKIN','eNN','eAT','HOMO_0','HOMO_1']
#temp=prop_list.append('BoB')
df_tr=df_tr[prop_list+['BoB']]
df_test=df_test[df_tr.columns]
latent_size=32#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])


dataset_1=pandas_to_dataset_mol_prop(df_tr,rep='BoB',property_list=prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,rep='BoB',property_list=prop_list) 

#latents=[8,12,16,24,32]
betas=[0.5,1,1.5,2,2.5]
alphas=[0.5,1,1.5,2,2.5]#[0.1,0.3,0.5,0.7,1]
test_error_glob=[]
latent_size=32
model_0 = toy_VAE()#input_dim=input_dim, latent_size=latent_size)
model = prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
print(32-len(prop_list))
model_prop_ls_2=prop_ls_lin(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))

device='cuda'
device_model=torch.device(device)
model.to(device_model)
model_0.to(device_model)
model_prop_ls_2.to(device_model)

for alpha in alphas:
    seed=0
    torch.manual_seed(seed)
    
    test_error=[]
    for beta in betas:
        property_type='global'
        log, model_0, model, test_error_1 = TRAIN_reco_prop_ls(model_0,model, dataset_1, dataset_2,beta=beta,alpha=alpha,lr_conv=1e-7,device=device,patience=10,reset=True, config_num=config_num,train_size=10000, batch_size=500, folder_name='VAE_prop_to_ls_NN/checkpoints_reco_prop_ls_{}'.format(property_type))
        log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_task_error","val_reco_error","lr"])
        #log.to_json('VAE_prop_to_ls_NN/log_{}2.json'.format(property_type))
        #print('check_test JPLNN: {}%_{}'.format(test_error_1,property_type))
        test_error.append(test_error_1.item())
    
    test_error_glob.append(test_error)




l=1
a=0.7
plt.figure('train_and_val')
plt.contourf(np.array(alphas), np.array(betas),np.log(np.array(test_error_glob,dtype=float)),cmap='RdGy')
plt.colorbar()
plt.xlabel('alphas')
plt.ylabel('betas')
plt.tight_layout()
plt.savefig('./hyperpar.pdf')



# log, model_0, model, test_error_1 = TRAIN_reco_prop_ls_extra(model_0,model,model_prop_ls_2,dataset_1, dataset_2,lr_conv=1e-5,device=device,patience=10,reset=True, config_num=config_num,train_size=10000, batch_size=500, folder_name='VAE_prop_to_ls_NN/checkpoints_reco_prop_ls_{}'.format(property_type))
# log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_task_error","val_reco_error","lr"])
# log.to_json('./log_{}2.json'.format(property_type))
# test_error.append(test_error_1.item())
# print('final_check_test JPLNN: {}%_{}'.format(test_error_1,property_type))
