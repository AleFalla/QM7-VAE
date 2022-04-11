from Train_Test_utils import TEST_prop_mol,TRAIN_reco_prop_ls
import pandas as pd
from separate_models import prop_ls_NN,prop_ls_NN_conv,prior
from Utils import split_train_test, pandas_to_dataset_mol_prop, generate_hCHG_rep, generate_MCM_rep, generate_adj_rep, generate_distance, generate_pol_rep
from VAE import base_VAE, toy_VAE, multi_input_VAE
import torch
from matplotlib import pyplot as plt
import numpy as np

seed=0
torch.manual_seed(seed)
device='cuda'
device_model=torch.device(device)

prop_list=['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'DIP', 'HLgap', 'HOMO_0', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#list(df.columns[df.columns != 'BoB'])[92:104]

def add_channel(x,y):
    image=[]
    s=np.shape(x)
    if len(s)<3:
        x=np.array(x).tolist()
        image.append(x)
    else:
        for i in range(0,s[0]):
            x=np.array(x)
            temp=x[i,:,:].tolist()
            image.append(temp)
    y=np.array(y)
    image.append(y.tolist())
    return image

df=pd.read_json('./dataset41537.json')#,usecols=fields)

hchg_mean=np.array([-1.30452077e-01, -3.13652837e-02, -3.64105838e-02, -5.26009989e-02,
       -6.25012193e-02, -7.20938399e-02, -1.34306107e-01,  6.30085101e-02,
        4.80373380e-02,  4.29486698e-02,  4.94381126e-02,  4.69107689e-02,
        4.92020861e-02,  4.77631522e-02,  4.60221417e-02,  3.94977769e-02,
        3.61250470e-02,  2.39005614e-02,  1.60888580e-02,  7.35463931e-03,
        2.68671305e-03,  6.59031682e-04,  8.17013022e-05])
        
hchg_std=np.array([0.06198044, 0.07543253, 0.07819971, 0.0932278 , 0.09316372,
       0.10044874, 0.09325014, 0.03791502, 0.02286283, 0.01560529,
       0.02318611, 0.02179252, 0.02702059, 0.02994096, 0.03412312,
       0.03718375, 0.04120263, 0.03730023, 0.03272621, 0.02271441,
       0.01340543, 0.00601317, 0.00171577])

pols_mean=np.array([8.69499105e+00, 9.67811470e+00, 9.65355178e+00, 9.25347047e+00,
       9.07288262e+00, 8.65906654e+00, 6.93069061e+00, 1.94862821e+00,
       1.84595878e+00, 1.80018947e+00, 1.73469629e+00, 1.75186887e+00,
       1.72909771e+00, 1.68147756e+00, 1.57730673e+00, 1.33835544e+00,
       1.14638218e+00, 7.61490485e-01, 5.13367333e-01, 2.47443651e-01,
       9.94026886e-02, 2.76110714e-02, 4.17295055e-03])
pols_std=np.array([2.08777737, 2.03262841, 2.04211536, 2.33677986, 2.36091205,
       2.27567909, 2.86002347, 0.18598961, 0.14090493, 0.15903706,
       0.24408032, 0.301619  , 0.41276242, 0.54822806, 0.70660885,
       0.8744611 , 0.95829958, 0.9513949 , 0.86160348, 0.65099468,
       0.42725207, 0.2276736 , 0.08764652])

atC6_mean=np.array([2.85804153e+01, 3.34035354e+01, 3.36631652e+01, 3.21080136e+01,
       3.12243338e+01, 2.91289043e+01, 2.20243168e+01, 1.49803671e+00,
       1.43972139e+00, 1.40187980e+00, 1.36544322e+00, 1.38908553e+00,
       1.37101194e+00, 1.33138548e+00, 1.24943975e+00, 1.05660397e+00,
       9.04636259e-01, 5.97483047e-01, 4.01752292e-01, 1.92792652e-01,
       7.72697901e-02, 2.14511214e-02, 3.24158402e-03])
atC6_std=np.array([ 8.38679854,  9.06545713, 10.1408944 , 11.44838   , 10.53001202,
        9.8227754 , 11.62598769,  0.20242199,  0.13932785,  0.15378311,
        0.22050933,  0.2753441 ,  0.35080388,  0.45044368,  0.57628183,
        0.70005932,  0.76411232,  0.75067573,  0.67668362,  0.50826658,
        0.33245806,  0.17698761,  0.06809436])

hchg=['hCHG_0','hCHG_1','hCHG_2','hCHG_3','hCHG_4','hCHG_5','hCHG_6','hCHG_7','hCHG_8','hCHG_9','hCHG_10','hCHG_11','hCHG_12','hCHG_13','hCHG_14','hCHG_15','hCHG_16','hCHG_17','hCHG_18','hCHG_19','hCHG_20','hCHG_21','hCHG_22']
pols=['atPOL_0','atPOL_1','atPOL_2','atPOL_3','atPOL_4','atPOL_5','atPOL_6','atPOL_7','atPOL_8','atPOL_9','atPOL_10','atPOL_11','atPOL_12','atPOL_13','atPOL_14','atPOL_15','atPOL_16','atPOL_17','atPOL_18','atPOL_19','atPOL_20','atPOL_21','atPOL_22']
atC6_list=['atC6_0','atC6_1','atC6_2','atC6_3','atC6_4','atC6_5','atC6_6','atC6_7','atC6_8','atC6_9','atC6_10','atC6_11','atC6_12','atC6_13','atC6_14','atC6_15','atC6_16','atC6_17','atC6_18','atC6_19','atC6_20','atC6_21','atC6_22']

#df[hchg]=df[hchg]*hchg_std+hchg_mean
df[pols]=df[pols]*pols_std+pols_mean
df[atC6_list]=df[atC6_list]*atC6_std+atC6_mean


el_list=['#S','#Cl','#O','#N','#C','#H']
at_nums=[16,17,8,7,6,1]
#df=generate_distance(df,el_list,at_nums)
df=generate_MCM_rep(df,el_list,at_nums)
df=generate_hCHG_rep(df,el_list,at_nums)
df=generate_adj_rep(df,el_list,at_nums)
#df=generate_pol_rep(df,el_list=el_list,at_nums=at_nums)
df['BoB']=df['CM'].apply(lambda x: np.reshape(x,(1,len(x),len(x))))##.combine(df['hCHG_mat'],lambda x,y: add_channel(x,y))#.apply(lambda x: add_channel(x))#.apply(lambda x: np.reshape(x,(1,len(x),len(x))))#
print('fatto')

df['BoB']=df['BoB'].combine(df['hCHG_mat'],lambda x,y: add_channel(x,y))#.apply(lambda x: add_channel(x))#.apply(lambda x: np.reshape(x,(1,len(x),len(x))))#
print('fatto')
#df['adj_mat']=df['adj_mat'].combine(df['CM'],lambda x,y: np.multiply(x,y))#.apply(lambda x: add_channel(x))#.apply(lambda x: np.reshape(x,(1,len(x),len(x))))#
#print('fatto')
#df['BoB']=df['BoB'].combine(df['adj_mat'],lambda x,y: add_channel(x,y))#.apply(lambda x: add_channel(x))#.apply(lambda x: np.reshape(x,(1,len(x),len(x))))#
#print('fatto')

#df['BoB']=df['BoB'].combine(df['adj_mat'],lambda x,y: add_channel(x,y))
#print('fatto')
# df['BoB']=df['BoB'].combine(df['distance_mat'],lambda x,y: add_channel(x,y))#df['distance_mat'].apply(lambda x: add_channel(x))
# print('fatto')

config_num=1

df_tr,df_test=split_train_test(df,config_num=config_num,save_to_file=False)
prop_list=['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'HLgap', 'HOMO_0']#,'DIP', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#['atPOL_0', 'atPOL_1', 'atPOL_2', 'atPOL_3', 'atPOL_4', 'atPOL_5', 'atPOL_6', 'atPOL_7', 'atPOL_8', 'atPOL_9', 'atPOL_10', 'atPOL_11', 'atPOL_12', 'atPOL_13', 'atPOL_14', 'atPOL_15']#, 'atPOL_16', 'atPOL_17', 'atPOL_18', 'atPOL_19', 'atPOL_20', 'atPOL_21', 'atPOL_22']#
df_tr=df_tr[prop_list+['BoB']]
df_test=df_test[df_tr.columns]
latent_size=32#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
print(input_dim)
print(df['BoB'][0])


dataset_1=pandas_to_dataset_mol_prop(df_tr,rep='BoB',property_list=prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,rep='BoB',property_list=prop_list) 

#latent_size=9#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=528#len(df['BoB'][0])
print('*********:',input_dim)
model_prop_ls=prop_ls_NN_conv(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
model_prop_ls_2=prior(latent_size=latent_size,prop_size=len(prop_list))#,extra_size=32-len(prop_list))#,coeff=coeff)
model_reco=multi_input_VAE(input_dim=input_dim, latent_size=latent_size)#toy_VAE(imgChannels=2,zDim=latent_size)#input_dim=input_dim, latent_size=latent_size) #
model_prop_ls.to(device_model)
model_prop_ls_2.to(device_model)
model_reco.to(device_model)


#for i in range(0,5):
log, model_reco, model_prop_ls, test_error_1 = TRAIN_reco_prop_ls(model_reco,model_prop_ls,dataset_1, dataset_2,lr_conv=1e-7,device=device,patience=10,beta=1,alpha=1,reset=True,config_num=config_num,train_size=10000, batch_size=500, folder_name='VAE_prop_to_ls_NN/checkpoints_reco_prop_ls_{}'.format(property_type))
log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_task_error","val_reco_error","lr"])
#log.to_json('VAE_prop_to_ls_NN/log_{}2.json'.format(property_type))
print('***',test_error_1.item(),'***')

# log, last_model_reco, last_model_prop_ls, test_error_1 = TRAIN_reco_prop_ls(model_reco,model_prop_ls,dataset_1, dataset_2,lr_conv=1e-7,device=device,patience=10,beta=1,reset=False,config_num=config_num,train_size=10000, batch_size=500, folder_name='VAE_prop_to_ls_NN/checkpoints_reco_prop_ls_{}'.format(property_type))
# log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_task_error","val_reco_error","lr"])
#log.to_json('VAE_prop_to_ls_NN/log_{}2.json'.format(property_type))

print('first_check_test JPLNN: {}%_{}'.format(test_error_1.item(),property_type))


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

