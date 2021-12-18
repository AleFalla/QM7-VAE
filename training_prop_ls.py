from Train_Test_utils import TRAIN_prop_ls, TEST_prop_mol
import pandas as pd
from Utils import pandas_to_dataset_ls_prop, pandas_to_dataset_mol_prop
from VAE import base_VAE
from separate_models import prop_ls_lin, prop_ls_NN2
import torch
from matplotlib import pyplot as plt

seed=0
torch.manual_seed(seed)
def perc_error(a,b):
    a=torch.Tensor(a).abs()
    b=torch.Tensor(b)
    numer=torch.sum((a-b).abs())
    denom=torch.sum(b.abs())
    return 100*torch.mean(numer/denom)

df=pd.read_json('./dataset41537.json') #put the name of the json file with the right data
df_tr=pd.read_json('./dataset30000_training.json') #put the name of the json file with the right data
df_test=pd.read_json('./dataset11537_test.json') #put the name of the json file with the right data
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
prop_list=list(df.columns[df.columns != 'BoB'])[46:69] #comment this if you want all the properties
latent_size=32#len(prop_list) #here the latent size is set to the lenght of the property list but this is not necessary. It would be nice for invertibility of linear mappings
input_dim=len(df['BoB'][0])
PATH_0='./main_testing/joint_local_2/vae_trained/trained'#./VAE_prop_to_ls_NN/reco_prop_ls_NN_trained_models/reco_config_num_{}_global'.format(config_num)#'./VAE_separate/onlyreco_trained_models/config_num_{}_50'.format(config_num) #
#./main_testing/joint_local_2/vae_trained/trained
decoder = base_VAE(input_dim=input_dim, latent_size=latent_size)
#device=torch.device('cpu')
decoder.load_state_dict(torch.load(PATH_0,map_location=torch.device('cpu')))

# def tmp2(x):

#         decoder.eval()

#         with torch.no_grad():
            
#             x=torch.reshape(x,(1,x.size()[0]))
#             mu,logvar=decoder.encode(x)
#             x_reco,logvar=decoder.decode(mu)

#         return x_reco.tolist()

# df['reconstructed']=df['BoB'].apply(lambda x: tmp2(torch.Tensor(x)))
# df['error_reco']=df[['reconstructed','BoB']].apply(lambda x: perc_error(*x),axis=1)
# df['error_reco']=df['error_reco'].apply(lambda x: x.item())
# print(df['error_reco'].mean())

def tmp(x):

        decoder.eval()

        with torch.no_grad():
            #x=torch.reshape(x,(1,x.size()[0]))
            mu_reco,logvar=decoder.encode(x)

        return mu_reco.tolist()

df_tr['latent_rep']=df_tr['BoB'].apply(lambda x: tmp(torch.Tensor(x))[0])
df_test['latent_rep']=df_test['BoB'].apply(lambda x: tmp(torch.Tensor(x))[0])
#print(df_tr['latent_rep'])
#print(df_test['latent_rep'])
dataset_1=pandas_to_dataset_ls_prop(df_tr,prop_list)
dataset_2=pandas_to_dataset_mol_prop(df_test,prop_list) 
#exit()


model=prop_ls_NN2(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
#PATH_0='./main_testing/joint_local_2/joint_trained/trained'#./VAE_prop_to_ls_NN/reco_prop_ls_NN_trained_models/reco_config_num_{}_global'.format(config_num)#'./VAE_separate/onlyreco_trained_models/config_num_{}_50'.format(config_num) #
#model.load_state_dict(torch.load(PATH_0,map_location=torch.device('cpu')))
#here put whichever decoder you want


log, last_model, test_error_1 = TRAIN_prop_ls(model,decoder,dataset_1, dataset_2, reset=True,config_num=config_num,patience=10, folder_name='VAE_separate/checkpoints_prop_ls')

def tmp(x):
        last_model.eval()
        decoder.eval()
        with torch.no_grad():
            #x=torch.reshape(x,(1,x.size()[0]))
            
            u_reco,logvar=model(x)
            x_reco,logvar2=model_0.decode(u_reco)
        return list(x_reco)

df['reconstructed']=df[prop_list].apply(lambda x: tmp(torch.Tensor(x))[0].tolist(),axis=1)
df['error_reco']=df[['reconstructed','BoB']].apply(lambda x: perc_error(*x),axis=1)
df['error_reco']=df['error_reco'].apply(lambda x: x.item())
df[['BoB','reconstructed','error_reco']].to_json('VAE_separate/log_joint_atPOL_refined.json')

# log=pd.DataFrame(log, columns=["epoch", "i", "train_loss","val_loss", "val_error","lr"])
# log.to_json('VAE_separate/log_{}.json'.format('prop_ls'))

# print('first_check_test: {}%'.format(test_error_1))

# l=1
# a=1
# plt.figure('train_and_val')
# fig, ax = plt.subplots()
# ax.plot(log['train_loss'],linestyle='solid',linewidth=l,label=r'train loss',alpha=a)
# ax.plot(log['val_loss'],linestyle='solid',linewidth=l,label=r'validation loss',alpha=a)
# leg = ax.legend(prop={'size': 10})
# text=leg.get_texts()
# for i in range(0,len(text)):
#     text[i].set_color('black')
# plt.xlabel('epochs/50')
# plt.ylabel('loss')
# plt.tight_layout()
# plt.savefig('./VAE_separate/plots/train_val_vs_epochs_prop_ls_NN.pdf')


# plt.figure('train_and_val')
# fig, ax = plt.subplots()
# ax.plot(log['val_error'],linestyle='solid',linewidth=l,label=r'val reco error',alpha=a)
# leg = ax.legend(prop={'size': 10})
# text=leg.get_texts()
# for i in range(0,len(text)):
#     text[i].set_color('black')
# plt.xlabel('epochs/50')
# plt.ylabel('validation recostruction error in %')
# plt.tight_layout()
# plt.savefig('./VAE_separate/plots/train_val_vs_epochs_prop_ls_NN.pdf')

# PATH='./VAE_separate/prop_ls_NN_trained_models/config_num_{}'.format(config_num)
# torch.save(last_model.state_dict(), PATH)
# model = prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list))
# model.load_state_dict(torch.load(PATH))

# test_error_2=TEST_prop_mol(model,decoder, dataset_2)
# print('second_check_test: {}%'.format(test_error_2))

# file = open("VAE_separate/test_error_prop_ls_mol_log.txt", "w")
# file.write("test error torch seed {}: ".format(seed) + '{}%'.format(test_error_2))
# file.close()

