from Train_Test_utils import TRAIN_reco_prop_ls_extra, TRAIN_prop_mol, TEST_reco, TRAIN_reco_prop_ls
import pandas as pd
from Utils import split_train_test, pandas_to_dataset_mol_prop
from VAE import base_VAE,conv_VAE
from separate_models import prop_mol_NN, prop_ls_NN, prop_ls_lin
import torch
from matplotlib import pyplot as plt
import numpy as np
import random 
import itertools

#global properties direct model increasing number of properties
def perc_error(a,b):
    a=torch.Tensor(a).abs()
    b=torch.Tensor(b)
    numer=torch.sum((a-b).abs())
    denom=torch.sum(b.abs())
    return 100*torch.mean(numer/denom)

device=torch.device('cuda')


seed=0
torch.manual_seed(seed)

df=pd.read_json('./dataset41537.json') #put the name of the json file with the right data
df_tr=pd.read_json('./dataset30000_training.json') #put the name of the json file with the right data
df_test=pd.read_json('./dataset11537_test.json') #put the name of the json file with the right data
input_dim=len(df['BoB'][0])
config_num=int(len(df)/41537)
prop_list=df.columns[df.columns != 'BoB']
prop_list=list(prop_list)
prop_list.remove('atom_numbers')
prop_list.remove('positions') 
test_error_direct_globals=[]
test_error_joint_globals=[]
test_error_joint_extra_globals=[]
dataset_error_joint_globals=[]
dataset_error_joint_extra_globals=[]
dataset_error_direct_globals=[]
latent_size=32
properties_left=[]

config_len=0
diff=1

prop_list=['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'DIP', 'HLgap', 'HOMO_0', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']#list(df.columns[df.columns != 'BoB'])[92:104]
lista=[]

for subset in itertools.combinations(prop_list, 5):
    lista.append(subset)

l=10

while config_len!=len(prop_list) and diff!=0:
    property_confs=random.sample(lista, l)
    prova2=np.reshape(property_confs,(10*l))
    uniques,counts=np.unique(prova2, return_counts=True)
    config_len=len(uniques)
    ones=np.ones_like(counts)
    diff=counts-counts[0]*ones
    diff=np.sum(diff)

prop_list_tot=prop_list

print(property_confs)

for i in range(0,l):

    prop_list=list(property_confs[i])
    #prop_list=list(df.columns[df.columns != 'BoB'])[92:104]
    properties_left.append('||'.join(prop_list))#[ele for ele in prop_list_tot if ele not in property_confs[i]][0])
    #del prop_list[i] 
    dataset_1=pandas_to_dataset_mol_prop(df_tr,prop_list)
    dataset_2=pandas_to_dataset_mol_prop(df_test,prop_list)

    #direct model
    model=prop_mol_NN(mol_size=input_dim,prop_size=len(prop_list),extra_size=32-len(prop_list))
    model.to(device)
    log, last_model, test_error_1 = TRAIN_prop_mol(model,dataset_1, dataset_2,device='cuda', config_num=config_num,patience=10, folder_name='main_testing/direct_global_1/checkpoints'.format(i))
    test_error_direct_globals.append(test_error_1.item())
    PATH='./main_testing/direct_global_1/trained_{}'.format(i)
    torch.save(last_model.state_dict(), PATH)
    model = prop_mol_NN(mol_size=input_dim,prop_size=len(prop_list),extra_size=32-len(prop_list))
    model.load_state_dict(torch.load(PATH))
    def tmp(x):
        model.eval()
        with torch.no_grad():
            x=torch.reshape(x,(1,x.size()[0]))
            x_reco,logvar=model(x)
        return x_reco
        
    df['reconstructed']=df[prop_list].apply(lambda x: tmp(torch.Tensor(x))[0].tolist(),axis=1)
    df['error_reco']=df[['reconstructed','BoB']].apply(lambda x: perc_error(*x),axis=1)
    df['error_reco']=df['error_reco'].apply(lambda x: x.item())
    dataset_error_direct_globals.append(list(df['error_reco']))
    
    #extra VAE
    model_prop_ls=prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
    model_prop_ls.to(device)
    model_prop_ls_2=prop_ls_lin(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
    model_prop_ls_2.to(device)
    model_reco=base_VAE(input_dim=input_dim, latent_size=latent_size)
    model_reco.to(device)
    log, last_model_reco, last_model_prop_ls, test_error_2 = TRAIN_reco_prop_ls_extra(model_reco,model_prop_ls,model_prop_ls_2,dataset_1, dataset_2,device='cuda',patience=10, config_num=config_num, folder_name='main_testing/joint_extra_global_1/checkpoints'.format(i))
    test_error_joint_extra_globals.append(test_error_2.item())
    PATH='./main_testing/joint_extra_global_1/vae_trained/trained_{}'.format(i)
    torch.save(last_model_reco.state_dict(), PATH)
    model_0 = base_VAE(input_dim=input_dim, latent_size=latent_size)
    model_0.load_state_dict(torch.load(PATH))
    PATH='./main_testing/joint_extra_global_1/joint_trained/trained_{}'.format(i)
    torch.save(last_model_prop_ls.state_dict(), PATH)
    model = prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
    model.load_state_dict(torch.load(PATH))
    
    def tmp(x):
        model.eval()
        model_0.eval()
        with torch.no_grad():
            x=torch.reshape(x,(1,x.size()[0]))
            u_reco,logvar=model(x)
            x_reco,logvar2=model_0.decode(u_reco)
        return x_reco

    df['reconstructed']=df[prop_list].apply(lambda x: tmp(torch.Tensor(x))[0].tolist(),axis=1)
    df['error_reco']=df[['reconstructed','BoB']].apply(lambda x: perc_error(*x),axis=1)
    df['error_reco']=df['error_reco'].apply(lambda x: x.item())
    dataset_error_joint_extra_globals.append(list(df['error_reco']))

    #regualar VAE
    model_prop_ls=prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
    model_prop_ls.to(device)
    model_reco=base_VAE(input_dim=input_dim, latent_size=latent_size)
    model_reco.to(device)
    log, last_model_reco, last_model_prop_ls, test_error_2 = TRAIN_reco_prop_ls(model_reco,model_prop_ls,dataset_1, dataset_2,device='cuda',patience=10, config_num=config_num, folder_name='main_testing/joint_global_1/checkpoints'.format(i))
    test_error_joint_globals.append(test_error_2.item())
    PATH='./main_testing/joint_global_1/vae_trained/trained_{}'.format(i)
    torch.save(last_model_reco.state_dict(), PATH)
    model_0 = base_VAE(input_dim=input_dim, latent_size=latent_size)
    model_0.load_state_dict(torch.load(PATH))
    PATH='./main_testing/joint_global_1/joint_trained/trained_{}'.format(i)
    torch.save(last_model_prop_ls.state_dict(), PATH)
    model = prop_ls_NN(latent_size=latent_size,prop_size=len(prop_list),extra_size=32-len(prop_list))
    model.load_state_dict(torch.load(PATH))
    
    def tmp(x):
        model.eval()
        model_0.eval()
        with torch.no_grad():
            x=torch.reshape(x,(1,x.size()[0]))
            u_reco,logvar=model(x)
            x_reco,logvar2=model_0.decode(u_reco)
        return x_reco

    df['reconstructed']=df[prop_list].apply(lambda x: tmp(torch.Tensor(x))[0].tolist(),axis=1)
    df['error_reco']=df[['reconstructed','BoB']].apply(lambda x: perc_error(*x),axis=1)
    df['error_reco']=df['error_reco'].apply(lambda x: x.item())
    dataset_error_joint_globals.append(list(df['error_reco']))
    

l=1
a=0.7
#x=[1,3,6,9,12]
plt.figure('train_and_val')
fig, ax = plt.subplots()

ax.plot(test_error_direct_globals,linestyle='solid',linewidth=l,label=r'direct test error',alpha=a)
ax.plot(test_error_joint_globals,linestyle='solid',linewidth=l,label=r'joint test error',alpha=a)
ax.plot(test_error_joint_extra_globals,linestyle='solid',linewidth=l,label=r'joint extra test error',alpha=a)

leg = ax.legend(prop={'size': 10})
text=leg.get_texts()
for i in range(0,len(text)):
    text[i].set_color('black')
plt.xlabel('left out property')
plt.ylabel('test error %')
plt.xticks(range(len(properties_left)), properties_left,rotation='vertical', size='small')
#plt.yscale('log')
plt.tight_layout()
plt.savefig('./main_testing/plots/test_error_directvsjointvsjextra_groups_global.pdf')


numpy_array = np.array(dataset_error_direct_globals)
transpose = numpy_array.T
transpose_list = transpose.tolist()
log=pd.DataFrame(transpose_list, columns=properties_left)
log.to_json('main_testing/log_direct.json')

numpy_array2 = np.array(dataset_error_joint_globals)
transpose2 = numpy_array2.T
transpose_list2 = transpose2.tolist()
log2=pd.DataFrame(transpose_list2, columns=properties_left)
log2.to_json('main_testing/log_joint.json')

numpy_array3 = np.array(dataset_error_joint_extra_globals)
transpose3 = numpy_array3.T
transpose_list3 = transpose3.tolist()
log3=pd.DataFrame(transpose_list3, columns=properties_left)
log3.to_json('main_testing/log_joint_extra.json')


numpy_array = np.array(test_error_direct_globals)
transpose = numpy_array.T
transpose_list = transpose.tolist()
log=pd.DataFrame(transpose_list, columns=properties_left)
log.to_json('main_testing/test_log_direct.json')

numpy_array2 = np.array(test_error_joint_globals)
transpose2 = numpy_array2.T
transpose_list2 = transpose2.tolist()
log2=pd.DataFrame(transpose_list2, columns=properties_left)
log2.to_json('main_testing/test_log_joint.json')

numpy_array3 = np.array(test_error_joint_extra_globals)
transpose3 = numpy_array3.T
transpose_list3 = transpose3.tolist()
log3=pd.DataFrame(transpose_list3, columns=properties_left)
log3.to_json('main_testing/test_log_joint_extra.json')


# plt.figure('train_and_val')
# fig, ax = plt.subplots()
# for i in range(0,len(dataset_error_direct_globals)):
#     ax.plot(dataset_error_direct_globals[i][:],linestyle='solid',linewidth=l,label='error on dataset {} props'.format(i*3),alpha=a)
# leg = ax.legend(prop={'size': 10})
# text=leg.get_texts()
# for i in range(0,len(text)):
#     text[i].set_color('black')
# plt.xlabel('molecule index')
# plt.ylabel('loss')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('./main_testing/plots/test_error_dataset_direct_global.pdf')



# plt.figure('train_and_val')
# fig, ax = plt.subplots()
# for i in range(0,len(dataset_error_joint_globals)):
#     ax.plot(dataset_error_joint_globals[i][:],linestyle='solid',linewidth=l,label='error on dataset {} props'.format(i*3),alpha=a)
# leg = ax.legend(prop={'size': 10})
# text=leg.get_texts()
# for i in range(0,len(text)):
#     text[i].set_color('black')
# plt.xlabel('molecule index')
# plt.ylabel('loss')
# plt.yscale('log')
# plt.tight_layout()
# plt.savefig('./main_testing/plots/test_error_dataset_joint_global.pdf')

# plt.figure('histos')
# fig, ax = plt.subplots()
# for i in range(0,len(dataset_error_joint_globals)):
#     ax.hist(dataset_error_direct_globals[i][:],bins=100,label='error on dataset {} props'.format(i*3),alpha=a)
# leg = ax.legend(prop={'size': 10})
# text=leg.get_texts()
# for i in range(0,len(text)):
#     text[i].set_color('black')
# plt.xlabel('molecule index')
# plt.ylabel('loss')
# plt.tight_layout()
# plt.savefig('./main_testing/plots/hist_test_error_dataset_direct_global.pdf')

# plt.figure('histos')
# fig, ax = plt.subplots()
# for i in range(0,len(dataset_error_joint_globals)):
#     ax.hist(dataset_error_joint_globals[i][:],bins=100,label='error on dataset {} props'.format(i*3),alpha=a)
# leg = ax.legend(prop={'size': 10})
# text=leg.get_texts()
# for i in range(0,len(text)):
#     text[i].set_color('black')
# plt.xlabel('molecule index')
# plt.ylabel('loss')
# plt.tight_layout()
# plt.savefig('./main_testing/plots/hist_test_error_dataset_joint_global.pdf')


    
