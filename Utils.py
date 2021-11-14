import torch 
from torch import nn
from schnetpack import AtomsData, AtomsLoader
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from qml.representations import *

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def split(data,train,test):
    """ 
        Splits dataset in two (can be test or validation)

     """
    train_set,test_set=torch.utils.data.random_split(data,[train,test])
    return train_set, test_set

def perc_error(a,b):
    numer=torch.sum((a-b).abs(),dim=1)
    denom=torch.sum(b.abs(),dim=1)
    return 100*torch.mean(numer/denom)

def get_unpacked_properties(dizionario,data):
    props_list=data.available_properties
    
    lista=data.available_properties.copy()
    for key in lista:
        k=0
        tmp=list(dizionario[key].size())
        print(key)
        if len(tmp)==3 and tmp[1]!=1:
            props_list.remove(key)
            for i in range(0,tmp[1]):
                dizionario[key+'_{}'.format(i)]=dizionario[key][:,i]
                props_list.insert(k,key+'_{}'.format(i))
                k=k+1
            del dizionario[key]
        
    return props_list

def normalization(dizionario,props_list):
    for key in props_list:
        tmp_avg=dizionario[key].mean().item()*torch.ones_like(dizionario[key])
        tmp_std=dizionario[key].std().item()*torch.ones_like(dizionario[key])
        dizionario[key]=(dizionario[key]-tmp_avg).div(tmp_std)
    return dizionario

def data_pre_preparation(data,shuffle=False,normalize=False,save_to_file=True):
    
    loader=AtomsLoader(data,batch_size=int(data.__len__()),shuffle=shuffle)
    dizionario=next(iter(loader))
    props_list=get_unpacked_properties(dizionario,data)
    
    if normalize==True:
        dizionario=normalization(dizionario, props_list)
        
    t=()
    for key in props_list:
        t=t+(dizionario[key],)
    
    df=pd.DataFrame(torch.cat(t,1).tolist(), columns = props_list)
    df['atom_numbers']=dizionario['_atomic_numbers'].tolist()
    df['positions']=dizionario['_positions'].tolist()
    df['#C']=df['atom_numbers'].apply(lambda x: np.count_nonzero(np.array(x)==6))
    df['#H']=df['atom_numbers'].apply(lambda x: np.count_nonzero(np.array(x)==1))
    df['#N']=df['atom_numbers'].apply(lambda x: np.count_nonzero(np.array(x)==7))
    df['#O']=df['atom_numbers'].apply(lambda x: np.count_nonzero(np.array(x)==8))
    df['#S']=df['atom_numbers'].apply(lambda x: np.count_nonzero(np.array(x)==16))
    df['#Cl']=df['atom_numbers'].apply(lambda x: np.count_nonzero(np.array(x)==17))
    max_asize={'C': df['#C'].max(), 'H': df['#H'].max(), 'N': df['#N'].max(), 'O': df['#O'].max(), 'S': df['#S'].max(),'Cl':df['#Cl'].max()}
    max_size=list(dizionario['_positions'].size())[1]
    df['BoB'] = df['atom_numbers'].combine(df['positions'], lambda x,y: generate_bob(x, y, x,size=max_size,asize=max_asize))
    #df=df.drop(columns=['atom_numbers','positions','#C','#H','#N','#O','#S','#Cl'])
    props_list.insert(0,'BoB')
    #df=df[props_list]
    if save_to_file==True:
        df.to_json('./dataset{}.json'.format(len(df)))
    return df

def data_preparation_latent_space(model,df,ID='',save_to_file=True):
    
    def tmp(x):
        z,null=model.encode(torch.Tensor(x))
        return z

    df['latent_rep'] = df['BoB'].apply(lambda x: tmp(x).tolist())
    df['latent_rep']=df['latent_rep'].apply(lambda x: x[0])

    if save_to_file==True:
        df.to_json('./dataset{}_ls_{}.json'.format(len(df),ID))

    return df

def split_train_test(df,config_num=1,save_to_file=True):
    len_group = config_num
    index_list = np.array(df.index)
    np.random.shuffle(np.reshape(index_list, (-1, len_group)))
    shuffled_df = df.loc[index_list, :]
    df=shuffled_df
    if save_to_file==True:
        df[0:config_num*30000].to_json('./dataset{}_training.json'.format(len(df[0:config_num*30000])))
        df[config_num*30000:].to_json('./dataset{}_test.json'.format(len(df[config_num*30000:])))
        
    return df[0:config_num*30000], df[config_num*30000:]

def pandas_to_dataset_mol_prop(df,property_list=['Eat','HLGAP','POL','C6']):
    
    dati= torch.Tensor(list(df['BoB']))
    labels = df [property_list]
    labels=torch.Tensor(labels.values)
    dataset = TensorDataset(dati,labels)
    return dataset

def pandas_to_dataset_ls_prop(df,property_list=['Eat','HLGAP','POL','C6']):
    
    dati= torch.Tensor(list(df['latent_rep']))
    labels = df [property_list]
    labels=torch.Tensor(labels.values)
    dataset = TensorDataset(dati,labels)
    return dataset
    
    


