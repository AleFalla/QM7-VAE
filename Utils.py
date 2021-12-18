import torch 
from torch import nn
from schnetpack import AtomsData, AtomsLoader
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from qml.representations import *
import math

#more used

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def perc_error(a,b):
    """ 
    :calculates relative error % (using L1 norm):
    
    """
    numer=torch.sum((a-b).abs(),dim=1)
    denom=torch.sum(b.abs(),dim=1)
    return 100*torch.mean(numer/denom)

def get_unpacked_properties(dizionario,data):
    """ 
    :linearizes vector properties into single dataset entries:
    :e.g. at_POL-> atPOL_0, at_POL_1, ... atPOL_N:
    
    """
    props_list=data.available_properties
    
    lista=data.available_properties.copy()
    for key in lista:
        k=0
        tmp=list(dizionario[key].size())
        #print(key)
        if len(tmp)>=2 and tmp[1]!=1:
            props_list.remove(key)
            for i in range(0,tmp[1]):
                dizionario[key+'_{}'.format(i)]=dizionario[key][:,i]
                if len(tmp)==2:
                    dizionario[key+'_{}'.format(i)]=torch.reshape(dizionario[key+'_{}'.format(i)],(dizionario[key+'_{}'.format(i)].size()[0],1))
                props_list.insert(k,key+'_{}'.format(i))
                k=k+1
            del dizionario[key]
        #k=k+1
    return props_list

def normalization(dizionario,props_list):

    for key in props_list:
        tmp_avg=dizionario[key].mean().item()*torch.ones_like(dizionario[key])
        tmp_std=dizionario[key].std().item()*torch.ones_like(dizionario[key])
        dizionario[key]=(dizionario[key]-tmp_avg).div(tmp_std)
    return dizionario

def ave_dist(Z, pos):
    
    """ 
    :calculates the distance between the heavy atoms:

    """
  dist = []

  for ii in range(len(Z)):
    if Z[ii] != 1:
      for jj in range(ii+1, len(Z)):
        if Z[jj] != 1:
          a = pos[ii]-pos[jj]
          dist.append(math.sqrt(np.dot(a,a)))
  if len(dist) == 0:
    return 0.0
  else:
    return np.amax(dist)

def newKSE(KSE, Egap, iorbs):
    """
    :param KSE:
    :param Egap:
    :param iorbs:
    :return:
    """
    kslen = len(KSE)
    for k in range(0,kslen-1):
        dE = abs(KSE[k+1]-KSE[k])
        if abs(float(dE) - float(Egap)) <= 0.01:
            HOMOs=[]
            LUMOs = []
            for ii in range(iorbs):
                HOMOs.append(KSE[k-ii])
                LUMOs.append(KSE[k+ii+1])
            break

    return np.sort(np.concatenate((HOMOs, LUMOs), axis=None))

def data_pre_preparation(data,shuffle=False,normalize=True,save_to_file=True):
    """ 
    :turns the db file to pandas dataset:

    """

    loader=AtomsLoader(data,batch_size=int(data.__len__()),shuffle=shuffle)
    dizionario=next(iter(loader))
    len_kse=dizionario['KSE'].size()[1]
    lista=[]
    
    #this is specific for kse eigenvalues, comment if not needed
    for i in range(0,len_kse):
        lista.append('KSE_{}'.format(i))        
    
    props_list=get_unpacked_properties(dizionario,data)
        
    #if normalize==True: #more efficient normalization, available if not using post processing operations
    #    dizionario=normalization(dizionario, props_list)
    
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
    
    #kse, distance and hlgap specific, comment if not needed
    df['KSE']=df[lista].values.tolist()
    df['HLeigs'] = df['KSE'].combine(df['HLgap'], lambda x,y: newKSE(x,y,iorbs=3))
    for i in range (0,3):
        df['HOMO_{}'.format(i)]=df['HLeigs'].apply(lambda x: x[i])
        df['LUMO_{}'.format(i)]=df['HLeigs'].apply(lambda x: x[i+3])
    df=df.drop(['KSE'],axis=1)
    df=df.drop(lista,axis=1)
    df=df.drop(['HLeigs'],axis=1)
    df['dimension']=df['atom_numbers'].combine(df['positions'], lambda x,y: ave_dist(np.array(x),np.array(y)))
    
    #other final stuff
    max_asize={'C': df['#C'].max(), 'H': df['#H'].max(), 'N': df['#N'].max(), 'O': df['#O'].max(), 'S': df['#S'].max(),'Cl':df['#Cl'].max()}
    max_size=list(dizionario['_positions'].size())[1]
    lista=[x for x in df.columns if x != "atom_numbers" and x != "positions" and x!='#C' and x!='#N' and x!='#O' and x!='#S'and x!='#Cl' and x!='#H']
    
    #in-pandas normalization, necessary with postprocessing such as HL levels calculation
    if normalize==True:
        for key in lista:
            df [key]=(df[key]-df[key].mean())/df[key].std()
    
    #BoB calculation
    df['BoB'] = df['atom_numbers'].combine(df['positions'], lambda x,y: generate_bob(x,y,x,size=max_size,asize=max_asize))
    props_list.insert(0,'BoB')
    
    if save_to_file==True:
        df.to_json('./dataset{}.json'.format(data.__len__()))
        
    return df

def split_train_test(df,config_num=1,save_to_file=True):
    """ 
    :splits pandas dataset in train and test and saves to file:

    """
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
    """ 
    :create torch dataset with molecules and properties:
    :next iter returns molecule, properties:

    """
    dati= torch.Tensor(list(df['BoB']))
    labels = df [property_list]
    labels=torch.Tensor(labels.values)
    dataset = TensorDataset(dati,labels)
    return dataset


#less used or deprecated

def init_weights(m):

    """ 
    :weight initialization:
    
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)
    
def split(data,train,test):
    """ 
        Splits torch dataset in two (can be test or validation)

     """
    train_set,test_set=torch.utils.data.random_split(data,[train,test])
    return train_set, test_set

def noneq_dataset(df_eq,df_noneq,config_num=1,save_to_file=True):  
    """ 
    :attempt to do something with the non equilibrium configurations:
    :adds a column with the correspondent equilibrium BoB rep:

    """  
    df_noneq['BoB_eq']=df2_eq['BoB'].loc[df2_eq['BoB'].index.repeat(config_num)].reset_index(drop=True)
    if save_to_file==True:
        df_noneq.to_json('./dataset{}.json'.format(len(df_noneq)))
    return df_noneq

def pandas_to_dataset_mol_prop_noneq(df,property_list=['Eat','HLGAP','POL','C6']):
    """ 
    :create torch dataset with the non equilibrium and equilibrium config:
    :next iter returns molecule, equilibrium molecule, properties:

    """
    dati= torch.Tensor(list(df['BoB']))
    dati_eq = torch.Tensor(list(df['BoB_eq']))
    labels = df [property_list]
    labels=torch.Tensor(labels.values)
    dataset = TensorDataset(dati,dati_eq,labels)
    return dataset

def pandas_to_dataset_ls_prop(df,property_list=['Eat','HLGAP','POL','C6']):
    """ 
    :create torch dataset with molecules and latent space representation:
    :next iter returns latent representation, properties:

    """
    dati= torch.Tensor(list(df['latent_rep']))
    labels = df [property_list]
    labels=torch.Tensor(labels.values)
    dataset = TensorDataset(dati,labels)
    return dataset


atomic_classes=[0,6,1,7,8,16,17]
tanh=nn.Tanh()
def mu_prep(mu):
    
    pos=mu[:,0:69].clone()
    pos=pos.detach()
    #print(pos)
    #pos=5*tanh(pos)
    pos=torch.reshape(pos,(pos.size()[0],23,3)).tolist()        
    atomic=mu[:,69:92].clone()
    atomic=atomic.detach()
    atomic=20*atomic
    atomic=atomic.apply_(lambda z: min(atomic_classes, key=lambda x:abs(x-z)))
    #print(atomic)
    lista=[]
    for i in range(0,mu.size()[0]):
        lista.append(generate_bob(atomic[i],pos[i],atomic[i],size=23,asize={'C': 7, 'H': 16, 'N': 3, 'O': 3, 'S': 1,'Cl':2}))
    mu=torch.Tensor(np.array(lista))
    return mu
