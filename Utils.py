import torch 
from torch import nn
from schnetpack import AtomsData, AtomsLoader
from torch.utils.data import TensorDataset
import pandas as pd
import numpy as np
from qml.representations import *
import scipy as sp
import scipy.spatial
import math

######################## utilities for making matrices and representations based on atomic properties


#adj matrix and utilities

dict_Z = {'H':1,'C':6,'N':7,'O':8,'S':16,'Cl':17}
dict_B = {'0-1':-1,'0-6':-1,'0-7':-1,'0-8':-1,'0-16':-1,'0-17':-1,'0-0':-1,'1-1':1.1,'1-6':1.77,'1-7':1.72,'1-8':1.72,'1-16':2.04,'1-17':2.07,'6-6':1.95,'6-7':2.0,'6-8':2.0,'6-16':2.22,'6-17':2.1,'7-7':2.05,'7-8':1.95,'7-16':2.33,'7-17':2.15,'8-8':1.80,'8-16':2.2,'8-17':2.12,'16-16':2.35,'16-17':2.47,'17-17':2.39}

def Graph(r,Z):
    matrix=scipy.spatial.distance_matrix(r,r)
    
    for i in range(0,len(Z)):
        for j in range(0,len(Z)):
            if '{}-{}'.format(Z[i],Z[j]) in dict_B:
                key='{}-{}'.format(Z[i],Z[j])
            else:
                key='{}-{}'.format(Z[j],Z[i])
            if matrix[i,j]<=dict_B[key]:
                matrix[i,j]=1
            
            else:
                matrix[i,j]=0
    return matrix

#computes adjacency matrix ordered for the dataset

def adj_mat(positions,atoms):
    """ computes adjacency matrix """
    
    #get atomic composition without zeros and hydrogens, then get sorting index by atomic number
    atoms=np.array(atoms)
    atoms=atoms[abs(atoms)!=0]
    indices=np.argsort(atoms)[::-1]
    
    #get only position of the remaining atoms and compute adj matrix
    positions=np.array(positions[0:len(atoms)])
    matrix=Graph(positions,atoms)
    
    #sort rows and columns in descending order of atomic number
    matrix=matrix[:,indices]
    matrix=matrix[indices,:]
    
    #return matrix sorted by diagonal element 
    return matrix


#computes coulomb matrix

def coulombize(positions, atoms):
    """ computes coulomb matrix """
    
    #get atomic composition without zeros and hydrogens, then get sorting index by atomic number
    atoms=np.array(atoms)
    atoms=atoms[abs(atoms)!=0]
    indices=np.argsort(atoms)[::-1]
    
    #get only position of the remaining atoms (heavy atoms) and compute distance matrix
    positions=positions[0:len(atoms)]
    matrix=scipy.spatial.distance_matrix(positions,positions)
    
    #compute coulomb matrix from distance and atomic composition
    for i in range(0,len(atoms)):
        matrix[i,i]=1
        matrix[i,:]=atoms[i]*np.multiply((matrix[i,:]**(-1)),atoms)
        matrix[i,i]=0.5*(abs(atoms[i])**2.4)
    
    #fix nans
    matrix=np.nan_to_num(matrix,posinf=0,neginf=0)
    
    #sort rows and columns in descending order of atomic number
    matrix=matrix[:,indices]
    matrix=matrix[indices,:]
    
    #return matrix sorted by diagonal element and clean atomic composition
    return matrix,atoms
    

#standardize (coulomb) matrix to the format with maximum atoms possible in the dataset -- used for all the matrices

def standardize_coulomb(coul_mat,atoms,master_vec):
    """ standardize coulomb matrices across dataset and adj too if needed"""
    #get atomic composition SORTED in descending order
    atoms=np.array(atoms)
    atoms=atoms[abs(atoms)!=0]
    atoms=np.sort(atoms)[::-1]
    
    #get maximum dimension from the master vector of maximum atoms per type and prepare a base for the standardized coulomb matrix
    max_len=len(master_vec)
    base=np.zeros((max_len,max_len))
    
    #create a zeros matrix max x max and insert coulomb matrix in the first nxn with n dimension of coulomb matrix
    padded=np.zeros((max_len,max_len))

    

    if len(atoms)<=max_len:
        padded[:len(atoms),:len(atoms)]=coul_mat
    
    else:
        padded=coul_mat[:max_len,:max_len]


    #atoms without zeros
    atoms_red=[x for x in atoms if x!=0]
    
    #counter for repeated species
    count=0
    
    #indices buffer
    indices=[]
    
    #starting atom number to check (null)
    atom_number=0
    
    #loop over atoms
    for i in range(0,len(atoms_red)):
        
        #check for repeated atom species,if not repeated reset counter
        if atom_number!=atoms_red[i]:
            count=0
        
        #set current atom number as the ith atom number in the list
        atom_number=atoms_red[i]
        
        #check where that species is in the master vector and save the index, go one after the other in case of repeated species
        index_list=[idx for idx,el in enumerate(master_vec) if el==atom_number]
        j=index_list[count]
        
        #save the index in indices and increase counter
        indices.append(j)
        count=count+1
        
    #order the coulomb matrix based on the master vector using the indices saved in the previous step (works because everything is sorted)
    for i in range(0,len(indices)):
        for j in range(0,len(indices)):
            base[indices[i],indices[j]]=padded[i,j]
    
    #return the standardized matrix
    return base


#computes standardized coulomb matrix alltogether

def MCM_st(atoms, positions, master_vec):
    """ does everything based on atoms and positions """
    
    #compute coulomb matrix
    coul_mat,atoms=coulombize(positions,atoms)
    
    #standardize it
    MCM_st=standardize_coulomb(coul_mat,atoms,master_vec)
    
    #if one wants can get only triupper, but for now I need the whole matrix
    #MCM_st=MCM_st[np.triu_indices(len(master_vec))]
    return MCM_st


#gets max dimension of coulomb matrix discarding zeros

def get_max(mat):
    """ get the maximum dimension of the coulomb matrix (throw away zeros) """
    diag=np.diagonal(mat)
    idx=np.argsort(diag)[::-1]
    diag=diag[idx]
    #A=mat[idx,:][:,idx]
    for i in range(0,len(diag)):
        if diag[i]==0:
            break
    return i


#compresses coulomb matrix to the minimum common dimension

def compress(mat,maxim):
    """ based on the maximum number of nonzeros across the dataset compress the coulomb matrix """
    diag=np.diagonal(mat)
    idx=np.argsort(diag)[::-1]
    A=mat[0:maxim,0:maxim]
    return A


#masks the equivalent of atomic numbers for atomic properties

def new_an(hchgsx,ans):
    mask=np.where(np.array(ans)>=1, 1, 0)
    return np.multiply(hchgsx,mask)


#compute distance matrix

def distances(x,at_nums):
    at_nums=np.array(at_nums)
    indices=np.argsort(at_nums)[::-1]
    matrix=scipy.spatial.distance_matrix(x,x)
    matrix=matrix[:,indices]
    matrix=matrix[indices,:]
    leng=np.count_nonzero(at_nums)
    matrix=matrix[:leng,:leng]
    return matrix


#does only coulomb matrix and does not return atomic composition

def coulombize_only(positions, atoms):
    M,atoms=coulombize(positions,atoms)
    return M


#computes matrix with correlation energies based on polarizabilities and c6 coefficients

def polarize(positions,atomspols):
    """ computes coulomb matrix """
    atoms=atomspols[0:int(len(atomspols)*0.5)]
    pols=atomspols[int(len(atomspols)*0.5)::]
    #get atomic composition without zeros and hydrogens, then get sorting index by atomic number
    atoms=np.array(atoms)
    atoms=atoms[abs(atoms)!=0]
    indices=np.argsort(atoms)[::-1]
    
    #get only position of the remaining atoms (heavy atoms) and compute distance matrix
    positions=positions[0:len(atoms)]
    matrix=scipy.spatial.distance_matrix(positions,positions)
    matrix=matrix**(-6)
    
    #fix nans
    matrix=np.nan_to_num(matrix,posinf=0,neginf=0)
    matrix=matrix+np.identity(len(atoms))
    
    tempo=np.zeros((len(atoms),len(atoms)))
    for i in range(0,len(atoms)):
        for j in range(0,len(atoms)):
            tempo[i,j]=2*pols[i]*pols[j]*atoms[i]*atoms[j]/(pols[j]**2*atoms[i]+pols[i]**2*atoms[j])
        
    matrix=np.multiply(matrix,tempo)
    
    
    #sort rows and columns in descending order of atomic number
    matrix=matrix[:,indices]
    matrix=matrix[indices,:]
    
    #return matrix sorted by diagonal element and clean atomic composition
    return matrix


#given a dataset adds a column with correlation energy matrix

def generate_pol_rep(df,el_list,at_nums):
    """ given a dataset adds a column with correlation energy matrix """
    
    pols=['atPOL_0','atPOL_1','atPOL_2','atPOL_3','atPOL_4','atPOL_5','atPOL_6','atPOL_7','atPOL_8','atPOL_9','atPOL_10','atPOL_11','atPOL_12','atPOL_13','atPOL_14','atPOL_15','atPOL_16','atPOL_17','atPOL_18','atPOL_19','atPOL_20','atPOL_21','atPOL_22']
    
    atC6_list=['atC6_0','atC6_1','atC6_2','atC6_3','atC6_4','atC6_5','atC6_6','atC6_7','atC6_8','atC6_9','atC6_10','atC6_11','atC6_12','atC6_13','atC6_14','atC6_15','atC6_16','atC6_17','atC6_18','atC6_19','atC6_20','atC6_21','atC6_22']
    
    
    #based on the list of atomic species gets the maximum number of appearence per each and builds a sorted master vector
    max_n=df[el_list].max().values
    tmp=[]
    a_n=at_nums
    
    #build master vector
    for i in range(0,len(a_n)):
        tmp=tmp+[a_n[i]]*max_n[i]
    master_vec=tmp
    master_vec.sort(reverse=True)

    #creates antry with partial charges vectors
    df['atC6_an']=(df[atC6_list].values).tolist()
    df['pols_an']=(df[pols].values).tolist()
    
    #eliminates charges relative to zero atom numbers
    df['atC6_an'] = df['atom_numbers'].combine(df['atC6_an'], lambda x,y: new_an(y,x))
    df['pols_an'] = df['atom_numbers'].combine(df['pols_an'], lambda x,y: new_an(y,x))
    
    #combine
    df['atC6_an'] = df['atC6_an'].combine(df['pols_an'], lambda x,y: x.tolist()+y.tolist())
    
    #creates partial charges coulomb matrix and standardizes it
    df['atC6_mat'] = df['positions'].combine(df['atC6_an'], lambda x,y: polarize(x,y))
    df['atC6_mat'] = df['atC6_mat'].combine(df['atom_numbers'], lambda x,y: standardize_coulomb(x,y,master_vec))

    #returns the dataset
    return df


#given a dataset adds a column with adjacency matrices

def generate_adj_rep(df,el_list,at_nums):
    """ given a dataset adds a column with adjacency matrices """
    
    #based on the list of atomic species gets the maximum number of appearence per each and builds a sorted master vector
    max_n=df[el_list].max().values
    tmp=[]
    a_n=at_nums
    
    #build master vector
    for i in range(0,len(a_n)):
        tmp=tmp+[a_n[i]]*max_n[i]
    master_vec=tmp
    master_vec.sort(reverse=True)

    
    #creates partial charges coulomb matrix and standardizes it
    df['adj_mat'] = df['positions'].combine(df['atom_numbers'], lambda x,y: 10*adj_mat(x,y))
    df['adj_mat'] = df['adj_mat'].combine(df['atom_numbers'], lambda x,y: standardize_coulomb(x,y,master_vec))

    #returns the dataset
    return df


#given a dataset adds a column with compressed coulomb matrices

def generate_MCM_rep_compressed(df,el_list,at_nums):
    """ given a dataset adds a column with compressed coulomb matrices """

    #based on the list of atomic species gets the maximum number of appearence per each and builds a sorted master vector
    max_n=df[el_list].max().values
    tmp=[]
    a_n=at_nums
    for i in range(0,len(a_n)):
        tmp=tmp+[a_n[i]]*max_n[i]
    master_vec=tmp
    master_vec.sort(reverse=True)

    #creates a column with CM representation
    df['CM_comp'] = df['positions'].combine(df['atom_numbers'], lambda x,y: MCM_st(y,x,master_vec))

    #get the maximum number of nonzero diagonal elements
    tmp=df['CM_comp'].apply(lambda x: get_max(x))
    maxim=tmp.max()

    #compresses the representation based on this maximum value
    df['CM_comp']=df['CM_comp'].apply(lambda x: compress(x,maxim))

    #returns the dataset
    return df
 

#given a dataset adds a column with standard coulomb matrices

def generate_MCM_rep(df,el_list,at_nums):
    """ given a dataset adds a column with standard coulomb matrices """

    #based on the list of atomic species gets the maximum number of appearence per each and builds a sorted master vector
    max_n=df[el_list].max().values
    tmp=[]
    a_n=at_nums
    
    #build master vector
    for i in range(0,len(a_n)):
        tmp=tmp+[a_n[i]]*max_n[i]
    master_vec=tmp
    master_vec.sort(reverse=True)

    #creates a column with CM representation
    df['CM'] = df['positions'].combine(df['atom_numbers'], lambda x,y: MCM_st(y,x,master_vec))

    #returns the dataset
    return df


#given a dataset adds a column with partial charges coulomb matrices

def generate_hCHG_rep(df,el_list,at_nums):
    """ given a dataset adds a column with partial charges coulomb matrices """
    hchg=['hCHG_0','hCHG_1','hCHG_2','hCHG_3','hCHG_4','hCHG_5','hCHG_6','hCHG_7','hCHG_8','hCHG_9','hCHG_10','hCHG_11','hCHG_12','hCHG_13','hCHG_14','hCHG_15','hCHG_16','hCHG_17','hCHG_18','hCHG_19','hCHG_20','hCHG_21','hCHG_22']

    #based on the list of atomic species gets the maximum number of appearence per each and builds a sorted master vector
    max_n=df[el_list].max().values
    tmp=[]
    a_n=at_nums
    
    #build master vector
    for i in range(0,len(a_n)):
        tmp=tmp+[a_n[i]]*max_n[i]
    master_vec=tmp
    master_vec.sort(reverse=True)

    #creates antry with partial charges vectors
    df['hCHG_an']=(df[hchg].values).tolist()
    print(len(df['atom_numbers']))
    #print(np.shape(np.array(list(df[hchg].values))))
    #eliminates charges relative to zero atom numbers
    df['hCHG_an'] = df['atom_numbers'].combine(df['hCHG_an'], lambda x,y: new_an(y,x))
    
    #creates partial charges coulomb matrix and standardizes it
    df['hCHG_mat'] = df['positions'].combine(df['hCHG_an'], lambda x,y: 10*coulombize_only(x,y))
    df['hCHG_mat'] = df['hCHG_mat'].combine(df['atom_numbers'], lambda x,y: standardize_coulomb(x,y,master_vec))

    #returns the dataset
    return df


#given a dataset adds a column with distance matrices

def generate_distance(df,el_list,at_nums):
    """ given a dataset adds a column with adjacency matrices """
    
    #based on the list of atomic species gets the maximum number of appearence per each and builds a sorted master vector
    max_n=df[el_list].max().values
    tmp=[]
    a_n=at_nums
    
    #build master vector
    for i in range(0,len(a_n)):
        tmp=tmp+[a_n[i]]*max_n[i]
    master_vec=tmp
    master_vec.sort(reverse=True)

    
    #creates partial charges coulomb matrix and standardizes it
    df['distance_mat'] = df['positions'].combine(df['atom_numbers'],lambda x,y: distances(x,y))
    df['distance_mat'] = df['distance_mat'].combine(df['atom_numbers'], lambda x,y: standardize_coulomb(x,y,master_vec))

    #returns the dataset
    return df


############################# other utilities for dataset and so on 

#weight resetting function

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()


#error percentage calculation

def perc_error(a,b):
    """ 
    :calculates relative error % (using L1 norm):
    
    """
    numer=torch.sum((a-b).abs(),dim=1)
    denom=torch.sum(b.abs(),dim=1)
    return 100*torch.mean(numer/denom)


#utility to unpack atomic properties into single entries

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


#normalization via dataloader (probably deprecated)

def normalization(dizionario,props_list):

    for key in props_list:
        tmp_avg=dizionario[key].mean().item()*torch.ones_like(dizionario[key])
        tmp_std=dizionario[key].std().item()*torch.ones_like(dizionario[key])
        dizionario[key]=(dizionario[key]-tmp_avg).div(tmp_std)
    return dizionario


#maximum distance computation function

def ave_dist(Z, pos):
    
    # """ 
    # :calculates the distance between the heavy atoms:

    # """
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


#get HOMO-LUMOs from Kohn-Sham eigenvalues
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


#this turns .db file to json (pandas readable) file with a bunch of options. Look inside the function to get how it works

def data_pre_preparation(data,atomic_keys=['C','O','N','S','Cl','F','H'],shuffle=False,normalize=True,save_to_file=True,fname='dataset',KSE_option=False,hCHG_full=True,CM_comp=True,CM_full=True):
    """ 
    :turns the db file to pandas dataset:

    """

    #define atomic dictionary
    atoms_dict={'C':6,'O':8,'N':7,'S':16,'Cl':17,'F':9,'H':1}

    #load the dataset with schnetpack dataloader
    loader=AtomsLoader(data,batch_size=int(data.__len__()),shuffle=shuffle)
    dizionario=next(iter(loader))
    
    #if Kohn-Sham eigvalues are in the dataset they must be extracted
    if KSE_option==True:
        len_kse=dizionario['KSE'].size()[1]
        lista=[]
        
        #this is specific for kse eigenvalues, comment if not needed
        for i in range(0,len_kse):
            lista.append('KSE_{}'.format(i))        
    
    #get the property list
    props_list=get_unpacked_properties(dizionario,data)
        
    #if normalize==True: #more efficient normalization, available if not using post processing operations
    #    dizionario=normalization(dizionario, props_list)
    

    #build the pandas dataset out of this (property only)
    t=()
    
    for key in props_list:
        t=t+(dizionario[key],)
        
    df=pd.DataFrame(torch.cat(t,1).tolist(), columns = props_list)

    #add atom numbers and positions
    df['atom_numbers']=dizionario['_atomic_numbers'].tolist()
    df['positions']=dizionario['_positions'].tolist()

    #add the count of single species
    for key in atomic_keys:
        df[key]=df['atom_numbers'].apply(lambda x: np.count_nonzero(np.array(x)==atoms_dict[key]))
        
    #kse, and hleigs if kse are used.... gets up to homo/lumo 2

    if KSE_option==True:
        df['KSE']=df[lista].values.tolist()
        df['HLeigs'] = df['KSE'].combine(df['HLgap'], lambda x,y: newKSE(x,y,iorbs=3))
        for i in range (0,3):
            df['HOMO_{}'.format(i)]=df['HLeigs'].apply(lambda x: x[i])
            df['LUMO_{}'.format(i)]=df['HLeigs'].apply(lambda x: x[i+3])
        df=df.drop(['KSE'],axis=1)
        df=df.drop(lista,axis=1)
        df=df.drop(['HLeigs'],axis=1)

    #computes the dimension, whatever it means
    df['dimension']=df['atom_numbers'].combine(df['positions'], lambda x,y: ave_dist(np.array(x),np.array(y)))
    
    #builds the max_asize vector or master vector
    max_asize={}
    for key in atomic_keys:
        max_asize[key]=df[key].max()
    
    #other final stuff
    max_size=list(dizionario['_positions'].size())[1]
    lista=props_list

    #in-pandas normalization, necessary with postprocessing such as HL levels calculation
    if normalize==True:
        for key in lista:
            df [key]=(df[key]-df[key].mean())/df[key].std()
    
    #Calculation of compressed and/or non compressed CM representation
    if CM_comp==True:

        at_nums=list(map(atoms_dict.get,atomic_keys))
        df=generate_MCM_rep(df,atomic_keys,at_nums)
        props_list.insert(0,'CM_comp')

    if CM_full==True:

        at_nums=list(map(atoms_dict.get,atomic_keys))
        df=generate_MCM_rep(df,atomic_keys,at_nums)
        props_list.insert(0,'CM')

    if hCHG_full==True:

        at_nums=list(map(atoms_dict.get,atomic_keys))
        df=generate_hCHG_rep(df,atomic_keys,at_nums)
        props_list.insert(0,'hCHG_CM')
        
    if save_to_file==True:
        df.to_json('./{}.json'.format(fname))
        
    return df


#train test splitting (from pandas dataset)

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
        df[0:config_num*30000].to_parquet('./dataset{}_training.parquet'.format(len(df[0:config_num*30000])))
        df[config_num*30000:].to_parquet('./dataset{}_test.parquet'.format(len(df[config_num*30000:])))
        
    return df[0:config_num*30000], df[config_num*30000:]




########################
#less used or deprecated

def find_nearest(array, value):
    #array = np.asarray(array)
    idx = ((array - value).abs()).argmin()
    return array[idx]

def recover_distance_mat(mcm):
    
    n=len(mcm)
    lun=int((-1+(1+2*4*n)**0.5)/2)
    M=torch.zeros((lun,lun),device='cuda')
    i,j=torch.triu_indices(lun,lun)
    M[i,j]=mcm
    M[j,i]=mcm
    idx=torch.argsort(torch.diag(-M))
    
    M=M[idx,:]
    M=M[:,idx]
    
    tmp=lun
    for i in range(0,lun):
        if M[i,i]<=0.25*(1**2.4):
            tmp=i
            break
    if tmp>=1:
        M=M[:tmp,:tmp]
    #set the device
    device=torch.device('cuda')
    species=torch.Tensor([17,16,8,7,6])
    species=species.to(device)
    master_vec=[]
    for i in range(0,tmp):
        
        Z=(2*M[i,i])**(1/2.4)
        Z=find_nearest(species,Z)
        master_vec.append(Z)
        
    for i in range(0,tmp):
        for j in range(0,tmp):
                if i==j:
                    M[i,j]=0
                else:
                    M[i,j]=(M[i,j]/(master_vec[i]*master_vec[j]))**(-1)

    return M,master_vec

def cartesian_recovery(distance_mat):
    D=distance_mat
    M=torch.zeros_like(D,device='cuda')
    
    for i in range(0,len(D)):
        for j in range(0,len(D)):
            M[i,j]=0.5*(D[0,i]**2+D[j,0]**2-D[i,j]**2)
            
    
    S=torch.linalg.eigvals(M)
    Q=S.real
    tmp,ind=torch.sort(Q)[::-1]
    if len(tmp)>3:    
        extra=Q[torch.abs(Q)<=torch.abs(tmp[3])]
        extra=torch.abs(extra).sum()
        cartesian=None
    else:
        cartesian=None
        extra=1000
    return cartesian, extra


def get_discard(mcm):
    total=0
    for i in range(0,mcm.size()[0]):
        dist,master=recover_distance_mat(mcm[i,:])
        
        tempo,extra=cartesian_recovery(dist)
        total=total+extra
    return total



def pandas_to_dataset_mol_prop(df,rep='CM',property_list=['Eat','HLGAP','POL','C6']):
    """ 
    :create torch dataset with molecules and properties:
    :next iter returns molecule, properties:

    """
    
    dati= torch.Tensor(list(df[rep]))
    labels = df [property_list]
    labels=torch.Tensor(list(labels.values))
    dataset = TensorDataset(dati,labels)
    return dataset


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
