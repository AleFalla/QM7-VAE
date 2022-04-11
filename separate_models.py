""" Here you will find the models used to predict latent space representation from molecular properties """

import torch 
from torch import nn
import torch.nn.functional as F
from VAE import PositionalEncoding, embedder

#these are just usual numbers but in the actual script you will redeclare these variables
input_dim=528
latent_size=30
prop_size=30
extra_size=1
total_size=32

GELU=nn.ReLU()

#convolutional model

class prop_ls_NN_conv(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()

        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size
        
        #this generates a set of extra_size properties that will be concatenated with the actual properties
        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,extra_size)
        )

        #convolutional module, takes in the properties 'image' which is just an outer product and outputs features
        self.encConv0 = nn.Sequential(
            #nn.BatchNorm2d(imgChannels),
            nn.Conv2d(1, 16, 11),
            nn.ReLU(),
            nn.Conv2d(16, 32, 11),
            nn.ReLU()
        )

        #feedforward module that takes in the convoluted features and outputs mean and logvar of the output 
        self.model=nn.Sequential(
            nn.Linear(32*12*12,2048),
            nn.Tanh(),
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Linear(1024,2*latent_size)              
        )
        
        

    def forward(self,x):

        #compute enhanced set of properties
        z=self.enhancer(x)
        #concatenate with original properties
        y=torch.cat((x,z),1)
        #outer product into an image
        y=torch.einsum('bp,bq->bpq', y, y)
        y=y.view(-1,1,32,32)
        #convolutional step
        y=self.encConv0(y)
        #reshape and feed to the feedforward module
        y=y.view(-1,32*12*12)
        mu_logvar=self.model(y).view(-1,2,self.latent_size)
        #output probabilistic prediction
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]

        return mu_p,logvar_p


#feedforward model

class prop_ls_NN(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()
        
        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size
  
        #this generates a set of extra_size properties that will be concatenated with the actual properties
        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Linear(128,extra_size),
            nn.BatchNorm1d(extra_size)
        )

        #feedforward module that takes in the properties and outputs mean and logvar of the output
        self.model=nn.Sequential(
            nn.Linear(prop_size+extra_size,2048),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,2*latent_size)              
        )
        
    
    def forward(self,x):

        #compute enhanced set of properties
        z=self.enhancer(x)
        #concatenate with original properties
        y=torch.cat((x,z),1)
        #compute output
        mu_logvar=self.model(y).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p



class mol_ls_NN(nn.Module):
    
    def __init__(self, latent_size=latent_size, input_dim=input_dim):
        
        super().__init__()
        
        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size

        #feedforward module that takes in the properties and outputs mean and logvar of the output
        self.model=nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,2*latent_size)              
        )
        
    
    def forward(self,x):

        #compute output
        mu_logvar=self.model(x).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]

        return mu_p,logvar_p



class ls_mol_NN(nn.Module):
    
    def __init__(self, latent_size=latent_size, input_dim=input_dim):
        
        super().__init__()
        
        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.input_dim=input_dim
  

        #feedforward module that takes in the properties and outputs mean and logvar of the output
        
        self.model=nn.Sequential(
            nn.Tanh(),
            nn.Linear(latent_size,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(1024),
            nn.Linear(1024,2048),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(2048),
            nn.Linear(2048,input_dim*2)
        )
        
    
    def forward(self,x):

        #compute output
        mu_logvar=self.model(x).view(-1,2,self.input_dim)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        
        return mu_p,logvar_p


class prop_ls_Transformer(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()
        
        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size
  
        #this generates a set of extra_size properties that will be concatenated with the actual properties
        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Linear(128,extra_size),
            nn.BatchNorm1d(extra_size)
        )

        
        self.embedd=embedder(3)

        self.pos_encoder = PositionalEncoding(prop_size+extra_size, dropout=0.5)

        encoder_layer = nn.TransformerEncoderLayer(d_model=prop_size+extra_size, nhead=8)
    
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.encoder = nn.Sequential(
            nn.Linear(prop_size+extra_size,2*latent_size)
        )
        
    
    def forward(self,x):

        #compute enhanced set of properties
        z=self.enhancer(x)
        #concatenate with original properties
        y=torch.cat((x,z),1)
        #compute output
        y=self.embedd(y)
        y=self.pos_encoder(y)
        transformed=self.transformer_enc(y).mean(1)
        mu_logvar=self.encoder(transformed.view(-1,self.prop_size+self.extra_size)).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p


class corrector(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size,output_size=528):
        
        super().__init__()
        
        #define a few variables
        self.output_size=output_size
        self.prop_size=prop_size
        self.extra_size=extra_size
        self.latent_size=latent_size
  
        #this generates a set of extra_size properties that will be concatenated with the actual properties
        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,extra_size)
        )

        #feedforward module that takes in the properties and outputs mean and logvar of the output
        self.model=nn.Sequential(
            
            nn.Linear(prop_size+extra_size+latent_size, 512),
            nn.Tanh(),
            nn.Linear(512,1024),
            nn.Tanh(),
            nn.Linear(1024,output_size)
        )
    
    def forward(self,x,lat):

        #compute enhanced set of properties
        z=self.enhancer(x)
        #concatenate with original properties
        y=torch.cat((x,z),1)
        y=torch.cat((y,lat),1)
        #compute output
        mu_logvar=self.model(y).view(-1,1,self.output_size)
        mu_p=mu_logvar[:,0,:]
        #logvar_p=mu_logvar[:,1,:]

        return mu_p#,logvar_p


#model that finds the prior based on properties

class prior(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size):
        
        super().__init__()
        
        #define a few variables
        self.latent_size=latent_size
        self.prop_size=prop_size
        

        #feedforward module that takes in the properties and outputs mean and logvar of the output
        self.model=nn.Sequential(
            nn.Linear(prop_size,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,2*latent_size)              
        )
        
    
    def forward(self,x):

        #compute output
        mu_logvar=self.model(x).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p


