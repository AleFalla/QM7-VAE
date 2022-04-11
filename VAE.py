import torch 
from torch import nn
from qml.representations import *
import torch.nn.functional as F
import math
#base VAE model

input_dim=528
latent_size=30
prop_size=3

class base_VAE(nn.Module):
    def __init__(self,input_dim=input_dim,latent_size=latent_size):
        super().__init__()
        
        self.input_dim=input_dim
        self.latent_size=latent_size
        

        self.encoder=nn.Sequential(
            #nn.BatchNorm1d(input_dim),
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
            nn.Linear(1024,latent_size*2),
            
        )
        
        
        self.decoder=nn.Sequential(
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
        
    

    def reparameterize(self,mu,logvar):
        if self.training:
            std=logvar.mul(0.5).exp()
            eps=std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        
        mu_logvar=self.encoder(x.view(-1,self.input_dim)).view(-1,2,self.latent_size)
        mu=mu_logvar[:,0,:]
        logvar=mu_logvar[:,1,:]
        return mu, logvar
    
    def decode(self,z):
        
        mu_logvar=self.decoder(z).view(-1,2,self.input_dim)
        mu=mu_logvar[:,0,:]
        logvar=mu_logvar[:,1,:]
        return mu, logvar
            
    def forward(self,x):
        
        mu,logvar=self.encode(x)
        z=self.reparameterize(mu,logvar)
        mu_x,logvar_x=self.decode(z)
        return mu_x,logvar_x,mu,logvar,z

    def sample(self, n_samples):
        z=torch.randn((n_samples,self.latent_size))
        mu,logvar=self.decode(z)
        return mu

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class embedder(nn.Module):

    def __init__(self,emb_dim):
        super().__init__()
        
        self.embed=nn.Linear(1,emb_dim)

    def forward(self, x):
        x = x.view(-1,x.size()[1],1)
        x = self.embed(x).transpose(1,2)
        return x

class TRANSFORMER_VAE(nn.Module):

    def __init__(self,input_dim=input_dim,latent_size=latent_size,dropout: float = 0.5):
        super().__init__()
        
        self.input_dim=input_dim
        self.latent_size=latent_size
        
        self.embedd=embedder(3)

        self.pos_encoder = PositionalEncoding(input_dim, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        decoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
        
        self.transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,2*latent_size)
        )
        
        self.decoder_pre=nn.Sequential(
            nn.Linear(latent_size,input_dim)
        )

        self.transformer_dec=nn.TransformerEncoder(decoder_layer, num_layers=6)

        self.decoder_post=nn.Sequential(
            nn.Linear(input_dim,input_dim*2)
        )


    def reparameterize(self,mu,logvar):

        if self.training:
            std=logvar.mul(0.5).exp()
            eps=std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        x=self.embedd(x)
        x=self.pos_encoder(x)
        transformed=self.transformer_enc(x).mean(1)
        mu_logvar=self.encoder(transformed.view(-1,self.input_dim)).view(-1,2,self.latent_size)
        mu=mu_logvar[:,0,:]
        logvar=mu_logvar[:,1,:]
        return mu, logvar
    
    def decode(self,z):
        decoded=self.decoder_pre(z).view(-1,self.input_dim)
        decoded=self.embedd(decoded)
        decoded=self.pos_encoder(decoded)
        decoded=self.transformer_dec(decoded).mean(1)
        mu_logvar=self.decoder_post(decoded).view(-1,2,self.input_dim)
        mu=mu_logvar[:,0,:]
        logvar=mu_logvar[:,1,:]
        return mu, logvar
            
    def forward(self,x):
        
        mu,logvar=self.encode(x)
        z=self.reparameterize(mu,logvar)
        mu_x,logvar_x=self.decode(z)
        return mu_x,logvar_x,mu,logvar,z

    def sample(self, n_samples):
        z=torch.randn((n_samples,self.latent_size))
        mu,logvar=self.decode(z)
        return mu



class multi_input_VAE(nn.Module):
    def __init__(self,input_dim=input_dim,latent_size=latent_size):
        super().__init__()
        
        self.input_dim=input_dim
        self.latent_size=latent_size
        self.soft=nn.Softmax(dim=1)
        

        

        self.encoder_0=nn.Sequential(
            
            nn.Linear(input_dim,2048),
            nn.Tanh(),  
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Linear(1024,int(latent_size)*2)
            
        )

        self.encoder_1=nn.Sequential(
            
            nn.Linear(input_dim,2048),
            nn.Tanh(),  
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Linear(1024,int(latent_size)*2)
            
        )

        self.encoder_2=nn.Sequential(
            
            nn.Linear(input_dim,2048),
            nn.Tanh(),  
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Linear(1024,int(latent_size)*2)
            
        )
        
        
        self.decoder=nn.Sequential(
            nn.Tanh(),
            nn.Linear(latent_size,1024),
            nn.Tanh(),
            nn.Linear(1024,2048),
            nn.Tanh(),
            nn.Linear(2048,input_dim*2)
        )
        
    

    def reparameterize(self,mu,logvar):
        if self.training:
            std=logvar.mul(0.5).exp()
            eps=std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        i,j=torch.torch.triu_indices(x.size()[2],x.size()[2])
        x=x[:,:,i,j]
        mu_logvar_0=self.encoder_0(x[:,0,:].view(-1,self.input_dim)).view(-1,2,int(self.latent_size))
        mu_0=mu_logvar_0[:,0,:]
        logvar_0=mu_logvar_0[:,1,:]
        mu_logvar_1=self.encoder_1(x[:,1,:].view(-1,self.input_dim)).view(-1,2,int(self.latent_size))
        mu_1=mu_logvar_1[:,0,:]
        logvar_1=mu_logvar_1[:,1,:]
        mu_logvar_2=self.encoder_2(x[:,2,:].view(-1,self.input_dim)).view(-1,2,int(self.latent_size))
        mu_2=mu_logvar_2[:,0,:]
        logvar_2=mu_logvar_2[:,1,:]
        mu=mu_1+mu_2+mu_0#torch.cat((mu_0,mu_1,mu_2),dim=1)
        logvar=logvar_0+logvar_1+logvar_2#torch.cat((logvar_0,logvar_1,logvar_2),dim=1)
        return mu, logvar
    
    def decode(self,z):
        
        mu_logvar=self.decoder(z).view(-1,2,self.input_dim)
        mu=mu_logvar[:,0,:]
        logvar=mu_logvar[:,1,:]
        return mu, logvar
            
    def forward(self,x):
        
        mu,logvar=self.encode(x)
        z=self.reparameterize(mu,logvar)
        mu_x,logvar_x=self.decode(z)
        return mu_x,logvar_x,mu,logvar,z

    def sample(self, n_samples):
        z=torch.randn((n_samples,self.latent_size))
        mu,logvar=self.decode(z)
        return mu

    
  
GELU=nn.ReLU()
layernorm=nn.LayerNorm([3, 32, 32],device='cuda')
class toy_VAE(nn.Module):
    def __init__(self, imgChannels=1, featureDim=32*12*12, zDim=32):
        super(toy_VAE, self).__init__()
        self.featureDim=featureDim
        self.imgChannels=imgChannels
        self.zDim=zDim
        
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv0 = nn.Sequential(

            nn.Conv2d(imgChannels, 16, 11),
            nn.ReLU(),
            nn.Conv2d(16, 32, 11),
            nn.ReLU(),
            #nn.Conv2d(16, 32, 9),
            #nn.ReLU(),
            
        )

        self.deConv0 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 11),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 2, 11),
            #nn.ReLU(),
            #nn.ConvTranspose2d(32, 2, 9),
            
        )

        self.encFC1 = nn.Sequential(
            nn.Linear(featureDim,2048),
            nn.Tanh(),
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Linear(1024,2*zDim)
            )

        # Initializing the fully-connected layer for decoder
        self.decFC1 = nn.Sequential(
            nn.Tanh(),
            nn.Linear(zDim,1024),
            nn.Tanh(),
            nn.Linear(1024,2048),
            nn.Tanh(),
            nn.Linear(2048,featureDim),
        )
        

    def encoder(self, x):
        
        x=self.encConv0(x)
        x = x.view(-1, self.featureDim)
        mu_log = self.encFC1(x).view(-1,2,self.zDim)
        mu=mu_log[:,0,:]
        logVar=mu_log[:,1,:]
        return mu, logVar

    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = GELU(self.decFC1(z))
        x = x.view(-1, 32, 12, 12)
        mu_log = self.deConv0(x).view(-1,2,32,32)
        i,j=torch.torch.triu_indices(32,32)
        mu=mu_log[:,0,:,:]
        mu=mu[:,i,j]
        logvar=mu_log[:,1,:,:]
        logvar=logvar[:,i,j]
        return mu,logvar

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out,logvarout = self.decode(z)
        return out,logvarout, mu, logVar, z

