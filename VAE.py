import torch 
from torch import nn
from qml.representations import *

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
            nn.Linear(input_dim,2048),
            nn.Tanh(),  
            nn.Linear(2048,1024),
            nn.Tanh(),         
            nn.Linear(1024,latent_size*2)
        )
        
        self.decoder=nn.Sequential(
            nn.Linear(latent_size,2048),
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




class conv_VAE(nn.Module):
    def __init__(self,input_dim=input_dim,latent_size=latent_size):
        super().__init__()
         
        self.input_dim=input_dim
        self.latent_size=latent_size

        self.encoder=nn.Sequential(
            nn.Conv1d(1,16,4),
            nn.GELU(),
            nn.Conv1d(16,32,4),
            nn.GELU(),
            nn.Conv1d(32,64,4),
            nn.AvgPool1d(16),
            nn.Flatten(),
            nn.Tanh(),
            nn.Linear(2048,512),
            nn.Tanh(),
            nn.Linear(512,latent_size*2)
        )
        
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(2,64,4),
            nn.GELU(),
            nn.ConvTranspose2d(64,32,4),
            nn.GELU(),
            nn.ConvTranspose2d(32,16,4),
            nn.GELU(),
            nn.ConvTranspose2d(16,8,4),
            #nn.MaxPool1d(3),
            nn.Flatten(),
            nn.Tanh(),
            nn.Linear(2048,input_dim*2),
            nn.ReLU()            
        )
        
    def reparameterize(self,mu,logvar):
        if self.training:
            std=logvar.mul(0.5).exp()
            eps=std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        y=torch.reshape(x,(x.size()[0],1,x.size()[1]))
        mu_logvar=self.encoder(y).view(-1,2,self.latent_size)
        #print(mu_logvar.size())
        mu=mu_logvar[:,0,:]
        logvar=mu_logvar[:,1,:]
        return mu, logvar
    
    def decode(self,z):
        y=torch.reshape(z,(z.size()[0],2,int((z.size()[1]/2)**0.5),int((z.size()[1]/2)**0.5)))
        out_logvar=self.decoder(y)#.view(-1,2,self.input_dim)
        logvar=out_logvar[:,0:528]
        mu=out_logvar[:,528:]
        #if mu.size()[1]!=528:
        #    mu=1e5*torch.ones((logvar.size()[0],528))
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
  


class special_VAE(nn.Module):
    def __init__(self,input_dim=input_dim,latent_size=latent_size):
        super().__init__()
         
        self.input_dim=input_dim
        self.latent_size=latent_size
        self.atomic_classes=[0,6,1,7,8,16,17]
        self.encoder=nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.Tanh(),
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Linear(1024,latent_size*2)
            )
        
        self.decoder=nn.Sequential(
            nn.Linear(latent_size,2048),
            nn.Tanh(),
            nn.Linear(2048,input_dim+69+23),          
        )
        
    def reparameterize(self,mu,logvar):
        if self.training:
            std=logvar.mul(0.5).exp()
            eps=std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        #y=torch.reshape(x,(x.size()[0],1,x.size()[1]))
        mu_logvar=self.encoder(x).view(-1,2,self.latent_size)
        #print(mu_logvar.size())
        mu=mu_logvar[:,0,:]
        logvar=mu_logvar[:,1,:]
        return mu, logvar
    
    def decode(self,z):
        #y=torch.reshape(z,(z.size()[0],1,z.size()[1]))
        out_logvar=self.decoder(z)#.view(-1,2,self.input_dim)
        logvar=out_logvar[:,0:528]
        mu=out_logvar[:,528:]
        #if mu.size()[1]!=528:
        #    mu=1e5*torch.ones((logvar.size()[0],528))
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
  


class prop_VAE(nn.Module):
    def __init__(self,input_dim=input_dim,latent_size=latent_size):
        super().__init__()
        
        self.input_dim=input_dim
        self.latent_size=latent_size

        self.encoder=nn.Sequential(
            nn.Linear(input_dim,2048),
            nn.Tanh(),
            nn.Linear(2048,1024),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),         
            nn.Linear(1024,latent_size*2)
        )
        
        self.decoder=nn.Sequential(
            nn.Linear(latent_size,2048),
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
