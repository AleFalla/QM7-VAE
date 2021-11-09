import torch 
from torch import nn
from VAE import base_VAE

input_dim=528
latent_size=30
prop_size=30

class VAE_prop_ls_NN(base_VAE):
    
    def __init__(self,input_dim=input_dim,latent_size=latent_size,prop_size=prop_size):
        super().__init__()

        self.predictor_mu=nn.Sequential(
            nn.Linear(prop_size,prop_size*2),
            nn.Tanh(),
            nn.Linear(prop_size*2,latent_size)            
        )
        
        self.predictor_logvar=nn.Sequential(
            nn.Linear(prop_size,prop_size*2),
            nn.Tanh(),
            nn.Linear(prop_size*2,latent_size)            
        )
                
    def predict(self, prop):
        mu_p=self.predictor_mu(prop)
        logvar_p=self.predictor_logvar(prop)
        return mu_p, logvar_p

    def forward(self,x,prop):
        mu,logvar=self.encode(x)
        mu_p,logvar_p=self.predict(prop)
        z=self.reparameterize(mu,logvar)
        mu_x,logvar_x=self.decode(z)
        return mu_x,logvar_x,mu,logvar,z,mu_p,logvar_p