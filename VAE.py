import torch 
from torch import nn

#base VAE model

input_dim=528
latent_size=30


class base_VAE(nn.Module):
    def __init__(self,input_dim=input_dim,latent_size=latent_size):
        super().__init__()
         
        self.input_dim=input_dim
        self.latent_size=latent_size

        self.encoder=nn.Sequential(
            nn.Linear(input_dim,2000),
            nn.Tanh(),
            nn.Linear(2000,2000),
            nn.Tanh(),
            nn.Linear(2000,latent_size*2)
        )
        
        self.decoder=nn.Sequential(
            nn.Linear(latent_size,2000),
            nn.Tanh(),
            nn.Linear(2000,input_dim*2)
            
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
  