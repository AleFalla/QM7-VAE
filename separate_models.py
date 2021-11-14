import torch 
from torch import nn

input_dim=528
latent_size=30
prop_size=30

class ls_prop_lin(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size):
        
        super().__init__()
        
        self.latent_size=latent_size
        self.prop_size=prop_size

        self.model_mu=nn.Sequential(
            nn.BatchNorm1d(latent_size),
            nn.Linear(latent_size, 2*prop_size),
        )


    def forward(self,x):
        
        mu_logvar=self.model_mu(x).view(-1,2,self.prop_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p


class prop_ls_lin(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size):
        
        super().__init__()

        self.latent_size=latent_size
        self.prop_size=prop_size

        self.model_mu=nn.Sequential(
            nn.BatchNorm1d(prop_size),
            nn.Linear(prop_size, 2*latent_size),
            #nn.Tanh()
        )

       
    def forward(self,x):
        
        mu_logvar=self.model_mu(x).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p


class prop_ls_NN(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size):
        
        super().__init__()

        self.latent_size=latent_size
        self.prop_size=prop_size
        
        self.model=nn.Sequential(
            #nn.Dropout(p=0.8),
            nn.Linear(prop_size, 500),
            nn.Tanh(),
            nn.Linear(500,300),
            nn.Tanh(),
            nn.Linear(300,100),
            nn.Tanh(),
            nn.Linear(100,2*latent_size)            
        )

    def forward(self,x):
        
        mu_logvar=self.model(x).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p


class ls_prop_NN(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size):
        
        super().__init__()

        self.latent_size=latent_size
        self.prop_size=prop_size
        
        self.model_mu=nn.Sequential(
            nn.Linear(latent_size, 500),
            nn.Tanh(),
            nn.Linear(500,300),
            nn.Tanh(),
            nn.Linear(300,100),
            nn.Tanh(),
            nn.Linear(100,2*prop_size)
        )

    def forward(self,x):

        mu_logvar=self.model(x).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p

mol_size=528

class prop_mol_NN(nn.Module):
    
    def __init__(self,mol_size=mol_size,prop_size=prop_size):
        
        super().__init__()

        self.mol_size=mol_size
        self.prop_size=prop_size
        
        self.model_mu=nn.Sequential(
            
            nn.Linear(prop_size, 100),
            nn.Tanh(),
            nn.Linear(100,300),
            nn.Tanh(),
            nn.Linear(300,500),
            nn.Tanh(),
            nn.Linear(500,2000),
            nn.Tanh(),
            nn.Linear(2000,mol_size*2),
        )

    def forward(self,x):

        mu_logvar=self.model_mu(x).view(-1,2,self.mol_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p

    


