import torch 
from torch import nn

input_dim=528
latent_size=30
prop_size=30
extra_size=1
total_size=32
class ls_prop_lin(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,total_size=total_size):
        
        super().__init__()
        
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.total_size=total_size

        self.model_mu=nn.Sequential(
            nn.Linear(latent_size, total_size),#latent_size),
            #nn.Tanh(),
            #nn.Linear(latent_size,prop_size)
        )

        self.model_logvar=nn.Sequential(
            nn.Linear(latent_size, total_size),#latent_size),
            #nn.Tanh(),
            #nn.Linear(latent_size,prop_size)
        )


    def forward(self,x):
        
        mup=self.model_mu(x)#.view(-1,2,self.prop_size)
        logvarp=self.model_logvar(x)
        mup=mup[:,0:self.prop_size]
        logvarp=logvarp[:,0:self.prop_size]

        return mup,logvarp


class prop_ls_lin(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()

        
        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size

        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,extra_size),
            #nn.Tanh()#nn.BatchNorm1d(extra_size)
        )

        self.model_mu=nn.Sequential(
            nn.Linear(prop_size+extra_size, 2048),
            nn.Linear(2048,latent_size),#prop_size),
            #nn.Tanh(),
            #nn.Linear(prop_size,latent_size)
        )

        self.model_logvar=nn.Sequential(
            nn.Linear(prop_size+extra_size, 2048),
            nn.Linear(2048,latent_size),#prop_size),
            #nn.Tanh(),
            #nn.Linear(prop_size,latent_size)
        )


    def forward(self,x):
        z=self.enhancer(x)
        y=torch.cat((x,z),1)
        mus=y#self.model_mu(y)#.view(-1,2,self.prop_size)
        logvars=self.model_logvar(y)
        return mus,logvars


class prop_ls_NN(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()

        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size
        self.f_sel=torch.nn.Parameter(torch.ones(prop_size))

        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,extra_size),
        
        )

        self.model=nn.Sequential(
            nn.Linear(prop_size+extra_size, 512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,2*latent_size)              
        )

    def forward(self,x):

        x=x*self.f_sel
        z=self.enhancer(x)
        y=torch.cat((x,z),1)
        mu_logvar=self.model(y).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p


class prop_ls_NN2(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()

        self.latent_size=latent_size
        self.prop_size=prop_size
        self.extra_size=extra_size

        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,extra_size)
        )

        self.model=nn.Sequential(
            nn.Linear(prop_size+extra_size, 2048),
            nn.Tanh(),
            nn.Linear(2048,1024),
            nn.Tanh(),
            nn.Linear(1024,2*latent_size)            
        )

    def forward(self,x):

        z=self.enhancer(x)
        y=torch.cat((x,z),1)
        mu_logvar=self.model(y).view(-1,2,self.latent_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p


class ls_prop_NN(nn.Module):
    
    def __init__(self,latent_size=latent_size,prop_size=prop_size,total_size=total_size):
        
        super().__init__()

        self.latent_size=latent_size
        self.prop_size=prop_size
        self.total_size=total_size
        
        self.model_mu=nn.Sequential(
            nn.Linear(latent_size, 2048),
            nn.Tanh(),
            nn.Linear(2048,2*total_size)
        )

    def forward(self,x):

        mu_logvar=self.model_mu(x).view(-1,2,self.total_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        mu_p=mu_p[:,0:self.prop_size]
        logvar_p=logvar_p[:,0:self.prop_size]
        return mu_p,logvar_p

mol_size=528

class prop_mol_NN(nn.Module):
    
    def __init__(self,mol_size=mol_size,prop_size=prop_size,extra_size=extra_size):
        
        super().__init__()

        self.mol_size=mol_size
        self.prop_size=prop_size
        self.extra_size=extra_size
        self.f_sel=torch.nn.Parameter(torch.ones(prop_size))

        self.enhancer=nn.Sequential(
            nn.Linear(prop_size, 128),
            nn.Tanh(),
            nn.Linear(128,128),
            nn.Tanh(),
            nn.Linear(128,extra_size)
        )

        self.model_mu=nn.Sequential(
            
            nn.Linear(prop_size+extra_size, 512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,512),
            nn.Tanh(),
            nn.Linear(512,2048),
            nn.Tanh(),
            nn.Linear(2048,mol_size*2)
        )

    def forward(self,x):
        #print(x.size())
        x=x*self.f_sel
        z=self.enhancer(x)
        y=torch.cat((x,z),1)
        mu_logvar=self.model_mu(y).view(-1,2,self.mol_size)
        mu_p=mu_logvar[:,0,:]
        logvar_p=mu_logvar[:,1,:]
        return mu_p,logvar_p

    


