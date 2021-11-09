import torch 
from torch import nn

def reconstruction(x,mu,logvar):
    logscale=nn.Parameter(logvar)
    scale=logscale.mul(0.5).exp()
    dist=torch.distributions.Normal(mu,torch.clamp(scale,min=1e-5,max=1e10))
    log_pxz=dist.log_prob(x)
    reco=log_pxz.sum(-1)
    return reco.mean()
    
def kl_divergence(z, mu, logvar):
    std = logvar.mul(0.5).exp()
    kl=0.5*torch.sum(logvar.exp()+mu.pow(2)-torch.ones_like(mu)-logvar,dim=1)
    return kl.mean()

def ELBO_beta(x,z,mu,logvar,mu_x,logvar_x,beta):
    return beta*kl_divergence(z=z,mu=mu,logvar=logvar)-reconstruction(x,mu_x,logvar_x)

def ELBO_beta_prop_ls(x,z,mu,logvar,mu_x,logvar_x,mu_p,logvar_p,beta):
    mus=torch.cat((mu_x,mu_p),dim=1)
    logvars=torch.cat((logvar_x,logvar_p),dim=1)
    targ=torch.cat((x,mu),dim=1)
    return beta*kl_divergence(z=z,mu=mu,logvar=logvar)-reconstruction(targ,mus,logvars)

def ELBO_beta_ls_prop(x,z,mu,logvar,mu_x,logvar_x,prop,mu_p,logvar_p,beta):
    mus=torch.cat((mu_x,mu_p),dim=1)
    logvars=torch.cat((logvar_x,logvar_p),dim=1)
    targ=torch.cat((x,prop),dim=1)
    return beta*kl_divergence(z=z,mu=mu,logvar=logvar)-reconstruction(targ,mus,logvars)

