import torch 
import Utils as Ut
from torch import nn


L1loss=nn.L1Loss(reduction='mean')

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

def ELBO_beta_special(x,z,mu,logvar,mu_x,logvar_x,beta):
    
    mu_x_2=Ut.mu_prep(mu_x)
    #print(mu_x_2.sum())
    return beta*kl_divergence(z=z,mu=mu,logvar=logvar)-reconstruction(x,mu_x_2,logvar_x)
""" 
def ELBO_beta_prop_ls(x,z,mu,logvar,mu_x,logvar_x,mu_p,logvar_p,beta,alpha=10):
    #mus=torch.cat((mu_x,mu_p),dim=1)
    #logvars=torch.cat((logvar_x,logvar_p),dim=1)
    #targ=torch.cat((x,mu),dim=1)
    return beta*kl_divergence(z=z,mu=mu,logvar=logvar)-reconstruction(x,mu_x,logvar_x)+alpha*L1loss(mu_p,mu) """


def kl_divergence_extra(mu, logvar,mu_2,logvar_2):
    std = logvar.mul(0.5).exp()
    std_2 = logvar.mul(0.5).exp()
    kl=torch.sum(logvar_2.mul(0.5)-logvar.mul(0.5)+(logvar.exp()+(mu-mu_2).pow(2))/(2*logvar_2.exp())-0.5*torch.ones_like(mu),dim=1)
    return kl.mean()

def symm_kld(mu, logvar,mu_2,logvar_2):
    return 0.5*(kl_divergence_extra(mu, logvar,mu_2,logvar_2)+kl_divergence_extra(mu_2, logvar_2,mu,logvar))

def ELBO_beta_extra(x,mu,logvar,mu_x,logvar_x,mu_2,logvar_2,beta):
    return beta*(symm_kld(mu=mu,logvar=logvar,mu_2=mu_2,logvar_2=logvar_2)+symm_kld(mu=torch.zeros_like(mu),logvar=torch.zeros_like(logvar),mu_2=mu_2,logvar_2=logvar_2))-reconstruction(x,mu_x,logvar_x)

def ELBO_beta_prop_ls_extra(x,z,mu,logvar,mu_x,logvar_x,mu_p,logvar_p,mu_2,logvar_2,beta):
    #print(x.size(),z.size())
    mus=torch.cat((mu_x,mu_p),dim=1)
    logvars=torch.cat((logvar_x,logvar_p),dim=1)
    targ=torch.cat((x,z),dim=1)
    return beta*(symm_kld(mu=mu,logvar=logvar,mu_2=mu_2,logvar_2=torch.zeros_like(logvar)))-reconstruction(targ,mus,logvars)#+symm_kld(mu=torch.zeros_like(mu),logvar=torch.zeros_like(logvar),mu_2=mu_2,logvar_2=torch.zeros_like(logvar))