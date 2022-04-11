import torch 
import Utils as Ut
from torch import nn


L1loss=nn.L1Loss(reduction='mean')
weights=torch.Tensor([0.86073257, 0.86116785, 0.86106419, 0.861002  , 0.86118858,
       0.86118858, 0.86081546, 0.86112638, 0.86118858, 0.86073257,
       0.86073257, 0.86073257, 0.86073257, 0.86110565, 0.86118858,
       0.86118858, 0.86116785, 0.86118858, 0.86118858, 0.86118858,
       0.86118858, 0.86116785, 0.86118858, 0.86118858, 0.86116785,
       0.86116785, 0.86116785, 0.86116785, 0.86118858, 0.86118858,
       0.86118858, 0.85020846, 0.85464122, 0.85581482, 0.858973  ,
       0.85425037, 0.85798094, 0.86118858, 0.85020846, 0.85020846,
       0.85020846, 0.85441492, 0.85994549, 0.86098127, 0.86118858,
       0.449667  , 0.71555394, 0.858973  , 0.58492151, 0.76620812,
       0.86118858, 0.449667  , 0.449667  , 0.44979693, 0.45305733,
       0.56518145, 0.75831709, 0.86118858, 0.71555394, 0.858973  ,
       0.80255313, 0.86118858, 0.86118858, 0.71555394, 0.71555394,
       0.71555394, 0.71909418, 0.77956777, 0.86118858, 0.86118858,
       0.858973  , 0.86118858, 0.86118858, 0.86118858, 0.858973  ,
       0.858973  , 0.858973  , 0.86118858, 0.86118858, 0.86118858,
       0.86118858, 0.43611592, 0.65853823, 0.85725829, 0.43611592,
       0.43611592, 0.43623143, 0.43825253, 0.55290738, 0.76278476,
       0.86118858, 0.65853823, 0.85725829, 0.65853823, 0.65853823,
       0.65853823, 0.65863337, 0.75809804, 0.86118858, 0.86118858,
       0.85725829, 0.85725829, 0.85725829, 0.85725829, 0.85725829,
       0.86118858, 0.86118858, 0.86118858, 0.31681357, 0.3168212 ,
       0.31695089, 0.31944825, 0.40936825, 0.63594619, 0.82199025,
       0.3168212 , 0.31695089, 0.31944825, 0.40936825, 0.63594619,
       0.82199025, 0.31695089, 0.31944825, 0.40936825, 0.63594619,
       0.82199025, 0.31944825, 0.40936825, 0.63594619, 0.82199025,
       0.40936825, 0.63594619, 0.82199025, 0.63594619, 0.82199025,
       0.82199025])

device='cpu'
device_model=torch.device(device)
weights=weights.to(device)

def reconstruction(x,mu,logvar):
    logscale=nn.Parameter(logvar)
    scale=logscale.mul(0.5).exp()
    dist=torch.distributions.Normal(mu,scale)#torch.clamp(scale,min=1e-5,max=1e10))
    log_pxz=dist.log_prob(x)
    # if log_pxz.size()[1]==136 and log_pxz.size()[0]==500:
    #     log_pxz=torch.mul(weights,log_pxz)
    reco=log_pxz.sum(-1)
    return reco.mean()

def contrastive_reco(x,mu,logvar):
    logscale=nn.Parameter(logvar)
    scale=logscale.mul(0.5).exp()
    dist=torch.distributions.Normal(mu,scale)#torch.clamp(scale,min=1e-5,max=1e10))
    log_pxz=dist.log_prob(x)
    x_rand=x[torch.randperm(x.shape[0]),:]
    cont_log_pxz=dist.log_prob(x_rand)
    weights=torch.norm(x-x_rand,dim=1)/torch.max(x,dim=1)[0]
    # if log_pxz.size()[1]==136 and log_pxz.size()[0]==500:
    #     log_pxz=torch.mul(weights,log_pxz)
    reco=log_pxz.sum(-1)
    cont_reco=torch.clamp(weights*cont_log_pxz.sum(-1),min=-1e2,max=1e30)
    return reco.mean()-cont_reco.mean()    

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

def ELBO_beta_prop_ls_extra(target_mol,target_latent,pred_mol_mean,pred_mol_logvar,pred_latent_mean,pred_latent_logvar,prior_mu,prior_logvar,latent_mean,latent_logvar,beta,alpha):
    preds=torch.cat((pred_mol_mean,pred_latent_mean),dim=1)
    logvars=torch.cat((pred_mol_logvar,pred_latent_logvar),dim=1)
    logvars=torch.clamp(logvars,min=-1e30,max=0)
    pred_mol_logvar=torch.clamp(pred_mol_logvar,min=-1e30,max=0)
    pred_latent_logvar=torch.clamp(pred_latent_logvar,min=-1e30,max=0)
    targs=torch.cat((target_mol,target_latent),dim=1)
    return beta*(kl_divergence_extra(mu=latent_mean,logvar=latent_logvar,mu_2=prior_mu,logvar_2=prior_logvar))-(reconstruction(target_mol,pred_mol_mean,pred_mol_logvar)+alpha*reconstruction(target_latent,pred_latent_mean,pred_latent_logvar))

