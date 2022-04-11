import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import math


softmax=nn.Softmax(dim=-1)

def Hopfield(a,b,beta=8):
    
    att = softmax(beta*torch.matmul(b,a.T))
    out = torch.matmul(b.T,att).T
    return F.normalize(out,dim=1)

def Loo_softmax(x,y,logvary=None,logvarx=None,tau=1/30):
    
    if logvarx==None:
        expmat = torch.matmul(x,y.T)
    else:
        expmat = inner_gaussian(x,logvarx,y,logvary)
    #print(expmat)
    mat = torch.exp((1/tau)*expmat)
    num = torch.diagonal(mat)
    mask = torch.ones(x.size()[0],x.size()[0],device=mat.device.type)
    mask = mask-torch.eye(x.size()[0],device=mat.device.type)
    denom = torch.matmul(mask,mat)
    denom = torch.diagonal(denom)
    torch.div(num,denom)
    return torch.div(num,denom)

def InfoLOOB_loss(Ux,Uy,Vx,Vy,logvarUx, logvarUy, logvarVx, logvarVy,tau):

    A = torch.clamp(Loo_softmax(Ux, Uy, logvarUx, logvarUy, tau),min=1e-300)
    B = torch.clamp(Loo_softmax(Vx, Vy, logvarVx, logvarVy, tau),min=1e-300)
    return -torch.mean(torch.log(A)+torch.log(B),dim=0)

class CLOOB_Loss(nn.Module):
    def __init__(self):
        super(CLOOB_Loss, self).__init__()
        self.tau=1/30
    
    def loob_loss(self, x, y, logvarx=None, logvary=None):
        if logvarx!=None:
            logvarUx = Hopfield(logvarx,logvarx)
            logvarUy = Hopfield(logvary,logvarx)
            logvarVx = Hopfield(logvarx,logvary)
            logvarVy = Hopfield(logvary,logvary)
        else:
            logvarUx = None
            logvarUy = None
            logvarVx = None
            logvarVy = None

        Ux = Hopfield(x,x)
        Uy = Hopfield(y,x)
        Vx = Hopfield(x,y)
        Vy = Hopfield(y,y)

        return InfoLOOB_loss(Ux, Uy, Vx, Vy, logvarUx, logvarUy, logvarVx, logvarVy, self.tau)

    def forward(self, molecule_embedding, properties_embedding, logvarmol=None, logvarprop=None):
        
        return self.loob_loss(molecule_embedding,properties_embedding, logvarmol, logvarprop)

def inner_gaussian(mu_f, logvar_f, mu_g, logvar_g):

    std_f_2 = logvar_f.exp()
    std_g_2 = logvar_g.exp()
    
    num=mu_f.reshape(mu_f.size()[0],-1, 1) - mu_g.reshape(-1, mu_g.size()[1],mu_g.size()[0])
    denom=std_f_2.reshape(std_f_2.size()[0],-1, 1) + std_g_2.reshape(-1, std_g_2.size()[1],std_g_2.size()[0])
    C=((2*math.pi*(denom))**(-0.5))
    S=C*torch.exp(-torch.div((num)**2,2*(denom)))
    S=torch.log(S)
    S=S.sum(dim=1)
    S=torch.exp(S)
    return S 