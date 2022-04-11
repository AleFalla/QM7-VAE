from sklearn import model_selection
import torch 
from torch import nn
from schnetpack import AtomsData, AtomsLoader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pandas as pd
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import Utils as Ut
import Losses as los
import numpy as np 
from CLIP_utils import CLOOB_Loss
from torch.nn import functional as F

#if I remember correctly these are the only functions used in the main testing, sorry for the unfortunate naming of validation and train batches/features
#if needed all of these functions return a log of the training, the structure varies but the gist of it is [epoch, step in epoch, train loss, validation loss, error prop to latent (if available), error reconstruction , learning rate value]

CLIP_loss=CLOOB_Loss()


def CLIP_training(model_list , dataset, reset=True, device='cpu', batch_size=500,beta=.1,alpha=1,lr_conv=1e-7,epochs_max=10000, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1):
    
    
    #set the device
    device=torch.device(device)
    
    #count number of batches in dataset
    chunks=int(len(dataset)/batch_size)
    
    #leave 4 batches in the validation set 
    train_size=(chunks-4)*batch_size
    trainset,valiset = Ut.split(dataset,train=config_num*train_size,test=config_num*(len(dataset)-train_size))
    print(len(trainset),len(valiset))

    #define validation dataloader and set it before and set tensor to device
    vali_dataloader = DataLoader(valiset, batch_size=config_num*(len(dataset)-train_size), shuffle=False)
    vali_ft,vali_lbs = next(iter(vali_dataloader))
    vali_ft=vali_ft.to(device)
    vali_lbs=vali_lbs.to(device)
    
    #define optimizer for training
    parameters=[]
    for model in model_list:
        parameters=parameters+list(model.parameters())
    optimizer = optim.AdamW(parameters , lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        for model in model_list:
            model.apply(Ut.reset_weights)
            print(model.__class__.__name__)
                
    
    e=0

    while (optimizer.param_groups[0]['lr'])>lr_conv and e<=epochs_max:

        #training dataloader
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for t in range(0,int(train_size/batch_size)):
            
            for model in model_list:
                model.train()
                
            #get next batch and set to device
            molecules,properties=next(iter(train_dataloader))
            molecules=molecules.to(device)
            properties=properties.to(device)

            #depending on the model we might want the matrix or the triupper
            temp=molecules.size()

            if len(temp)==3:
                i,j=torch.torch.triu_indices(temp[1],temp[1])
                ref=molecules[:,i,j]

            #get only the CM if convolutional
            if len(temp)==4:
                i,j=torch.torch.triu_indices(temp[2],temp[2])
                ref=molecules[:,0,i,j]

            #prediction property->latent space
            mu_p,logvar_p=model_list[1](properties)
            
            #VAE reconstruction
            mu_x, logvar_x = model_list[0](ref)

            std_p=logvar_p.mul(0.5).exp()
            eps_p=std_p.data.new(std_p.size()).normal_()
            z_p = eps_p.mul(std_p).add_(mu_p)

            std_x=logvar_x.mul(0.5).exp()
            eps_x=std_x.data.new(std_x.size()).normal_()
            z_x = eps_x.mul(std_x).add_(mu_x)
            
            z_x=F.normalize(z_x, dim=1)
            z_p=F.normalize(z_p, dim=1)

            loss_tr=CLIP_loss(z_x, z_p)+beta*(los.kl_divergence_extra(mu_x,logvar_x,torch.zeros_like(mu_x),(-0.5)*torch.ones_like(logvar_x))+los.kl_divergence_extra(mu_p,logvar_p,torch.zeros_like(mu_p),(-0.5)*torch.ones_like(logvar_p)))#Ut.get_discard(mu_x)            

            #backprop
            optimizer.zero_grad()
            loss=loss_tr
            loss.backward()
            optimizer.step()

        #it's validation time
        if e%(1)==0:

            for model in model_list:
                model.eval()

            with torch.no_grad():

                #depending on the model we might want the matrix or the triupper
                temp=vali_ft.size()

                if len(temp)==3:
                    i,j=torch.torch.triu_indices(temp[1],temp[1])
                    ref=vali_ft[:,i,j]

                #get only the CM if convolutional
                if len(temp)==4:
                    i,j=torch.torch.triu_indices(temp[2],temp[2])
                    ref=vali_ft[:,0,i,j]

                #predictions
                mu_p,logvar_p=model_list[1](vali_lbs)
                mu_x, logvar_x = model_list[0](ref)
                
                std_p=logvar_p.mul(0.5).exp()
                eps_p=std_p.data.new(std_p.size()).normal_()
                z_p = eps_p.mul(std_p).add_(mu_p)

                std_x=logvar_x.mul(0.5).exp()
                eps_x=std_x.data.new(std_x.size()).normal_()
                z_x = eps_x.mul(std_x).add_(mu_x)

                z_x=F.normalize(z_x, dim=1)
                z_p=F.normalize(z_p, dim=1)

                penalty=CLIP_loss(z_x, z_p)
                kld=beta*(los.kl_divergence_extra(mu_x,logvar_x,torch.zeros_like(mu_x),(-0.5)*torch.ones_like(logvar_x))+los.kl_divergence_extra(mu_p,logvar_p,torch.zeros_like(mu_p),(-0.5)*torch.ones_like(logvar_p)))#Ut.get_discard(mu_x)            

                #loss calculation
                loss_tst_reco=penalty+kld#los.ELBO_beta_prop_ls_extra(ref,z,mu_x,logvar_x,mu_p,logvar_p,prior_mu,prior_logvar,mu,logvar,beta=beta,alpha=alpha)#-los.reconstruction(ref, mu_x, logvar_x)+100*penalty#
                
                #loss and status print plu lr schedule step
                #loss_tst_reco=los.ELBO_beta_prop_ls_extra(ref,mu,mu_x,logvar_x,mu_p,logvar_p,torch.zeros_like(mu),torch.zeros_like(logvar),mu,logvar,beta=beta,alpha=alpha)
                loss_tst=loss_tst_reco
                print(e,t,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),penalty.item(),kld.item(),optimizer.param_groups[0]['lr'])
                scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            # PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            # torch.save({
            # 'epoch': e,
            # 'model_prop_ls_state_dict': model_prop_ls.state_dict(),
            # 'model_reco_state_dict': model_reco.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss_tst,
            # }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),optimizer.param_groups[0]['lr']])
        e=e+1
    
    return model_list


def decoding_training(encoder, decoder, mol_embedder , dataset, dataset2, reset=True, device='cpu', batch_size=500,beta=.1,alpha=1,lr_conv=1e-7,epochs_max=10000, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1):
    
    
    #set the device
    device=torch.device(device)
    
    #count number of batches in dataset
    chunks=int(len(dataset)/batch_size)
    
    #leave 4 batches in the validation set 
    train_size=(chunks-4)*batch_size
    trainset,valiset = Ut.split(dataset,train=config_num*train_size,test=config_num*(len(dataset)-train_size))
    print(len(trainset),len(valiset))

    #define validation dataloader and set it before and set tensor to device
    vali_dataloader = DataLoader(valiset, batch_size=config_num*(len(dataset)-train_size), shuffle=False)
    vali_ft,vali_lbs = next(iter(vali_dataloader))
    vali_ft=vali_ft.to(device)
    vali_lbs=vali_lbs.to(device)
    
    #define optimizer for training
    parameters=[]
    for model in [encoder, decoder]:
        parameters=parameters+list(model.parameters())
    optimizer = optim.AdamW(parameters , lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        for model in [encoder, decoder]:
            model.apply(Ut.reset_weights)
            print(model.__class__.__name__)
                
    
    e=0

    mol_embedder.eval()

    while (optimizer.param_groups[0]['lr'])>lr_conv and e<=epochs_max:

        #training dataloader
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for t in range(0,int(train_size/batch_size)):
            
            for model in [encoder, decoder]:
                model.train()
                
            #get next batch and set to device
            molecules,properties=next(iter(train_dataloader))
            molecules=molecules.to(device)
            properties=properties.to(device)

            #depending on the model we might want the matrix or the triupper
            temp=molecules.size()

            if len(temp)==3:
                i,j=torch.torch.triu_indices(temp[1],temp[1])
                ref=molecules[:,i,j]

            #get only the CM if convolutional
            if len(temp)==4:
                i,j=torch.torch.triu_indices(temp[2],temp[2])
                ref=molecules[:,0,i,j]

            with torch.no_grad():
                mol_embedding,_=mol_embedder(ref)

            #prediction property->latent space
            mu_p,logvar_p=encoder(properties)
            
            #reconstruction
            mu_x, logvar_x = decoder(mu_p)
            
            loss_tr=-los.reconstruction(ref, mu_x, logvar_x)-los.reconstruction(mol_embedding, mu_p, logvar_p)            

            #backprop
            optimizer.zero_grad()
            loss=loss_tr
            loss.backward()
            optimizer.step()

        #it's validation time
        if e%(1)==0:

            for model in [encoder,decoder]:
                model.eval()

            with torch.no_grad():

                #depending on the model we might want the matrix or the triupper
                temp=vali_ft.size()

                if len(temp)==3:
                    i,j=torch.torch.triu_indices(temp[1],temp[1])
                    ref=vali_ft[:,i,j]

                #get only the CM if convolutional
                if len(temp)==4:
                    i,j=torch.torch.triu_indices(temp[2],temp[2])
                    ref=vali_ft[:,0,i,j]

                mol_embedding,_=mol_embedder(ref)

                #prediction property->latent space
                mu_p,logvar_p=encoder(vali_lbs)
                
                #reconstruction
                mu_x, logvar_x = decoder(mu_p)
                
                error_3=Ut.perc_error(mu_x[:,0:528],ref[:,0:528])

                #loss calculation
                loss_tst_reco=-los.reconstruction(ref, mu_x, logvar_x)-los.reconstruction(mol_embedding, mu_p, logvar_p)            

                
                #loss and status print plu lr schedule step
                #loss_tst_reco=los.ELBO_beta_prop_ls_extra(ref,mu,mu_x,logvar_x,mu_p,logvar_p,torch.zeros_like(mu),torch.zeros_like(logvar),mu,logvar,beta=beta,alpha=alpha)
                loss_tst=loss_tst_reco
                print(e,t,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'final:{}%'.format(error_3),optimizer.param_groups[0]['lr'])
                scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            # PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            # torch.save({
            # 'epoch': e,
            # 'model_prop_ls_state_dict': model_prop_ls.state_dict(),
            # 'model_reco_state_dict': model_reco.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss_tst,
            # }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),error_3.item(),optimizer.param_groups[0]['lr']])
        e=e+1

    error = TEST([decoder, encoder], dataset2, device = device)
    model_list=[decoder,encoder]
    return model_list, error


def TRAIN_reco_prop_ls(model_list , dataset, dataset_2,reset=True,device='cpu', batch_size=500,beta=1,alpha=1,lr_conv=1e-7,epochs_max=10000, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    number_of_models=len(model_list)

    #set the device
    device=torch.device(device)
    
    #count number of batches in dataset
    chunks=int(len(dataset)/batch_size)
    
    #leave 4 batches in the validation set 
    train_size=(chunks-4)*batch_size
    trainset,valiset = Ut.split(dataset,train=config_num*train_size,test=config_num*(len(dataset)-train_size))
    print(len(trainset),len(valiset))

    #define validation dataloader and set it before and set tensor to device
    vali_dataloader = DataLoader(valiset, batch_size=config_num*(len(dataset)-train_size), shuffle=False)
    vali_ft,vali_lbs = next(iter(vali_dataloader))
    vali_ft=vali_ft.to(device)
    vali_lbs=vali_lbs.to(device)
    
    #define optimizer for training
    parameters=[]
    for model in model_list:
        parameters=parameters+list(model.parameters())
    optimizer = optim.AdamW(parameters , lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        for model in model_list:
            model.apply(Ut.reset_weights)
            print(model.__class__.__name__)
                
    
    e=0

    while (optimizer.param_groups[0]['lr'])>lr_conv and e<=epochs_max:

        #training dataloader
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for t in range(0,int(train_size/batch_size)):
            
            for model in model_list:
                model.train()
                
            #get next batch and set to device
            train_ft,train_lbs=next(iter(train_dataloader))
            train_ft=train_ft.to(device)
            train_lbs=train_lbs.to(device)

            #depending on the model we might want the matrix or the triupper
            temp=train_ft.size()
            if len(temp)==3:
                i,j=torch.torch.triu_indices(temp[1],temp[1])
                ref=train_ft[:,i,j]

            #get only the CM if convolutional
            if len(temp)==4:
                i,j=torch.torch.triu_indices(temp[2],temp[2])
                ref=train_ft[:,0,i,j]#.view(-1,32*32)
                #ref=ref.view(ref.size()[0],-1)

            #prediction property->latent space
            mu_p,logvar_p=model_list[1](train_lbs)
            
            #VAE reconstruction
            mu, logvar = model_list[0].encode(ref)
            z = model_list[0].reparameterize(mu, logvar)
            mu_x, logvar_x = model_list[0].decode(z)
            #mu_x,logvar_x,mu,logvar,z=model_list[0](ref)
            
            
            

            if number_of_models!=2:
                
                if model_list[2].__class__.__name__=='prior':
                    prior_mu,prior_logvar=model_list[2](train_lbs)
                
                if model_list[2].__class__.__name__=='corrector':
                    prior_mu=torch.zeros_like(mu)
                    prior_logvar=torch.zeros_like(logvar)
                    correction=model_list[2](train_lbs,z)
                    mu_x=mu_x+correction

            if number_of_models==2:
                prior_mu=torch.zeros_like(mu)
                prior_logvar=torch.zeros_like(logvar)

             
            #penalty=CLIP_loss(mu, mu_p)#, logvar, logvar_p)#Ut.get_discard(mu_x)            

            #loss calculation
            loss_tr_reco=los.ELBO_beta_prop_ls_extra(ref,z,mu_x,logvar_x,mu_p,logvar_p,prior_mu,prior_logvar,mu,logvar,beta=beta,alpha=alpha)#-los.reconstruction(ref, mu_x, logvar_x)+100*penalty
            
            #backprop
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()

        #it's validation time
        if e%(1)==0:

            for model in model_list:
                model.eval()

            with torch.no_grad():

                #depending on the model we might want the matrix or the triupper
                temp=vali_ft.size()
                if len(temp)==3:
                    i,j=torch.torch.triu_indices(temp[1],temp[1])
                    ref=vali_ft[:,i,j]

                #get only the CM if convolutional
                if len(temp)==4:
                    i,j=torch.torch.triu_indices(temp[2],temp[2])
                    ref=vali_ft[:,0,i,j]

                #predictions
                mu_p,logvar_p=model_list[1](vali_lbs)
                
                mu, logvar = model_list[0].encode(ref)
                z = model_list[0].reparameterize(mu, logvar)
                mu_x, logvar_x = model_list[0].decode(z)
                
                mu_f,_=model_list[0].decode(mu_p)

                #compute correction
                if number_of_models==3 and model_list[2].__class__.__name__=='corrector':
                    correction=model_list[2](vali_lbs,z2)
                    mu_x=mu_x+correction
                    correction2=model_list[2](vali_lbs,mu_p)
                    mu_f=mu_f+correction2

                #prop ls reco loss
                error_1=-los.reconstruction(mu,mu_p,logvar_p)

                #moleecule reco loss
                error_2=-los.reconstruction(ref,mu_x,logvar_x)

                #prop to mol reco error %
                
                error_3=Ut.perc_error(mu_f[:,0:528],ref[:,0:528])
                
            
                #penalty=CLIP_loss(mu, mu_p)#, logvar, logvar_p)#Ut.get_discard(mu_x)            

                #loss calculation
                loss_tst_reco=los.ELBO_beta_prop_ls_extra(ref,z,mu_x,logvar_x,mu_p,logvar_p,prior_mu,prior_logvar,mu,logvar,beta=beta,alpha=alpha)#-los.reconstruction(ref, mu_x, logvar_x)+100*penalty#
                
                #loss and status print plu lr schedule step
                #loss_tst_reco=los.ELBO_beta_prop_ls_extra(ref,mu,mu_x,logvar_x,mu_p,logvar_p,torch.zeros_like(mu),torch.zeros_like(logvar),mu,logvar,beta=beta,alpha=alpha)
                loss_tst=loss_tst_reco
                print(e,t,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'pr_ls:{}'.format(error_1),'reco:{}'.format(error_2),'final:{}%'.format(error_3),optimizer.param_groups[0]['lr'])
                scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            # PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            # torch.save({
            # 'epoch': e,
            # 'model_prop_ls_state_dict': model_prop_ls.state_dict(),
            # 'model_reco_state_dict': model_reco.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss_tst,
            # }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),error_1.item(),error_2.item(),optimizer.param_groups[0]['lr']])
        e=e+1
    
    if number_of_models==3 and model_list[2].__class__.__name__=='corrector':
        error=TEST_prop_mol(model_list, dataset_2,device=device)
    else:
        error=TEST_prop_mol(model_list[0:2], dataset_2,device=device)
    
    return log, model_list, error


def TEST(model_list, dataset,device='cpu'):
    device=torch.device(device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    temp=x.size()

    if len(temp)==4:
        i,j=torch.torch.triu_indices(temp[2],temp[2])
        ref=x[:,0,i,j]
        x=ref
    
    if len(temp)==3:
        i,j=torch.torch.triu_indices(temp[1],temp[1])
        ref=x[:,i,j]
        x=ref

    x=x.to(device)
    props=props.to(device)
    for model in model_list:
        model.eval()
    
    with torch.no_grad():
        mu,logvar=model_list[1](props)
        #mu=F.normalize(mu, dim=1)
        mu_x,logvar_x=model_list[0](mu)
        if len(model_list)==3 and model_list[2].__class__.__name__=='corrector':
            correction=model_list[2](props,mu)
            mu_x=mu_x+correction
        mu_x[mu_x<=1e-2]=0
        error=Ut.perc_error(mu_x,x)

    return error



def TEST_prop_mol(model_list, dataset,device='cpu'):
    device=torch.device(device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    temp=x.size()

    if len(temp)==4:
        i,j=torch.torch.triu_indices(temp[2],temp[2])
        ref=x[:,0,i,j]
        x=ref
    
    if len(temp)==3:
        i,j=torch.torch.triu_indices(temp[1],temp[1])
        ref=x[:,i,j]
        x=ref

    x=x.to(device)
    props=props.to(device)
    for model in model_list:
        model.eval()
    
    with torch.no_grad():
        mu,logvar=model_list[1](props)
        #mu=F.normalize(mu, dim=1)
        mu_x,logvar_x=model_list[0].decode(mu)
        if len(model_list)==3 and model_list[2].__class__.__name__=='corrector':
            correction=model_list[2](props,mu)
            mu_x=mu_x+correction
        mu_x[mu_x<=1e-2]=0
        error=Ut.perc_error(mu_x,x)

    return error

def TEST_prop_mol_direct(model, dataset,device='cpu'):
    device=torch.device(device)    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    x=x.to(device)
    props=props.to(device)
    model.eval()
    with torch.no_grad():
        mu,logvar=model(props)
        mu[mu<=1e-2]=0
        error=Ut.perc_error(mu.abs(),x)
    return error

def TRAIN_prop_mol(model, dataset, dataset_2, reset=True,device='cpu', batch_size=500, train_size=28000,lr_conv=1e-7, learning_rate=1e-3,epochs_max=10000, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    #set the device
    device=torch.device(device)
    
    #count number of batches in dataset
    chunks=int(len(dataset)/batch_size)
    
    #leave 4 batches in the validation set 
    train_size=(chunks-4)*batch_size
    trainset,valiset = Ut.split(dataset,train=config_num*train_size,test=config_num*(len(dataset)-train_size))
    
    #define validation dataloader and set it before and set tensor to device
    vali_dataloader = DataLoader(valiset, batch_size=config_num*(len(dataset)-train_size), shuffle=False)
    vali_ft,vali_lbs = next(iter(vali_dataloader))
    vali_ft=vali_ft.to(device)
    vali_lbs=vali_lbs.to(device)

    #set optimizer and lr scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    #reset the model
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>lr_conv and e<=epochs_max:

        #dataloader for training
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            
            #get training batch and set to device
            train_ft,train_lbs=next(iter(train_dataloader))
            train_ft=train_ft.to(device)
            train_lbs=train_lbs.to(device)

            #model prediction
            mu_x,logvar_x=model(train_lbs)

            #loss calculation
            loss_tr_reco=-los.reconstruction(x=train_ft,mu=mu_x,logvar=logvar_x)

            #backprop
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()

            #it's validation time
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    #prediction
                    mu_x,logvar_x=model(vali_lbs)
                    mu_x[mu_x<=5e-1]=0
                    
                    #error perc
                    error=Ut.perc_error(mu_x,vali_ft)

                    #loss log and scheduler
                    loss_tst_reco=-los.reconstruction(x=vali_ft,mu=mu_x,logvar=logvar_x)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            # """ PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            # torch.save({
            # 'epoch': e,
            # 'model_state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss_tst,
            # }, PATH) """
            log.append([e,i,loss.item(),loss_tst.item(),error.item(),optimizer.param_groups[0]['lr']])
        e=e+1
    
    error=TEST_prop_mol_direct(model, dataset_2,device=device)
    
    return log, model, error


#####EXTRA OR DEPRECATED######


L1loss=nn.L1Loss(reduction='sum')

def TRAIN_reco(model, dataset, dataset_2, reset=True, batch_size=500, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch_eq,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-7:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            train_ft_eq,train_lb=next(iter(train_dataloader))
            train_ft_eq_in=randomize_rowise(train_ft_eq)#train_ft_eq[:,torch.randperm(train_ft_eq.size()[1])]
            mu_x,logvar_x,mu,logvar,z=model(train_ft_eq_in)
            loss_tr_reco=los.ELBO_beta(x=train_ft_eq,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,beta=1)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    mu_x,logvar_x,mu,logvar,z=model(test_batch_eq)
                    error=Ut.perc_error(mu_x,test_batch_eq)
                    loss_tst_reco=los.ELBO_beta(x=test_batch_eq,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,beta=1)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            #break
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),error.item(),optimizer.param_groups[0]['lr']])
            
            
        e=e+1
    
    error=TEST_reco(model, dataset_2)
    
    return log, model, error

def TEST_reco(model,dataset):
    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,x_eq,props=next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        x_reco,logvar,mu,logvar,z=model(x)
        error=Ut.perc_error(x_reco.abs(),x_eq)
    return error

def TRAIN_prop_ls(model,decoder, dataset, dataset_2, reset=True, batch_size=500, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    #val_error=[]
    
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-7:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            train_ft,train_lb=next(iter(train_dataloader))
            mu_x,logvar_x=model(train_lb)
            loss_tr_reco=-los.reconstruction(x=train_ft,mu=mu_x,logvar=logvar_x)#L1loss(mu_x,train_ft)#
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    mu_x,logvar_x=model(test_lbs)
                    loss_tst_reco=-los.reconstruction(x=test_batch,mu=mu_x,logvar=logvar_x)#L1loss(mu_x,test_batch)#
                    loss_tst=loss_tst_reco
                    error=Ut.perc_error(mu_x,test_batch)
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])#'{}%'.format(error),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),error.item(),optimizer.param_groups[0]['lr']])
            #val_error.append(error)
        e=e+1
    
    error=TEST_prop_mol(model, decoder, dataset_2)
    
    return log, model, error
