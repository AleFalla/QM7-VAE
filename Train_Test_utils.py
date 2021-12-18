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

#if I remember correctly these are the only functions used in the main testing, sorry for the unfortunate naming of validation and train batches/features
#if needed all of these functions return a log of the training, the structure varies but the gist of it is [epoch, step in epoch, train loss, validation loss, error prop to latent (if available), error reconstruction , learning rate value]

def TRAIN_reco_prop_ls_extra(model_reco, model_prop_ls, model_prop_ls_2, dataset, dataset_2, device='cpu',reset=True, batch_size=500, train_size=28000,beta=1, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    device=torch.device(device)
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(len(dataset)-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(len(dataset)-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    test_batch=test_batch.to(device)
    test_lbs=test_lbs.to(device)
    optimizer = optim.AdamW(list(model_reco.parameters()) + list(model_prop_ls.parameters()) + list(model_prop_ls_2.parameters()) , lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        model_reco.apply(Ut.reset_weights)
        model_prop_ls.apply(Ut.reset_weights)
        model_prop_ls_2.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-7:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model_reco.train()
            model_prop_ls.train()
            train_ft,train_lb=next(iter(train_dataloader))
            train_ft=train_ft.to(device)
            train_lb=train_lb.to(device)
            mu_2,logvar_2=model_prop_ls_2(train_lb)
            mu_x,logvar_x,mu,logvar,z=model_reco(train_ft)
            #print(train_ft.size(),z.size())
            mu_p,logvar_p=model_prop_ls(train_lb)
            loss_tr_reco=los.ELBO_beta_prop_ls_extra(x=train_ft,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,mu_p=mu_p,logvar_p=logvar_p,mu_2=mu_2,logvar_2=logvar_2,beta=beta)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model_reco.eval()
                model_prop_ls.eval()
                with torch.no_grad():
                    mu_x,logvar_x,mu,logvar,z=model_reco(test_batch)
                    mu_p,logvar_p=model_prop_ls(test_lbs)
                    mu_2,logvar_2=model_prop_ls_2(test_lbs)
                    error_1=Ut.perc_error(mu_p,mu)
                    error_2=Ut.perc_error(mu_x,test_batch)
                    loss_tst_reco=los.ELBO_beta_prop_ls_extra(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,mu_p=mu_p,logvar_p=logvar_p,mu_2=mu_2,logvar_2=logvar_2,beta=beta)
                    loss_tst=loss_tst_reco
                    print(e,i,torch.mean(mu_2[:,30]).item(),torch.std(mu_2[:,30]).item(),'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'prop ls error:{}%'.format(error_1),'reco error:{}%'.format(error_2),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            # PATH='./{}/data_config_num_{}_epoch{}2'.format(folder_name,config_num,e)
            # torch.save({
            # 'epoch': e,
            # 'model_prop_ls_state_dict': model_prop_ls.state_dict(),
            # 'model_reco_state_dict': model_reco.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss_tst,
            # }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),error_1.item(),error_2.item(),optimizer.param_groups[0]['lr']])
        e=e+1
    
    error=TEST_prop_mol(model_prop_ls,model_reco, dataset_2,device=device)
    
    return log, model_reco, model_prop_ls, error

def TRAIN_reco_prop_ls(model_reco, model_prop_ls, dataset, dataset_2, reset=True,device='cpu', batch_size=500,beta=1, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    device=torch.device(device)
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    test_batch=test_batch.to(device)
    test_lbs=test_lbs.to(device)
    optimizer = optim.AdamW(list(model_reco.parameters()) + list(model_prop_ls.parameters()) , lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        model_reco.apply(Ut.reset_weights)
        model_prop_ls.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-7:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model_reco.train()
            model_prop_ls.train()
            train_ft,train_lb=next(iter(train_dataloader))
            train_ft=train_ft.to(device)
            train_lb=train_lb.to(device)
            mu_x,logvar_x,mu,logvar,z=model_reco(train_ft)
            mu_p,logvar_p=model_prop_ls(train_lb)
            loss_tr_reco=los.ELBO_beta_prop_ls(x=train_ft,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,mu_p=mu_p,logvar_p=logvar_p,beta=beta)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model_reco.eval()
                model_prop_ls.eval()
                with torch.no_grad():
                    mu_x,logvar_x,mu,logvar,z=model_reco(test_batch)
                    mu_p,logvar_p=model_prop_ls(test_lbs)
                    error_1=Ut.perc_error(mu_p,mu)
                    error_2=Ut.perc_error(mu_x,test_batch)
                    loss_tst_reco=los.ELBO_beta_prop_ls(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,mu_p=mu_p,logvar_p=logvar_p,beta=beta)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'prop ls error:{}%'.format(error_1),'reco error:{}%'.format(error_2),optimizer.param_groups[0]['lr'])
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
    
    error=TEST_prop_mol(model_prop_ls,model_reco, dataset_2,device=device)
    
    return log, model_reco, model_prop_ls, error

def TRAIN_prop_mol(model, dataset, dataset_2, reset=True,device='cpu', batch_size=500, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    device=torch.device(device)
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(len(dataset)-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(len(dataset)-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    test_batch=test_batch.to(device)
    test_lbs=test_lbs.to(device)
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
            train_ft,train_lb=next(iter(train_dataloader))
            train_ft=train_ft.to(device)
            train_lb=train_lb.to(device)
            mu_x,logvar_x=model(train_lb)
            loss_tr_reco=-los.reconstruction(x=train_ft,mu=mu_x,logvar=logvar_x)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    mu_x,logvar_x=model(test_lbs)
                    error=Ut.perc_error(mu_x,test_batch)
                    loss_tst_reco=-los.reconstruction(x=test_batch,mu=mu_x,logvar=logvar_x)
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

def TEST_prop_mol(model, decoder, dataset,device='cpu'):
    device=torch.device(device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    x=x.to(device)
    props=props.to(device)
    model.eval()
    decoder.eval()
    with torch.no_grad():
        mu,logvar=model(props)
        mu_x,logvar_x=decoder.decode(mu)
        error=Ut.perc_error(mu_x.abs(),x)
    return error



#those other are tests and other stuff

def randomize_rowise(tensor):
    a=tensor
    for i in range (0,tensor.size()[0]):
        a[i,:]=a[i,torch.randperm(a.size()[1])]
    return a

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

def TRAIN_reco_special(model, dataset, dataset_2, reset=True, batch_size=500, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        model.apply(Ut.reset_weights)
        #model.apply(Ut.init_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-7:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            train_ft,train_lb=next(iter(train_dataloader))
            #print(train_ft.sum())
            mu_x,logvar_x,mu,logvar,z=model(train_ft)
            loss_tr_reco=los.ELBO_beta_special(x=train_ft,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,beta=1)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    mu_x,logvar_x,mu,logvar,z=model(test_batch)
                    error=Ut.perc_error(Ut.mu_prep(mu_x),test_batch)
                    loss_tst_reco=los.ELBO_beta_special(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x,logvar_x=logvar_x,beta=1)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])
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

def TRAIN_reco_ls_prop(model_reco, model_ls_prop, dataset, dataset_2, reset=True, batch_size=500, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW( list(model_reco.parameters()) + list(model_ls_prop.parameters()) , lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    if reset==True:
        model_reco.apply(Ut.reset_weights)
        model_ls_prop.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-7:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model_reco.train()
            model_ls_prop.train()
            train_ft,train_lb=next(iter(train_dataloader))
            mu_x,logvar_x,mu,logvar,z=model_reco(train_ft)
            mu_p,logvar_p=model_ls_prop(z)
            loss_tr_reco=los.ELBO_beta_ls_prop(x=train_ft,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,prop=train_lb,mu_p=mu_p,logvar_p=logvar_p,beta=1)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model_reco.eval()
                model_ls_prop.eval()
                with torch.no_grad():
                    mu_x,logvar_x,mu,logvar,z=model_reco(test_batch)
                    mu_p,logvar_p=model_ls_prop(mu)
                    error_1=Ut.perc_error(mu_p,test_lbs)
                    error_2=Ut.perc_error(mu_x.abs(),test_batch)
                    loss_tst_reco=los.ELBO_beta_ls_prop(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,prop=test_lbs,mu_p=mu_p,logvar_p=logvar_p,beta=1)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'ls prop error:{}%'.format(error_1),'reco error:{}%'.format(error_2),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_prop_ls_state_dict': model_ls_prop.state_dict(),
            'model_reco_state_dict': model_reco.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),error_1.item(),error_2.item(),optimizer.param_groups[0]['lr']])
        e=e+1

    error=TEST_reco(model_reco, dataset_2)
    
    return log, model_reco, model_ls_prop, error

def TEST_prop_mol_direct(model, dataset,device='cpu'):
    device=torch.device(device)    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    x=x.to(device)
    props=props.to(device)
    model.eval()
    with torch.no_grad():
        mu,logvar=model(props)
        error=Ut.perc_error(mu.abs(),x)
    return error

def reco_pretrain(model, reset=True, batch_size=50, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    log=[]
    test_batch=torch.rand(10000,528)*1e3
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)

    while (optimizer.param_groups[0]['lr'])>1e-7:
        
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            train_ft=torch.rand(500,528)*1e3
            mu_x,logvar_x,mu,logvar,z=model(train_ft)
            
            loss_tr_reco=los.ELBO_beta(x=train_ft,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,beta=1)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    mu_x,logvar_x,mu,logvar,z=model(test_batch)
                    loss_tst_reco=los.ELBO_beta(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,beta=1)
                    loss_tst=loss_tst_reco
                    error=Ut.perc_error(mu_x.abs(),test_batch)
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%10==0 and e!=0:
            PATH='./{}/data_pretrain_epoch{}'.format(folder_name,e)
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            log.append([e,i,loss.item(),loss_tst.item(),error.item(),optimizer.param_groups[0]['lr']])
        e=e+1
    
    
    
    return log, model, error

def TRAIN_reco_inj(model, dataset, dataset_2, reset=True, batch_size=500, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    log=[]
    
    
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-8:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            train_ft,train_lb=next(iter(train_dataloader))
            #input=torch.cat((train_ft,train_lb),1)
            mu_x,logvar_x,mu,logvar,z=model(train_ft,train_lb)
            loss_tr_reco=los.ELBO_beta(x=train_ft,z=z,mu=mu,logvar=logvar,mu_x=(mu_x.abs()>1e-2).float(),logvar_x=logvar_x,beta=1)
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    #input=torch.cat((test_batch,test_lbs),1)
                    mu_x,logvar_x,mu,logvar,z=model(test_batch,test_lbs)
                    error=Ut.perc_error((mu_x.abs()>1e-2).float(),test_batch)
                    loss_tst_reco=los.ELBO_beta(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=(mu_x.abs()>1e-2).float(),logvar_x=logvar_x,beta=1)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])
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
        e=e+1
    
    error=TEST_reco_inj(model, dataset_2)
    
    return log, model, error

def TEST_reco_inj(model,dataset):
    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        x_reco,logvar,mu,logvar,z=model(x,props)
        error=Ut.perc_error(x_reco.abs(),x)
    return error


