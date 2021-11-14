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



L1loss=nn.L1Loss(reduction='sum')


def TRAIN_reco(model, dataset, dataset_2, reset=True, batch_size=500, train_size=20000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=.9,patience=2)
    train_losses=[]
    val_losses=[]
    val_error=[]
    
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-6:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            train_ft,train_lb=next(iter(train_dataloader))
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
                    error=Ut.perc_error(mu_x.abs(),test_batch)
                    loss_tst_reco=los.ELBO_beta(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,beta=1)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%50==0 and e!=0:
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            train_losses.append(loss_tr_reco.item())
            val_losses.append(loss_tst.item())
            val_error.append(error)
        e=e+1
    
    error=TEST_reco(model, dataset_2)
    
    return train_losses, val_losses, val_error, model, error




def TEST_reco(model,dataset):
    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        x_reco,logvar,mu,logvar,z=model(x)
        error=Ut.perc_error(x_reco.abs(),x)
    return error


def TRAIN_prop_ls(model,decoder, dataset, dataset_2, reset=True, batch_size=500, train_size=20000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    train_losses=[]
    val_losses=[]
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
            loss_tr_reco=-los.reconstruction(x=train_ft,mu=mu_x,logvar=logvar_x)#L1loss(train_ft,mu_x)#
            optimizer.zero_grad()
            loss=loss_tr_reco
            loss.backward()
            optimizer.step()
            if i%(50)==0:
                model.eval()
                with torch.no_grad():
                    mu_x,logvar_x=model(test_lbs)
                    loss_tst_reco=-los.reconstruction(x=test_batch,mu=mu_x,logvar=logvar_x)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(Ut.perc_error(mu_x,test_batch)),optimizer.param_groups[0]['lr'])#'{}%'.format(error),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%50==0 and e!=0:
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            train_losses.append(loss_tr_reco.item())
            val_losses.append(loss_tst.item())
            #val_error.append(error)
        e=e+1
    
    error=TEST_prop_mol(model, decoder, dataset_2)
    
    return train_losses, val_losses, model, error



def TEST_prop_mol(model, decoder, dataset):
    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    model.eval()
    decoder.eval()
    with torch.no_grad():
        mu,logvar=model(props)
        mu_x,logvar_x=decoder.decode(mu)
        error=Ut.perc_error(mu_x.abs(),x)
    return error




def TRAIN_reco_prop_ls(model_reco, model_prop_ls, dataset, dataset_2, reset=True, batch_size=500, train_size=28000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    params = list(model_reco.parameters()) + list(model_prop_ls.parameters()) 
    optimizer = optim.AdamW(params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    train_losses=[]
    val_losses=[]
    val_error=[]
    
    if reset==True:
        model_reco.apply(Ut.reset_weights)
        model_prop_ls.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-6:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model_reco.train()
            model_prop_ls.train()
            train_ft,train_lb=next(iter(train_dataloader))
            mu_x,logvar_x,mu,logvar,z=model_reco(train_ft)
            mu_p,logvar_p=model_prop_ls(train_lb)
            loss_tr_reco=los.ELBO_beta_prop_ls(x=train_ft,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,mu_p=mu_p,logvar_p=logvar_p,beta=1)
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
                    error_2=Ut.perc_error(mu_x.abs(),test_batch)
                    loss_tst_reco=los.ELBO_beta_prop_ls(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,mu_p=mu_p,logvar_p=logvar_p,beta=1)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'prop ls error:{}%'.format(error_1),'reco error:{}%'.format(error_2),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%50==0 and e!=0:
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_prop_ls_state_dict': model_prop_ls.state_dict(),
            'model_reco_state_dict': model_reco.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            train_losses.append(loss_tr_reco.item())
            val_losses.append(loss_tst.item())
            error=[error_1,error_2]
            val_error.append(error)
        e=e+1
    
    error=TEST_prop_mol(model_prop_ls,model_reco, dataset_2)
    
    return train_losses, val_losses, val_error, model_reco, model_prop_ls, error






def TRAIN_reco_ls_prop(model_reco, model_ls_prop, dataset, dataset_2, reset=True, batch_size=500, train_size=20000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    params = list(model_reco.parameters()) + list(model_ls_prop.parameters()) 
    optimizer = optim.AdamW(params, lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=factor,patience=patience)
    train_losses=[]
    val_losses=[]
    val_error=[]
    
    if reset==True:
        model_reco.apply(Ut.reset_weights)
        model_ls_prop.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-6:
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
                    mu_p,logvar_p=model_ls_prop(z)
                    error_1=Ut.perc_error(mu_p,test_lbs)
                    error_2=Ut.perc_error(mu_x.abs(),test_batch)
                    loss_tst_reco=los.ELBO_beta_ls_prop(x=test_batch,z=z,mu=mu,logvar=logvar,mu_x=mu_x.abs(),logvar_x=logvar_x,prop=test_lbs,mu_p=mu_p,logvar_p=logvar_p,beta=1)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'ls prop error:{}%'.format(error_1),'reco error:{}%'.format(error_2),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%50==0 and e!=0:
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_prop_ls_state_dict': model_ls_prop.state_dict(),
            'model_reco_state_dict': model_reco.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            train_losses.append(loss_tr_reco.item())
            val_losses.append(loss_tst.item())
            error=[error_1,error_2]
            val_error.append(error)
        e=e+1
    
    error=TEST_reco(model_reco, dataset_2)
    
    return train_losses, val_losses, val_error, model_reco, model_ls_prop, error




def TRAIN_prop_mol(model, dataset, dataset_2, reset=True, batch_size=500, train_size=20000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    trainset,testset = Ut.split(dataset,train=config_num*train_size,test=config_num*(30000-train_size))
    test_dataloader = DataLoader(testset, batch_size=config_num*(30000-train_size), shuffle=False)
    test_batch,test_lbs = next(iter(test_dataloader))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=.9,patience=2)
    train_losses=[]
    val_losses=[]
    val_error=[]
    
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    while (optimizer.param_groups[0]['lr'])>1e-6:
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    
        for i in range(0,int(train_size/batch_size)):
            
            model.train()
            train_ft,train_lb=next(iter(train_dataloader))
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
                    error=Ut.perc_error(mu_x.abs(),test_batch)
                    loss_tst_reco=-los.reconstruction(x=test_batch,mu=mu_x,logvar=logvar_x)
                    loss_tst=loss_tst_reco
                    print(e,i,'train loss:{}'.format(loss.item()),'val loss:{}'.format(loss_tst.item()),'{}%'.format(error),optimizer.param_groups[0]['lr'])
                    scheduler.step(loss_tst)
            
        if e%50==0 and e!=0:
            PATH='./{}/data_config_num_{}_epoch{}'.format(folder_name,config_num,e)
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            train_losses.append(loss_tr_reco.item())
            val_losses.append(loss_tst_reco.item())
            val_error.append(error)
        e=e+1
    
    error=TEST_prop_mol_direct(model, dataset_2)
    
    return train_losses, val_losses, val_error, model, error



def TEST_prop_mol_direct(model, dataset):
    
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    x,props=next(iter(dataloader))
    model.eval()
    with torch.no_grad():
        mu,logvar=model(props)
        error=Ut.perc_error(mu.abs(),x)
    return error
    




def reco_pretrain(model, reset=True, batch_size=50, train_size=20000, learning_rate=1e-3, factor=.9, patience=2, config_num=1, folder_name='checkpoints'):
    
    
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    train_losses=[]
    val_losses=[]
    val_error=[]
    test_batch=torch.rand(10000,528)*1e3
    if reset==True:
        model.apply(Ut.reset_weights)
    
    e=0

    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=.9,patience=2)

    while (optimizer.param_groups[0]['lr'])>1e-6:
        
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
            
        if e%50==0 and e!=0:
            PATH='./{}/data_pretrain_epoch{}'.format(folder_name,e)
            torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tst,
            }, PATH)
            train_losses.append(loss_tr_reco.item())
            val_losses.append(loss_tst.item())
            val_error.append(error)
        e=e+1
    
    
    
    return train_losses, val_losses, val_error, model, error

    