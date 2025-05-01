import sys
# append the path of the parent directory
sys.path.append("..")

import torch
import random
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
import wandb
import argparse
from train_SP_with_CV import SigmaProfileGCN, AccumulationMeter, dataset_Sigma_Profiles, CustomLoss, collate_SP

os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())

def train(epoch, train_loader, model, loss_fn, l2_coef, optimizer, wandb_logs=False):
    stage = "train"
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()

    # Set model to training mode
    model.train()
    for i, component_data in enumerate(train_loader):
        end = time.time()
        sigma_profile = component_data['SP'].float().cuda() 

        # Model predictions and gradients
        y = None
        with torch.backends.cudnn.flags(enabled=False): #disables the CuDNN during prediction, this is could have different reasons: 1. reproducibiltiy issues, 2. performance issues, 3. speed issues
            y = model(component_data)   
      
        # Prediction loss
        loss = loss_fn(y,sigma_profile)
        #Add L2 Regularization
        l2_penalty = l2_coef * sum([(p**2).sum() for p in model.parameters()])
        loss = loss + l2_penalty
        
        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stats
        loss_accum.update(loss.item(),sigma_profile.size(0))
        batch_time.update(time.time() - end)

        if i % 100 == 0:
            print('Epoch [{}][{}/{}]'
                'Time {:.3f} ({:.3f})\t'
                'Loss {:.3f} ({:.3f})\t'.format(
                epoch, i, len(train_loader), 
                batch_time.sum, batch_time.avg,
                loss_accum.value, loss_accum.avg,))

    # Logging
    print("[Stage {}]: Epoch {} finished at time={:.3f} with loss={:.3f}".format(
    stage, epoch, batch_time.sum, loss_accum.avg))
    if wandb_logs:
        wandb.log({
                f"epoch": epoch,
                f"train_loss_accum": loss_accum.avg,
            }, step=epoch)
            
    return [loss_accum.avg]

def main(hyperparameter):
    all_start = time.time()

    # fix seed
    seed = hyperparameter.seed

    # model parameters
    model_type = hyperparameter.model_type
    mlp_activation = hyperparameter.mlp_activation
    enc_activation = hyperparameter.enc_activation

    # training parameters 
    batch_size = hyperparameter.batch_size
    lr = hyperparameter.lr
    use_lr_scheduler = hyperparameter.use_lr_scheduler
    epochs = hyperparameter.epochs
    early_stopping = hyperparameter.early_stopping
    l2_coef = hyperparameter.l2_coef

    # data parameters
    data = hyperparameter.data

    save_add = f"_{data}_{model_type}_act{mlp_activation}_encAct{enc_activation}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}_L2coef_{l2_coef}_batchsize{batch_size}_earlystopping_{early_stopping}"

    
    config = hyperparameter
    wandb_logs = config.wandb_logs

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # read dataset file
    dataset_path = 'C:\\Users\\kverg\\GDI-NN\\SigmaProfileModel\\Databases\\MMFF_spDatabase_Train.csv'
    dataset = dataset_Sigma_Profiles(
        input_file_path=dataset_path,
        generate_all=True)
    dataset_size = len(dataset)
        
    # print dataset size
    print('dataset size: {}'.format(dataset_size))

    train_indices = np.arange(dataset_size)
    # Dataloader
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   collate_fn=collate_SP,
                                                   shuffle=False,
                                                   drop_last=True)
        
    if model_type == "Sigma_profile_prediction": 
        model = SigmaProfileGCN(in_dim=50).cuda()

    print(model)
        
    loss_fn = CustomLoss().cuda()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    if use_lr_scheduler:
        scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
        
    # Training
    best_loss = 1000000
    train_loss_save = []
    
    for epoch in range(1, epochs+1):

        # Train epoch
        train_loss = train(
            epoch=epoch,
            train_loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            l2_coef=l2_coef 
            )
        train_loss_save.append(train_loss[0])

        # LR scheduler
        if use_lr_scheduler:
            scheduler.step(train_loss[0])
            
        best_loss = min(train_loss[0], best_loss)
                               
            
    # Save Final model and stats 
    torch.save({
            'epoch': epoch,
            'model_arch': model_type,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
            }, 'C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_final_model_{}.pth'.format(save_add))
        
    np.save('C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_train_loss_{}.npy'.format(save_add),np.array(train_loss_save))
  
    #print(index_list_train)
    #index_list_train = np.array(index_list_train, dtype=object)
    #np.save(f'C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/train_ind_list{save_add}.npy',index_list_train)
    
    # Plotting for all cv runs
    ## Training and validation loss

    plt.figure(figsize=(16,8))
    train_losses = np.load('C:/Users/kverg/GDI-NN/SigmaProfileModel//results_SP/Complete_trainset_train_loss_{}.npy'.format(save_add))
    plt.plot(train_losses,label="train loss")
    plt.xlabel("epoch (training iteration)")
    plt.ylabel("loss")
    plt.legend(loc="best")
    train_error = train_losses[-1]
    train_error = np.array(train_error)
    error_str = (r'Train LOSS = {:.2f} $\pm$ {:.2f}'.format(
            np.mean(train_error), np.std(train_error)))
    plt.subplot(2,3,6)
    plt.text(0,0.5, error_str, fontsize=12)
    plt.axis('off')
    plt.savefig(f'C:/Users/kverg/GDI-NN/SigmaProfileModel//results_SP/Complete_trainset_loss_{save_add}_test.png',dpi=300)            
    
    all_end = time.time() - all_start
    print(all_end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument('--model_type', default="Sigma_profile_prediction", type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--mlp_activation', default="relu", type=str)
    parser.add_argument('--enc_activation', default="relu", type=str)

    # Training
    parser.add_argument('--lr', default=1.51e-3, type=float)                   #ORIGINAL: 1e-3
    parser.add_argument('--use_lr_scheduler', default="False", type=str)
    parser.add_argument('--epochs', default=700, type=int)
    parser.add_argument('--early_stopping', default='False', type=str)
    parser.add_argument('--l2_coef', default=7.79e-6, type=str)

    # Data, split, and logs
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--data', default="MMFF_sp_Database", type=str)
    parser.add_argument('--wandb_logs', default="False", type=str)

    hyperparameter = parser.parse_args()

    if hyperparameter.use_lr_scheduler == "False": hyperparameter.use_lr_scheduler = False
    if hyperparameter.use_lr_scheduler == "True": hyperparameter.use_lr_scheduler = True
    if hyperparameter.wandb_logs == "False": hyperparameter.wandb_logs = False
    if hyperparameter.wandb_logs == "True": hyperparameter.wandb_logs = True
    if hyperparameter.early_stopping == "False": hyperparameter.early_stopping = False
    if hyperparameter.early_stopping == "True": hyperparameter.early_stopping = True

    print(hyperparameter)

    main(hyperparameter=hyperparameter)
