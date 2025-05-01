from __future__ import absolute_import
import os


import sys, random, pickle, csv, time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
import matplotlib.pyplot as plt
import wandb
import argparse

# internal imports
from train_water_solubility import AccumulationMeter
from util_water.generate_dataset_for_training_water import solute_dataset_water, solute_dataset_water_SP, collate_solute_water, collate_solute_water_SP
from model.model_GNN_water import water_gegnn, water_gegnn_no_Pooling, water_gegnn_SigmaProfile_noPooling, RittigPure, AbranchesPure, Rittig_Abranches, RA_var_opt, RA_opt, RittigPure_opt

device = torch.device("cuda:0")
os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())

def train(epoch, train_loader, empty_solvsys, model, loss_fn, l2_coef, optimizer, device, wandb_logs=False):
    stage = "train"
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()

    # Set model to training mode
    model.train()
    for i, component_data in enumerate(train_loader):
        end = time.time()
        lablogS = component_data['LogS'].float().cuda() 

        # Model predictions and gradients
        y = None
        with torch.backends.cudnn.flags(enabled=False): #disables the CuDNN during prediction, this is could have different reasons: 1. reproducibiltiy issues, 2. performance issues, 3. speed issues
            if empty_solvsys != None:
                y = model(component_data, empty_solvsys, device)
            else:
                y = model(component_data)   
      
        # Prediction loss
        loss = loss_fn(y[:,0],lablogS)
        #Add L2 Regularization
        if l2_coef != 0:
            l2_penalty = l2_coef * sum([(p**2).sum() for p in model.parameters()])
            loss = loss + l2_penalty
        
        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update stats
        loss_accum.update(loss.item(),lablogS.size(0))
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
    w_init = hyperparameter.weight_init
    torch.manual_seed(seed)
    if hyperparameter.weight_init == 'He_uniform':
        weight_init_fn = torch.nn.init.kaiming_uniform_
    elif hyperparameter.weight_init == 'uniform':
        weight_init_fn = torch.nn.init.uniform_
    else:
        weight_init_fn = None

    # model parameters
    model_type = hyperparameter.model_type
    mlp_dropout_rate = hyperparameter.mlp_dropout_rate
    mlp_activation = hyperparameter.mlp_activation
    enc_activation = hyperparameter.enc_activation
    mlp_num_hid_layers = hyperparameter.mlp_num_hid_layers
    gcn_dim = hyperparameter.gcn_dim
    mpnn_dim = hyperparameter.mpnn_dim
    mlp_dim = hyperparameter.mlp_dim
    num_step_message_passing = hyperparameter.num_step_message_passing
    n_conv_filters = hyperparameter.n_conv_filters
    kernel_size = hyperparameter.kernel_size
    pool_size = hyperparameter.pool_size
    SP_dim = hyperparameter.SP_dim
    hidden_dim = hyperparameter.hidden_dim

    # training parameters 
    batch_size = hyperparameter.batch_size
    #batch_adding = hyperparameter.batch_adding
    lr = hyperparameter.lr
    use_lr_scheduler = hyperparameter.use_lr_scheduler
    epochs = hyperparameter.epochs
    l2_coef = hyperparameter.l2_coef
    early_stopping = hyperparameter.early_stopping
    SP = hyperparameter.SP
    normalize = hyperparameter.normalize
    use_solvsys = hyperparameter.use_solvsys

    # data parameters
    data = hyperparameter.data

    # checkpoint
    checkpoint_path = hyperparameter.checkpoint_path
    if checkpoint_path == 'None':
        checkpoint_path = None
        check = False
    else:
        check = True

    #save_add = f"_{data}_{model_type}_act{mlp_activation}_encAct{enc_activation}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}_L2coef{l2_coef}_batchsize{batch_size}_earlystop{early_stopping}_norm{normalize}_weight{w_init}_checkpoint{check}"
    save_add = f"_{model_type}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}_L2coef{l2_coef}_batchsize{batch_size}__weight{w_init}_gcnDim{gcn_dim}_mpnnDim{mpnn_dim}_mlpDim{mlp_dim}"
    #_nconvfilters{n_conv_filters}_kernelsize{kernel_size}_poolsize{pool_size}_SPdim{SP_dim}"
    
    config = hyperparameter
    wandb_logs = config.wandb_logs

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #Determine dataset class and collate functions, depending on wether to include Sigma Profiles in the model or not
    if SP==True:
        dataset_class = solute_dataset_water_SP
        collate_fn = collate_solute_water_SP
        sigma_profile_model_path = "C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_final_model__MMFF_sp_Database_Sigma_profile_prediction_actrelu_encActrelu_lrschedFalse_epochs700_lr0.00151_L2coef_7.79e-06_batchsize16_earlystopping_False.pth"
    else:
        dataset_class = solute_dataset_water
        collate_fn = collate_solute_water
    
    # read dataset file
    dataset_path = './data/solubility_water/COMPATIBLE_unique_train_water_with_hybrid_classes.csv'
    solute_list_path = './data/solubility_water/solute_list_with_polarity_and_size.csv'
    solvent_list_path = './data/solubility_water/solvent_list_water_with_polarity_and_size.csv'
    dataset = dataset_class(
        input_file_path=dataset_path,
        solute_list_path=solute_list_path,
        solvent_list_path=solvent_list_path,
        generate_all=True,
        normalize=normalize)
    dataset_size = len(dataset)
        
    # print dataset size
    print('dataset size: {}'.format(dataset_size))

    train_indices = np.arange(dataset_size)

    # Dataloader
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   collate_fn=collate_fn,
                                                   shuffle=False,
                                                   drop_last=True)
    
    #Determine if empty solute-solvent systems are necessary
    if use_solvsys:
        empty_solvsys = dataset.generate_solvsys(batch_size).to("cuda")
    else:
        empty_solvsys = None
    
    #Define model
    if model_type == "water_gegnn_SigmaProfile_noPooling": 
        model = water_gegnn_SigmaProfile_noPooling(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_dropout_rate=mlp_dropout_rate, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers, num_step_message_passing=num_step_message_passing).cuda()
    elif model_type == "RittigPure": 
        model = RittigPure(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_dropout_rate=mlp_dropout_rate, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers, num_step_message_passing=num_step_message_passing).cuda()
    elif model_type == "RittigPure_optimized": 
        model = RittigPure_opt(in_dim=74, gcn_dim=gcn_dim, mpnn_dim=mpnn_dim, mlp_dim=mlp_dim, n_classes=1, mpnn_activation=enc_activation, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers,
                               num_step_message_passing=num_step_message_passing, device=device).cuda()
    elif model_type == "AbranchesPure":
        model = AbranchesPure(in_dim=50, n_classes=1, weight_init_fn=weight_init_fn, sigma_profile_model_path=sigma_profile_model_path, sigma_profile_length=51).cuda()
    elif model_type == "Rittig_Abranches":
        model = Rittig_Abranches(in_dim_rit=74, in_dim_abr=50, n_classes=1, sigma_profile_length=51, hidden_dim=hidden_dim, weight_init_fn=weight_init_fn,
                                     mpnn_activation=enc_activation, num_step_message_passing=num_step_message_passing, mlp_dropout_rate=mlp_dropout_rate,
                                     mlp_num_hid_layers=mlp_num_hid_layers, mlp_activation='softplus',
                                     sigma_profile_model_path=sigma_profile_model_path).cuda()
    elif model_type == "Rittig_Abranches_optimized":
        model = RA_opt(in_dim_rit=74, in_dim_abr=50, n_classes=1, sigma_profile_length=51, weight_init_fn=weight_init_fn,
                        mpnn_activation= enc_activation, num_step_message_passing=num_step_message_passing, mlp_activation=mlp_activation,
                        mlp_num_hid_layers=mlp_num_hid_layers,
                        gcn_dim=gcn_dim, mpnn_dim=mpnn_dim, mlp_dim=mlp_dim,
                        n_conv_filters=n_conv_filters, kernel_size=kernel_size, pool_size=pool_size, SP_dim=SP_dim,
                        device=device,
                        sigma_profile_model_path=sigma_profile_model_path).to(device)
    elif model_type == "Rittig_Abranches_variant_optimized":
        model = RA_var_opt(in_dim_rit=74, in_dim_abr=50, n_classes=1, sigma_profile_length=51, weight_init_fn=weight_init_fn,
                            mpnn_activation=enc_activation, num_step_message_passing=num_step_message_passing, mlp_activation=mlp_activation,
                            mlp_num_hid_layers=mlp_num_hid_layers,
                            gcn_dim=gcn_dim, mpnn_dim=mpnn_dim, mlp_dim=mlp_dim,
                            device=device,
                            sigma_profile_model_path=sigma_profile_model_path).to(device)  

    print(model)
    #Load model weights
    if checkpoint_path != None:
        model.load_state_dict(torch.load(checkpoint_path)["model_state_dict"])

    #Training parameters
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.9))
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
            empty_solvsys = empty_solvsys,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            l2_coef=l2_coef,
            device=device 
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
            }, 'C:/Users/kverg/GDI-NN/results_water/Complete_trainset_final_model_{}.pth'.format(save_add))
        
    np.save('C:/Users/kverg/GDI-NN/results_water/Complete_trainset_train_loss_{}.npy'.format(save_add),np.array(train_loss_save))

    
    # Plotting for all cv runs
    ## Training and validation loss

    plt.figure(figsize=(16,8))
    train_losses = np.load('C:/Users/kverg/GDI-NN/results_water/Complete_trainset_train_loss_{}.npy'.format(save_add))
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
    plt.savefig(f'C:/Users/kverg/GDI-NN/results_water/Complete_trainset_loss_{save_add}_test.png',dpi=300)            
    
    all_end = time.time() - all_start
    print(all_end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument('--model_type', default="RittigPure_optimized", type=str)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--mlp_dropout_rate', default=0, type=float)
    parser.add_argument('--mlp_activation', default="softplus", type=str)
    parser.add_argument('--enc_activation', default="relu", type=str)
    parser.add_argument('--mlp_num_hid_layers', default=2, type=int) 
    parser.add_argument('--gcn_dim', default=156, type=int) 
    parser.add_argument('--mpnn_dim', default=128, type=int) 
    parser.add_argument('--mlp_dim', default=134, type=int) 
    parser.add_argument('--num_step_message_passing', default=1, type=int) 
    parser.add_argument('--n_conv_filters', default=6, type=int) 
    parser.add_argument('--kernel_size', default=4, type=int) 
    parser.add_argument('--pool_size', default=10, type=int) 
    parser.add_argument('--SP_dim', default=7, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)

    # Training
    parser.add_argument('--lr', default=1e-3, type=float)                   
    parser.add_argument('--use_lr_scheduler', default="True", type=str)
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--early_stopping', default='False', type=str)
    parser.add_argument('--l2_coef', default=1e-5, type=str)
    parser.add_argument('--SP', default="True", type=str)
    parser.add_argument('--normalize', default='False', type=str)
    parser.add_argument('--use_solvsys', default='True', type=str)

    # Data, split, and logs
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--weight_init', default='None', type=str)
    parser.add_argument('--data', default="RAvarOpt", type=str)
    parser.add_argument('--wandb_logs', default="False", type=str)

    # load checkpoint (for fine-tuning)
    parser.add_argument('--checkpoint_path', default='None', type=str)

    hyperparameter = parser.parse_args()

    if hyperparameter.use_lr_scheduler == "False": hyperparameter.use_lr_scheduler = False
    if hyperparameter.use_lr_scheduler == "True": hyperparameter.use_lr_scheduler = True
    if hyperparameter.wandb_logs == "False": hyperparameter.wandb_logs = False
    if hyperparameter.wandb_logs == "True": hyperparameter.wandb_logs = True
    if hyperparameter.early_stopping == "False": hyperparameter.early_stopping = False
    if hyperparameter.early_stopping == "True": hyperparameter.early_stopping = True
    if hyperparameter.SP == "False": hyperparameter.SP = False
    if hyperparameter.SP == "True": hyperparameter.SP = True
    if hyperparameter.use_solvsys == "False": hyperparameter.use_solvsys = False
    if hyperparameter.use_solvsys == "True": hyperparameter.use_solvsys = True

    print(hyperparameter)

    main(hyperparameter=hyperparameter)
