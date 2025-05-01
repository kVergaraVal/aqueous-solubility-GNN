
from __future__ import absolute_import
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
# external imports
import sys, random, pickle, csv, time
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
import matplotlib.pyplot as plt
import wandb
import argparse

# internal imports
from util_water.generate_dataset_for_training_water import solute_dataset_water, collate_solute_water, solute_dataset_water_SP, collate_solute_water_SP
from model.model_GNN_water import water_gegnn, water_gegnn_no_Pooling, water_gegnn_SigmaProfile_noPooling, RittigPure, AbranchesPure, Rittig_Abranches
from util_water import data_splitting_water


os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())

class AccumulationMeter(object):
    #################################################################
    #                                                               #
    # This class initiates an object of a determined metric         #
    #     which updates on each iteration of a batch                #
    #                                                               #
    # input: essentially, it *should* only take an empty argument   #
    #                                                               #
    # output: an object with different statistical attributes       #
    #################################################################
    def __init__(self): #Initializes the metric with 0-value statistical attributes
        self.reset()

    def reset(self): #Resets the statistical attributes to 0
        self.value = 0.0
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0

    def update(self, value, n=1): #updates each statistical attribute after n iterations of a batch
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def train(cv_index, epoch, train_loader, empty_solvsys, model, loss_fn, l2_coef, optimizer, wandb_logs=False):
    ###################################################################################
    #
    # This method executes the training of the model                                    
    #
    # input:
    #   cv_index: 
    #   epoch:
    #   train_loader:
    #   empty_solvsys:
    #   model:
    #   loss_fn1:
    #   loss_fn2:
    #   optimizer:
    #   pinn_lambda:
    #   batch_adding:
    #   wandb_logs:
    #
    # output:
    ###################################################################################
    stage = "train"
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()

    # Set model to training mode
    model.train()
    for i, component_data in enumerate(train_loader):
        end = time.time()
        lablogS = component_data['LogS'].float().cuda() # logS

        # Model predictions and gradients
        y = None
        with torch.backends.cudnn.flags(enabled=False): #disables the CuDNN during prediction, this is could have different reasons: 1. reproducibiltiy issues, 2. performance issues, 3. speed issues
            if empty_solvsys != None:
                y = model(component_data, empty_solvsys)
            else:
                y = model(component_data)
            
            # Batch adding: Pseudo data is added to data samples in each batch -> only considered for Gibbs-Duhem loss
            #if batch_adding:
            #    # randomly sample x1-x2-pairs from uniform distribution
            #    add_x1 = torch.distributions.uniform.Uniform(0,1).sample([solvdata["solv1_x"].shape[0],]).cuda()
            #    add_x2 = 1 - add_x1
            #    # replace data x1 with pseudo x1
            #    solvdata["solv1_x"] = add_x1
            #    # reevaluate model and get gradients
            #    _, add_y1_x1, add_y2_x1 = model(solvdata, empty_solvsys, gamma_grad=True)
            #    # Add pseudo data and additional gradients for loss calculation
            #    x1, x2 = torch.cat([x1, add_x1]), torch.cat([x2, add_x2])
            #    y1_x1, y2_x1 = torch.cat([y1_x1, add_y1_x1]), torch.cat([y2_x1, add_y2_x1])
      
        # Prediction loss
        loss = loss_fn(y[:,0],lablogS) # loss logS
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
                f"train_loss_accum_cv{cv_index}": loss_accum.avg,
            }, step=epoch)
            
    return [loss_accum.avg]


def validate(cv_index, epoch, val_loader, empty_solvsys, model, loss_fn, wandb_logs=False):
    stage = 'validate'
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()

    # Set model to eval mode
    model.eval()

    with torch.set_grad_enabled(True):
        for i, component_data in enumerate(val_loader):
            end = time.time()
            lablogS = component_data['LogS'].float().cuda() # logS

            # Model predictions and gradients
            y = None
            with torch.backends.cudnn.flags(enabled=False):
                if empty_solvsys != None:
                    y = model(component_data, empty_solvsys)
                else:
                    y = model(component_data)

            # Prediction loss
            loss = loss_fn(y[:,0],lablogS) # loss logS
            
            # Update stats
            loss_accum.update(loss.item(),lablogS.size(0))
            batch_time.update(time.time() - end)

            if i % 100 == 0:
                print('Epoch [{}][{}/{}]'
                    'Time {:.3f} ({:.3f})\t'
                    'Loss {:.3f} ({:.3f})\t'.format(
                    epoch, i, len(val_loader), 
                    batch_time.sum, batch_time.avg,
                    loss_accum.value, loss_accum.avg))

    # Logging
    print("[Stage {}]: Epoch {} finished at time={:.3f} with loss={:.3f}".format(
            stage, epoch, batch_time.sum, loss_accum.avg))
    if wandb_logs:
        wandb.log({
                    f"epoch": epoch,
                    f"val_loss_accum_cv{cv_index}": loss_accum.avg,
                }, step=epoch)

    return [loss_accum.avg]


def main(hyperparameter):
    all_start = time.time()

    # fix seed
    seed = hyperparameter.seed
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
    hidden_dim = hyperparameter.hidden_dim
    num_step_message_passing = hyperparameter.num_step_message_passing

    # training parameters 
    batch_size = hyperparameter.batch_size
    #batch_adding = hyperparameter.batch_adding
    lr = hyperparameter.lr
    use_lr_scheduler = hyperparameter.use_lr_scheduler
    epochs = hyperparameter.epochs
    l2_coef = hyperparameter.l2_coef

    # data parameters
    data = hyperparameter.data
    data_split_mode = hyperparameter.data_split_mode
    n_splits = hyperparameter.num_splits
    SP = hyperparameter.SP
    normalize = hyperparameter.normalize
    w_init = hyperparameter.weight_init
    use_solvsys = hyperparameter.use_solvsys

    save_add = f"_{data}_split-{data_split_mode}_nums{n_splits}-{model_type}_hidDim_{hidden_dim}_dropout{mlp_dropout_rate}_act{mlp_activation}_encAct{enc_activation}_nhl{mlp_num_hid_layers}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}_L2coef{l2_coef}_batchsize{batch_size}_mespass{num_step_message_passing}_norm{normalize}_WI{w_init}"

    
    config = hyperparameter
    wandb_logs = config.wandb_logs
    #if wandb_logs:
    #    if "binaryGamma" == config.data:
    #        project = "binaryGamma"
    #    else:
    #        raise ValueError(f"Wandb project not available for dataset {config.data}")
    #    wandb.init(config=config, project=project, dir=f"./wandb/{config.data}")


    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if SP==True:
        dataset_class = solute_dataset_water_SP
        collate_fn = collate_solute_water_SP
        sigma_profile_model_path = "C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_final_model__MMFF_sp_Database_Sigma_profile_prediction_actrelu_encActrelu_lrschedFalse_epochs700_lr0.00151_L2coef_7.79e-06_batchsize16_earlystopping_False.pth"
    else:
        dataset_class = solute_dataset_water
        collate_fn = collate_solute_water
    
    # read dataset file
    if data_split_mode == 'standard':
        dataset_path = './data/solubility_water/COMPATIBLE_unique_train_water_final.csv'
        solute_list_path = './data/solubility_water/solute_list.csv'
        solvent_list_path = './data/solubility_water/solvent_list_water.csv'
        dataset = dataset_class(
            input_file_path=dataset_path,
            solute_list_path=solute_list_path,
            solvent_list_path=solvent_list_path,
            generate_all=True,
            normalize=normalize)
        dataset_size = len(dataset)
    
        # get data splits and load dataset
        train_indices_splits, val_indices_splits = data_splitting_water.data_split_standard(
            dataset=dataset, 
            n_splits=n_splits, 
            seed=seed
            )
        
    elif data_split_mode in ['stratified_hybrid', 'stratHybrid', 'stratHyb']:
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
    
        # get data splits and load dataset
        train_indices_splits, val_indices_splits = data_splitting_water.data_split_stratified_hybrid(
            dataset=dataset, 
            n_splits=n_splits, 
            seed=seed
            )
        
    # print dataset size
    print('dataset size: {}'.format(dataset_size))

    cv_index = 0
    index_list_train = []
    index_list_valid = []
    for train_indices, val_indices in zip(train_indices_splits, val_indices_splits):
        index_list_train.append(train_indices)
        index_list_valid.append(val_indices)

        # Dataloader
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   collate_fn=collate_fn,
                                                   shuffle=False,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=valid_sampler,
                                                 collate_fn=collate_fn,
                                                 shuffle=False,
                                                 drop_last=True)
        
        if use_solvsys:
            empty_solvsys = dataset.generate_solvsys(batch_size).to("cuda")
        else:
            empty_solvsys = None
        
        # initialize model
        if model_type == "water_gegnn":
            model = water_gegnn(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_activation=mlp_activation, mpnn_activation=enc_activation).cuda()
        elif model_type == "water_gegnn_no_Pooling": 
            model = water_gegnn_no_Pooling(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_dropout_rate=mlp_dropout_rate, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers).cuda()
        elif model_type == "water_gegnn_SigmaProfile_noPooling": 
            model = water_gegnn_SigmaProfile_noPooling(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_dropout_rate=mlp_dropout_rate, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers, num_step_message_passing=num_step_message_passing).cuda()
        elif model_type == "RittigPure":
            model = RittigPure(in_dim=74, hidden_dim=hidden_dim, n_classes=1, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2, num_step_message_passing=1).cuda()
        elif model_type == "AbranchesPure":
            model = AbranchesPure(in_dim=50, n_classes=1, weight_init_fn=weight_init_fn, sigma_profile_model_path=sigma_profile_model_path, sigma_profile_length=51).cuda()
        elif model_type == "Rittig_Abranches":
            model = Rittig_Abranches(in_dim_rit=74, in_dim_abr=50, n_classes=1, sigma_profile_length=51, hidden_dim=hidden_dim, weight_init_fn=weight_init_fn,
                                     mpnn_activation=enc_activation, num_step_message_passing=num_step_message_passing, mlp_dropout_rate=mlp_dropout_rate,
                                     mlp_num_hid_layers=mlp_num_hid_layers, mlp_activation='softplus',
                                     sigma_profile_model_path=sigma_profile_model_path).cuda()
        print(model)
        
        loss_fn = nn.MSELoss().cuda()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.9))
        if use_lr_scheduler:
            scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
        
        # Training
        best_loss = 1000000
        train_loss_save = []
        val_loss_save = []
    
        for epoch in range(1, epochs+1):

            # Train epoch
            train_loss = train(
                cv_index=cv_index, 
                epoch=epoch,
                train_loader=train_loader,
                empty_solvsys=empty_solvsys,
                model=model,
                loss_fn=loss_fn,
                l2_coef=l2_coef,
                optimizer=optimizer, 
                #batch_adding=batch_adding, 
                )
            train_loss_save.append(train_loss[0])

            # Val epoch
            val_loss = validate(
                cv_index=cv_index, 
                epoch=epoch,
                val_loader=val_loader,
                empty_solvsys=empty_solvsys,
                model=model,
                loss_fn=loss_fn,
                )
            
            val_loss_save.append(val_loss[0])

            # LR scheduler
            if use_lr_scheduler:
                scheduler.step(train_loss[0])
            
            is_best = val_loss[0] < best_loss
            best_loss = min(val_loss[0], best_loss)
                               
            if val_loss[0] == best_loss:
                print(f'new best_loss: {best_loss} at epoch {epoch}')
            # Save Final model and stats
                torch.save({
                        'epoch': epoch,
                        'model_arch': model_type,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': best_loss
                        }, './results_water/final_model_cv{}{}.pth'.format(cv_index, save_add))
        
        np.save('./results_water/train_loss_cv{}{}.npy'.format(cv_index, save_add),np.array(train_loss_save))
        np.save('./results_water/val_loss_cv{}{}.npy'.format(cv_index, save_add),np.array(val_loss_save))

        cv_index += 1

    index_list_train = np.array(index_list_train, dtype=object)
    np.save(f'./results_water/train_ind_list{save_add}.npy',index_list_train)
    index_list_valid = np.array(index_list_valid, dtype=object)
    np.save(f'./results_water/valid_ind_list{save_add}.npy',index_list_valid)
    
    # Plotting for all cv runs
    ## Training and validation loss
    train_mse_min = []
    valid_mse_min = []
    train_mse_last = []
    valid_mse_last = []
    train_mse_avg = []
    valid_mse_avg = []
    #best_epochs = []
    plt.figure(figsize=(16,8))
    plt_rows = 2 if n_splits > 4 else 1
    for cv_index in range(n_splits):
        train_losses = np.load('./results_water/train_loss_cv{}{}.npy'.format(cv_index, save_add))
        valid_losses = np.load('./results_water/val_loss_cv{}{}.npy'.format(cv_index, save_add))
        plt.subplot(plt_rows,int(n_splits/2+0.5)+1,cv_index+1)
        plt.plot(train_losses,label="train loss cv{}".format(cv_index))
        plt.plot(valid_losses,label="valid loss cv{}".format(cv_index))
        plt.xlabel("epoch (training iteration)")
        plt.ylabel("loss")
        plt.legend(loc="best")
        best_model_val = np.min(valid_losses)
        best_model_val_index = np.argmin(valid_losses)
        train_mse_min.append(train_losses[best_model_val_index])
        valid_mse_min.append(best_model_val)
        train_mse_last.append(train_losses[-1])
        valid_mse_last.append(valid_losses[-1])
        train_mse_avg.append(np.mean(train_losses))
        valid_mse_avg.append(np.mean(valid_losses))
        #best_epochs.append(best_model_val_index+1)
    train_mse_min = np.sqrt(np.array(train_mse_min))
    valid_mse_min = np.sqrt(np.array(valid_mse_min))
    train_mse_last = np.sqrt(np.array(train_mse_last))
    valid_mse_last = np.sqrt(np.array(valid_mse_last))
    train_mse_avg = np.sqrt(np.array(train_mse_avg))
    valid_mse_avg = np.sqrt(np.array(valid_mse_avg))
    rmse_str = (r'best model Train RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'best model Val RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'last model Train RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'last model Val RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'average Train RMSE = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'average Train RMSE = {:.2f} $\pm$ {:.2f}'.format(
            np.mean(train_mse_min), np.std(train_mse_min), np.mean(valid_mse_min), np.std(valid_mse_min),
            np.mean(train_mse_last), np.std(train_mse_last), np.mean(valid_mse_last), np.std(valid_mse_last),
            np.mean(train_mse_avg), np.std(train_mse_avg), np.mean(valid_mse_avg), np.std(valid_mse_avg)))
    plt.subplot(2,3,6)
    plt.text(0,0.5, rmse_str, fontsize=12)
    plt.axis('off')
    plt.savefig(f'./results_water/cvloss{save_add}_test.png',dpi=300)            
    
    all_end = time.time() - all_start
    print(all_end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument('--model_type', default="Rittig_Abranches", type=str) # options: water_gegnn, water_gegnn_no_Pooling, water_gegnn_SigmaProfile_noPooling, RittigPure, AbranchesPure, RittigNoSolvSys, Rittig_Abranches
    parser.add_argument('--batch_size', default=100, type=int) ##ORIGINAL: 100 // checkpoint: 512
    parser.add_argument('--mlp_dropout_rate', default=0, type=float)
    parser.add_argument('--mlp_activation', default="softplus", type=str) #ORIGINAL: softplus 
    parser.add_argument('--enc_activation', default="relu", type=str) #ORIGINAL: relu
    parser.add_argument('--mlp_num_hid_layers', default=2, type=int) ##ORIGINAL: 2 // checkpoint: 0? 
    parser.add_argument('--hidden_dim', default=32, type=int) ##ORIGINAL: 256 // checkpoint: 32
    parser.add_argument('--num_step_message_passing', default=0, type=int) #ORIGINAL: 1 (HYPERPARAMETER NOT CONSIDERED IN ORIGINAL)

    # Training
    parser.add_argument('--lr', default=1e-3, type=float)                   #ORIGINAL: 1e-3
    parser.add_argument('--use_lr_scheduler', default="True", type=str)
    parser.add_argument('--epochs', default=100, type=int)  ##ORIGINAL: 100
    parser.add_argument('--l2_coef', default=1e-5, type=float)
    parser.add_argument('--SP', default='True', type=str)
    parser.add_argument('--normalize', default='False', type=str)
    parser.add_argument('--use_solvsys', default='True', type=str)

    # Data, split, and logs
    parser.add_argument('--seed', default=2021, type=int)
    parser.add_argument('--weight_init', default='uniform', type=str)
    parser.add_argument('--data', default="solubWater", type=str)
    parser.add_argument('--data_split_mode', default="stratHyb", type=str)
    parser.add_argument('--num_splits', default=5, type=int)
    parser.add_argument('--wandb_logs', default="False", type=str)

    hyperparameter = parser.parse_args()

    # Workaround to set booleans via bash
    if hyperparameter.use_lr_scheduler == "False": hyperparameter.use_lr_scheduler = False
    if hyperparameter.use_lr_scheduler == "True": hyperparameter.use_lr_scheduler = True
    if hyperparameter.wandb_logs == "False": hyperparameter.wandb_logs = False
    if hyperparameter.wandb_logs == "True": hyperparameter.wandb_logs = True
    if hyperparameter.SP == "False": hyperparameter.SP = False
    if hyperparameter.SP == "True": hyperparameter.SP = True
    if hyperparameter.normalize == "False": hyperparameter.normalize = False
    if hyperparameter.normalize == "True": hyperparameter.normalize = True
    if hyperparameter.use_solvsys == "False": hyperparameter.use_solvsys = False
    if hyperparameter.use_solvsys == "True": hyperparameter.use_solvsys = True

    print(hyperparameter)

    main(hyperparameter=hyperparameter)
