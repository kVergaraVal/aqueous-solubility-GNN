
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
import xgboost

# internal imports
from util_water.generate_dataset_for_training_water import solute_dataset_water, collate_solute_water, solute_dataset_water_SP, collate_solute_water_SP
from model.model_GNN_water import water_gegnn, water_gegnn_no_Pooling, water_gegnn_SigmaProfile_noPooling, RittigPure, AbranchesPure, Rittig_Abranches, RA_var_opt
from util_water import data_splitting_water


os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())

def main(hyperparameter):
    all_start = time.time()
    #device = torch.device("cuda:0")
    device = torch.device('cpu')

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
    gcn_dim = hyperparameter.gcn_dim
    enc_activation = hyperparameter.enc_activation
    mpnn_dim = hyperparameter.mpnn_dim
    mlp_activation = hyperparameter.mlp_activation
    mlp_num_hid_layers = hyperparameter.mlp_num_hid_layers
    mlp_dim = hyperparameter.mlp_dim
    num_step_message_passing = hyperparameter.num_step_message_passing

    # training parameters 
    batch_size = hyperparameter.batch_size
    lr = hyperparameter.lr
    use_lr_scheduler = hyperparameter.use_lr_scheduler
    epochs = hyperparameter.epochs
    l2_coef = hyperparameter.l2_coef

    # data parameters
    SP = hyperparameter.SP
    normalize = hyperparameter.normalize
    w_init = hyperparameter.weight_init
    use_solvsys = hyperparameter.use_solvsys

    #XGB
    n_estimators = hyperparameter.n_estimators
    eta = hyperparameter.eta
    max_depth = hyperparameter.max_depth
    n_jobs = hyperparameter.n_jobs

    save_add = f"hybXGB_nestim{n_estimators}_eta{eta}_maxdepth{max_depth}_gcndim{gcn_dim}_mpnndim{mpnn_dim}_mlpdim{mlp_dim}_act{mlp_activation}_encAct{enc_activation}_nhl{mlp_num_hid_layers}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}_L2coef{l2_coef}_batchsize{batch_size}_mespass{num_step_message_passing}_norm{normalize}_WI{w_init}"

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
    
    train_indices = np.arange(dataset_size)
    batch_size = dataset_size
        
    # print dataset size
    print('dataset size: {}'.format(dataset_size))

    # Dataloader
    train_sampler = SubsetRandomSampler(train_indices)
        
    if use_solvsys:
        empty_solvsys = dataset.generate_solvsys(batch_size).to("cuda")
    else:
        empty_solvsys = None
        
    # initialize model
    model_RA = RA_var_opt(in_dim_rit=74, in_dim_abr=50, n_classes=1, sigma_profile_length=51, weight_init_fn=weight_init_fn,
                              mpnn_activation=enc_activation, num_step_message_passing=num_step_message_passing, mlp_activation=mlp_activation, mlp_num_hid_layers=mlp_num_hid_layers,
                              gcn_dim=gcn_dim, mpnn_dim=mpnn_dim, mlp_dim=mlp_dim,
                              device=device,
                              sigma_profile_model_path=sigma_profile_model_path).to(device)
    model_xgb = xgboost.XGBRegressor(n_estimators=n_estimators,
                                     learning_rate=eta,
                                     max_depth=max_depth,
                                     n_jobs=n_jobs,
                                     device='cpu')

    print(model_xgb)

    model_RA_best = model_RA.cpu()
    RA_best_load_str = 'C:/Users/kverg/GDI-NN/results_water/Complete_trainset_final_model__Rittig_Abranches_variant_optimized_lrschedTrue_epochs100_lr0.0004173_L2coef4.046e-10_batchsize80__weightNone_gcnDim282_mpnnDim198_mlpDim119.pth'
    model_RA_best.load_state_dict(torch.load(RA_best_load_str, weights_only=True)["model_state_dict"])

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
        
    model_RA_best.mfp_trans.register_forward_hook(get_activation('mfp_trans'))

    # XGB training
    model_RA_best.eval()
    train_loader_xgb = torch.utils.data.DataLoader(dataset, batch_size=len(train_indices),
                                                   sampler=train_sampler,
                                                   collate_fn=collate_fn,
                                                   shuffle=False,
                                                   drop_last=True)
    #training_subset = torch.utils.data.Subset(dataset, train_indices)
    output_array_RA_best = torch.tensor([])
    target_array = torch.tensor([])
    XGB_input_array = torch.tensor([])
    empty_solvsys = dataset.generate_solvsys(len(train_indices)).to("cpu")
    print('beginning XGB training...')
    count=0
    with torch.set_grad_enabled(True):
        for _, component_data in enumerate(train_loader_xgb):
            if count % 500==0:
                print(f'[{count}]/[{len(train_indices)}]')
            target = component_data['LogS']
            with torch.backends.cudnn.flags(enabled=False):
                output_RA_best = model_RA_best(component_data, empty_solvsys, device=torch.device('cpu'))
                XGB_input = activation['mfp_trans'].detach()
            output_array_RA_best = torch.concatenate([output_array_RA_best, output_RA_best])
            target_array = torch.concatenate([target_array, target])
            XGB_input_array = torch.concatenate([XGB_input_array, XGB_input])
            count+=1

    target_array = target_array.numpy()
    XGB_input_array = XGB_input_array.numpy()

    model_xgb.fit(XGB_input, target_array)

    # XGB evaluation
    #valid_subset = torch.utils.data.Subset(dataset, val_indices)
    val_indices = train_indices
    val_loader_xgb = train_loader_xgb
    output_array_RA_best_valid = torch.tensor([])
    target_array_valid = torch.tensor([])
    XGB_input_array_valid = torch.tensor([])
    empty_solvsys = dataset.generate_solvsys(len(val_indices)).to("cpu")
    print('beginning XGB evaluation...')
    count=0
    with torch.set_grad_enabled(True):
        for _, component_data in enumerate(val_loader_xgb):
            if count % 500==0:
                print(f'[{count}]/[{len(val_indices)}]')
            target_valid = component_data['LogS']
            with torch.backends.cudnn.flags(enabled=False):
                output_RA_best_valid = model_RA_best(component_data, empty_solvsys, device=torch.device('cpu'))
                XGB_input_valid = activation['mfp_trans'].detach()
            output_array_RA_best_valid = torch.concatenate([output_array_RA_best_valid, output_RA_best_valid])
            target_array_valid = torch.concatenate([target_array_valid, target_valid])
            XGB_input_array_valid = torch.concatenate([XGB_input_array_valid, XGB_input_valid])
            count+=1

        target_array_valid = target_array_valid.numpy()
        XGB_input_array_valid = XGB_input_array_valid.numpy()

        predicted_array_valid = model_xgb.predict(XGB_input_array_valid)

        xgb_mse = ((target_array_valid - predicted_array_valid)**2).mean()

        print(f'hybrid model final loss is MSE={xgb_mse}')

    # Save model
    filename = 'C:/Users/kverg/GDI-NN/results_hybrids/' + save_add + '.bst'
    model_xgb.save_model(filename)

    all_end = time.time() - all_start
    print(all_end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model architecture
    parser.add_argument('--gcn_dim', default=282, type=int)
    parser.add_argument('--enc_activation', default="relu", type=str) #ORIGINAL: relu
    parser.add_argument('--mpnn_dim', default=198, type=int)
    parser.add_argument('--mlp_activation', default="softplus", type=str) #ORIGINAL: softplus 
    parser.add_argument('--mlp_num_hid_layers', default=2, type=int) ##ORIGINAL: 2 // checkpoint: 0? 
    parser.add_argument('--mlp_dim', default=119, type=int) ##ORIGINAL: 256 // checkpoint: 32
    parser.add_argument('--num_step_message_passing', default=1, type=int) #ORIGINAL: 1 (HYPERPARAMETER NOT CONSIDERED IN ORIGINAL)

    # Training
    parser.add_argument('--lr', default=4.173e-4, type=float)                   #ORIGINAL: 1e-3
    parser.add_argument('--use_lr_scheduler', default="True", type=str)
    parser.add_argument('--batch_size', default=80, type=int) ##ORIGINAL: 100 // checkpoint: 512
    parser.add_argument('--epochs', default=100, type=int)  ##ORIGINAL: 100
    parser.add_argument('--l2_coef', default=4.046e-10, type=float)
    parser.add_argument('--SP', default='True', type=str)
    parser.add_argument('--normalize', default='False', type=str)
    parser.add_argument('--use_solvsys', default='True', type=str)

    # Data, split, and logs
    parser.add_argument('--seed', default=2025, type=int)
    parser.add_argument('--weight_init', default='uniform', type=str)
    parser.add_argument('--data', default="solubWater", type=str)
    parser.add_argument('--data_split_mode', default="stratHyb", type=str)
    parser.add_argument('--num_splits', default=5, type=int)
    parser.add_argument('--wandb_logs', default="False", type=str)

    #XGB
    parser.add_argument('--n_estimators', default=49, type=int)
    parser.add_argument('--eta', default=7.6343e-2, type=float)
    parser.add_argument('--max_depth', default=4, type=int)
    parser.add_argument('--n_jobs', default=8, type=int)

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

'''    
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
'''
