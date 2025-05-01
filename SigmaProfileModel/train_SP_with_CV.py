import sys
# append the path of the parent directory
sys.path.append("..")

import torch
import random
import torch.nn as nn
import torch.optim
import torch.utils.data
from dgl.nn.pytorch import GraphConv, SumPooling
import torch.nn.functional as F
import pandas as pd
import dgl
from util_SP.atom_feat_encoding_SP import MMFFAtomFeaturizer
from util_SP.molecular_graph_SP import mol_to_bigraph
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import KFold, StratifiedKFold
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
import wandb
import argparse
from rdkit import Chem

class SigmaProfileGCN(nn.Module):
    def __init__(self, in_dim):
        super(SigmaProfileGCN, self).__init__()
        self.conv1 = GraphConv(in_dim, 300)
        self.conv2 = GraphConv(300, 153)
        self.conv3 = GraphConv(153, 161)
        #self.global_conv1 = GraphConv(hidden_dim+1, hidden_dim)
        self.classify = nn.Linear(161, 51)
        self.globalpooling = SumPooling()

    def forward(self, molecule_data):
        graph = molecule_data['g'].to("cuda")
        with graph.local_scope():
            h1 = graph.ndata['h'].float().cuda()
                
            h1_temp = F.relu(self.conv1(graph, h1))
            h1_temp = F.relu(self.conv2(graph, h1_temp))
            h1_temp = F.relu(self.conv3(graph, h1_temp))
            graph.ndata['h'] = h1_temp
                
            #hg1 = dgl.mean_nodes(graph, 'h')
            #hg1 = torch.cat((hg1, None), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
            #hg = hg1

            hg = h1_temp
            output = F.relu(self.classify(hg))
            output = self.globalpooling(graph, output)                        
            #output = torch.cat(
            #        (output[0:len(output)//2,:],
            #        output[len(output)//2:,:]),axis=1)
            return output
        
class dataset_Sigma_Profiles:

    def __init__(self, input_file_path = "C:\\Users\\kverg\\GDI-NN\\SigmaProfileModel\\Databases\\MMFF_spDatabase_Train.csv",
                 generate_all = False):
        if input_file_path:
            self.dataset = pd.read_csv(input_file_path, index_col='VT-2005 Index')
            self.dataset_smiles = self.dataset['SMILES'].to_dict()              #Dictionary of smiles of the entire dataset
        self.component_data = {}
        if generate_all:
            self.generate_all()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()                  
        data = self.dataset.iloc[idx]           
        id = self.dataset.index[idx]
        #print(f"self.component_data is: {self.component_data}")  
        compound = self.component_data[id]    
        sample['g'] = compound[0]                
        sample["compound_id"] = int(id)
        SP_vect = []
        for col in self.dataset.columns[1:]:
            SP_vect.append(data[col])
        sample['SP'] = SP_vect
        return sample
    
    def generate_sample(self,chemical_id,smiles,SP=None):
        component_data = {}
        component_data[chemical_id] = []
        sml = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
        mol = Chem.MolFromSmiles(sml)
        component_data[chemical_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                            node_featurizer=MMFFAtomFeaturizer(),
                                            edge_featurizer=None,
                                            canonical_atom_order=False,
                                            explicit_hydrogens=True,
                                            num_virtual_nodes=0
                                            ))
        sample = {}
        compound = component_data[chemical_id]
        sample['g'] = compound[0]
        if SP:            
            sample['SP'] = SP
        return sample
    
    def generate_all(self):
        for mol_id in self.dataset_smiles:
            mol = Chem.MolFromSmiles(self.dataset_smiles[mol_id])
            self.component_data[mol_id] = []
            self.component_data[mol_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                node_featurizer=MMFFAtomFeaturizer(),
                                                                edge_featurizer=None,
                                                                canonical_atom_order=False,
                                                                explicit_hydrogens=True,
                                                                num_virtual_nodes=0
                                                                ))

def collate_SP(batch):
    #print(f'batch keys are {batch[0].keys()}')
    keys = list(batch[0].keys())[2:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    #print(f'samples are {samples}')
    #print("modified samples are:")
    #print(samples)
    batched_sample = {}
    batched_sample['g'] = dgl.batch(samples[0])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+2])
        batched_sample[key] = torch.tensor(samples[i+2])
    return batched_sample

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, inputs, targets):
        criterion = nn.L1Loss()
        MAE = criterion(inputs,targets)
        buff = 0.1
        inputs = inputs+buff
        targets = targets+buff
        LE = torch.log(targets)-torch.log(inputs)
        SLE = torch.square(LE)
        MSLE = SLE.mean()
        loss = MAE + MSLE
        return loss
    
def data_split_Kfold(dataset, n_splits=5, seed=2021):
    train_indices_splits = []
    val_indices_splits = []
    
    dataset_size = len(dataset)
    all_ind = np.arange(dataset_size)
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_indices, valid_indices in kf.split(all_ind):
        train_indices_splits.append(train_indices)
        val_indices_splits.append(valid_indices)
    return train_indices_splits, val_indices_splits


def data_split_stratified_Kfold(dataset, n_splits=5, seed=2021):
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = str(torch.cuda.current_device())

class AccumulationMeter(object):
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

def train(cv_index, epoch, train_loader, model, loss_fn, l2_coef, optimizer, wandb_logs=False):
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
                f"train_loss_accum_cv{cv_index}": loss_accum.avg,
            }, step=epoch)
            
    return [loss_accum.avg]


def validate(cv_index, epoch, val_loader, model, loss_fn, wandb_logs=False):
    stage = 'validate'
    batch_time = AccumulationMeter()
    loss_accum = AccumulationMeter()

    # Set model to eval mode
    model.eval()

    with torch.set_grad_enabled(True):
        for i, component_data in enumerate(val_loader):
            end = time.time()
            sigma_profile = component_data['SP'].float().cuda() # logS

            # Model predictions and gradients
            y = None
            with torch.backends.cudnn.flags(enabled=False):
                y = model(component_data)   

            # Prediction loss
            loss = loss_fn(y,sigma_profile) # loss logS
            
            # Update stats
            loss_accum.update(loss.item(),sigma_profile.size(0))
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
    data_split_mode = hyperparameter.data_split_mode
    n_splits = hyperparameter.num_splits

    save_add = f"_{data}_split-{data_split_mode}_nums{n_splits}-{model_type}_act{mlp_activation}_encAct{enc_activation}_lrsched{use_lr_scheduler}_epochs{epochs}_lr{lr}_L2coef_{l2_coef}_batchsize{batch_size}_earlystopping_{early_stopping}"

    
    config = hyperparameter
    wandb_logs = config.wandb_logs

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # read dataset file
    if data_split_mode == 'standard':
        dataset_path = 'C:\\Users\\kverg\\GDI-NN\\SigmaProfileModel\\Databases\\MMFF_spDatabase_Train.csv'
        dataset = dataset_Sigma_Profiles(
            input_file_path=dataset_path,
            generate_all=True)
        dataset_size = len(dataset)
    
        # get data splits and load dataset
        train_indices_splits, val_indices_splits = data_split_Kfold(
            dataset=dataset, 
            n_splits=n_splits, 
            seed=seed
            )
        
    elif data_split_mode == 'stratified_hybrid':
        dataset_path = 'C:\\Users\\kverg\\GDI-NN\\SigmaProfileModel\\Databases\\MMFF_spDatabase_Train.csv'
        dataset = dataset_Sigma_Profiles(
            input_file_path=dataset_path,
            generate_all=True)
        dataset_size = len(dataset)
    
        # get data splits and load dataset
        train_indices_splits, val_indices_splits = data_split_stratified_Kfold(
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
                                                   collate_fn=collate_SP,
                                                   shuffle=False,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 sampler=valid_sampler,
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
        val_loss_save = []
    
        for epoch in range(1, epochs+1):

            # Train epoch
            train_loss = train(
                cv_index=cv_index, 
                epoch=epoch,
                train_loader=train_loader,
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                l2_coef=l2_coef 
                )
            train_loss_save.append(train_loss[0])

            # Val epoch
            val_loss = validate(
                cv_index=cv_index, 
                epoch=epoch,
                val_loader=val_loader,
                model=model,
                loss_fn=loss_fn,
                )
            val_loss_save.append(val_loss[0])

            # LR scheduler
            if use_lr_scheduler:
                scheduler.step(train_loss[0])
            
            is_best = val_loss[0] < best_loss
            best_loss = min(val_loss[0], best_loss)
                               
            
        # Save Final model and stats 
        torch.save({
                'epoch': epoch,
                'model_arch': model_type,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss
                }, 'C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/final_model_cv{}{}.pth'.format(cv_index, save_add))
        
        np.save('C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/train_loss_cv{}{}.npy'.format(cv_index, save_add),np.array(train_loss_save))
        np.save('C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/val_loss_cv{}{}.npy'.format(cv_index, save_add),np.array(val_loss_save))

        cv_index += 1
    
    print(index_list_train)
    index_list_train = np.array(index_list_train, dtype=object)
    np.save(f'C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/train_ind_list{save_add}.npy',index_list_train)
    index_list_valid = np.array(index_list_valid, dtype=object)
    np.save(f'C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/valid_ind_list{save_add}.npy',index_list_valid)
    
    # Plotting for all cv runs
    ## Training and validation loss
    train_mse = []
    valid_mse = []    
    plt.figure(figsize=(16,8))
    plt_rows = 2 if n_splits > 4 else 1
    for cv_index in range(n_splits):
        train_losses = np.load('C:/Users/kverg/GDI-NN/SigmaProfileModel//results_SP/train_loss_cv{}{}.npy'.format(cv_index, save_add))
        valid_losses = np.load('C:/Users/kverg/GDI-NN/SigmaProfileModel//results_SP/val_loss_cv{}{}.npy'.format(cv_index, save_add))
        plt.subplot(plt_rows,int(n_splits/2+0.5)+1,cv_index+1)
        plt.plot(train_losses,label="train loss cv{}".format(cv_index))
        plt.plot(valid_losses,label="valid loss cv{}".format(cv_index))
        plt.xlabel("epoch (training iteration)")
        plt.ylabel("loss")
        plt.legend(loc="best")
        train_mse.append(train_losses[-1])
        valid_mse.append(valid_losses[-1])
    train_mse = np.sqrt(np.array(train_mse))
    valid_mse = np.sqrt(np.array(valid_mse))
    rmse_str = (r'Train LOSS = {:.2f} $\pm$ {:.2f}'
                '\n'
                r'Val LOSS = {:.2f} $\pm$ {:.2f}'.format(
            np.mean(train_mse), np.std(train_mse), np.mean(valid_mse), np.std(valid_mse)))
    plt.subplot(2,3,6)
    plt.text(0,0.5, rmse_str, fontsize=12)
    plt.axis('off')
    plt.savefig(f'C:/Users/kverg/GDI-NN/SigmaProfileModel//results_SP/cvloss{save_add}_test.png',dpi=300)            
    
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
    parser.add_argument('--data_split_mode', default="standard", type=str)
    parser.add_argument('--num_splits', default=5, type=int)
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
