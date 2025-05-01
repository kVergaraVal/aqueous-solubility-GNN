import torch
import optuna
import torch.nn as nn
import numpy as np
import pandas as pd
import dgl
import dgl.backend as F
import itertools
import math
from sklearn.model_selection import StratifiedKFold
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from collections import defaultdict
from rdkit.Chem import rdmolfiles, rdmolops
from functools import partial
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau as reduce_lr
from dgl.nn.pytorch import GraphConv, NNConv, SumPooling
import torch.nn.functional as AF
import plotly

# Global variables
EPOCHS = 100
CV_SPLITS = 5
USE_LR_SCHEDULER = True
#SIGMA_PROFILE_MODEL_PATH = "/home/kevinvergara/Physics-informed-Thermodynamics/SigmaProfileModel/results_SP/Complete_trainset_final_model__MMFF_sp_Database_Sigma_profile_prediction_actrelu_encActrelu_lrschedFalse_epochs700_lr0.00151_L2coef_7.79e-06_batchsize16_earlystopping_False.pth"
DATASET_PATH = '/home/kevinvergara/Physics-informed-Thermodynamics/data/solubility_water/COMPATIBLE_unique_train_water_with_hybrid_classes.csv'
SOLUTE_LIST_PATH = '/home/kevinvergara/Physics-informed-Thermodynamics/data/solubility_water/solute_list_with_polarity_and_size.csv'
SOLVENT_LIST_PATH = '/home/kevinvergara/Physics-informed-Thermodynamics/data/solubility_water/solvent_list_water_with_polarity_and_size.csv'
MODEL_TYPE = "RittigPure"

# Ensure reproducibility
SEED = 2025
torch.manual_seed(SEED)
np.random.seed(SEED)


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

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    return [atom.GetFormalCharge()]

def atom_num_radical_electrons(atom):
    return [atom.GetNumRadicalElectrons()]

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    return [atom.GetIsAromatic()]

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

class ConcatFeaturizer(object):
    def __init__(self, func_list):
        self.func_list = func_list

    def __call__(self, x):
        return list(itertools.chain.from_iterable(
            [func(x) for func in self.func_list]))

class BaseAtomFeaturizer(object):
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

    def __call__(self, mol):
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features

class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_one_hot,
                 atom_degree_one_hot,
                 atom_implicit_valence_one_hot,
                 atom_formal_charge,
                 atom_num_radical_electrons,
                 atom_hybridization_one_hot,
                 atom_is_aromatic,
                 atom_total_num_H_one_hot]
            )})

def atom_type_MMFF_one_hot(atomType, allowable_set=None, encode_unknown=False):
    if allowable_set is None:
        allowable_set = list(range(1, 51))
    return one_hot_encoding(atomType, allowable_set, encode_unknown)


class MMFFAtomFeaturizer(BaseAtomFeaturizer):

    def __init__(self, atom_data_field='h'):
        super(MMFFAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer(
                [atom_type_MMFF_one_hot]
            )})

def construct_bigraph_from_mol(mol, add_self_loop=False):
    g = dgl.graph(([], []), idtype=torch.int32)
    #g = dgl.DGLGraph()
    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))
#    g.add_edges(torch.LongTensor(src_list), torch.LongTensor(dst_list))

    return g

def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
                 canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
    if mol is None:
        print('Invalid mol found')
        return None

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)      #Asignación del ordenamiento canónico a cada átomo, según el paper https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00543
        mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)                              #Construcción del grafo

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if num_virtual_nodes > 0:
        num_real_nodes = g.num_nodes()
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(num_virtual_nodes)

        # Change Topology
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)

        for nk, nv in g.ndata.items():
            nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
            nv[-num_virtual_nodes:, -1] = 1
            g.ndata[nk] = nv

        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=True,
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes)


class solute_dataset_water:
    #####################################################################################
    #                                                                                   #
    #   Major class for embedding the dataset into graphs.                              #
    #           First done by Qin et al., 2023.                                         #
    #                                                                                   #
    #   The dataset is a Map-style dataset (https://pytorch.org/docs/stable/data.html)  #
    #                                                                                   #
    #####################################################################################

    def __init__(self, input_file_path,
                 solute_list_path,
                 solvent_list_path, 
                 generate_all = False,
                 return_hbond=True,
                 normalize=False):
        if input_file_path:
            solvent_list = pd.read_csv(solvent_list_path, index_col='solvent_id')
            solute_list = pd.read_csv(solute_list_path, index_col='solute_id')
            self.dataset = pd.read_csv(input_file_path)                             #Dataframe of the solute and their respective outputs (known aqueous solubility)
            #self.solvent_names = solvent_list['solvent_name'].to_dict()             #Dictionary of solvent names of the entire dataset
            self.solvent_smiles = solvent_list['smiles_canon'].to_dict()              #Dictionary of smiles of the entire dataset
            #self.solute_names = solute_list['solute_name'].to_dict()             #Dictionary of solute names of the entire dataset
            self.solute_smiles = solute_list['smiles_canon'].to_dict()              #Dictionary of smiles of the entire dataset
        if normalize==True:
            self.dataset['LogS'] = [math.pow(10,x)/0.997 for x in self.dataset['LogS'].tolist()]
            logS_mod = np.array([math.log(x,10) for x in self.dataset['LogS'].tolist()])
            self.dataset['LogS'] = [(i-np.mean(logS_mod))/np.std(logS_mod) for i in logS_mod]
        self.component_data = {}
        self.return_hbond = return_hbond
        if generate_all:
            self.generate_all()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self,idx):
        #################################################################
        #                                                               #
        #       Method that allows indexing an object (dataset)         #
        #                                                               #
        # input:                                                        #
        #       idx: index of the object (dataset, similar to pandas)   #
        # output: an item (sample)                                      #
        #################################################################
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()                  #Transforms tensor object to list
        data = self.dataset.iloc[idx]           #Obtain a row from the dataset pandas dataframe
        ids = [data['solute_id'], data['solvent_id']]    #solute and solvent columns contain the info about the index for solute_list.csv and solvent_list_water.CSv
        solute = self.component_data[ids[0]]       #Gets the component data from the component ID
        solvent = self.component_data[ids[1]]
        sample['g1'] = solute[0]                 
        sample['g2'] = solvent[0]
        sample['intra_hb1'] = solute[3]          
        sample['intra_hb2'] = solvent[3]
        sample['inter_hb'] = min(solute[1],solvent[2]) + min(solute[2],solvent[1])
        sample["solute_id"] = int(ids[0].split("_")[1])
        sample["sovlent_id"] = int(ids[1].split("_")[1])
        sample['LogS'] = data['LogS']
        return sample
    
    def generate_sample(self,chemical_list,smiles_list,solv1_x,logS=None):
        component_data = {}
        for i,sml in enumerate(smiles_list):
            component_data[chemical_list[i]] = []
            sml = Chem.MolToSmiles(Chem.MolFromSmiles(sml))
            mol = Chem.MolFromSmiles(sml)
            component_data[chemical_list[i]].append(mol_to_bigraph(mol,add_self_loop=True,
                                               node_featurizer=CanonicalAtomFeaturizer(),
                                               edge_featurizer=None,
                                               canonical_atom_order=False,
                                               explicit_hydrogens=False,
                                               num_virtual_nodes=0
                                               ))
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            component_data[chemical_list[i]].append(hba)
            component_data[chemical_list[i]].append(hbd)
            component_data[chemical_list[i]].append(min(hba,hbd))
        sample = {}
        solute = component_data[chemical_list[0]]
        solvent = component_data[chemical_list[1]]
        sample['g1'] = solute[0]
        sample['g2'] = solvent[0]
        sample['intra_hb1'] = solute[3]
        sample['intra_hb2'] = solvent[3]
        sample['inter_hb'] = min(solute[1],solvent[2]) + min(solute[2],solvent[1])
        if logS:            
            sample['LogS'] = logS
        return sample
    
    def generate_all(self):
        for solvent_id in self.solvent_smiles:
            mol = Chem.MolFromSmiles(self.solvent_smiles[solvent_id])
            self.component_data[solvent_id] = []
            self.component_data[solvent_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                node_featurizer=CanonicalAtomFeaturizer(),
                                                                edge_featurizer=None,
                                                                canonical_atom_order=False,
                                                                explicit_hydrogens=False,
                                                                num_virtual_nodes=0
                                                                ))
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            self.component_data[solvent_id].append(hba)
            self.component_data[solvent_id].append(hbd)
            self.component_data[solvent_id].append(min(hba,hbd))


        for solute_id in self.solute_smiles:
            if self.solute_smiles[solute_id] != "CC1=CC=C[NH++]([O-])[CH-]1" and self.solute_smiles[solute_id] != "OC(=O)C1=C[NH++]([O-])[CH-]C=C1":
                mol = Chem.MolFromSmiles(self.solute_smiles[solute_id])
                self.component_data[solute_id] = []
                self.component_data[solute_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                    node_featurizer=CanonicalAtomFeaturizer(),
                                                                    edge_featurizer=None,
                                                                    canonical_atom_order=False,
                                                                    explicit_hydrogens=False,
                                                                    num_virtual_nodes=0
                                                                    ))
                hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
                hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
                self.component_data[solute_id].append(hba)
                self.component_data[solute_id].append(hbd)
                self.component_data[solute_id].append(min(hba,hbd))

    def generate_solvsys(self,batch_size=5):
        n_comp = 2
        solvsys = dgl.DGLGraph()
        solvsys.add_nodes(n_comp*batch_size)
        src = torch.arange(batch_size)
        dst = torch.arange(batch_size,n_comp*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        solvsys.add_edges(torch.arange(n_comp*batch_size),torch.arange(n_comp*batch_size))    
        return solvsys

def collate_solute_water(batch):
    keys = list(batch[0].keys())[2:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['g1'] = dgl.batch(samples[0])
    batched_sample['g2'] = dgl.batch(samples[1])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+2])
        batched_sample[key] = torch.tensor(samples[i+2])
    return batched_sample


def data_split_stratified_hybrid(dataset, n_splits=CV_SPLITS, seed=SEED):
    train_indices_splits = []
    val_indices_splits = []
    
    dataset_size = len(dataset)
    all_ind = np.arange(dataset_size)
    hybrid_class = dataset.dataset['hybrid_class']
    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_indices, valid_indices in kf.split(all_ind,hybrid_class):
        train_indices_splits.append(train_indices)
        val_indices_splits.append(valid_indices)
    return train_indices_splits, val_indices_splits



def get_activation(activation, get_nn=False):
    if (activation == None) or (activation in ["relu", "ReLU", "RELU"]):
        if get_nn: return nn.ReLU
        return AF.relu
    elif activation in ["elu", "ELU"]:
        if get_nn: return nn.ELU
        return AF.elu
    elif activation in ["LeakyReLU", "LeakyRELU", "leakyReLU", "leakyrelu", "leakyRELU", "leaky_relu", "Leaky_ReLU", "Leaky_RELU"]:
        if get_nn: return nn.LeakyReLU
        return AF.leaky_relu
    elif activation in ["sigmoid", "Sigmoid", "SIGMOID"]:
        if get_nn: return nn.Sigmoid
        return AF.sigmoid
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn: return nn.Softplus
        return AF.softplus
    elif activation in ["silu", "SiLU", "SILU", "swish", "Swish", "SWISH"]:
        if get_nn: return nn.SiLU
        return AF.silu
    elif activation in ["tanh", "Tanh", "TANH"]:
        if get_nn: return nn.Tanh
        return AF.tanh

class MPNNconv(nn.Module):
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=128,
                 edge_hidden_feats=32, num_step_message_passing=6, activation="relu"):
        super(MPNNconv, self).__init__() #Inherits methods and attributes from parent classs nn.Module

        self.mpnn_activation = get_activation(activation)       #Initiates activation function for the message passing algorithm portion

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),           #Adds linear layer (y=x*A^T+b)
            get_activation(activation, get_nn=True)()           #Adds activation function layer
        )

        self.num_step_message_passing = num_step_message_passing

        edge_network = nn.Sequential(                                       
            nn.Linear(edge_in_feats, edge_hidden_feats),                    #Adds linear layer
            get_activation(activation, get_nn=True)(),                      #Adds activation function layer
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)   #Adds linear layer, with the output being size (node_out_feats)^2
        )

        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )
        self.gru = nn.GRU(node_out_feats, node_out_feats)

    def reset_parameters(self):                         #Self explanatory
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        self.gru.reset_parameters()

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.project_node_feats(node_feats)    #As I understand, this simply maps the node features to hidden node features
        hidden_feats = node_feats.unsqueeze(0)              #unsqueeze transforms 3d tensor to 2d tensor, unsqueeze is the inverse transformation

        for _ in range(self.num_step_message_passing):
            node_feats = self.mpnn_activation(self.gnn_layer(g, node_feats, edge_feats))    #Message update function
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)      #Node update function as explicited by Qin et al.
            node_feats = node_feats.squeeze(0)                                              #Return to original format
        return node_feats 

class RittigPure(nn.Module):
    def __init__(self, in_dim, n_classes,
                 gcn_dim, mpnn_dim, mlp_dim,
                 mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None,
                 gcn_layers=2, mlp_num_hid_layers=2, num_step_message_passing=1):
        super(RittigPure, self).__init__()
        self.gcn_layers = gcn_layers
        self.conv1 = GraphConv(in_dim, gcn_dim)
        if self.gcn_layers==2:
            self.conv2 = GraphConv(gcn_dim, gcn_dim)
        elif self.gcn_layers != 1:
            raise ValueError(f"gcn_layers has invalid value. Choose either 1 or 2.")
        self.global_conv1 = MPNNconv(node_in_feats=gcn_dim,
                                     edge_in_feats=1,
                                     node_out_feats=mpnn_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mfp_trans = nn.Linear(mpnn_dim+0, mlp_dim+0)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(mlp_dim+0, mlp_dim)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(mlp_dim, mlp_dim)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(mlp_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, device):
        g1 = solvdata['g1'].to(device)
        g2 = solvdata['g2'].to(device)
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().to(device)
                h2 = g2.ndata['h'].float().to(device)
                inter_hb = solvdata['inter_hb'].float().to(device)
                intra_hb1 = solvdata['intra_hb1'].float().to(device)
                intra_hb2 = solvdata['intra_hb2'].float().to(device)

                h1_temp = AF.relu(self.conv1(g1, h1))
                h2_temp = AF.relu(self.conv1(g2, h2))
                if self.gcn_layers == 2:
                    h1_temp = AF.relu(self.conv2(g1, h1_temp))
                    h2_temp = AF.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').to(device)
                hg2 = dgl.mean_nodes(g2, 'h').to(device)
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 

                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                # SLP
                hg1_temp = self.mlp_activation(self.mfp_trans(hg[0:len(hg)//2,:]))
                hg2_temp = self.mlp_activation(self.mfp_trans(hg[len(hg)//2:,:]))

                # Pooling
                hg_temp = (hg1_temp + hg2_temp) / 2
                
                output = self.mlp_dropout(hg_temp)
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_dropout(output)
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                              
            return output    



dataset_class = solute_dataset_water
collate_fn = collate_solute_water
weight_init_fn = torch.nn.init.uniform_


# read dataset file
dataset = dataset_class(
    input_file_path=DATASET_PATH,
    solute_list_path=SOLUTE_LIST_PATH,
    solvent_list_path=SOLVENT_LIST_PATH,
    generate_all=True,
    normalize=False)



def train(train_loader, empty_solvsys, model, loss_fn, l2_coef, optimizer, device):
    loss_accum = AccumulationMeter()
    model.train()
    for i, component_data in enumerate(train_loader):
        lablogS = component_data['LogS'].float().to(device) # logS

        # Model predictions and gradients
        y = None
        with torch.backends.cudnn.flags(enabled=False): #disables the CuDNN during prediction, this is could have different reasons: 1. reproducibiltiy issues, 2. performance issues, 3. speed issues
            if empty_solvsys != None:
                y = model(component_data, empty_solvsys, device)
            else:
                y = model(component_data, device)

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
    return [loss_accum.avg]


def validate(val_loader, empty_solvsys, model, loss_fn, device):
    loss_accum = AccumulationMeter()
    model.eval()

    with torch.set_grad_enabled(True):
        for i, component_data in enumerate(val_loader):
            lablogS = component_data['LogS'].float().to(device) # logS

            # Model predictions and gradients
            y = None
            with torch.backends.cudnn.flags(enabled=False):
                if empty_solvsys != None:
                    y = model(component_data, empty_solvsys, device)
                else:
                    y = model(component_data, device)

            # Prediction loss
            loss = loss_fn(y[:,0],lablogS) # loss logS
            
            # Update stats
            loss_accum.update(loss.item(),lablogS.size(0))

    return [loss_accum.avg]


def train_and_evaluate(trial, gcn_dim, mpnn_dim, mlp_dim, gcn_layers, mlp_num_hid_layers, learning_rate, batch_size, l2_coef, gpu_id):
    """
    Trains the model and evaluates it using 5-fold cross-validation.
    Uses a specific GPU if available.
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f'Device for trial {trial.number} is {device}')
    
    # get data splits and load dataset
    train_indices_splits, val_indices_splits = data_split_stratified_hybrid(dataset=dataset, 
                                                                            n_splits=CV_SPLITS, 
                                                                            seed=SEED)
    
    cv_index = 0
    best_fold_losses = []
    for train_indices, val_indices in zip(train_indices_splits, val_indices_splits):

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
        
        empty_solvsys = dataset.generate_solvsys(batch_size).to(device)
        

        enc_activation = 'relu'
        num_step_message_passing = 1
        mlp_activation = 'softplus'

        # initialize model
        if MODEL_TYPE == "RittigPure":
            model = RittigPure(in_dim=74, n_classes=1,
                                gcn_dim=gcn_dim, mpnn_dim=mpnn_dim, mlp_dim=mlp_dim, 
                                mlp_dropout_rate=0, mlp_activation=mlp_activation, mpnn_activation=enc_activation,
                                gcn_layers=gcn_layers, mlp_num_hid_layers=mlp_num_hid_layers,
                                num_step_message_passing=num_step_message_passing).to(device)
        
        loss_fn = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
        if USE_LR_SCHEDULER:
            scheduler = reduce_lr(optimizer, mode='min', factor=0.8, patience=3, min_lr=1e-7, verbose=False)
        
        # Training
        best_val_loss = float("inf")
    
        for epoch in range(EPOCHS):
            
            train_loss = None
            
            while train_loss == None:
            # Train epoch
                train_loss = train(
                    train_loader=train_loader,
                    empty_solvsys=empty_solvsys,
                    model=model,
                    loss_fn=loss_fn,
                    l2_coef=l2_coef,
                    optimizer=optimizer,
                    device=device
                    )
                if train_loss == None:
                    print(f'Something is wrong at epoch {epoch} for split {cv_index} in tiral {trial.number}, training step resetted')

            if epoch % 4 == 0 or epoch == 50:
                print(f'Finished training epoch {epoch} for split {cv_index} in trial {trial.number} with loss: {train_loss[0]}')

            val_loss = None
            while val_loss == None: 
                # Val epoch
                val_loss = validate(
                    val_loader=val_loader,
                    empty_solvsys=empty_solvsys,
                    model=model,
                    loss_fn=loss_fn,
                    device=device
                    )
                if train_loss == None:
                    print(f'Something is wrong at epoch {epoch} for split {cv_index} in tiral {trial.number}, validation step resetted')

            if epoch % 4 == 0 or epoch == 50:
                print(f'Finished validation epoch {epoch} for split {cv_index} in trial {trial.number} with loss: {val_loss[0]}')

            # LR scheduler
            if USE_LR_SCHEDULER:
                scheduler.step(train_loss[0])
            
            best_val_loss = min(val_loss[0], best_val_loss)
            

            #if cv_index == 0:       
            #    trial.report(val_loss[0], epoch) 
            #    if trial.should_prune():
            #        raise optuna.exceptions.TrialPruned() 
        
        best_fold_losses.append(best_val_loss)
        cv_index += 1

    return np.mean(best_fold_losses)


def objective(trial):
    """
    Defines the hyperparameter search space and runs training/evaluation.
    """
    #Architecture
    GCN_dim = 156
    MPNN_dim = 128
    MLP_dim = 134
    GCN_layers = 2
    MLP_layers = 2

    #Training
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    l2_coef = trial.suggest_float("l2_coef", 1e-10, 1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 256)
    use_scheduler = trial.suggest_categorical("use_lr_scheduler", [0, 1])

    # Assign GPU for this trial
    gpu_id = trial.number % torch.cuda.device_count()
    print(f'torch cuda device count is {torch.cuda.device_count()}')
    #gpu_id = 1

    #try:
    #    loss = train_and_evaluate(trial, GCN_dim, MPNN_dim, MLP_dim, learning_rate, batch_size, l2_coef, gpu_id)
    #except:
    #    loss = None
    #    print(f'trial {trial.number} skipped due to unknown reasons')

    loss = train_and_evaluate(trial, GCN_dim, MPNN_dim, MLP_dim, GCN_layers, MLP_layers, learning_rate, batch_size, l2_coef, gpu_id)

    return loss



# Set up Optuna study with SQLite storage for checkpointing
sampler = optuna.samplers.TPESampler()

study = optuna.create_study(
    study_name="Test_RittigPure_fittingparam",
    direction="minimize",
    storage="sqlite:////home/kevinvergara/Physics-informed-Thermodynamics/TRIALS_BO_rittigPure_fittingparam.db",  # Checkpointing database
    load_if_exists=True,
    sampler=sampler
)

# Run optimization with 2 parallel jobs (for 2 GPUs)
study.optimize(objective, n_trials=100, n_jobs=2)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

#plot_param_importances = optuna.visualization.plot_param_importances(study)
#plot_param_importances.write_image("fig1.png")
#plot_param_importances.write_image("fig1.svg")33222222222ww22k2w2w2w2wwwwww222w2222222222222222222222wwwwww222222222wwwww22w2w2273=03966k123