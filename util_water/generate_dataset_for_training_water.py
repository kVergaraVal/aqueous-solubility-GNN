import os
import pandas as pd
import torch
import dgl
from util_water.atom_feat_encoding_water import CanonicalAtomFeaturizer, MMFFAtomFeaturizer
from util.molecular_graph import mol_to_bigraph
from rdkit import Chem, DataStructs
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import AllChem
import numpy as np
from util.antoine_coeff import get_antoine_coef
import matplotlib
import matplotlib.pyplot as plt
#import ternary
#import win32com.client as win32
import time
import math
from model.model_GNN_water import SigmaProfileGCN, SigmaProfileGCN_from_graph

class solute_dataset_water_SP:
    def __init__(self, input_file_path = "./data/solubility_water/COMPATIBLE_unique_train_water_with_hybrid_classes.csv",
                 solute_list_path = "./data/solubility_water/solute_list_with_polarity_and_size.csv",
                 solvent_list_path = "./data/solubility_water/solvent_list_water_with_polarity_and_size.csv",
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
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()                  #Transforms tensor object to list
        data = self.dataset.iloc[idx]           #Obtain a row from the dataset pandas dataframe
        ids = [data['solute_id'], data['solvent_id']]    #solute and solvent columns contain the info about the index for solute_list.csv and solvent_list_water.csv
        solute = self.component_data[ids[0]]       #Gets the component data from the component ID
        solvent = self.component_data[ids[1]]
        sample['g1'] = solute[0]                 
        sample['g2'] = solvent[0]
        sample['g_sp1'] = solute[1]
        sample['g_sp2'] = solvent[1]
        sample['intra_hb1'] = solute[4]          
        sample['intra_hb2'] = solvent[4]
        sample['inter_hb'] = min(solute[2],solvent[3]) + min(solute[3],solvent[2])
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
            component_data[chemical_list[i]].append(mol_to_bigraph(mol,add_self_loop=True,
                                                node_featurizer=MMFFAtomFeaturizer(),
                                                edge_featurizer=None,
                                                canonical_atom_order=False,
                                                explicit_hydrogens=True,
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
        sample['g_sp1'] = solute[1]
        sample['g_sp2'] = solvent[1]
        sample['intra_hb1'] = solute[4]
        sample['intra_hb2'] = solvent[4]
        sample['inter_hb'] = min(solute[2],solvent[3]) + min(solute[3],solvent[2])
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
            self.component_data[solvent_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                node_featurizer=MMFFAtomFeaturizer(),
                                                                edge_featurizer=None,
                                                                canonical_atom_order=False,
                                                                explicit_hydrogens=True,
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
                self.component_data[solute_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                    node_featurizer=MMFFAtomFeaturizer(),
                                                                    edge_featurizer=None,
                                                                    canonical_atom_order=False,
                                                                    explicit_hydrogens=True,
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

def collate_solute_water_SP(batch):
    keys = list(batch[0].keys())[4:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['g1'] = dgl.batch(samples[0])
    batched_sample['g2'] = dgl.batch(samples[1])
    batched_sample['g_sp1'] = dgl.batch(samples[2])
    batched_sample['g_sp2'] = dgl.batch(samples[3])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+4])
        batched_sample[key] = torch.tensor(samples[i+4])
    return batched_sample



####################################################################################################################################################


class solute_dataset_water:
    def __init__(self, input_file_path = "./data/solubility_water/COMPATIBLE_unique_train_water_with_hybrid_classes.csv",
                 solute_list_path = "./data/solubility_water/solute_list_with_polarity_and_size.csv",
                 solvent_list_path = "./data/solubility_water/solvent_list_water_with_polarity_and_size.csv",
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


####################################################################################################################################################

class solub_cosolvent_dataset:
    def __init__(self, input_file_path = "C:/Users/kverg/GDI-NN/data/solubility_cosolvents/COMPATIBLEFORMAT_systems_train_298K_0109molfrac.csv",
                 solute_list_path = "C:/Users/kverg/GDI-NN/data/solubility_cosolvents/COMPATIBLEFORMAT_solvents_list.csv",
                 solvent_list_path = "C:/Users/kverg/GDI-NN/data/solubility_cosolvents/COMPATIBLEFORMAT_solvents_list.csv",
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
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()                 
        data = self.dataset.iloc[idx]          
        ids = [data['solvent_1_id'], data['solvent_2_id'], data['solute_id']]   
        solv1 = self.component_data[ids[0]]
        solv2 = self.component_data[ids[1]] 
        solute = self.component_data[ids[2]]       
        sample['g1'] = solv1[0]                 
        sample['g2'] = solv2[0]
        sample['g3'] = solute[0]
        sample['g_sp1'] = solv1[1]
        sample['g_sp2'] = solv2[1]
        sample['g_sp3'] = solute[1]       
        sample['intra_hb1'] = solv1[4]
        sample['intra_hb2'] = solv2[4]
        sample['intra_hb3'] = solute[4] 
        sample['inter_hb12'] = min(solv1[2],solv2[3]) + min(solv1[3],solv2[2])
        sample['inter_hb13'] = min(solv1[2],solute[3]) + min(solv1[3],solute[2])
        sample['inter_hb23'] = min(solv2[2],solute[3]) + min(solv2[3],solute[2])
        sample['solv1_x'] = data['Solvent_1_mol_fraction']
        sample['solv2_x'] = 1-data['Solvent_1_mol_fraction']
        sample["solute_id"] = data['solute_id']
        sample['solvent_1_id'] = data['solvent_1_id']
        sample['solvent_2_id'] = data['solvent_2_id']
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
            component_data[chemical_list[i]].append(mol_to_bigraph(mol,add_self_loop=True,
                                                node_featurizer=MMFFAtomFeaturizer(),
                                                edge_featurizer=None,
                                                canonical_atom_order=False,
                                                explicit_hydrogens=True,
                                                num_virtual_nodes=0
                                                ))
            hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
            hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
            component_data[chemical_list[i]].append(hba)
            component_data[chemical_list[i]].append(hbd)
            component_data[chemical_list[i]].append(min(hba,hbd))
        sample = {}
        solv1 = component_data[chemical_list[0]]
        solv2 = component_data[chemical_list[1]]
        solute = component_data[chemical_list[2]]
        sample['g1'] = solv1[0]                 
        sample['g2'] = solv2[0]
        sample['g3'] = solute[0]
        sample['g_sp1'] = solv1[1]
        sample['g_sp2'] = solv2[1]
        sample['g_sp3'] = solute[1]  
        sample['intra_hb1'] = solv1[4]
        sample['intra_hb2'] = solv2[4]
        sample['intra_hb3'] = solute[4] 
        sample['inter_hb12'] = min(solv1[2],solv2[3]) + min(solv1[3],solv2[2])
        sample['inter_hb13'] = min(solv1[2],solute[3]) + min(solv1[3],solute[2])
        sample['inter_hb23'] = min(solv2[2],solute[3]) + min(solv2[3],solute[2])
        sample['solv1_x'] = solv1_x
        sample['solv2_x'] = 1-solv1_x
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
            self.component_data[solvent_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                    node_featurizer=MMFFAtomFeaturizer(),
                                                                    edge_featurizer=None,
                                                                    canonical_atom_order=False,
                                                                    explicit_hydrogens=True,
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
                self.component_data[solute_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                    node_featurizer=MMFFAtomFeaturizer(),
                                                                    edge_featurizer=None,
                                                                    canonical_atom_order=False,
                                                                    explicit_hydrogens=True,
                                                                    num_virtual_nodes=0
                                                                    ))
                hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
                hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
                self.component_data[solute_id].append(hba)
                self.component_data[solute_id].append(hbd)
                self.component_data[solute_id].append(min(hba,hbd))

    def generate_solvsys_water(self,batch_size=5):
        n_comp = 2
        solvsys = dgl.DGLGraph()
        solvsys.add_nodes(n_comp*batch_size)
        src = torch.arange(batch_size)
        dst = torch.arange(batch_size,n_comp*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        solvsys.add_edges(torch.arange(n_comp*batch_size),torch.arange(n_comp*batch_size))    
        return solvsys

    def generate_solvsys_tricomponent(self,batch_size=5):  
        n_solv = 3
        solvsys = dgl.DGLGraph()
        solvsys.add_nodes(n_solv*batch_size)
        src = torch.arange(batch_size)
        dst = torch.arange(batch_size,2*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        src = torch.arange(batch_size)
        dst = torch.arange(2*batch_size,3*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        src = torch.arange(batch_size,2*batch_size)
        dst = torch.arange(2*batch_size,3*batch_size)
        solvsys.add_edges(torch.cat((src,dst)),torch.cat((dst,src)))
        solvsys.add_edges(torch.arange(n_solv*batch_size),torch.arange(n_solv*batch_size))    
        return solvsys

def collate_solute_cosolvent_SP(batch):
    keys = list(batch[0].keys())[6:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['g1'] = dgl.batch(samples[0])
    batched_sample['g2'] = dgl.batch(samples[1])
    batched_sample['g3'] = dgl.batch(samples[2])
    batched_sample['g_sp1'] = dgl.batch(samples[3])
    batched_sample['g_sp2'] = dgl.batch(samples[4])
    batched_sample['g_sp3'] = dgl.batch(samples[5])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+6])
        batched_sample[key] = torch.tensor(samples[i+6])
    return batched_sample



####

class solute_dataset_water_SP_cosolvent_compatibleFormat:

    def __init__(self, input_file_path = "./data/solubility_water/COMPATIBLE_unique_train_water_with_hybrid_classes.csv",
                 solute_list_path = "./data/solubility_water/solute_list_with_polarity_and_size.csv",
                 solvent_list_path = "./data/solubility_water/solvent_list_water_cosolventFormat.csv",
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
        sample = {}
        if torch.is_tensor(idx):
            idx = idx.tolist()                  #Transforms tensor object to list
        data = self.dataset.iloc[idx]           #Obtain a row from the dataset pandas dataframe
        ids = [data['solute_id'], data['water_id']]    #solute and solvent columns contain the info about the index for solute_list.csv and solvent_list_water.CSv
        solute = self.component_data[ids[0]]       #Gets the component data from the component ID
        solvent = self.component_data[ids[1]]
        sample['g1'] = solute[0]                 
        sample['g2'] = solvent[0]
        sample['g_sp1'] = solute[1]
        sample['g_sp2'] = solvent[1]
        sample['intra_hb1'] = solute[4]          
        sample['intra_hb2'] = solvent[4]
        sample['inter_hb'] = min(solute[2],solvent[3]) + min(solute[3],solvent[2])
        #sample["solute_id"] = int(ids[0].split("_")[1])
        #sample["sovlent_id"] = int(ids[1].split("_")[1])
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
            component_data[chemical_list[i]].append(mol_to_bigraph(mol,add_self_loop=True,
                                                node_featurizer=MMFFAtomFeaturizer(),
                                                edge_featurizer=None,
                                                canonical_atom_order=False,
                                                explicit_hydrogens=True,
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
        sample['g_sp1'] = solute[1]
        sample['g_sp2'] = solvent[1]
        sample['intra_hb1'] = solute[4]
        sample['intra_hb2'] = solvent[4]
        sample['inter_hb'] = min(solute[2],solvent[3]) + min(solute[3],solvent[2])
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
            self.component_data[solvent_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                node_featurizer=MMFFAtomFeaturizer(),
                                                                edge_featurizer=None,
                                                                canonical_atom_order=False,
                                                                explicit_hydrogens=True,
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
                self.component_data[solute_id].append(mol_to_bigraph(mol,add_self_loop=True,
                                                                    node_featurizer=MMFFAtomFeaturizer(),
                                                                    edge_featurizer=None,
                                                                    canonical_atom_order=False,
                                                                    explicit_hydrogens=True,
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

def collate_solute_water_SP(batch):
    keys = list(batch[0].keys())[4:]
    samples = list(map(lambda sample: sample.values(), batch))
    samples = list(map(list,zip(*samples)))
    batched_sample = {}
    batched_sample['g1'] = dgl.batch(samples[0])
    batched_sample['g2'] = dgl.batch(samples[1])
    batched_sample['g_sp1'] = dgl.batch(samples[2])
    batched_sample['g_sp2'] = dgl.batch(samples[3])
    for i,key in enumerate(keys):        
        batched_sample[key] = torch.tensor(samples[i+4])
        batched_sample[key] = torch.tensor(samples[i+4])
    return batched_sample