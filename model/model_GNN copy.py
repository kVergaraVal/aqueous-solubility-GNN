# -*- coding: utf-8 -*-

#**********************************************************************************
# Copyright (c) 2024 Process Systems Engineering (AVT.SVT), RWTH Aachen University
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0.
#
# SPDX-License-Identifier: EPL-2.0
#
# The source code can be found here:
# https://git.rwth-aachen.de/avt-svt/public/GDI-NN
#
# Notes:
# - This code was adpated from the original implementation by Qin, S., Jiang, S., Li, J., Balaprakash, P., Van Lehn, R. C., & Zavala, V. M. (2023). Capturing molecular interactions in graph neural networks: a case study in multi-component phase equilibrium. Digital Discovery, 2(1), 138-151.
# - The original implementation can be found here: https://github.com/zavalab/ML/tree/master/SolvGNN
#
#*********************************************************************************


import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import dgl
from dgl.nn.pytorch import GraphConv, NNConv
import torch.nn.functional as F


def get_n_params(model):
    n_params = 0
    for item in list(model.parameters()):
        item_param = 1
        for dim in list(item.size()):
            item_param = item_param*dim
        n_params += item_param
    return n_params

def get_activation(activation, get_nn=False):
    if (activation == None) or (activation in ["relu", "ReLU", "RELU"]):
        if get_nn: return nn.ReLU
        return F.relu
    elif activation in ["elu", "ELU"]:
        if get_nn: return nn.ELU
        return F.elu
    elif activation in ["LeakyReLU", "LeakyRELU", "leakyReLU", "leakyrelu", "leakyRELU", "leaky_relu", "Leaky_ReLU", "Leaky_RELU"]:
        if get_nn: return nn.LeakyReLU
        return F.leaky_relu
    elif activation in ["sigmoid", "Sigmoid", "SIGMOID"]:
        if get_nn: return nn.Sigmoid
        return F.sigmoid
    elif activation in ["softplus", "Softplus", "SOFTPLUS"]:
        if get_nn: return nn.Softplus
        return F.softplus
    elif activation in ["silu", "SiLU", "SILU"]:
        if get_nn: return nn.SiLU
        return F.silu
    elif activation in ["tanh", "Tanh", "TANH"]:
        if get_nn: return nn.Tanh
        return F.tanh

class MPNNconv(nn.Module):
    #########################################################################
    #                                                                       #
    #   This class corresponds to the global message passing algorithm      #
    #       contained in the GNN as depicted in Qin et al., 2023.           #
    #                                                                       #
    #########################################################################
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=128,
                 edge_hidden_feats=32, num_step_message_passing=6, activation="relu"):
        #####################################################################################
        #                                                                                   #
        #   This method initializes the architecture of the global message                  #
        #               passing portion of the neural network                               #
        #                                                                                   #
        #   input:                                                                          #
        #       node_in_feats: number of starting features for each node                    #
        #       edge_in_feats: number of starting features for each edge                    #
        #       node_out_feats: number of final features for each node                      #
        #       edge_hidden_feats: number of transient features for each edge               #
        #       num_step_message_passing: number of steps in the message passing algorithm  #
        #       activation: activation function used                                        #
        #                                                                                   #
        #   output: a global convolution with message passing layer object                  #
        #####################################################################################

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
        #########################################################
        #                                                       #
        #   Method that executes the message passing algorithm  #
        #                                                       #
        #   Equations are depicted in Eq (5) and Eq (6)         #
        #              from Qin et al., (page 142)              #
        #                                                       #
        #   input:                                              #
        #       g: graph object (contains architecture info)    #
        #       node_feats: initial node features               #
        #       edge_feats: initial edge features               #
        #                                                       #
        #   output:                                             #
        #       node_feats: updated node features               #
        #########################################################

        node_feats = self.project_node_feats(node_feats)    #As I understand, this simply maps the node features to hidden node features
        hidden_feats = node_feats.unsqueeze(0)              #unsqueeze transforms 3d tensor to 2d tensor, unsqueeze is the inverse transformation

        for _ in range(self.num_step_message_passing):
            node_feats = self.mpnn_activation(self.gnn_layer(g, node_feats, edge_feats))    #Message update function
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)      #Node update function as explicited by Qin et al.
            node_feats = node_feats.squeeze(0)                                              #Return to original format
        return node_feats    

class solvgnn_binary(nn.Module):
    ############################################################
    #                                                          #
    #    Corresponds to the GNN model for binary mixtures      #
    #           as depicted in Qin et al., 2023.               #
    #    This model contains the Message Passing algorithm     #
    #                                                          #
    ############################################################

    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2):
        #####################################################################################
        #                                                                                   #
        #   This method initializes the architecture of GNN model with Message Passing.     #
        #      The purpose of Message Passing is to effectively take into account           #                               #
        # intermolecular bond information. Qin et al. specifically embedded H-bonding info. #
        #                                                                                   #
        #   input:                                                                          #
        #       in_dim: input dimension lenght                                              #
        #       hidden_dim: hidden dimension length                                         #
        #       n_classes: number of classes at the output                                  #
        #       mlp_dropout_rate: dropout rate after layer (not used in Rittig et al.)      #
        #       mlp_activation: activation function at the output of the MLP layers         #
        #       mpnn_activation: activation function at the output of the MPNN portion      #
        #       mlp_num_hid_layers: number of hidden layers of the MLP (not used)           #
        #                                                                                   #
        #   output: a GNN object                                                            #
        #####################################################################################

        super(solvgnn_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)  
        #GraphConv is defined as:
        #   h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ji}}h_j^{(l)}W^{(l)})
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim+1,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        #self.dropout1 = nn.Dropout(0.2)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        #self.dropout2 = nn.Dropout(0.2)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                #solv2x = 1 - solv1x
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()
                
                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
        
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                output = self.mlp_activation(self.classify1(hg))
                #output = self.dropout1(output)
                output = self.mlp_activation(self.classify2(output))
                #output = self.dropout2(output)
                output = self.classify3(output)                        
                output = torch.cat(
                    (output[0:len(output)//2,:],
                    output[len(output)//2:,:]),axis=1)      
                    
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]   
                    return output, y1_x1, y2_x1                          
            return output         


class solvgnn_xMLP_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_xMLP_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim+1, hidden_dim)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg, torch.cat((solv1x, 1-solv1x))[:, None]), axis=1)
                
                output = self.mlp_dropout(hg)
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_dropout(output)
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                        
                output = torch.cat(
                    (output[0:len(output)//2,:],
                    output[len(output)//2:,:]),axis=1)    
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    


class solvgnn_onexMLP_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_onexMLP_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim*2+2, hidden_dim*2)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                #hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                #hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg[0:len(hg)//2,:], solv1x[:, None], hg[len(hg)//2:,:], 1-solv1x[:, None]),axis=1)   
                
                output = self.mlp_dropout(hg)
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_dropout(output)
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    

class solvgnn_onexMLP_share1layer_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_onexMLP_share1layer_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim*2+2, hidden_dim*2)
        if self.mlp_num_hid_layers == 2:
            self.classify2_1 = nn.Linear(hidden_dim*2, hidden_dim*2)
            self.classify2_2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3_1 = nn.Linear(hidden_dim*2, n_classes)
        self.classify3_2 = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                #hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                #hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg[0:len(hg)//2,:], solv1x[:, None], hg[len(hg)//2:,:], 1-solv1x[:, None]),axis=1)   
                
                output = self.mlp_activation(self.classify1(hg))
                output_y1 = self.mlp_activation(self.classify2_1(output))
                output_y1 = self.classify3_1(output_y1)
                output_y2 = self.mlp_activation(self.classify2_2(output))
                output_y2 = self.classify3_2(output_y2)
                output = torch.cat([output_y1, output_y2], dim=1)  
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    

class solvgnn_onexMLP_share2layer_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mlp_num_hid_layers=2):
        super(solvgnn_onexMLP_share2layer_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim*2+2, hidden_dim*2)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim*2, hidden_dim*2)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3_1 = nn.Linear(hidden_dim*2, n_classes)
        self.classify3_2 = nn.Linear(hidden_dim*2, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                #hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                #hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                # hg1 = solv1x[:,None]*hg1
                # hg2 = solv2x[:,None]*hg2
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                hg = torch.cat((hg[0:len(hg)//2,:], solv1x[:, None], hg[len(hg)//2:,:], 1-solv1x[:, None]),axis=1)   
                
                output = self.mlp_activation(self.classify1(hg))
                output = self.mlp_activation(self.classify2(output))
                output_y1 = self.classify3_1(output)
                output_y2 = self.classify3_2(output)
                output = torch.cat([output_y1, output_y2], dim=1)  
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(output[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(output[:,1].sum(), solv1x, create_graph=True)[0]
                    return output, y1_x1, y2_x1           
            return output    
        
class gegnn_binary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2, num_step_message_passing=1):
        super(gegnn_binary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        self.mfp_trans = nn.Linear(hidden_dim+1, hidden_dim+1)
        self.classify1 = nn.Linear(hidden_dim+1, hidden_dim)
        self.dropout1 = nn.Dropout(mlp_dropout_rate)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(mlp_dropout_rate)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, gamma_grad=False):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                solv1x = solvdata['solv1_x'].float().cuda()
                solv1x.requires_grad = True
                inter_hb = solvdata['inter_hb'].float().cuda()
                intra_hb1 = solvdata['intra_hb1'].float().cuda()
                intra_hb2 = solvdata['intra_hb2'].float().cuda()
                
                # Molecule embedding
                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                # Molecular interaction
                hg1 = dgl.mean_nodes(g1, 'h').cuda()
                hg2 = dgl.mean_nodes(g2, 'h').cuda()
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))
                # SLP
                hg1_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[0:len(hg)//2,:], solv1x[:, None]), axis=1)))
                hg2_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[len(hg)//2:,:], 1-solv1x[:, None]), axis=1)))
                # Pooling
                hg_temp = (hg1_temp + hg2_temp) / 2
                
                # MLP
                output = self.mlp_activation(self.classify1(hg_temp))
                output = self.dropout1(output)
                output = self.mlp_activation(self.classify2(output))
                output = self.dropout2(output)
                output = self.classify3(output) 
                # Derive act. coeff. based on shared Gibbs Excess Energy output node 
                G_dx1 = torch.autograd.grad(output.sum(), solv1x, create_graph=True)[0] 
                #G_dx2 = torch.autograd.grad(output.sum(), -solv1x, create_graph=True)[0] 
                gamma_1 = output.squeeze(-1) + (1-solv1x) * G_dx1    
                gamma_2 = output.squeeze(-1) + (solv1x) * (-1) * G_dx1                   
                gamma = torch.cat((gamma_1.unsqueeze(-1), gamma_2.unsqueeze(-1)),axis=1)  
                    
                if gamma_grad:
                    y1_x1 = torch.autograd.grad(gamma[:,0].sum(), solv1x, create_graph=True)[0] 
                    y2_x1 = torch.autograd.grad(gamma[:,1].sum(), solv1x, create_graph=True)[0]   
                    return gamma, y1_x1, y2_x1                          
            return gamma, 0, 0      





class Qin_ternary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Qin_ternary, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim+1,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=1)
        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys, device, gamma_grad=False):

        g1 = solvdata['g1'].to(device)
        g2 = solvdata['g2'].to(device)
        g3 = solvdata['g3'].to(device)
        with g1.local_scope():
            with g2.local_scope():
                with g3.local_scope():
                    h1 = g1.ndata['h'].float().to(device)
                    h2 = g2.ndata['h'].float().to(device)
                    h3 = g3.ndata['h'].float().to(device)
                    solv1x = solvdata['solv1_x'].float().to(device)
                    solv1x.requires_grad = True
                    #print(f'solv1x is {solv1x}')
                    inter_hb12 = solvdata['inter_hb12'].float().to(device)
                    inter_hb13 = solvdata['inter_hb13'].float().to(device)
                    inter_hb23 = solvdata['inter_hb23'].float().to(device)
                    intra_hb1 = solvdata['intra_hb1'].float().to(device)
                    intra_hb2 = solvdata['intra_hb2'].float().to(device)
                    intra_hb3 = solvdata['intra_hb3'].float().to(device)
                    
                    h1_temp = F.relu(self.conv1(g1, h1))
                    h1_temp = F.relu(self.conv2(g1, h1_temp))
                    h2_temp = F.relu(self.conv1(g2, h2))
                    h2_temp = F.relu(self.conv2(g2, h2_temp))
                    h3_temp = F.relu(self.conv1(g3, h3))
                    h3_temp = F.relu(self.conv2(g3, h3_temp))
                    g1.ndata['h'] = h1_temp
                    g2.ndata['h'] = h2_temp
                    g3.ndata['h'] = h3_temp        
            
                    hg1 = dgl.mean_nodes(g1, 'h')
                    hg2 = dgl.mean_nodes(g2, 'h')
                    hg3 = dgl.mean_nodes(g3, 'h')
                    hg1 = torch.cat((hg1, solv1x[:, None]), axis=1)
                    hg2 = torch.cat((hg2, 1-solv1x[:, None]), axis=1)
                    frac_0 = torch.zeros(solv1x.shape).to(device)
                    hg3 = torch.cat((hg3, frac_0[:,None]), axis=1)
                    #print(f'solv1x[:,None] is {solv1x[:,None]}')
                    # hg1 = solv1x[:,None]*hg1
                    # hg2 = solv2x[:,None]*hg2
                    # hg3 = solv3x[:,None]*hg3
            
                    hg = self.global_conv1(empty_solvsys, 
                                           torch.cat((hg1,hg2,hg3),axis=0), 
                                           torch.cat((inter_hb12.repeat(2),inter_hb13.repeat(2),
                                                      inter_hb23.repeat(2),
                                                      intra_hb1,intra_hb2,intra_hb3)).unsqueeze(1))
                    output = F.relu(self.classify1(hg))
                    print(f'output shape after classify1 is {output.shape}')
                    output = F.relu(self.classify2(output))
                    print(f'output shape after classify2 is {output.shape}')
                    output = self.classify3(output)
                    print(f'output shape after classify3 is {output.shape}')
                    output = torch.cat(
                        (output[0:len(output)//3,:],
                        output[len(output)//3:2*len(output)//3,:],
                        output[2*len(output)//3:,:]),axis=1)
                    print(f'new output shape is {output.shape}')
                    gibbs = output[:,2:]

                    # Derive act. coeff. based on shared Gibbs Excess Energy output node 
                    G_dx1 = torch.autograd.grad(gibbs.sum(), solv1x, create_graph=True)[0] 
                    #G_dx2 = torch.autograd.grad(output.sum(), -solv1x, create_graph=True)[0] 
                    gamma_1 = gibbs.squeeze(-1) + (1-solv1x) * G_dx1    
                    gamma_2 = gibbs.squeeze(-1) + (solv1x) * (-1) * G_dx1                   
                    gamma = torch.cat((gamma_1.unsqueeze(-1), gamma_2.unsqueeze(-1)),axis=1)
                    gamma = torch.cat((gamma, output[:,1]), axis=1)  

                    if gamma_grad:
                        y1_x1 = torch.autograd.grad(gamma[:,0].sum(), solv1x, create_graph=True)[0] 
                        y2_x1 = torch.autograd.grad(gamma[:,1].sum(), solv1x, create_graph=True)[0]   
                        return gamma, y1_x1, y2_x1  
                    return gamma, 0, 0  