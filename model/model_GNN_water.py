
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import dgl
from dgl.nn.pytorch import GraphConv, NNConv, SumPooling
import torch.nn.functional as F
import math

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
    elif activation in ["silu", "SiLU", "SILU", "swish", "Swish", "SWISH"]:
        if get_nn: return nn.SiLU
        return F.silu
    elif activation in ["tanh", "Tanh", "TANH"]:
        if get_nn: return nn.Tanh
        return F.tanh

def calculate_same_padding_conv1d(kernel_size, stride, input_length):
    return math.ceil(((stride - 1) * input_length + kernel_size - stride) / 2)

def padding_pool(pool_size):
    pool_stride = pool_size 
    total_pad = pool_size - 1
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    return (left_pad, right_pad)

def apply_same_padding_pool(pool_size, stride, input_tensor):
    total_pad = max(pool_size - stride, 0)
    left_pad = total_pad // 2
    right_pad = total_pad - left_pad
    padded_input = F.pad(input_tensor, (left_pad, right_pad), mode='replicate')
    return padded_input

class SigmaProfileGCN(nn.Module):
    def __init__(self, in_dim):
        super(SigmaProfileGCN, self).__init__()
        self.conv1 = GraphConv(in_dim, 300)
        self.conv2 = GraphConv(300, 153)
        self.conv3 = GraphConv(153, 161)
        self.classify = nn.Linear(161, 51)
        self.globalpooling = SumPooling()

    def forward(self, molecule_data, device):
        graph = molecule_data['g_sp1'].to(device)
        with graph.local_scope():
            h1 = graph.ndata['h'].float().to(device)
                
            h1_temp = F.relu(self.conv1(graph, h1))
            h1_temp = F.relu(self.conv2(graph, h1_temp))
            h1_temp = F.relu(self.conv3(graph, h1_temp))
            graph.ndata['h'] = h1_temp

            hg = h1_temp
            output = F.relu(self.classify(hg))
            output = self.globalpooling(graph, output)
            return output
    
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

class water_gegnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2, num_step_message_passing=1):
        super(water_gegnn, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        #self.mfp_trans = nn.Linear(hidden_dim+0, hidden_dim+0)  #The +0 replaces the +1 from the original Rittig architecture, which corresponded to solvent molar fraction
        self.mfp_trans = nn.Linear(hidden_dim+0, hidden_dim+0)
        self.classify1 = nn.Linear(hidden_dim+0, hidden_dim)
        self.dropout1 = nn.Dropout(mlp_dropout_rate)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(mlp_dropout_rate)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, component_data, empty_solvsys):
        g1 = component_data['g1'].to("cuda")
        g2 = component_data['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                inter_hb = component_data['inter_hb'].float().cuda()
                intra_hb1 = component_data['intra_hb1'].float().cuda()
                intra_hb2 = component_data['intra_hb2'].float().cuda()
                
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
                hg1_temp = self.mlp_activation(self.mfp_trans(hg[0:len(hg)//2,:]))
                hg2_temp = self.mlp_activation(self.mfp_trans(hg[len(hg)//2:,:]))
                # Pooling
                hg_temp = hg1_temp
                hg_temp = (hg1_temp + hg2_temp) / 2

                # MLP
                output = self.mlp_activation(self.classify1(hg_temp))
                output = self.dropout1(output)
                output = self.mlp_activation(self.classify2(output))
                output = self.dropout2(output)
                output = self.classify3(output) 
                
            return output     

class water_gegnn_no_Pooling(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2, num_step_message_passing=1):
        super(water_gegnn_no_Pooling, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        #self.mfp_trans = nn.Linear(hidden_dim+0, hidden_dim+0)  #The +0 replaces the +1 from the original Rittig architecture, which corresponded to solvent molar fraction
        self.mfp_trans = nn.Linear(hidden_dim+0, hidden_dim+0)
        self.classify1 = nn.Linear(hidden_dim+0, hidden_dim)
        self.dropout1 = nn.Dropout(mlp_dropout_rate)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(mlp_dropout_rate)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, component_data, empty_solvsys):
        g1 = component_data['g1'].to("cuda")
        g2 = component_data['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                inter_hb = component_data['inter_hb'].float().cuda()
                intra_hb1 = component_data['intra_hb1'].float().cuda()
                intra_hb2 = component_data['intra_hb2'].float().cuda()
                
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
                hg1_temp = self.mlp_activation(self.mfp_trans(hg[0:len(hg)//2,:]))
                #g2_temp = self.mlp_activation(self.mfp_trans(hg[len(hg)//2:,:]))
                # Pooling
                hg_temp = hg1_temp
                #hg_temp = (hg1_temp + hg2_temp) / 2

                # MLP
                output = self.mlp_activation(self.classify1(hg_temp))
                output = self.dropout1(output)
                output = self.mlp_activation(self.classify2(output))
                output = self.dropout2(output)
                output = self.classify3(output) 
                
            return output     

class water_gegnn_SigmaProfile_noPooling(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2, num_step_message_passing=1,
                 sigma_profile_model_path="C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_final_model__MMFF_sp_Database_Sigma_profile_prediction_actrelu_encActrelu_lrschedFalse_epochs700_lr0.00151_L2coef_7.79e-06_batchsize16_earlystopping_False.pth"):
        super(water_gegnn_SigmaProfile_noPooling, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        self.mfp_trans = nn.Linear(hidden_dim+51, hidden_dim+51)
        self.classify1 = nn.Linear(hidden_dim+51, hidden_dim)
        self.dropout1 = nn.Dropout(mlp_dropout_rate)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(mlp_dropout_rate)
        self.classify3 = nn.Linear(hidden_dim, n_classes)

        ## Sigma Profile Module
        self.SP_Module = SigmaProfileGCN(in_dim=50).cuda()
        self.SP_Module.load_state_dict(torch.load(sigma_profile_model_path)["model_state_dict"])
        self.SP_Module.eval()

    def forward(self, component_data, empty_solvsys):
        g1 = component_data['g1'].to("cuda")
        g2 = component_data['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                inter_hb = component_data['inter_hb'].float().cuda()
                intra_hb1 = component_data['intra_hb1'].float().cuda()
                intra_hb2 = component_data['intra_hb2'].float().cuda()
                
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

                # Sigma Profile prediction
                sp = self.SP_Module(component_data)


                # SLP
                hg1_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[0:len(hg)//2,:], sp), axis=1)))
                hg_temp = hg1_temp

                # MLP
                output = self.mlp_activation(self.classify1(hg_temp))
                output = self.dropout1(output)
                output = self.mlp_activation(self.classify2(output))
                output = self.dropout2(output)
                output = self.classify3(output) 
                
            return output
        

        
class RittigPure(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2, num_step_message_passing=1):
        super(RittigPure, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mfp_trans = nn.Linear(hidden_dim+0, hidden_dim+0)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim+0, hidden_dim)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, solvdata, empty_solvsys):
        g1 = solvdata['g1'].to("cuda")
        g2 = solvdata['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
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

class AbranchesPure(nn.Module):
    def __init__(self, in_dim=50, n_classes=1, sigma_profile_length=51, weight_init_fn=None,
                 sigma_profile_model_path="C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_final_model__MMFF_sp_Database_Sigma_profile_prediction_actrelu_encActrelu_lrschedFalse_epochs700_lr0.00151_L2coef_7.79e-06_batchsize16_earlystopping_False.pth"):
        super(AbranchesPure, self).__init__()

        ## Sigma Profile Module
        self.SP_Module = SigmaProfileGCN(in_dim=in_dim).cuda()
        self.SP_Module.load_state_dict(torch.load(sigma_profile_model_path)["model_state_dict"])
        self.SP_Module.eval()

        conv_1_param = [1, 10, 5]       #Channels (filters), kernel, stride
        self.pool_1_size = 8
        self.pool_1_stride = self.pool_1_size
        conv_2_param = [3, 8, 4]
        self.pool_2_size = 5
        self.pool_2_stride = self.pool_2_size
        dense_1_nodes = 2
        dense_2_nodes = 4

        padding_conv_1 = calculate_same_padding_conv1d(kernel_size=conv_1_param[1],
                                                       stride=conv_1_param[2],
                                                       input_length=sigma_profile_length)
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                                out_channels=conv_1_param[0],
                                kernel_size=conv_1_param[1],
                                stride=conv_1_param[2],
                                padding=padding_conv_1)

        self.pool_1 = nn.AvgPool1d(kernel_size=self.pool_1_size,
                                 padding=0,
                                 stride=self.pool_1_stride)
        
        padding_conv_2 = calculate_same_padding_conv1d(kernel_size=conv_2_param[1],
                                                       stride=conv_2_param[2],
                                                       input_length=sigma_profile_length)

        self.conv1d_2 = nn.Conv1d(in_channels=conv_1_param[0],
                                out_channels=conv_2_param[0],
                                kernel_size=conv_2_param[1],
                                stride=conv_2_param[2],
                                padding=padding_conv_2)
        
        self.pool_2 = nn.AvgPool1d(kernel_size=self.pool_2_size,
                                 padding=0,
                                 stride=self.pool_2_stride)

        #self.dense_1 = nn.Linear(in_features=math.ceil(sigma_profile_length/self.pool_2_stride)*conv_2_param[0], out_features=dense_1_nodes)
        self.dense_1 = nn.Linear(in_features=24, out_features=dense_1_nodes)
        self.dense_2 = nn.Linear(in_features=dense_1_nodes, out_features=dense_2_nodes)

        self.output = nn.Linear(in_features=dense_2_nodes, out_features=n_classes)

        if weight_init_fn == torch.nn.init.kaiming_uniform_:
            weight_init_fn(self.conv1d_1.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.conv1d_2.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.dense_1.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.dense_2.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif weight_init_fn == torch.nn.init.uniform_:
            weight_init_fn(self.conv1d_1.weight)
            weight_init_fn(self.conv1d_2.weight)
            weight_init_fn(self.dense_1.weight)
            weight_init_fn(self.dense_2.weight)
        #if weight_init_fn != None:
        #    torch.nn.init.zeros_(self.conv1d_1.bias)
        #    torch.nn.init.zeros_(self.conv1d_2.bias)
        #    torch.nn.init.zeros_(self.dense_1.bias)
        #    torch.nn.init.zeros_(self.dense_2.bias)


    def forward(self, component_data):
        
        sp = self.SP_Module(component_data)
        sp = sp.unsqueeze(0)
        sp = sp.permute(1, 0, 2)

        x = F.silu(self.conv1d_1(sp))
        padded_x = F.pad(x, padding_pool(self.pool_1_size), mode='replicate')
        x = self.pool_1(padded_x)

        x = F.silu(self.conv1d_2(x))
        padded_x = F.pad(x, padding_pool(self.pool_2_size), mode='replicate')
        x = self.pool_2(padded_x)

        flattened_x = x.view(x.shape[0], -1)

        x = F.silu(self.dense_1(flattened_x))
        x = F.silu(self.dense_2(x))

        output = self.output(x)

        return output

class RittigNoSolvSys(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, mlp_dropout_rate=0, mlp_activation=None, mpnn_activation=None, mlp_num_hid_layers=2, num_step_message_passing=1,
                 sigma_profile_model_path="C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_final_model__MMFF_sp_Database_Sigma_profile_prediction_actrelu_encActrelu_lrschedFalse_epochs700_lr0.00151_L2coef_7.79e-06_batchsize16_earlystopping_False.pth"):
        super(RittigNoSolvSys, self).__init__()


    def forward(self, component_data, empty_solvsys):
        pass


class Rittig_Abranches(nn.Module):
    def __init__(self, in_dim_rit=74, in_dim_abr=50, n_classes=1, sigma_profile_length=51, hidden_dim=256, weight_init_fn=None,
                 mpnn_activation='relu', num_step_message_passing=1, mlp_dropout_rate=0.0, mlp_num_hid_layers=2, mlp_activation='softplus',
                 sigma_profile_model_path="C:/Users/kverg/GDI-NN/SigmaProfileModel/results_SP/Complete_trainset_final_model__MMFF_sp_Database_Sigma_profile_prediction_actrelu_encActrelu_lrschedFalse_epochs700_lr0.00151_L2coef_7.79e-06_batchsize16_earlystopping_False.pth"):
        super(Rittig_Abranches, self).__init__()

    ## Abranches Module
        self.SP_Module = SigmaProfileGCN(in_dim=in_dim_abr).cuda()
        self.SP_Module.load_state_dict(torch.load(sigma_profile_model_path)["model_state_dict"])
        self.SP_Module.eval()

        conv_1_param = [1, 10, 5]       #Channels (filters), kernel, stride
        self.pool_1_size = 8
        self.pool_1_stride = self.pool_1_size
        conv_2_param = [3, 8, 4]
        self.pool_2_size = 5
        self.pool_2_stride = self.pool_2_size
        dense_1_nodes = 2
        dense_2_nodes = 4

        padding_conv_1 = calculate_same_padding_conv1d(kernel_size=conv_1_param[1],
                                                       stride=conv_1_param[2],
                                                       input_length=sigma_profile_length)
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                                out_channels=conv_1_param[0],
                                kernel_size=conv_1_param[1],
                                stride=conv_1_param[2],
                                padding=padding_conv_1)

        self.pool_1 = nn.AvgPool1d(kernel_size=self.pool_1_size,
                                 padding=0,
                                 stride=self.pool_1_stride)
        
        padding_conv_2 = calculate_same_padding_conv1d(kernel_size=conv_2_param[1],
                                                       stride=conv_2_param[2],
                                                       input_length=sigma_profile_length)

        self.conv1d_2 = nn.Conv1d(in_channels=conv_1_param[0],
                                out_channels=conv_2_param[0],
                                kernel_size=conv_2_param[1],
                                stride=conv_2_param[2],
                                padding=padding_conv_2)
        
        self.pool_2 = nn.AvgPool1d(kernel_size=self.pool_2_size,
                                 padding=0,
                                 stride=self.pool_2_stride)
        
        self.dense_1 = nn.Linear(in_features=24, out_features=dense_1_nodes)
        self.dense_2 = nn.Linear(in_features=dense_1_nodes, out_features=dense_2_nodes)

        #self.output = nn.Linear(in_features=dense_2_nodes, out_features=n_classes)
        if weight_init_fn == torch.nn.init.kaiming_uniform_:
            weight_init_fn(self.conv1d_1.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.conv1d_2.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.dense_1.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.dense_2.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif weight_init_fn == torch.nn.init.uniform_:
            weight_init_fn(self.conv1d_1.weight)
            weight_init_fn(self.conv1d_2.weight)
            weight_init_fn(self.dense_1.weight)
            weight_init_fn(self.dense_2.weight)
        #if weight_init_fn != None:
        #    torch.nn.init.zeros_(self.conv1d_1.bias)
        #    torch.nn.init.zeros_(self.conv1d_2.bias)
        #    torch.nn.init.zeros_(self.dense_1.bias)
        #    torch.nn.init.zeros_(self.dense_2.bias)


    # Rittig Module
        self.conv1 = GraphConv(in_dim_rit, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.global_conv1 = MPNNconv(node_in_feats=hidden_dim,
                                     edge_in_feats=1,
                                     node_out_feats=hidden_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        #self.mfp_trans = nn.Linear(hidden_dim+0, hidden_dim+0)
        self.mlp_dropout = nn.Dropout(mlp_dropout_rate)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(hidden_dim+4, hidden_dim)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(hidden_dim, n_classes)



    def forward(self, component_data, empty_solvsys):
    
    # Abranches Module
        sp = self.SP_Module(component_data)
        sp = sp.unsqueeze(0)
        sp = sp.permute(1, 0, 2)

        x = F.silu(self.conv1d_1(sp))
        padded_x = F.pad(x, padding_pool(self.pool_1_size), mode='replicate')
        x = self.pool_1(padded_x)

        x = F.silu(self.conv1d_2(x))
        padded_x = F.pad(x, padding_pool(self.pool_2_size), mode='replicate')
        x = self.pool_2(padded_x)

        flattened_x = x.view(x.shape[0], -1)

        x = F.silu(self.dense_1(flattened_x))
        x = F.silu(self.dense_2(x))
    
    # Rittig Module
        g1 = component_data['g1'].to("cuda")
        g2 = component_data['g2'].to("cuda")
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().cuda()
                h2 = g2.ndata['h'].float().cuda()
                inter_hb = component_data['inter_hb'].float().cuda()
                intra_hb1 = component_data['intra_hb1'].float().cuda()
                intra_hb2 = component_data['intra_hb2'].float().cuda()

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

                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))  ## NEED TO CONCATENATE THE X FROM THE SP
                # SLP
                #hg1_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[0:len(hg)//2,:], solv1x[:, None]), axis=1)))
                #hg2_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[len(hg)//2:,:], 1-solv1x[:, None]), axis=1)))
                hg_temp = torch.concat((hg[0:len(hg)//2,:], x), axis=1)

                
                output = self.mlp_dropout(hg_temp)
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_dropout(output)
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                              
            return output



class SigmaProfileGCN_from_graph(nn.Module):
    def __init__(self, in_dim):
        super(SigmaProfileGCN, self).__init__()
        self.conv1 = GraphConv(in_dim, 300)
        self.conv2 = GraphConv(300, 153)
        self.conv3 = GraphConv(153, 161)
        self.classify = nn.Linear(161, 51)
        self.globalpooling = SumPooling()

    def forward(self, graph):
        with graph.local_scope():
            h1 = graph.ndata['h'].float().cuda()
                
            h1_temp = F.relu(self.conv1(graph, h1))
            h1_temp = F.relu(self.conv2(graph, h1_temp))
            h1_temp = F.relu(self.conv3(graph, h1_temp))
            graph.ndata['h'] = h1_temp

            hg = h1_temp
            output = F.relu(self.classify(hg))
            output = self.globalpooling(graph, output)
            return output
        
class RA_opt(nn.Module):
    def __init__(self, in_dim_rit, in_dim_abr, n_classes, sigma_profile_length, weight_init_fn,
                mpnn_activation, num_step_message_passing, mlp_activation,
                mlp_num_hid_layers,
                gcn_dim, mpnn_dim, mlp_dim,
                n_conv_filters, kernel_size, pool_size, SP_dim,
                device,
                sigma_profile_model_path):
        super(RA_opt, self).__init__()
    ## Abranches Module
        self.SP_Module = SigmaProfileGCN(in_dim=in_dim_abr).cuda()
        self.SP_Module.load_state_dict(torch.load(sigma_profile_model_path)["model_state_dict"])
        self.SP_Module.eval()

        conv_1_param = [n_conv_filters, kernel_size, kernel_size//2]       #Channels (filters), kernel, stride
        self.pool_1_size = pool_size
        self.pool_1_stride = self.pool_1_size
        conv_2_param = [n_conv_filters, kernel_size, kernel_size//2]
        self.pool_2_size = pool_size
        self.pool_2_stride = self.pool_2_size
        dense_1_nodes = SP_dim
        dense_2_nodes = SP_dim

        padding_conv_1 = calculate_same_padding_conv1d(kernel_size=conv_1_param[1],
                                                       stride=conv_1_param[2],
                                                       input_length=sigma_profile_length)
        self.conv1d_1 = nn.Conv1d(in_channels=1,
                                out_channels=conv_1_param[0],
                                kernel_size=conv_1_param[1],
                                stride=conv_1_param[2],
                                padding=padding_conv_1)

        self.pool_1 = nn.AvgPool1d(kernel_size=self.pool_1_size,
                                 padding=0,
                                 stride=self.pool_1_stride)
        
        padding_conv_2 = calculate_same_padding_conv1d(kernel_size=conv_2_param[1],
                                                       stride=conv_2_param[2],
                                                       input_length=sigma_profile_length)

        self.conv1d_2 = nn.Conv1d(in_channels=conv_1_param[0],
                                out_channels=conv_2_param[0],
                                kernel_size=conv_2_param[1],
                                stride=conv_2_param[2],
                                padding=padding_conv_2)
        
        self.pool_2 = nn.AvgPool1d(kernel_size=self.pool_2_size,
                                 padding=0,
                                 stride=self.pool_2_stride)
        
        self.dense_1 = nn.LazyLinear(out_features=dense_1_nodes)
        self.dense_2 = nn.Linear(in_features=dense_1_nodes, out_features=dense_2_nodes)

        #self.output = nn.Linear(in_features=dense_2_nodes, out_features=n_classes)
        if weight_init_fn == torch.nn.init.kaiming_uniform_:
            weight_init_fn(self.conv1d_1.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.conv1d_2.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.dense_1.weight, mode='fan_in', nonlinearity='leaky_relu')
            weight_init_fn(self.dense_2.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif weight_init_fn == torch.nn.init.uniform_:
            weight_init_fn(self.conv1d_1.weight)
            weight_init_fn(self.conv1d_2.weight)
            weight_init_fn(self.dense_1.weight)
            weight_init_fn(self.dense_2.weight)
        #if weight_init_fn != None:
        #    torch.nn.init.zeros_(self.conv1d_1.bias)
        #    torch.nn.init.zeros_(self.conv1d_2.bias)
        #    torch.nn.init.zeros_(self.dense_1.bias)
        #    torch.nn.init.zeros_(self.dense_2.bias)


    # Rittig Module
        self.conv1 = GraphConv(in_dim_rit, gcn_dim)
        self.conv2 = GraphConv(gcn_dim, gcn_dim)
        self.global_conv1 = MPNNconv(node_in_feats=gcn_dim,
                                     edge_in_feats=1,
                                     node_out_feats=mpnn_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        #self.mfp_trans = nn.Linear(hidden_dim+0, hidden_dim+0)
        self.mlp_activation = get_activation(mlp_activation)
        self.mlp_num_hid_layers = mlp_num_hid_layers
        self.classify1 = nn.Linear(mpnn_dim+SP_dim, mlp_dim)
        if self.mlp_num_hid_layers == 2:
            self.classify2 = nn.Linear(mlp_dim, mlp_dim)
        elif self.mlp_num_hid_layers != 1:
            raise ValueError(f"num_mlp_hid_layers has invalid value. Choose either 1 or 2.")
        self.classify3 = nn.Linear(mlp_dim, n_classes)



    def forward(self, component_data, empty_solvsys, device):
    
    # Abranches Module
        sp = self.SP_Module(component_data, device)
        sp = sp.unsqueeze(0)
        sp = sp.permute(1, 0, 2)

        x = F.silu(self.conv1d_1(sp))
        padded_x = F.pad(x, padding_pool(self.pool_1_size), mode='replicate')
        x = self.pool_1(padded_x)

        x = F.silu(self.conv1d_2(x))
        padded_x = F.pad(x, padding_pool(self.pool_2_size), mode='replicate')
        x = self.pool_2(padded_x)

        flattened_x = x.view(x.shape[0], -1)

        x = F.silu(self.dense_1(flattened_x))
        x = F.silu(self.dense_2(x))
    
    # Rittig Module
        g1 = component_data['g1'].to(device)
        g2 = component_data['g2'].to(device)
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().to(device)
                h2 = g2.ndata['h'].float().to(device)
                inter_hb = component_data['inter_hb'].float().to(device)
                intra_hb1 = component_data['intra_hb1'].float().to(device)
                intra_hb2 = component_data['intra_hb2'].float().to(device)

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                hg1 = dgl.mean_nodes(g1, 'h').to(device)
                hg2 = dgl.mean_nodes(g2, 'h').to(device)
                
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 

                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))  ## NEED TO CONCATENATE THE X FROM THE SP
                # SLP
                #hg1_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[0:len(hg)//2,:], solv1x[:, None]), axis=1)))
                #hg2_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[len(hg)//2:,:], 1-solv1x[:, None]), axis=1)))
                hg_temp = torch.concat((hg[0:len(hg)//2,:], x), axis=1)

                
                output = hg_temp
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                              
            return output



class RA_var_opt(nn.Module):
    def __init__(self, in_dim_rit, in_dim_abr, n_classes, sigma_profile_length, weight_init_fn,
                    mpnn_activation, num_step_message_passing, mlp_activation,
                    mlp_num_hid_layers,
                    gcn_dim, mpnn_dim, mlp_dim,
                    device, 
                    sigma_profile_model_path):
        super(RA_var_opt, self).__init__()
        self.conv1 = GraphConv(in_dim_rit, gcn_dim)
        self.conv2 = GraphConv(gcn_dim, gcn_dim)
        self.global_conv1 = MPNNconv(node_in_feats=gcn_dim,
                                     edge_in_feats=1,
                                     node_out_feats=mpnn_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mlp_activation = get_activation(mlp_activation)
        self.mfp_trans = nn.Linear(mpnn_dim+sigma_profile_length, mlp_dim)
        self.mlp2 = nn.Linear(mlp_dim, mlp_dim)
        self.mlp3 = nn.Linear(mlp_dim, mlp_dim)
        self.output_layer = nn.Linear(mlp_dim, n_classes)

        ## Sigma Profile Module
        self.SP_Module = SigmaProfileGCN(in_dim=in_dim_abr).to(device)
        self.SP_Module.load_state_dict(torch.load(sigma_profile_model_path, weights_only=True)["model_state_dict"])
        self.SP_Module.eval()

    def forward(self, component_data, empty_solvsys, device):
        g1 = component_data['g1'].to(device)
        g2 = component_data['g2'].to(device)
        with g1.local_scope():
            with g2.local_scope():
                
                h1 = g1.ndata['h'].float().to(device)
                h2 = g2.ndata['h'].float().to(device)
                inter_hb = component_data['inter_hb'].float().to(device)
                intra_hb1 = component_data['intra_hb1'].float().to(device)
                intra_hb2 = component_data['intra_hb2'].float().to(device)
                
                # Molecule embedding
                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
                g1.ndata['h'] = h1_temp
                g2.ndata['h'] = h2_temp
                
                # Molecular interaction
                hg1 = dgl.mean_nodes(g1, 'h').to(device)
                hg2 = dgl.mean_nodes(g2, 'h').to(device)
                hg = self.global_conv1(empty_solvsys, 
                                       torch.cat((hg1,hg2),axis=0), 
                                       torch.cat((inter_hb.repeat(2),intra_hb1,intra_hb2)).unsqueeze(1))

                # Sigma Profile prediction
                sp = self.SP_Module(component_data, device)


                # SLP
                hg1_temp = self.mlp_activation(self.mfp_trans(torch.cat((hg[0:len(hg)//2,:], sp), axis=1)))
                hg_temp = hg1_temp

                # MLP
                output = self.mlp_activation(self.mlp2(hg_temp))
                output = self.mlp_activation(self.mlp3(output))
                output = self.output_layer(output) 

            return output

class RittigPure_opt(nn.Module):
    def __init__(self, in_dim, gcn_dim, mpnn_dim, mlp_dim, n_classes, mlp_activation, mpnn_activation, mlp_num_hid_layers, num_step_message_passing, device):
        super(RittigPure_opt, self).__init__()
        self.conv1 = GraphConv(in_dim, gcn_dim)
        self.conv2 = GraphConv(gcn_dim, gcn_dim)
        self.global_conv1 = MPNNconv(node_in_feats=gcn_dim,
                                     edge_in_feats=1,
                                     node_out_feats=mpnn_dim,
                                     edge_hidden_feats=32,
                                     num_step_message_passing=num_step_message_passing,
                                     activation=mpnn_activation)
        self.mfp_trans = nn.Linear(mpnn_dim+0, mlp_dim+0)
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

                h1_temp = F.relu(self.conv1(g1, h1))
                h1_temp = F.relu(self.conv2(g1, h1_temp))
                h2_temp = F.relu(self.conv1(g2, h2))
                h2_temp = F.relu(self.conv2(g2, h2_temp))
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
                
                output = hg_temp
                output = self.mlp_activation(self.classify1(output))
                if self.mlp_num_hid_layers == 2:
                    output = self.mlp_activation(self.classify2(output))
                output = self.classify3(output)                              
            return output    

class Qin_ternary(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, device):
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

    def forward(self, solvdata, empty_solvsys, device):

        g1 = solvdata['g1']
        g2 = solvdata['g2']
        g3 = solvdata['g3']
        with g1.local_scope():
            with g2.local_scope():
                with g3.local_scope():
                    h1 = g1.ndata['h'].float().to(device)
                    h2 = g2.ndata['h'].float().to(device)
                    h3 = g3.ndata['h'].float().to(device)
                    solv1x = solvdata['solv1_x'].float().to(device)
                    solv2x = solvdata['solv2_x'].float().to(device)
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
                    hg2 = torch.cat((hg2, solv2x[:, None]), axis=1)
                    # hg1 = solv1x[:,None]*hg1
                    # hg2 = solv2x[:,None]*hg2
                    # hg3 = solv3x[:,None]*hg3
            
                    hg = self.global_conv1(empty_solvsys, 
                                           torch.cat((hg1,hg2,hg3),axis=0), 
                                           torch.cat((inter_hb12.repeat(2),inter_hb13.repeat(2),
                                                      inter_hb23.repeat(2),
                                                      intra_hb1,intra_hb2,intra_hb3)).unsqueeze(1))
                    output = F.relu(self.classify1(hg))
                    output = F.relu(self.classify2(output))
                    output = self.classify3(output)
                    output = torch.cat(
                        (output[0:len(output)//3,:],
                        output[len(output)//3:2*len(output)//3,:],
                        output[2*len(output)//3:,:]),axis=1)

                    return output