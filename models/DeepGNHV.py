import gc
import time
import random

import numpy as np
from tqdm import tqdm
from pathlib import Path
# from einops import rearrange, reduce, repeat
from typing import Type,Union,List,Optional

import torch
import torch.optim as optim
from torch import nn

from torch.nn import functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, GraphSAGE, GIN, global_mean_pool, TopKPooling, global_max_pool
from torch_geometric.nn.pool import SAGPooling
from torch_geometric.nn.norm import GraphNorm


class DeepGNHV(nn.Module):
    def __init__(self
                , Graph_Type         = Type[Union["Semidirectedgraph", 'Undirectedgraph', 'Surfacegraph']]
                , GNN_Type           = Type[Union["GCNConv", "GATConv", "TransformerConv", "GraphSAGE", "GIN"]]
                , Attention_Heads    = [12, 12]
                , Dropout_Rates      = 0.5
                , Dimensions         = [1024, 512, 256]
                , num_classes        = 2
                , num_GNN_Layers     = 2
                , num_MLP_Layers     = 1
                , Concat             = False
                , GNNBN              = True
                , GNNDP              = False
                , MLPBN              = True
                , MLPDP              = False
                , Pool_Type          = 'max'
                , MLPDecrease        = 2
                ):
        super().__init__()
        
        self.Graph_Type        = Graph_Type
        self.GNN_Type          = GNN_Type
        self.Attention_Heads   = Attention_Heads
        self.Dropout_Rates     = Dropout_Rates
        self.GNNDimensions     = Dimensions
        self.num_classes       = num_classes
        self.num_GNN_Layers    = num_GNN_Layers
        self.num_MLP_Layers    = num_MLP_Layers
        self.Concat            = Concat
        self.Pool_Type         = Pool_Type
        self.GNNBN             = GNNBN
        self.GNNDP             = GNNDP
        self.MLPBN             = MLPBN
        self.MLPDP             = MLPDP
        self.MLPDecrease       = MLPDecrease
        
        self.encoder = Encoder(GNN_Type        = self.GNN_Type,
                               GNNDimensions   = self.GNNDimensions,
                               Attention_Heads = self.Attention_Heads,
                               num_GNN_Layers  = self.num_GNN_Layers,
                               Graph_Type      = self.Graph_Type,
                               GNNBN           = self.GNNBN,
                               GNNDP           = self.GNNDP,
                               Dropout_Rates   = self.Dropout_Rates)

        self.decoder = Decoder(Concat          = self.Concat, 
                                Pool_Type      = self.Pool_Type , 
                                MLPBN          = self.MLPBN, 
                                MLPDP          = self.MLPDP,
                                MLPDecrease    = self.MLPDecrease, 
                                num_MLP_Layers = self.num_MLP_Layers, 
                                num_classes    = self.num_classes, 
                                Dropout_Rates  = self.Dropout_Rates,
                                output_dim     = self.GNNDimensions[-1],
                                )
        

    def forward(self, protein_pair_idx,
                    protein1_feat, protein1_edge_index, protein1_nodes_batch,
                    protein2_feat, protein2_edge_index, protein2_nodes_batch,
                    protein1_edge_weight = None, protein2_edge_weight = None,
                    return_attention_weights = False):
        
        protein1_feat, protein1_attention_list = self.encoder(protein1_feat, protein1_edge_index, protein1_nodes_batch,
                                                                protein1_edge_weight, return_attention_weights)
        protein2_feat, protein2_attention_list = self.encoder(protein2_feat, protein2_edge_index, protein2_nodes_batch,
                                                                protein2_edge_weight, return_attention_weights)
        
        output = self.decoder(protein1_feat, protein2_feat, protein1_nodes_batch, protein2_nodes_batch)
        
        
        if return_attention_weights:
            return output, protein1_attention_list, protein2_attention_list
        else:
            return output, None, None
        
    
    # def SAG_Container_build(self, Dimensions, num_Layers):
    #     SAG_Container = nn.ModuleList()
    #     for Layers_id in range(num_Layers):
    #         SAG_Container.append(SAGPooling(Dimensions[Layers_id + 1], 0.8))
    #     return SAG_Container

    # def MLP_Dim_Generate(self, MLPDecrease, MLP_start_dim, num_MLP_Layers):
    #     MLPDimensions = [MLP_start_dim]
    #     current_value = MLP_start_dim
    #     for i in range(num_MLP_Layers):
    #         current_value //= MLPDecrease
    #         MLPDimensions.append(current_value)
    #     return MLPDimensions



class Encoder(nn.Module):
    def __init__(self, GNN_Type         = Type[Union["GCNConv", "GATConv", "TransformerConv", "GraphSAGE", "GIN"]],
                        GNNDimensions   = [1024, 720, 360],
                        Attention_Heads = [12, 12],
                        num_GNN_Layers  = 2,
                        Graph_Type      = 'Undirectedgraph',
                        GNNBN           = True,
                        GNNDP           = False,
                        Dropout_Rates   = 0.3):
        super(Encoder, self).__init__()
        
        self.GNN_Type        = GNN_Type
        self.GNNDimensions   = GNNDimensions
        self.Attention_Heads = Attention_Heads
        self.num_GNN_Layers  = num_GNN_Layers
        self.Graph_Type      = Graph_Type
        self.Dropout_Rates   = Dropout_Rates
        self.GNNBN           = GNNBN
        self.GNNDP           = GNNDP
        self.GNN_Container   = self.GNN_Container_build(GNN_Type          = self.GNN_Type
                                                        , GNNDimensions   = self.GNNDimensions
                                                        , Attention_Heads = self.Attention_Heads
                                                        , num_Layers      = self.num_GNN_Layers
                                                        , Graph_Type      = self.Graph_Type
                                                        , Dropout_Rates   = self.Dropout_Rates)
        self.GNN_Norm_Container = self.BatchNorm_Container_build(self.GNNDimensions, self.num_GNN_Layers) if self.GNNBN else None
        self.Activation         = nn.Mish()   # nn.LeakyReLU(negative_slope = 0.1)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     for layer in self.GNN_Container:
    #         torch.nn.init.xavier_uniform_(layer.weight)
    #         if layer.bias is not None:
    #             layer.bias.data.fill_(0)

    def forward(self, protein_feat, protein_edge_index, protein_nodes_batch,
                        protein_edge_weight = None,
                        return_attention_weights = False):

        protein_attention_list = [] if return_attention_weights else None
        
        if self.Graph_Type == "Semidirectedgraph":
            protein_edge_index_retain = protein_edge_index
            
        for layer_id, GNN_Layer in enumerate(self.GNN_Container):
            if self.Graph_Type == "Semidirectedgraph":
                if self.num_GNN_Layers - 1 == layer_id:
                    protein_edge_index_mask = protein_edge_index_retain[2, :] == 1
                    protein_edge_index = protein_edge_index_retain[:2, protein_edge_index_mask]
                else:
                    protein_edge_index = protein_edge_index_retain[:2]


            if return_attention_weights:
                protein_feat, (_, protein_edges_attention_weight) = GNN_Layer(x = protein_feat, edge_index = protein_edge_index, return_attention_weights = return_attention_weights)
                protein_attention_list.append(protein_edges_attention_weight)

            else:
                protein_feat = GNN_Layer(x = protein_feat, edge_index = protein_edge_index)

            if self.GNNBN:
                protein_feat = self.GNN_Norm_Container[layer_id](protein_feat)
  
            protein_feat = self.GNNDP_Layer(self.Activation(protein_feat)) if self.GNNDP else self.Activation(protein_feat)
            
        return protein_feat, protein_attention_list
    
    def BatchNorm_Container_build(self, Dimensions, num_Layers):
        BatchNorm_Container = nn.ModuleList()
        for Layers_id in range(num_Layers):
            bn_layer = nn.BatchNorm1d(Dimensions[Layers_id + 1])
            # self.init_bn_layer(bn_layer)
            BatchNorm_Container.append(bn_layer)
        return BatchNorm_Container
    
    def GraphNorm_Container_build(self, Dimensions, num_Layers):
        GraphNorm_Container = nn.ModuleList()
        for Layers_id in range(num_Layers):
            GraphNorm_Container.append(GraphNorm(Dimensions[Layers_id + 1]))
        return GraphNorm_Container

    # def init_bn_layer(self, bn_layer):
    #     nn.init.constant_(bn_layer.weight, 1)
    #     nn.init.constant_(bn_layer.bias, 0)

    def GNN_Container_build(self, GNN_Type        = Type[Union["GCNConv", "GATConv", "TransformerConv", "GraphSAGE", "GIN"]]
                                , GNNDimensions   = [1024, 720, 360]
                                , Attention_Heads = [12, 12]
                                , num_Layers      = 2
                                , Graph_Type      = 'Undirectedgraph'
                                , Dropout_Rates   = 0.3):
        
        
        GNN_Container = nn.ModuleList()
        for Layers_id in range(num_Layers):
            add_self_loops = True if Graph_Type == 'Semidirectedgraph' and Layers_id == num_Layers-1 else False
            print(f'Layers_id:{Layers_id} add_self_loops:{add_self_loops}')
            if GNN_Type == "GraphSAGE":
                GNN = GraphSAGE(in_channels   = GNNDimensions[Layers_id], hidden_channels = GNNDimensions[Layers_id + 1], out_channels = GNNDimensions[Layers_id + 1], num_layers = 1)
            elif GNN_Type == "GIN":
                GNN = GIN(in_channels   = GNNDimensions[Layers_id], hidden_channels  = GNNDimensions[Layers_id + 1], out_channels = GNNDimensions[Layers_id + 1], num_layers = 1)
            elif GNN_Type == "GCNConv":
                GNN = GCNConv(in_channels   = GNNDimensions[Layers_id], out_channels = GNNDimensions[Layers_id + 1], add_self_loops = add_self_loops)
            elif GNN_Type == "GATConv":
                GNN = GATConv(in_channels  = GNNDimensions[Layers_id], 
                              out_channels = int(GNNDimensions[Layers_id + 1]/Attention_Heads[Layers_id]),
                              heads        = Attention_Heads[Layers_id], 
                              concat       = True,
                              add_self_loops = add_self_loops)
            elif GNN_Type == "TransformerConv":
                GNN = TransformerConv(in_channels  = GNNDimensions[Layers_id], 
                                      out_channels = int(GNNDimensions[Layers_id + 1]/Attention_Heads[Layers_id]),
                                      heads        = Attention_Heads[Layers_id], 
                                      concat       = True)
            else:
                print('Please input right Graph Neural model')
            GNN_Container.append(GNN)
            
        return GNN_Container

class Decoder(nn.Module):
    def __init__(self, Concat         = False, 
                       Pool_Type      = 'max', 
                       MLPBN          = False, 
                       MLPDP          = True,
                       MLPDecrease    = 2, 
                       num_MLP_Layers = 1, 
                       num_classes    = 2, 
                       Dropout_Rates  = 0.3,
                       output_dim     = 360,
                       ):
        super(Decoder, self).__init__()
        self.Concat         = Concat
        self.Pool_Type      = Pool_Type
        self.MLPBN          = MLPBN
        self.MLPDP          = MLPDP
        self.MLPDecrease    = MLPDecrease
        self.num_MLP_Layers = num_MLP_Layers
        self.num_classes    = num_classes
        self.Dropout_Rates  = Dropout_Rates
        self.output_dim     = output_dim
        self.pooling        = global_max_pool if Pool_Type == 'max' else global_mean_pool
        self.MLP_start_dim  = int(self.output_dim * 2) if self.Concat else int(self.output_dim)
        
        if self.num_MLP_Layers > 0:
            self.Linear_Container, self.final_dim = self.MLP_Container_build(self.MLP_start_dim, self.num_MLP_Layers, self.MLPDecrease, self.Dropout_Rates, self.MLPBN, self.MLPDP)
        else:
            self.Linear_Container, self.final_dim = None, self.MLP_start_dim
        
        self.classifier = nn.Linear(self.final_dim, self.num_classes)


    def forward(self, protein1_feat, protein2_feat, protein1_nodes_batch, protein2_nodes_batch):
        protein1_feat_afterPooling = self.pooling(protein1_feat, protein1_nodes_batch)
        protein2_feat_afterPooling = self.pooling(protein2_feat, protein2_nodes_batch)
        
        Global_embedding = torch.cat([protein1_feat_afterPooling, protein2_feat_afterPooling], dim = -1) if self.Concat else torch.mul(protein1_feat_afterPooling, protein2_feat_afterPooling)
        
        if self.num_MLP_Layers > 0:
            for layer_id, Linear_Layer in enumerate(self.Linear_Container):
                Global_embedding = Linear_Layer(Global_embedding)
        
        output = self.classifier(Global_embedding)
        
        return output
        
       
    def MLP_Container_build(self, MLP_start_dim, num_MLP_Layers, MLPDecrease, Dropout_Rates, MLPBN, MLPDP):
        MLP_Container = nn.ModuleList()
        input_dim     = MLP_start_dim
        for Layers_id in range(num_MLP_Layers):
            Sequential_Module = nn.Sequential(nn.Linear(int(input_dim), int(input_dim/MLPDecrease)))
            if MLPBN:
                Sequential_Module.add_module('MLP_BatchNorm1d', nn.BatchNorm1d(int(input_dim/MLPDecrease)))
                # Sequential_Module.add_module('LeakyReLU', nn.LeakyReLU(negative_slope = 0.05))
            Sequential_Module.add_module('Mish', nn.Mish())
            if MLPDP:
                Sequential_Module.add_module('MLP_Dropout', nn.Dropout(p = Dropout_Rates))
            self.init_mlp_layer(Sequential_Module)
            MLP_Container.append(Sequential_Module)
            input_dim = int(input_dim/MLPDecrease)
        return MLP_Container, input_dim
    
    def init_mlp_layer(self, Sequential_Module):
        for layer in Sequential_Module:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
                


    

        