import os
import pickle
import math
import time
import numpy as np
import networkx as nx
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim as optim
from torch_geometric.data import Data, Batch

import random
random.seed(100)

def explainer_main(config, logger, model, Datasets_Eval, DataLoader_Eval):
    explainer   = Explainer(model = model)
    explain_dir = Path(config.DATA.Eval.Explain_Dir)
    os.makedirs(explain_dir, exist_ok=True)
    
    Graph_Type         = config.MODEL.DeepGNHV.Graph_Type
    
    Explain_indices = list(range(len(Datasets_Eval)))
    random.shuffle(Explain_indices)

    print('Explain_LR: ',     config.DATA.Eval.Explain_LR)
    print('Explain_Epochs: ', config.DATA.Eval.Explain_Epochs)

    for idx in Explain_indices:
        idx, Protein1_Length, Protein2_Length, Protein1_Embedding_Graph, Protein2_Embedding_Graph, Label = Datasets_Eval[idx]
        if Label[1] == 1:
            print(Label)
            Protein1_name, Protein2_name, _ = Datasets_Eval.Sample_List[idx]
            print(f'Start treating {Protein1_name} and {Protein2_name}')
            explainer.explain_graphs(Protein1_name   = Protein1_name
                                    , Protein2_name  = Protein2_name
                                    , Protein1_Embedding_graphs = Batch.from_data_list([Protein1_Embedding_Graph])
                                    , Protein2_Embedding_graphs = Batch.from_data_list([Protein2_Embedding_Graph])
                                    , Label           = Label
                                    , explain_dir     = explain_dir
                                    , Graph_Type      = Graph_Type
                                    , DataLoader_Eval = DataLoader_Eval
                                    , explain_lr      = float(config.DATA.Eval.Explain_LR)
                                    , explain_epochs  = int(config.DATA.Eval.Explain_Epochs))
            print(f'------------------------------------------------------------------------------------------------------------------------')
            print(f'------------------------------------------------------------------------------------------------------------------------')

class Explainer:
    def __init__(self, model):
        self.model = model
        
    def adjacency_matrix_symmetrize(self, adj_PyG):
        graph_node_num = int(torch.max(adj_PyG).detach() + 1)
        adj_Symmetric = torch.zeros([graph_node_num,graph_node_num], dtype=torch.float).cuda(non_blocking=True)
        adj_Symmetric[adj_PyG[1],adj_PyG[0]] = 1
        return torch.unsqueeze(adj_Symmetric, dim=0)
    
    def edge_weight_assign_symmetrize(self, edge_weight, adj_PyG):
        graph_node_num = int(torch.max(adj_PyG).detach() + 1)
        masked_adj_forward     = torch.zeros([1, graph_node_num, graph_node_num], dtype=torch.float).cuda(non_blocking=True)
        masked_adj_forward[0, adj_PyG[1,:], adj_PyG[0,:]] = edge_weight
        masked_adj_reverse = masked_adj_forward.transpose(1,2)
        masked_adj = (masked_adj_forward + masked_adj_reverse)/2
        return masked_adj
    
    def save_results(self, save_file_path, result_dict):
        with open(save_file_path, "wb") as h:
            pickle.dump(result_dict, h)

    def explain_graphs(self,Protein1_name, Protein2_name
                            , Protein1_Embedding_graphs
                            , Protein2_Embedding_graphs
                            , Label
                            , explain_dir
                            , Graph_Type
                            , DataLoader_Eval
                            , explain_lr
                            , explain_epochs):
        
        Protein1_feat_weight, Protein2_feat_weight = self.explain(Protein1_Embedding_graphs, Protein2_Embedding_graphs
                                                                    , Label
                                                                    , explain_lr
                                                                    , explain_epochs)

        save_file_path = explain_dir/f'{Protein1_name}_{Protein2_name}.{Graph_Type}_explain_result'
        result_dict = {"Protein1_name"        : Protein1_name,
                       "Protein2_name"        : Protein2_name,
                       "Protein1_feat_weight" : np.array(Protein1_feat_weight),
                       "Protein2_feat_weight" : np.array(Protein2_feat_weight),
                       "explain_lr"           : explain_lr,
                       "explain_epochs"       : explain_epochs
                    }
        
        self.save_results(save_file_path, result_dict)

    
    def Explainer_retraining(self, explainer_module, explain_epochs, Protein1_feat_Train, Protein2_feat_Train):
        for epoch in range(1, explain_epochs + 1):
            explainer_module.zero_grad()
            if Protein1_feat_Train and not Protein2_feat_Train:
                explainer_module.protein1_optimizer.zero_grad()
            elif not Protein1_feat_Train and Protein2_feat_Train:
                explainer_module.protein2_optimizer.zero_grad()
            elif Protein1_feat_Train and Protein2_feat_Train:
                explainer_module.protein_optimizer.zero_grad()
            if epoch == 1:
                initial_yhat = explainer_module.forward(False, False).detach()
                pred_score = round(float(torch.softmax(initial_yhat, dim=1)[:,1].detach().cpu()), 4)
                print(f'pred_score:{pred_score}')

            else:
                explainer_pred = explainer_module.forward(Protein1_feat_Train, Protein2_feat_Train)
                loss = explainer_module.loss(explainer_pred, initial_yhat, epoch, Protein1_feat_Train, Protein2_feat_Train)
                loss.backward()
                if loss.item() <= 0.05 * (Protein1_feat_Train + Protein2_feat_Train):
                    print("Early stopping at epoch:", epoch)
                    break

                if Protein1_feat_Train and not Protein2_feat_Train:
                    explainer_module.protein1_optimizer.step()
                elif not Protein1_feat_Train and Protein2_feat_Train:
                    explainer_module.protein2_optimizer.step()
                elif Protein1_feat_Train and Protein2_feat_Train:
                    explainer_module.protein_optimizer.step()

    def explain(self, Protein1_Embedding_graphs
                    , Protein2_Embedding_graphs
                    , Label
                    , explain_lr
                    , explain_epochs):

        Protein1_feat          = Protein1_Embedding_graphs.x.cuda(non_blocking=True)
        Protein2_feat          = Protein2_Embedding_graphs.x.cuda(non_blocking=True)
        Protein1_batch         = Protein1_Embedding_graphs.batch.cuda(non_blocking=True)
        Protein2_batch         = Protein2_Embedding_graphs.batch.cuda(non_blocking=True)
        Protein1_edge_index    = Protein1_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein2_edge_index    = Protein2_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein1_num_nodes     = Protein1_feat.shape[0]
        Protein2_num_nodes     = Protein2_feat.shape[0]

        Label                  = torch.tensor(Label, dtype=torch.float32).reshape(-1,2).cuda(non_blocking=True)
        
        # self.model.eval()
        
        explainer_module = ExplainModule(Protein1_edge_index   = Protein1_edge_index
                                        , Protein2_edge_index  = Protein2_edge_index
                                        , Protein1_feat        = Protein1_feat
                                        , Protein2_feat        = Protein2_feat
                                        , Protein1_batch       = Protein1_batch
                                        , Protein2_batch       = Protein2_batch
                                        , Protein1_num_nodes   = Protein1_num_nodes
                                        , Protein2_num_nodes   = Protein2_num_nodes
                                        , explain_lr           = explain_lr
                                        , model                = self.model
                                        , Label                = Label
                                        )
        explainer_module = explainer_module.cuda()
        explainer_module.eval()


        if explainer_module.combine:
            self.Explainer_retraining(explainer_module, explain_epochs, True, True)
        else:
            self.Explainer_retraining(explainer_module, explain_epochs, True, False)
            self.Explainer_retraining(explainer_module, explain_epochs, False, True)

        Protein1_feat_weight   = explainer_module.smooth_weights(explainer_module.Protein1_feat_weight, explainer_module.window_size).detach().cpu()
        Protein2_feat_weight   = explainer_module.smooth_weights(explainer_module.Protein2_feat_weight, explainer_module.window_size).detach().cpu()
        
        return Protein1_feat_weight, Protein2_feat_weight

        


class ExplainModule(nn.Module):
    def __init__(self
                , Protein1_edge_index
                , Protein2_edge_index
                , Protein1_feat
                , Protein2_feat
                , Protein1_batch
                , Protein2_batch
                , Protein1_num_nodes
                , Protein2_num_nodes
                , explain_lr
                , model
                , Label
            ):
        super(ExplainModule, self).__init__()
        
        self.Protein1_edge_index   = Protein1_edge_index
        self.Protein2_edge_index   = Protein2_edge_index
        self.Protein1_feat         = Protein1_feat
        self.Protein2_feat         = Protein2_feat
        self.Protein1_feat.requires_grad = True
        self.Protein2_feat.requires_grad = True
        self.Protein1_batch        = Protein1_batch
        self.Protein2_batch        = Protein2_batch
        self.Protein1_num_nodes    = Protein1_num_nodes
        self.Protein2_num_nodes    = Protein2_num_nodes
        self.explain_lr            = explain_lr
        self.model                 = model
        self.Label                 = Label

        init_strategy = "normal"
        
        
        self.Protein1_num_edges = self.Protein1_edge_index.shape[-1]
        self.Protein2_num_edges = self.Protein2_edge_index.shape[-1]

        self.Protein1_feat_weight = self.construct_feat_weight(self.Protein1_num_nodes)
        self.Protein2_feat_weight = self.construct_feat_weight(self.Protein2_num_nodes)
        
        Protein1_params = [self.Protein1_feat_weight]
        Protein2_params = [self.Protein2_feat_weight]
        Protein_params  = [self.Protein1_feat_weight, self.Protein2_feat_weight]

        self.protein1_optimizer = optim.AdamW(filter(lambda p : p.requires_grad, Protein1_params),
                                    lr    = self.explain_lr,
                                    eps   = 1e-9,
                                    betas = (0.9, 0.999),
                                    weight_decay = 0)
        self.protein2_optimizer = optim.AdamW(filter(lambda p : p.requires_grad, Protein2_params),
                                    lr    = self.explain_lr,
                                    eps   = 1e-9,
                                    betas = (0.9, 0.999),
                                    weight_decay = 0)

        self.protein_optimizer = optim.AdamW(filter(lambda p : p.requires_grad, Protein_params),
                                    lr    = self.explain_lr,
                                    eps   = 1e-9,
                                    betas = (0.9, 0.999),
                                    weight_decay = 0)
        
        self.window_size = 5
        self.combine     = False
        self.coeffs      = {"feat_size": 0.5, "feat_ent": 0.5}
        self.criterion = nn.MSELoss().cuda() 


    def construct_feat_weight(self, num_nodes):
        torch.manual_seed(100)
        a = 0
        b = 0.2
        feat_weight = nn.Parameter(torch.empty(num_nodes))
        nn.init.uniform_(feat_weight, a=a, b=b)
        return feat_weight


    def smooth_weights(self, raw_weights, window_size):
        left_pad  = window_size // 2
        right_pad = window_size - left_pad - 1
        window_kernel = torch.ones(1, 1, window_size, device=raw_weights.device) / window_size
        normalized_weights = torch.sigmoid(raw_weights).unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        padded_weights     = F.pad(normalized_weights, (left_pad, right_pad), mode='constant', value=0)
        smoothed_weights   = F.conv1d(padded_weights, window_kernel).squeeze()
        return smoothed_weights.unsqueeze(-1)


    def forward(self, Protein1_feat_Train, Protein2_feat_Train):

        Protein1_feat = self.Protein1_feat * self.smooth_weights(self.Protein1_feat_weight, self.window_size) if Protein1_feat_Train else self.Protein1_feat
        Protein2_feat = self.Protein2_feat * self.smooth_weights(self.Protein2_feat_weight, self.window_size) if Protein2_feat_Train else self.Protein2_feat

        Protein1_edge_weight = None
        Protein2_edge_weight = None
        protein_pair_idx = torch.tensor([0.]).cuda(non_blocking=True)
        
        ypred, _, _ = self.model(protein_pair_idx
                            , Protein1_feat, self.Protein1_edge_index, self.Protein1_batch
                            , Protein2_feat, self.Protein2_edge_index, self.Protein2_batch
                            , Protein1_edge_weight, Protein2_edge_weight)
        return ypred


    def loss(self, explainer_pred, initial_yhat, epoch, Protein1_feat_Train = True, Protein2_feat_Train = True):
        pred_loss                = self.criterion(explainer_pred, initial_yhat)
        feat_size_loss           = 0
        feat_weight_entropy_loss = 0

        if Protein1_feat_Train:
            Protein1_feat_weight = self.smooth_weights(self.Protein1_feat_weight, self.window_size)

            feat_size_loss += self.coeffs["feat_size"] * torch.mean(Protein1_feat_weight)

            Protein1_feat_weight_entropy = - Protein1_feat_weight * torch.log(Protein1_feat_weight + 1e-8) - (1 - Protein1_feat_weight) * torch.log(1 - Protein1_feat_weight + 1e-8)
            feat_weight_entropy_loss += self.coeffs["feat_ent"] * torch.mean(Protein1_feat_weight_entropy)
            
            if (epoch-2) % 250 == 0:
                print(f'Epoch:{epoch-1}', end = " ")
                print(f'High:{sum(Protein1_feat_weight>0.5).item()}|Low:{sum(Protein1_feat_weight<0.5).item()}', end = '\t')
                print(f'Protein1_feat_weight max:{float(Protein1_feat_weight.max().detach().cpu()):.3f}', end=' ')
                print(f'Protein1_feat_weight min:{float(Protein1_feat_weight.min().detach().cpu()):.3f}')

        if Protein2_feat_Train:
            Protein2_feat_weight = self.smooth_weights(self.Protein2_feat_weight, self.window_size)

            feat_size_loss += self.coeffs["feat_size"] * torch.mean(Protein2_feat_weight)

            Protein2_feat_weight_entropy = - Protein2_feat_weight * torch.log(Protein2_feat_weight + 1e-8) - (1 - Protein2_feat_weight) * torch.log(1 - Protein2_feat_weight + 1e-8)
            feat_weight_entropy_loss += self.coeffs["feat_ent"] * torch.mean(Protein2_feat_weight_entropy)
            
            if (epoch-2) % 250 == 0:
                print(f'Epoch:{epoch-1}', end = " ")
                print(f'High:{sum(Protein2_feat_weight>0.5).item()}|Low:{sum(Protein2_feat_weight<0.5).item()}', end = '\t')
                print(f'Protein2_feat_weight max:{float(Protein2_feat_weight.max().detach().cpu()):.3f}', end=' ')
                print(f'Protein2_feat_weight min:{float(Protein2_feat_weight.min().detach().cpu()):.3f}')

        
        All_loss = pred_loss + feat_size_loss + feat_weight_entropy_loss
        
        pred_label = int(torch.max(explainer_pred, dim=1)[1].detach().cpu())
        pred_score = round(float(torch.softmax(explainer_pred, dim=1)[:,1].detach().cpu()), 4)
        
        if (epoch-2) % 250 == 0:
            
            print(f'pred_score:{pred_score:.3f}\tpred_label:{pred_label}', end = " ")
            print(f'ALL_loss:{All_loss.item():.3f}\t', end = " ")
            print(f'pred_loss:{pred_loss.item():.3f}\t', end = " ")
            print(f'feat_size_loss:{feat_size_loss.item():.3f}\t', end = " ")
            print(f'feat_weight_entropy_loss:{feat_weight_entropy_loss.item():.3f}')
        
        
        return All_loss