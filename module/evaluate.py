import os
import gc
import random
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pickle
import datetime
from timm.utils import AverageMeter

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from build.Pair_dataset import Pair_dataset
from build.model_build import build_model
from module.utils import *
from module.explainer_utils import *



def evaluate(config, logger):
    DataLoader_Eval, Datasets_Eval = build_Eval_datasets(config, logger)
    
    model            = build_model(config, logger)
    model_state_Path = os.path.join(config.DATA.Eval.Output_Dir, f'{config.DATA.Eval.Eval_Type}_savedict.pth')
    model_state      = torch.load(model_state_Path, map_location='cpu')
    model.load_state_dict(model_state['model'])
    
    
    if logger != None:
        logger.info(f'Start Evaluating')
        logger.info(f'Evaluating Type:          {config.DATA.Eval.Eval_Type}')

        logger.info(f'Eval model_state_Path:    {model_state_Path}')
        logger.info(f'Eval_Output_Dir:          {config.DATA.Eval.Output_Dir}')
        logger.info(f'Savefile_Name:            {config.DATA.Eval.Savefile_Name}')
        
        logger.info(f'return_attention_weights: {config.MODEL.DeepGNHV.return_attention_weights}')
        # logger.info(f'Global Node:              {config.MODEL.DeepGNHV.Global_Node}')


    if config.Train.Pos_Weight != 1:
        logger.info(f'Specify label weight')
        weight = torch.tensor([1., config.Train.Pos_Weight]).cuda()
        criterion = nn.CrossEntropyLoss(reduction = config.Train.Criterion_Reduction, weight = weight).cuda()
    else:
        logger.info(f'Not specify label weight')
        criterion = nn.CrossEntropyLoss(reduction = config.Train.Criterion_Reduction,).cuda() 
    
    
    if config.MODEL.DeepGNHV.return_attention_weights:
        model.cuda().eval()
        explaination_from_attention(config, logger, model, Datasets_Eval)
    elif config.DATA.Eval.Explain:
        model.cuda()
        explainer_main(config, logger, model, Datasets_Eval, DataLoader_Eval)
    else:
        model.cuda().eval()
        evaluate_one_epoch(config, logger, model, DataLoader_Eval, Datasets_Eval, criterion)

@torch.no_grad()
def evaluate_one_epoch(config, logger, model, DataLoader_Eval, Datasets_Eval, criterion):
    Mission_Start_Time = time.time()
    
    model.eval()
    Batch_time                      = AverageMeter()
    num_steps                       = len(DataLoader_Eval)
    
    Eval_Score_ThisEpoch            = []
    Eval_Label_ThisEpoch            = []
    Eval_Protein_Pair_Name          = []
    All_Samples_Eval_ThisEpoch      = 0
    Correct_Samples_Eval_ThisEpoch  = 0
    Loss_Eval_ThisEpoch             = 0

    end = time.time()
    for batch_idx, (Samples_and_Labels) in enumerate(DataLoader_Eval):
        Protein_Pair_idx, Protein1_length, Protein2_length, Protein1_Embedding_graphs, Protein2_Embedding_graphs, Labels = Samples_and_Labels
        Protein_Pair_name = [Datasets_Eval.Sample_List[id] for id in Protein_Pair_idx]
        Eval_Protein_Pair_Name.extend(Protein_Pair_name)

        Protein1_feat       = Protein1_Embedding_graphs.x.cuda(non_blocking=True)
        Protein2_feat       = Protein2_Embedding_graphs.x.cuda(non_blocking=True)
        Protein1_edge_index = Protein1_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein2_edge_index = Protein2_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein1_batch      = Protein1_Embedding_graphs.batch.cuda(non_blocking=True)
        Protein2_batch      = Protein2_Embedding_graphs.batch.cuda(non_blocking=True)
        
        Samples = (Protein_Pair_idx
                    , Protein1_feat, Protein1_edge_index, Protein1_batch
                    , Protein2_feat, Protein2_edge_index, Protein2_batch
                    , None, None
                    , config.MODEL.DeepGNHV.return_attention_weights)
            
        Labels = Labels.type(torch.float32).reshape(-1,2).cuda(non_blocking=True)
        

        Sigma, Correct, Loss, protein1_edges_attention_weight_list, protein2_edges_attention_weight_list = EvalOnce(model, criterion, Samples, Labels)

        
        Eval_Score_ThisEpoch.extend(Sigma.detach().cpu().numpy())
        Eval_Label_ThisEpoch.extend(Labels[:,1].detach().cpu().numpy())
        
        All_Samples_Eval_ThisEpoch     += Labels.size(0)
        Loss_Eval_ThisEpoch            += Loss.item() * Labels.size(0) if config.Train.Criterion_Reduction == 'mean' else Loss.item()
        
        Correct_Samples_Eval_ThisEpoch += Correct.item()
        
        Batch_time.update(time.time() - end)
        end = time.time()
        
        # Output results every x batches
        if batch_idx % config.Settings.Print_freq == 0:
            Memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = Batch_time.avg * (num_steps - batch_idx)
            logger.info(
                f'Eval:\t[1][{batch_idx + 1}/{num_steps}]\t'
                f'Eta\t{datetime.timedelta(seconds=int(etas))}\t'
                f'Batchtime\t{Batch_time.val:.4f} ({Batch_time.avg:.4f})\t'
                f'Mem\t{Memory:.0f}MB')


    Total_Time = time.time() - Mission_Start_Time
    Total_Time_Str = str(datetime.timedelta(seconds=int(Total_Time)))
    logger.info('Total evaluating time {}'.format(Total_Time_Str))

    Eval_Acc_ThisEpoch  = float(Correct_Samples_Eval_ThisEpoch*100)/All_Samples_Eval_ThisEpoch
    Eval_Loss_ThisEpoch = float(Loss_Eval_ThisEpoch)/All_Samples_Eval_ThisEpoch
    
    Eval_Auroc_Area_ThisEpoch, \
    Eval_Aupr_Area_ThisEpoch, \
            Eval_Fpr_ThisEpoch, \
            Eval_Tpr_ThisEpoch, \
            Eval_Precision_ThisEpoch,\
            Eval_Recall_ThisEpoch = ROC_curve(Eval_Score_ThisEpoch, Eval_Label_ThisEpoch)
            
    logger.info(f'Mission_Name:   {config.Settings.Mission_Name}')
    logger.info(f'Savefile_Name:  {config.DATA.Eval.Savefile_Name}')
    logger.info(f'Eval_Output_Dir:{config.DATA.Eval.Output_Dir}')
    logger.info(f'Test Acc: {Eval_Acc_ThisEpoch:.4f}%\tTest Loss: {Eval_Loss_ThisEpoch:.4f}\tTest AUROC: {Eval_Auroc_Area_ThisEpoch:.4f}\tTest AUPR: {Eval_Aupr_Area_ThisEpoch:.4f}')
    
    Eval_Output_Dict_ThisEpoch = dict(Eval_Score_ThisEpoch           = Eval_Score_ThisEpoch
                                        , Eval_Label_ThisEpoch       = Eval_Label_ThisEpoch
                                        , Eval_Protein_Pair_Name     = Eval_Protein_Pair_Name
                                        , Eval_Acc_ThisEpoch         = Eval_Acc_ThisEpoch
                                        , Eval_Loss_ThisEpoch        = Eval_Loss_ThisEpoch
                                        , Eval_Auroc_Area_ThisEpoch  = Eval_Auroc_Area_ThisEpoch
                                        , Eval_Aupr_Area_ThisEpoch   = Eval_Aupr_Area_ThisEpoch
                                        , Eval_Fpr_ThisEpoch         = Eval_Fpr_ThisEpoch
                                        , Eval_Tpr_ThisEpoch         = Eval_Tpr_ThisEpoch
                                        , Eval_Precision_ThisEpoch   = Eval_Precision_ThisEpoch
                                        , Eval_Recall_ThisEpoch      = Eval_Recall_ThisEpoch
                                        , Eval_Output_Dir            = config.DATA.Eval.Output_Dir
                                        , Mission_Name               = config.Settings.Mission_Name
                                        , Savefile_Name              = config.DATA.Eval.Savefile_Name
                                        , protein1_edges_attention_weight_list = protein1_edges_attention_weight_list
                                        , protein2_edges_attention_weight_list = protein2_edges_attention_weight_list
                                        )
    
    Save_Eval_Output(config.DATA.Eval.Eval_Type, Eval_Output_Dict_ThisEpoch)
    del Eval_Output_Dict_ThisEpoch, Samples, Labels, Sigma, Correct, Loss, Loss_Eval_ThisEpoch, All_Samples_Eval_ThisEpoch, Correct_Samples_Eval_ThisEpoch, Eval_Score_ThisEpoch, Eval_Label_ThisEpoch
    gc.collect()
    torch.cuda.empty_cache() 
                
def save_edge_attention_weight(Protein1_edges_attention_weight, Protein2_edges_attention_weight
                                , Protein1_num_edges, Protein2_num_edges
                                , Protein1_edge_index, Protein2_edge_index
                                , Protein_Pair_name
                                , Eval_Output_Dir, Savefile_Name):
    Protein1_splited_edge_index = torch.split(Protein1_edge_index, Protein1_num_edges, dim = 1)
    Protein2_splited_edge_index = torch.split(Protein2_edge_index, Protein2_num_edges, dim = 1)
    Protein1_spliter_edge_attention_weight = torch.split(Protein1_edges_attention_weight, Protein1_num_edges, dim = 0)
    Protein2_spliter_edge_attention_weight = torch.split(Protein2_edges_attention_weight, Protein2_num_edges, dim = 0)
    Eval_Save_dir = Path(Eval_Output_Dir)/Savefile_Name
    Eval_Save_dir.mkdir(exist_ok=True)
    for i,item in enumerate(Protein_Pair_name):
        Protein1_name, Protein2_name, Label_type = item
        if Label_type == 1:
            Eval_transformerconv_attention_Dict = dict(Protein1_name   = Protein1_name
                                                    , Protein2_name = Protein2_name
                                                    , Protein1_edge_index = Protein1_splited_edge_index[i]
                                                    , Protein2_edge_index = Protein2_splited_edge_index[i]
                                                    , Protein1_edge_attention_weight = Protein1_spliter_edge_attention_weight[i]
                                                    , Protein2_edge_attention_weight = Protein2_spliter_edge_attention_weight[i]
                                                    , label = Label_type
                                                    )
            Eval_Save_Path = (Eval_Save_dir/(Protein1_name + "_" + Protein2_name)).with_suffix(".TransformerConv_attention")
            torch.save(Eval_transformerconv_attention_Dict, Eval_Save_Path)

def EvalOnce(model, criterion, Samples, Labels):
    
    zhat, protein1_edges_attention_weight_list, protein2_edges_attention_weight_list = model.forward(*Samples)
    Loss = criterion(zhat,Labels)
    Sigma   = torch.softmax(zhat, dim=1)[:,1]
    yhat    = torch.max(zhat, dim=1)[1]
    Correct = torch.sum(yhat == Labels[:,1])
    
    return Sigma, Correct, Loss, protein1_edges_attention_weight_list, protein2_edges_attention_weight_list


def Save_Eval_Output(Eval_Type, Eval_Output_Dict_ThisEpoch):

    Eval_Save_Prediction_Path  = Path(Eval_Output_Dict_ThisEpoch['Eval_Output_Dir'])/f'{Eval_Type}Model_{Eval_Output_Dict_ThisEpoch["Savefile_Name"]}_prediction.txt'
    Eval_Save_Savedict_Path    = Path(Eval_Output_Dict_ThisEpoch['Eval_Output_Dir'])/f'{Eval_Type}Model_{Eval_Output_Dict_ThisEpoch["Savefile_Name"]}_savedict.pth'

    torch.save(Eval_Output_Dict_ThisEpoch, Eval_Save_Savedict_Path)
    
    with open(Eval_Save_Prediction_Path, "w") as h:
        h.write(f'human_protein\tvirus_protein\tTrue_label\tProbability_not_interact\tProbability_interact\n')
        for Protein_Pair, Prediction_Score in zip(Eval_Output_Dict_ThisEpoch['Eval_Protein_Pair_Name'], Eval_Output_Dict_ThisEpoch['Eval_Score_ThisEpoch']):
            Protein1_name, Protein2_name, label = Protein_Pair
            h.write(f'{Protein1_name}\t{Protein2_name}\t{label[1]}\t{str(round(1-Prediction_Score, 4))}\t{str(round(Prediction_Score, 4))}\n')


def build_Eval_datasets(config, logger):
    File_Suffix      = config.DATA.Processing.File_Suffix
    # transform        = transforms.Compose([transforms.Resize((256,256),antialias=True)
    #                                     ,transforms.CenterCrop((224,224))])
    Data_Process     = config.DATA.Processing.Data_Process
    transform        = transforms.CenterCrop((224,224))
    Positive_Catalog = Path(config.DATA.Eval.Positive_Catalog)
    Negative_Catalog = Path(config.DATA.Eval.Negative_Catalog)
    Protein1_Dir     = Path(config.DATA.Eval.Protein1_Dir)
    Protein2_Dir     = Path(config.DATA.Eval.Protein2_Dir)
    Numbers_Positive = config.DATA.Eval.Numbers_Positive
    Numbers_Negative = config.DATA.Eval.Numbers_Negative
    Fix_Length       = config.DATA.Processing.Fix_Length
    # Global_Node      = config.MODEL.DeepGNHV.Global_Node
    
    Datasets_Eval = Pair_dataset(Positive_Catalog  = Positive_Catalog
                                , Negative_Catalog = Negative_Catalog
                                , Protein1_Dir     = Protein1_Dir
                                , Protein2_Dir     = Protein2_Dir
                                , logger           = logger
                                , Data_Process     = Data_Process
                                , transform        = transform
                                , Numbers_Positive = Numbers_Positive
                                , Numbers_Negative = Numbers_Negative
                                , File_Suffix      = File_Suffix
                                , Fix_Length       = Fix_Length
                                # , Global_Node      = Global_Node
                                )
    
    DataLoader_Eval = DataLoader(Datasets_Eval
                                    ,batch_size = config.Train.Batch_Size
                                    ,shuffle    = False
                                    ,drop_last  = False
                                    ,pin_memory = True
                                    ,collate_fn = Graph_Collect_Func)
    return DataLoader_Eval, Datasets_Eval


# """
def explaination_from_attention(config, logger, model, Datasets_Eval):
    Eval_Explain_Dir = config.DATA.Eval.Explain_Dir
    Explaination_Protein_Pair = dict()
    DataLoader_Eval_explaination = DataLoader(Datasets_Eval
                                            , batch_size = 1
                                            , shuffle    = False
                                            , drop_last  = False
                                            , pin_memory = True
                                            , collate_fn = Graph_Collect_Func)

    for Samples_and_Labels in DataLoader_Eval_explaination:
        Protein_Pair_idx, Protein1_length, Protein2_length, Protein1_Embedding_graphs, Protein2_Embedding_graphs, Labels = Samples_and_Labels
        Labels              = Labels.type(torch.float32).reshape(-1,2).cuda(non_blocking=True)
        if Labels[1] == 1:
            Protein1_Name, Protein2_Name, Label = [Datasets_Eval.Sample_List[id] for id in Protein_Pair_idx][0]
            # Explaination_Protein_Pair.extend(Protein_Pair_Name[0])
            
            Protein1_feat       = Protein1_Embedding_graphs.x.cuda(non_blocking=True)
            Protein2_feat       = Protein2_Embedding_graphs.x.cuda(non_blocking=True)
            Protein1_edge_index = Protein1_Embedding_graphs.edge_index.cuda(non_blocking=True)
            Protein2_edge_index = Protein2_Embedding_graphs.edge_index.cuda(non_blocking=True)
            Protein1_batch      = Protein1_Embedding_graphs.batch.cuda(non_blocking=True)
            Protein2_batch      = Protein2_Embedding_graphs.batch.cuda(non_blocking=True)
            
            
            Samples = (Protein_Pair_idx
                        , Protein1_feat, Protein1_edge_index, Protein1_batch
                        , Protein2_feat, Protein2_edge_index, Protein2_batch
                        , None, None
                        , True)
            
            protein1_edges_attention_weight_list_numpy = []
            protein2_edges_attention_weight_list_numpy = []
                
            zhat, protein1_edges_attention_weight_list, protein2_edges_attention_weight_list = model.forward(*Samples)

            for item in protein1_edges_attention_weight_list:
                protein1_edges_attention_weight_list_numpy.append(item.cpu().numpy())
            for item in protein2_edges_attention_weight_list:
                protein2_edges_attention_weight_list_numpy.append(item.cpu().numpy())
            
            Explaination_Protein_Pair[f'{Protein1_Name}_{Protein2_Name}'] = dict(protein1_edges_attention_weight_list_numpy  = protein1_edges_attention_weight_list_numpy
                                                                                , protein2_edges_attention_weight_list_numpy = protein2_edges_attention_weight_list_numpy
                                                                                , Protein1_edge_index = Protein1_edge_index.cpu().numpy()
                                                                                , Protein2_edge_index = Protein2_edge_index.cpu().numpy()
                                                                                , Protein1_length = int(Protein1_length.cpu()), Protein2_length = int(Protein2_length.cpu())
                                                                                )
    Eval_Explain_File_path = os.path.join(Eval_Explain_Dir, f'{config.MODEL.DeepGNHV.Graph_Type}_explaination.pth')
    with open(Eval_Explain_File_path, 'wb') as h:
        pickle.dump(Explaination_Protein_Pair, h)
    print(f'Successfully treated {Eval_Explain_File_path}')
 
# """