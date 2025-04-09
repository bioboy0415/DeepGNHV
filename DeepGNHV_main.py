import os
script_path = os.path.abspath(__file__)
script_dir  = os.path.dirname(script_path)

import sys
sys.path.append(script_dir)

import random
import time
import argparse
import datetime
import gc
import numpy as np

import torch
torch.set_num_threads(35)
import torch.backends.cudnn as cudnn
from torch import nn

from timm.utils import AverageMeter

from build.model_build import build_model
from build.dataset_build import build_loader

from module.config import get_config
import torch.optim.lr_scheduler as lr_scheduler
from module.optimizer import build_optimizer
from module.logger import create_logger
from module.utils import *
from module.evaluate import evaluate


def parse_option():
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--Output_Dir',   type = str, default  = 'Output', help = 'Path for output directory.')
    parser.add_argument('--Mission_Name', type = str, required = True,     help = 'User-specified mission name.')
    parser.add_argument('--File_Suffix' , type = str, required = True,     help = 'Protein graph file suffix.')
    
    parser.add_argument('--Model_Name', type = str, required = True, choices = ['DeepGNHV'] , help = 'Specify the model name.')
    parser.add_argument('--GNN_Type',   type = str, required = True, choices = ['GCNConv', 'GATConv', 'TransformerConv', 'GraphSAGE', 'GIN'], help = 'Specify the graph neural network type.')
    parser.add_argument('--Graph_Type', type = str, required = True, choices = ['Undirectedgraph', 'Semidirectedgraph', 'Surfacegraph'],      help = 'Specify the method of information flow.')
    parser.add_argument('--Graph_RSA_Threshold',      type = str, default = '0.3', help = 'Specify the relative solvent accessibility threshold for surface amino acid residues if needed.')
    parser.add_argument('--Graph_Distance_Threshold', type = str, default = '8',   help = 'Specify the threshold for constructing the adjacency matrix in the graph neural network.')

    parser.add_argument('--Optimizer',       type = str, default = 'adamw',        help = 'Specify the optimizer type.')
    parser.add_argument('--Dropout_Rates',   type = str, default = '0.3',          help = 'Specify the dropout rates.')
    parser.add_argument('--Weight_Decay',    type = str, default = '0.0',          help = 'Specify weight decay to prevent overfitting.')
    parser.add_argument('--Attention_Heads', type = str, default = '12,12',        help = 'Specify the attention heads if needed.')
    parser.add_argument('--Dimensions',      type = str, default = '1024,720,360', help = 'Specify the variations in dimensions.')
    
    parser.add_argument('--Concat',          type = str, default = 'False',        help = 'Whether to concatenate embedding vectors.')
    # parser.add_argument('--Global_Node',     type = str, default = 'True' )
    parser.add_argument('--num_GNN_Layers',  type = str, default = '2',            help = 'Specify the number of graph neural layers.')
    parser.add_argument('--num_MLP_Layers',  type = str, default = '1',            help = 'Specify the number of linear layers.')
    parser.add_argument('--MLPDecrease',     type = str, default = '2',            help = 'Specify the magnitude of dimension reduction in the linear layers.')
    parser.add_argument('--GNNBN',           type = str, default = 'True',         help = 'Specify whether to use batch normalization on GNNLayers.')
    parser.add_argument('--GNNDP',           type = str, default = 'False',        help = 'Specify whether to use dropout on GNNLayers.')
    parser.add_argument('--MLPBN',           type = str, default = 'False',        help = 'Specify whether to use batch normalization on MLPLayers.')
    parser.add_argument('--MLPDP',           type = str, default = 'True',         help = 'Specify whether to use dropout on MLPLayers.')
    parser.add_argument('--Pool_Type',       type = str, default = 'max',  choices = ['max', 'mean'],  help = 'Specify the type of global pooling.')
    parser.add_argument('--return_attention_weights', type = str, default = 'False',  help = 'Specify whether to return attention list.')
    
    parser.add_argument('--Data_Process', type = str, default = 'None', choices = ['transform', 'fix_length', 'None'], help = 'Specify the method of transforming the raw data.')
    parser.add_argument('--Fix_Length',   type = int, default = 2000) 
    parser.add_argument('--Data_Concat',  action='store_true' )
    
    
    parser.add_argument('--Train_Numbers_Positive', type = int, help = 'Numbers of positive samples used in the training datasets.')
    parser.add_argument('--Train_Numbers_Negative', type = int, help = 'Numbers of negative samples used in the training datasets.')
    parser.add_argument('--Test_Numbers_Positive',  type = int, help = 'Numbers of positive samples used in the testing datasets.')
    parser.add_argument('--Test_Numbers_Negative',  type = int, help = 'Numbers of negative samples used in the testing datasets.')
    
    parser.add_argument('--Train_Positive_Catalog', type = str, help = 'The catalog for positive training samples.')
    parser.add_argument('--Train_Negative_Catalog', type = str, help = 'The catalog for negative training samples.')
    parser.add_argument('--Test_Positive_Catalog',  type = str, help = 'The catalog for positive testing samples.')
    parser.add_argument('--Test_Negative_Catalog',  type = str, help = 'The catalog for negative testing samples.')
    
    parser.add_argument('--Train_Protein1_Dir', type = str, help = 'The directory for protein1 monomer in the training datasets.')
    parser.add_argument('--Train_Protein2_Dir', type = str, help = 'The directory for protein2 monomer in the training datasets.')
    parser.add_argument('--Test_Protein1_Dir',  type = str, help = 'The directory for protein1 monomer in the testing datasets.')
    parser.add_argument('--Test_Protein2_Dir',  type = str, help = 'The directory for protein2 monomer in the testing datasets.')
    
    parser.add_argument('--GPU_Id', type = str, default = 0,  help = 'Which gpu is used for computations.')
    
    parser.add_argument('--Lr',          type = float, default = 0.0003,  help = 'Learning rates for training.')
    parser.add_argument('--Max_Epochs',  type = int,   default = 100,     help = 'Max epochs for training.')
    parser.add_argument('--Batch_Size',  type = int,   default = 64,      help = 'Batch size for single gpu.')
    parser.add_argument('--Criterion_Reduction',  type = str, default = 'sum',  choices = ['sum', 'mean'], help = 'Specify the Criterion Reduction.')
    parser.add_argument('--Use_scheduler',  type = str, default = 'True',  choices = ['True', 'False'], help = 'Specify whether to use lr scheduler.')
    parser.add_argument('--Pos_Weight',     type = str, default = '1' ,    help = 'Specify the weight of positive samples.')
    
    # Eval arguments
    parser.add_argument('--Eval',            default = False, help = 'Enter evaluating mode.')
    parser.add_argument('--Eval_Type',       choices = ['Highest', 'Final'], help = 'Choose trained prediction model type.')
    parser.add_argument('--Eval_Output_Dir', type = str, default = 'Output', help = 'Path of output directory.')
    
    parser.add_argument('--Eval_Numbers_Positive' , type = int, help = 'Numbers of positive samples used in the evaluating datasets.')
    parser.add_argument('--Eval_Numbers_Negative' , type = int, help = 'Numbers of negative samples used in the evaluating datasets.')
    parser.add_argument('--Eval_Positive_Catalog' , type = str, help = 'The catalog for positive evaluating samples.')
    parser.add_argument('--Eval_Negative_Catalog' , type = str, help = 'The catalog for negative evaluating samples.')
 
    parser.add_argument('--Eval_Protein1_Dir',  type = str, help = 'The directory for protein1 monomer in the evaluating datasets.')
    parser.add_argument('--Eval_Protein2_Dir',  type = str, help = 'The directory for protein2 monomer in the evaluating datasets.')
    parser.add_argument('--Eval_Savefile_Name', type = str, default = 'Eval_Savefile_Name', help = 'The file name for storing prediction results.')
    
    parser.add_argument('--Explain',            type = str, default = "False", help = 'Enter explainer mode.')
    parser.add_argument('--Eval_Explain_Dir',   type = str, help = 'The file path for model explaination analysis.')
    parser.add_argument('--Eval_Explain_LR',    type = str, help = 'The explain learning rates for explainer.')
    parser.add_argument('--Eval_Explain_Epochs',type = str, help = 'The explain epochs for explainer.')
    args, unparsed = parser.parse_known_args()
    config = get_config(args)

    return args, config

def main(config):
    logger.info(f'Mission_Name:     {config.Settings.Mission_Name}')
    logger.info(f'Using GPU:        {config.Settings.GPU_Id}')
    logger.info(f'Creating model:   {config.MODEL.Model_Name}')

    
    if config.DATA.Eval.Bool   == True:
        logger.info(f'Start Evaluating')
        evaluate(config, logger)
        return None
    
    logger.info(f'------------------------------------------------------------------------------------')
    
    
    logger.info(f'Start Training')
    logger.info(f'Max_Epochs:          {config.Train.Max_Epochs}')
    logger.info(f'Pos_Weight:          {config.Train.Pos_Weight}')
    logger.info(f'Batch_Size:          {config.Train.Batch_Size}')
    logger.info(f'Use_scheduler:       {config.Train.Use_scheduler}')
    logger.info(f'Criterion_Reduction: {config.Train.Criterion_Reduction}')


    logger.info(f'------------------------------------------------------------------------------------')
    

    DataLoader_Train, DataLoader_Test, Datasets_Train, Datasets_Test = build_loader(config, logger)
    
    model = build_model(config, logger)
    model.cuda()

    optimizer = build_optimizer(config, model, logger)
    if config.Train.Use_scheduler:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=config.Train.Min_Lr, verbose=True)
    
    num_params = sum(params.numel() for params in model.parameters() if params.requires_grad)
    logger.info(f'Numbers of Params: {num_params}')
    
    if config.Train.Pos_Weight != 1:
        logger.info(f'Specify label weight')
        weight = torch.tensor([1., config.Train.Pos_Weight]).cuda()
        criterion = nn.CrossEntropyLoss(reduction = config.Train.Criterion_Reduction, weight = weight).cuda()
    else:
        logger.info(f'Not specify label weight')
        criterion = nn.CrossEntropyLoss(reduction = config.Train.Criterion_Reduction).cuda() 
    
    ALL_Information_Dict = dict(Train_Auroc_PerEpoch   = [], Train_Aupr_PerEpoch  = [], Train_Acc_PerEpoch   = [], Train_Loss_PerEpoch  = []
                                , Test_Auroc_PerEpoch  = [], Test_Aupr_PerEpoch   = [], Test_Acc_PerEpoch    = [], Test_Loss_PerEpoch   = [])
    
    Early_Stopping = EarlyStopping(logger     = logger
                                   , Patience = config.Early_Stop.Patience
                                   , Tol      = config.Early_Stop.Tol)
    Highest_Aupr = None
    Mission_Start_Time = time.time()

    for Epoch in range(config.Train.Max_Epochs):
        This_Epoch_Start_Time = time.time() 
        
        Train_Output_Dict_ThisEpoch = Train_one_Epoch(config, model, criterion, DataLoader_Train, Datasets_Train, optimizer, Epoch)

        ALL_Information_Dict['Train_Acc_PerEpoch'].append(Train_Output_Dict_ThisEpoch['Train_Acc_ThisEpoch'])
        ALL_Information_Dict['Train_Loss_PerEpoch'].append(Train_Output_Dict_ThisEpoch['Train_Loss_ThisEpoch'])
        ALL_Information_Dict['Train_Auroc_PerEpoch'].append(Train_Output_Dict_ThisEpoch['Train_Auroc_Area_ThisEpoch'])
        ALL_Information_Dict['Train_Aupr_PerEpoch'].append(Train_Output_Dict_ThisEpoch['Train_Aupr_Area_ThisEpoch'])


        Test_Output_Dict_ThisEpoch = Test_one_Epoch(config, model, criterion, DataLoader_Test, Datasets_Test, Epoch)
        
        ALL_Information_Dict['Test_Acc_PerEpoch'].append(Test_Output_Dict_ThisEpoch['Test_Acc_ThisEpoch'])
        ALL_Information_Dict['Test_Loss_PerEpoch'].append(Test_Output_Dict_ThisEpoch['Test_Loss_ThisEpoch'])
        ALL_Information_Dict['Test_Auroc_PerEpoch'].append(Test_Output_Dict_ThisEpoch['Test_Auroc_Area_ThisEpoch'])
        ALL_Information_Dict['Test_Aupr_PerEpoch'].append(Test_Output_Dict_ThisEpoch['Test_Aupr_Area_ThisEpoch'])
        
        This_Epoch_Cost_Time     = time.time() - This_Epoch_Start_Time
        This_Epoch_Cost_Time_str = str(datetime.timedelta(seconds=int(This_Epoch_Cost_Time)))
        
        
        
        logger.info(f'Epoch:{Epoch+1}\tHighest Aupr:{Highest_Aupr or Test_Output_Dict_ThisEpoch["Test_Aupr_Area_ThisEpoch"]:.4f}')
        logger.info(f'This Epoch takes:{This_Epoch_Cost_Time_str}')
        
        
        logger.info(f'Train Acc: {Train_Output_Dict_ThisEpoch["Train_Acc_ThisEpoch"]:.4f}%\tTrain Loss: {Train_Output_Dict_ThisEpoch["Train_Loss_ThisEpoch"]:.4f}\tTrain AUROC: {Train_Output_Dict_ThisEpoch["Train_Auroc_Area_ThisEpoch"]:.4f}\tTrain AUPR:  {Train_Output_Dict_ThisEpoch["Train_Aupr_Area_ThisEpoch"]:.4f}')
        
        logger.info(f'Test Acc:  {Test_Output_Dict_ThisEpoch["Test_Acc_ThisEpoch"]:.4f}%\tTest Loss:  {Test_Output_Dict_ThisEpoch["Test_Loss_ThisEpoch"]:.4f}\tTest AUROC:  {Test_Output_Dict_ThisEpoch["Test_Auroc_Area_ThisEpoch"]:.4f}\tTest AUPR:   {Test_Output_Dict_ThisEpoch["Test_Aupr_Area_ThisEpoch"]:.4f}')
        
        if Highest_Aupr == None:
            Highest_Aupr = Test_Output_Dict_ThisEpoch["Test_Aupr_Area_ThisEpoch"]
        elif Highest_Aupr < Test_Output_Dict_ThisEpoch["Test_Aupr_Area_ThisEpoch"]:
            Highest_Aupr = Test_Output_Dict_ThisEpoch["Test_Aupr_Area_ThisEpoch"]
            Save_Checkpoint("Highest", config, Epoch, model, optimizer, logger, Train_Output_Dict_ThisEpoch, Test_Output_Dict_ThisEpoch, ALL_Information_Dict)
            logger.info('Weight Saved')

        Early_Stop, Counter = Early_Stopping(Test_Output_Dict_ThisEpoch["Test_Aupr_Area_ThisEpoch"])

        if Early_Stop == True:
            logger.info('Early stop !!!')
            break

        if config.Train.Use_scheduler:
            scheduler.step(Test_Output_Dict_ThisEpoch["Test_Aupr_Area_ThisEpoch"])

    Total_Time = time.time() - Mission_Start_Time
    Total_Time_Str = str(datetime.timedelta(seconds=int(Total_Time)))
    logger.info('Total training time {}'.format(Total_Time_Str))
    
    Save_Checkpoint("Final", config, Epoch, model, optimizer, logger, Train_Output_Dict_ThisEpoch, Test_Output_Dict_ThisEpoch, ALL_Information_Dict)
    
    del Train_Output_Dict_ThisEpoch, Test_Output_Dict_ThisEpoch, ALL_Information_Dict
    gc.collect()
    torch.cuda.empty_cache()


def Train_one_Epoch(config, model, criterion, DataLoader_Train, Datasets_Train, optimizer, Epoch):
    model.train()
    optimizer.zero_grad()
    num_steps = len(DataLoader_Train)  # Get the number of batches
    
    Batch_time = AverageMeter()
    Train_Score_ThisEpoch   = []
    Train_Label_ThisEpoch   = []
    Train_Protein_Pair_Name = []
    All_Samples_Train_ThisEpoch     = 0
    Correct_Samples_Train_ThisEpoch = 0
    Loss_Train_ThisEpoch            = 0

    start = time.time()
    end   = time.time()
    
    for batch_idx, (Samples_and_Labels) in enumerate(DataLoader_Train):
        Protein_Pair_idx, Protein1_length, Protein2_length, Protein1_Embedding_graphs, Protein2_Embedding_graphs, Labels = Samples_and_Labels
        
        Protein_Pair_Name = [Datasets_Train.Sample_List[id] for id in Protein_Pair_idx]
        # Protein_Pair_Name = Datasets_Train.Sample_List[Protein_Pair_idx] 
        Train_Protein_Pair_Name.extend(Protein_Pair_Name)
        
        Protein1_feat       = Protein1_Embedding_graphs.x.cuda(non_blocking=True)
        Protein2_feat       = Protein2_Embedding_graphs.x.cuda(non_blocking=True)
        Protein1_edge_index = Protein1_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein2_edge_index = Protein2_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein1_batch      = Protein1_Embedding_graphs.batch.cuda(non_blocking=True)
        Protein2_batch      = Protein2_Embedding_graphs.batch.cuda(non_blocking=True)
        Labels              = Labels.type(torch.float32).reshape(-1,2).cuda(non_blocking=True)
        Samples             = (Protein_Pair_idx
                                , Protein1_feat, Protein1_edge_index, Protein1_batch
                                , Protein2_feat, Protein2_edge_index, Protein2_batch
                                , None, None
                                , config.MODEL.DeepGNHV.return_attention_weights)
        
        Sigma, Correct, Loss, protein1_attention_list, protein2_attention_list = IterOnce(model, criterion, optimizer, Samples, Labels)
        torch.cuda.synchronize()
        
        Train_Score_ThisEpoch.extend(Sigma.detach().cpu().numpy())
        Train_Label_ThisEpoch.extend(Labels[:,1].detach().cpu().numpy())
        
        All_Samples_Train_ThisEpoch     += Labels.size(0)
        Loss_Train_ThisEpoch            += Loss.item() * Labels.size(0) if config.Train.Criterion_Reduction == 'mean' else Loss.item()
        Correct_Samples_Train_ThisEpoch += Correct.item()

        Batch_time.update(time.time() - end)
        end = time.time()

        # Output results every x batches
        if batch_idx % config.Settings.Print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            Memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = Batch_time.avg * (num_steps - batch_idx)
            logger.info(
                f'Train:\t[{Epoch+1}/{config.Train.Max_Epochs}][{batch_idx + 1}/{num_steps}]\t'
                f'Eta\t{datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'Batchtime\t{Batch_time.val:.4f} ({Batch_time.avg:.4f})\t'
                f'Mem\t{Memory:.0f}MB')
            
    Train_Acc_ThisEpoch  = float(Correct_Samples_Train_ThisEpoch*100)/All_Samples_Train_ThisEpoch
    Train_Loss_ThisEpoch = float(Loss_Train_ThisEpoch)/All_Samples_Train_ThisEpoch
    
    Train_Auroc_Area_ThisEpoch, \
    Train_Aupr_Area_ThisEpoch, \
            Train_Fpr_ThisEpoch, \
            Train_Tpr_ThisEpoch, \
            Train_Precision_ThisEpoch,\
            Train_Recall_ThisEpoch = ROC_curve(Train_Score_ThisEpoch, Train_Label_ThisEpoch)
    
    Epoch_time = time.time() - start
    logger.info(f'Epoch:{Epoch+1}\tTraining takes {datetime.timedelta(seconds=int(Epoch_time))}')
    
    Train_Output_Dict_ThisEpoch = dict(Train_Score_ThisEpoch         = Train_Score_ThisEpoch 
                                        , Train_Label_ThisEpoch      = Train_Label_ThisEpoch
                                        , Train_Protein_Pair_Name    = Train_Protein_Pair_Name
                                        , Train_Acc_ThisEpoch        = Train_Acc_ThisEpoch
                                        , Train_Loss_ThisEpoch       = Train_Loss_ThisEpoch
                                        , Train_Auroc_Area_ThisEpoch = Train_Auroc_Area_ThisEpoch
                                        , Train_Aupr_Area_ThisEpoch  = Train_Aupr_Area_ThisEpoch
                                        , Train_Fpr_ThisEpoch        = Train_Fpr_ThisEpoch
                                        , Train_Tpr_ThisEpoch        = Train_Tpr_ThisEpoch
                                        , Train_Precision_ThisEpoch  = Train_Precision_ThisEpoch
                                        , Train_Recall_ThisEpoch     = Train_Recall_ThisEpoch
                                        )
    return Train_Output_Dict_ThisEpoch
           

@torch.no_grad()
def Test_one_Epoch(config, model, criterion, DataLoader_Test, Datasets_Test, Epoch):
    model.eval()

    Batch_time = AverageMeter()
    Test_Score_ThisEpoch   = []
    Test_Label_ThisEpoch   = []
    Test_Protein_Pair_Name = []
    All_Samples_Test_ThisEpoch      = 0
    Correct_Samples_test_ThisEpoch  = 0
    Loss_test_ThisEpoch             = 0
    end = time.time()
    
    for batch_idx, (Samples_and_Labels) in enumerate(DataLoader_Test):
        Protein_Pair_idx, Protein1_length, Protein2_length, Protein1_Embedding_graphs, Protein2_Embedding_graphs, Labels = Samples_and_Labels
        # Protein_Pair_Name = Datasets_Test.Sample_List[Protein_Pair_idx]
        Protein_Pair_Name = [Datasets_Test.Sample_List[id] for id in Protein_Pair_idx]
        Test_Protein_Pair_Name.extend(Protein_Pair_Name)

        Protein1_feat       = Protein1_Embedding_graphs.x.cuda(non_blocking=True)
        Protein2_feat       = Protein2_Embedding_graphs.x.cuda(non_blocking=True)
        Protein1_edge_index = Protein1_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein2_edge_index = Protein2_Embedding_graphs.edge_index.cuda(non_blocking=True)
        Protein1_batch      = Protein1_Embedding_graphs.batch.cuda(non_blocking=True)
        Protein2_batch      = Protein2_Embedding_graphs.batch.cuda(non_blocking=True)
        Labels = Labels.type(torch.float32).reshape(-1,2).cuda(non_blocking=True)
        
        Samples = (Protein_Pair_idx
                    , Protein1_feat, Protein1_edge_index, Protein1_batch
                    , Protein2_feat, Protein2_edge_index, Protein2_batch
                    , None, None
                    , config.MODEL.DeepGNHV.return_attention_weights)
            
        
        
        Sigma, Correct, Loss, protein1_attention_list, protein2_attention_list = TestOnce(model, criterion, Samples, Labels)
            
        Test_Score_ThisEpoch.extend(Sigma.detach().cpu().numpy())
        Test_Label_ThisEpoch.extend(Labels[:,1].detach().cpu().numpy())

        All_Samples_Test_ThisEpoch     += Labels.size(0)
        Loss_test_ThisEpoch            += Loss.item() * Labels.size(0) if config.Train.Criterion_Reduction == 'mean' else Loss.item()
        
        Correct_Samples_test_ThisEpoch += Correct.item()
        
        Batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % config.Settings.Print_freq == 0:
            Memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test:\t[{batch_idx+1}/{len(DataLoader_Test)}]\t'
                f'Batchtime:\t{Batch_time.val:.4f} ({Batch_time.avg:.4f})\t'
                f'Mem\t{Memory:.0f}MB')
            
    Test_Acc_ThisEpoch  = float(Correct_Samples_test_ThisEpoch*100)/All_Samples_Test_ThisEpoch
    Test_Loss_ThisEpoch = float(Loss_test_ThisEpoch)/All_Samples_Test_ThisEpoch
    
    Test_Auroc_Area_ThisEpoch, \
    Test_Aupr_Area_ThisEpoch, \
            Test_Fpr_ThisEpoch, \
            Test_Tpr_ThisEpoch, \
            Test_Precision_ThisEpoch,\
            Test_Recall_ThisEpoch = ROC_curve(Test_Score_ThisEpoch, Test_Label_ThisEpoch)
    
    Test_Output_Dict_ThisEpoch = dict(Test_Score_ThisEpoch           = Test_Score_ThisEpoch
                                        , Test_Label_ThisEpoch       = Test_Label_ThisEpoch
                                        , Test_Protein_Pair_Name     = Test_Protein_Pair_Name
                                        , Test_Acc_ThisEpoch         = Test_Acc_ThisEpoch
                                        , Test_Loss_ThisEpoch        = Test_Loss_ThisEpoch
                                        , Test_Auroc_Area_ThisEpoch  = Test_Auroc_Area_ThisEpoch
                                        , Test_Aupr_Area_ThisEpoch   = Test_Aupr_Area_ThisEpoch
                                        , Test_Fpr_ThisEpoch         = Test_Fpr_ThisEpoch
                                        , Test_Tpr_ThisEpoch         = Test_Tpr_ThisEpoch
                                        , Test_Precision_ThisEpoch   = Test_Precision_ThisEpoch
                                        , Test_Recall_ThisEpoch      = Test_Recall_ThisEpoch
                                        )
    
    return Test_Output_Dict_ThisEpoch
           
           
           
if __name__ == '__main__':
    
    _, config = parse_option()
    # torch.cuda.set_device(config.Settings.GPU_Id)
    cudnn.benchmark = True
    Random_State_Set(config.Seed)

    output_dir = config.DATA.Eval.Output_Dir if config.DATA.Eval.Bool else config.Settings.Output_Dir
    log_type = 'Eval' if config.DATA.Eval.Bool else 'Train'

    os.makedirs(output_dir, exist_ok=True)
    logger = create_logger(output_dir=output_dir, log_type=log_type, name=f'DeepGNHV')

    main(config)
    