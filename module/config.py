import os
import yaml
from pathlib import Path
from yacs.config import CfgNode as CN

# Define config container
_C = CN()

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
_C.Settings = CN()
_C.Settings.GPU_Id       = 0
_C.Settings.PIN_MEMORY   = True
_C.Settings.Print_freq   = 200
_C.Settings.Output_Dir   = ''
_C.Settings.Mission_Name = 'Mission_Name'
# -----------------------------------------------------------------------------
# Data 
# -----------------------------------------------------------------------------
_C.DATA = CN()

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------
_C.DATA.Processing = CN()
_C.DATA.Processing.Data_Process = None
_C.DATA.Processing.Data_Concat  = False
_C.DATA.Processing.File_Suffix  = None
_C.DATA.Processing.Fix_Length   = 2000


_C.DATA.Train = CN()
_C.DATA.Train.Positive_Catalog = None
_C.DATA.Train.Negative_Catalog = None
_C.DATA.Train.Numbers_Positive = None
_C.DATA.Train.Numbers_Negative = None
_C.DATA.Train.Protein1_Dir     = None
_C.DATA.Train.Protein2_Dir     = None

_C.DATA.Test = CN()
_C.DATA.Test.Positive_Catalog = None
_C.DATA.Test.Negative_Catalog = None
_C.DATA.Test.Numbers_Positive = None
_C.DATA.Test.Numbers_Negative = None
_C.DATA.Test.Protein1_Dir     = None
_C.DATA.Test.Protein2_Dir     = None

_C.DATA.Eval = CN()
_C.DATA.Eval.Bool             = False
_C.DATA.Eval.Model_Type       = None
_C.DATA.Eval.Output_Dir       = None
_C.DATA.Eval.Positive_Catalog = None
_C.DATA.Eval.Negative_Catalog = None
_C.DATA.Eval.Numbers_Positive = None
_C.DATA.Eval.Numbers_Negative = None
_C.DATA.Eval.Protein1_Dir     = None
_C.DATA.Eval.Protein2_Dir     = None

_C.DATA.Eval.Savefile_Name    = None
_C.DATA.Eval.Explain_Dir      = None
_C.DATA.Eval.Explain_LR       = None
_C.DATA.Eval.Explain_Epochs   = None




# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.Model_Name      = 'DeepGNHV'


# DeepGNHV
_C.MODEL.DeepGNHV = CN()
_C.MODEL.DeepGNHV.Dropout_Rates     = 0.5
_C.MODEL.DeepGNHV.Dimensions        = [1024, 720, 360]
_C.MODEL.DeepGNHV.Concat            = False
_C.MODEL.DeepGNHV.num_GNN_Layers    = 2
_C.MODEL.DeepGNHV.num_MLP_Layers    = 1
_C.MODEL.DeepGNHV.Pool_Type         = 'max'
_C.MODEL.DeepGNHV.Attention_Heads   = [12, 12]
_C.MODEL.DeepGNHV.return_attention_weights   = False

_C.MODEL.DeepGNHV.GNN_Type                  = 'GATConv'          # 'GCNConv', 'GATConv', 'TransformerConv', 'GraphSAGE', 'GIN'
_C.MODEL.DeepGNHV.Graph_Type                = 'Undirectedgraph'  # 'Semidirectedgraph', 'Undirectedgraph', 'Surfacegraph'
_C.MODEL.DeepGNHV.Graph_RSA_Threshold       = '0.3'      # '0.2 0.25 0.3 0.35 0.4'
_C.MODEL.DeepGNHV.Graph_Distance_Threshold  = '8'        # '4,6,8,10,12'



# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.Train = CN()

_C.Train.Max_Epochs  = 100
_C.Train.Batch_Size  = 64
_C.Train.Lr          = 0.0003
_C.Train.Min_Lr      = 0.00005
_C.Train.Pos_Weight  = 1
_C.Train.Use_scheduler       = True
_C.Train.Criterion_Reduction = 'sum'


_C.Train.Optimizer = CN()
_C.Train.Optimizer.Name         = 'adamw'
_C.Train.Optimizer.Eps          = 1e-9
_C.Train.Optimizer.Betas        = (0.9, 0.999)
_C.Train.Optimizer.Momentum     = 0.9
_C.Train.Optimizer.Weight_Decay = 0.0


# -----------------------------------------------------------------------------
# Seed
# -----------------------------------------------------------------------------
_C.Seed = 10000

# -----------------------------------------------------------------------------
# Early stopping
# -----------------------------------------------------------------------------
_C.Early_Stop = CN()
_C.Early_Stop.Patience = 6
_C.Early_Stop.Tol = 0.0001


# -----------------------------------------------------------------------------
# Updating
# -----------------------------------------------------------------------------
def update_config(config, args):
    config.defrost()
    if args.GPU_Id:
        config.Settings.GPU_Id = int(args.GPU_Id)
    if args.Mission_Name:
        config.Settings.Mission_Name = args.Mission_Name
    if args.Output_Dir:
        config.Settings.Output_Dir = os.path.join(args.Output_Dir, args.Mission_Name)
        
    if args.Model_Name == "DeepGNHV":
        config.MODEL.Model_Name = args.Model_Name
        
        if args.GNN_Type:
            config.MODEL.DeepGNHV.GNN_Type   = args.GNN_Type
        if args.Graph_Type:
            config.MODEL.DeepGNHV.Graph_Type = args.Graph_Type
        if args.Graph_RSA_Threshold:
            config.MODEL.DeepGNHV.Graph_RSA_Threshold = args.Graph_RSA_Threshold
        if args.Graph_Distance_Threshold:
            config.MODEL.DeepGNHV.Graph_Distance_Threshold = args.Graph_Distance_Threshold
        if args.Dropout_Rates:
            config.MODEL.DeepGNHV.Dropout_Rates     = float(args.Dropout_Rates)

        if args.Attention_Heads:
            config.MODEL.DeepGNHV.Attention_Heads   = [int(x.strip('')) for x in args.Attention_Heads.split(',') if x != '']
        if args.Dimensions:
            config.MODEL.DeepGNHV.Dimensions        = [int(x.strip('')) for x in args.Dimensions.split(',') if x != '']
        if args.Concat:
            config.MODEL.DeepGNHV.Concat            = True if args.Concat == "True" else False
        if args.Pool_Type:
            config.MODEL.DeepGNHV.Pool_Type         = 'max' if args.Pool_Type == "max" else 'mean'
        # if args.Global_Node:
        #     config.MODEL.DeepGNHV.Global_Node       = True if args.Global_Node == "True" else False
        if args.return_attention_weights:
            config.MODEL.DeepGNHV.return_attention_weights   = True if args.return_attention_weights == "True" else False
        if args.num_GNN_Layers:
            config.MODEL.DeepGNHV.num_GNN_Layers = int(args.num_GNN_Layers)
        if args.num_MLP_Layers:
            config.MODEL.DeepGNHV.num_MLP_Layers = int(args.num_MLP_Layers)
        if args.MLPDecrease:
            config.MODEL.DeepGNHV.MLPDecrease    = float(args.MLPDecrease)
        if args.GNNBN:
            config.MODEL.DeepGNHV.GNNBN          = True if args.GNNBN == "True" else False
        if args.GNNDP:
            config.MODEL.DeepGNHV.GNNDP          = True if args.GNNDP == "True" else False
        if args.MLPBN:
            config.MODEL.DeepGNHV.MLPBN          = True if args.MLPBN == "True" else False
        if args.MLPDP:
            config.MODEL.DeepGNHV.MLPDP          = True if args.MLPDP == "True" else False




    if args.File_Suffix:
        config.DATA.Processing.File_Suffix  = args.File_Suffix
    if args.Data_Process:
        config.DATA.Processing.Data_Process = args.Data_Process
    if args.Data_Concat:
        config.DATA.Processing.Data_Concat  = args.Data_Concat
    if args.Fix_Length:
        config.DATA.Processing.Fix_Length   = args.Fix_Length
        
    if args.Lr:
        config.Train.Lr = float(args.Lr)
    if args.Max_Epochs:
        config.Train.Max_Epochs = args.Max_Epochs
    if args.Batch_Size:
        config.Train.Batch_Size = args.Batch_Size
    if args.Criterion_Reduction:
        config.Train.Criterion_Reduction = args.Criterion_Reduction
    if args.Use_scheduler:
        config.Train.Use_scheduler = True if args.Use_scheduler == "True" else False
    if args.Pos_Weight:
        config.Train.Pos_Weight = float(args.Pos_Weight)
        
    if args.Weight_Decay:
        config.Train.Optimizer.Weight_Decay = float(args.Weight_Decay)
    if args.Optimizer:
        config.Train.Optimizer.Name         = args.Optimizer
     
    if args.Train_Numbers_Positive:
        config.DATA.Train.Numbers_Positive  = args.Train_Numbers_Positive
    if args.Train_Numbers_Negative:
        config.DATA.Train.Numbers_Negative  = args.Train_Numbers_Negative
    if args.Test_Numbers_Positive:
        config.DATA.Test.Numbers_Positive   = args.Test_Numbers_Positive
    if args.Test_Numbers_Negative:
        config.DATA.Test.Numbers_Negative   = args.Test_Numbers_Negative
    
    if args.Train_Positive_Catalog:
        config.DATA.Train.Positive_Catalog = args.Train_Positive_Catalog
    if args.Train_Negative_Catalog:
        config.DATA.Train.Negative_Catalog = args.Train_Negative_Catalog
    if args.Test_Positive_Catalog:
        config.DATA.Test.Positive_Catalog  = args.Test_Positive_Catalog
    if args.Test_Negative_Catalog:
        config.DATA.Test.Negative_Catalog  = args.Test_Negative_Catalog

    if args.Train_Protein1_Dir:
        config.DATA.Train.Protein1_Dir = args.Train_Protein1_Dir
    if args.Train_Protein2_Dir:
        config.DATA.Train.Protein2_Dir = args.Train_Protein2_Dir
    if args.Test_Protein1_Dir:
        config.DATA.Test.Protein1_Dir  = args.Test_Protein1_Dir
    if args.Test_Protein2_Dir:
        config.DATA.Test.Protein2_Dir  = args.Test_Protein2_Dir
        
    
    if args.Eval:
        config.DATA.Eval.Bool             = True
        config.DATA.Eval.Eval_Type        = args.Eval_Type
        config.DATA.Eval.Positive_Catalog = args.Eval_Positive_Catalog
        config.DATA.Eval.Negative_Catalog = args.Eval_Negative_Catalog
        config.DATA.Eval.Numbers_Positive = args.Eval_Numbers_Positive
        config.DATA.Eval.Numbers_Negative = args.Eval_Numbers_Negative
        config.DATA.Eval.Protein1_Dir     = args.Eval_Protein1_Dir
        config.DATA.Eval.Protein2_Dir     = args.Eval_Protein2_Dir
        config.DATA.Eval.Output_Dir       = os.path.join(args.Eval_Output_Dir, args.Mission_Name)
        config.DATA.Eval.Savefile_Name    = args.Eval_Savefile_Name
        
    if args.Explain:
        config.DATA.Eval.Explain          = True if args.Explain == "True" else False
    if args.Eval_Explain_Dir:
        config.DATA.Eval.Explain_Dir      = args.Eval_Explain_Dir
    if args.Eval_Explain_LR:
        config.DATA.Eval.Explain_LR       = float(args.Eval_Explain_LR)
    if args.Eval_Explain_Epochs:
        config.DATA.Eval.Explain_Epochs   = int(args.Eval_Explain_Epochs)

    config.freeze()


def get_config(args):
    config = _C.clone()
    update_config(config, args)

    return config