import os
import random
import numpy as np
import torch
from torch_geometric.data import Data, Batch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
def Save_Checkpoint(save_type
                    , config, Epoch, model, optimizer, logger
                    , Train_Output_Dict_ThisEpoch, Test_Output_Dict_ThisEpoch, ALL_Information_Dict):
    """
    Save the model checkpoint and predictions to disk.

    Args:
    - save_type: Type of checkpoint to save ('Highest', 'Final').
    - config:    Configuration settings.
    - Epoch:     Current epoch number.
    - model:     The neural network model.
    - optimizer: The optimizer used for training.
    - logger:    Logger for logging information.
    - Train_Output_Dict_ThisEpoch: Dictionary containing training outputs for the current epoch.
    - Test_Output_Dict_ThisEpoch:  Dictionary containing test outputs for the current epoch.
    - ALL_Information_Dict:        Dictionary containing all relevant information.

    """
    
    save_state = {'model':       model.state_dict()
                  , 'optimizer': optimizer.state_dict()
                  , 'Epoch':     Epoch+1
                  , 'config':    config
                  , 'Train_Output_Dict_ThisEpoch': Train_Output_Dict_ThisEpoch
                  , 'Test_Output_Dict_ThisEpoch':  Test_Output_Dict_ThisEpoch
                  , 'ALL_Information_Dict':        ALL_Information_Dict
                  }

    save_dict_path = os.path.join(config.Settings.Output_Dir, f'{save_type}_savedict.pth')
    save_file_path = os.path.join(config.Settings.Output_Dir, f'{save_type}_prediction.txt')
    
    torch.save(save_state, save_dict_path)
    
    with open(save_file_path, "w") as h:
        h.write(f'human_Protein\tvirus_Protein\tTrue_Label\tProbability_not_interact\tProbability_interact\n')
        for Protein_pair, prediction_score in zip(Test_Output_Dict_ThisEpoch["Test_Protein_Pair_Name"], Test_Output_Dict_ThisEpoch["Test_Score_ThisEpoch"]):
            Protein1_Name, Protein2_Name, Label = Protein_pair
            h.write(f'{Protein1_Name}\t{Protein2_Name}\t{Label[1]}\t{str(round(1-prediction_score, 4))}\t{str(round(prediction_score, 4))}\n')
            
    logger.info(f'Saved {save_type} savedict   from Epoch:{Epoch+1} in {save_dict_path} !!!')
    logger.info(f'Saved {save_type} prediction from Epoch:{Epoch+1} in {save_file_path} !!!')
    
class EarlyStopping():
    """
    Stop the training process when performance on the validation dataset no longer improves.

    Attributes:
    - Patience:     The number of epochs with no improvement after which training will be stopped.
    - Tol:          Minimum change in the monitored quantity to qualify as an improvement.
    - Counter:      Tracks the number of consecutive epochs without improvement.
    - Highest_Aupr: Highest AUPR (Area Under the Precision-Recall Curve) observed so far.
    - Early_Stop:   A boolean indicating whether to stop training.
    - logger:       Logger object for recording information.

    Methods:
    - __init__: Initialize the EarlyStopping object.
    - __call__: Check the validation AUPR and decide whether to stop the training.

    """
    def __init__(self, logger, Patience = 5, Tol = 0.001):
        self.Patience = Patience
        self.Tol      = Tol 
        self.Counter  = 0
        self.Highest_Aupr = None
        self.Early_Stop   = False
        self.logger       = logger
        self.logger.info(f'EarlyStopping Patience is {self.Patience}')
        self.logger.info(f'EarlyStopping Tol is {self.Tol}')

    def __call__(self, Test_Aupr):
        if self.Highest_Aupr == None:
            self.Highest_Aupr = Test_Aupr
        elif Test_Aupr - self.Highest_Aupr > self.Tol:
            self.logger.info(f'This Epoch Aupr is {Test_Aupr}')
            self.logger.info(f'The Highest Aupr before this Epoch is {self.Highest_Aupr}')
            self.logger.info(f'Keep Training')
            self.Highest_Aupr = Test_Aupr
            self.Counter = 0
        elif Test_Aupr - self.Highest_Aupr <= self.Tol:
            self.Counter += 1
            self.logger.info("Notice: Early stopping Counter {} of {}.".format(self.Counter, self.Patience))
            if self.Counter >= self.Patience:
                self.logger.info("Notice: Early Stopping Actived.")
                self.Early_Stop = True
        return self.Early_Stop, self.Counter

def Random_State_Set(seed):
    """
    Set random seed for reproducibility across multiple runs.

    Args:
    - seed: Random seed value.

    Note:
    This function sets the random seed for various libraries to ensure reproducibility:
    - Python's built-in random module
    - NumPy
    - PyTorch's CPU and GPU
    - CuDNN for PyTorch (to ensure deterministic behavior)
    """
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def IterOnce(model, criterion, optimizer, Samples, Labels):
    """
    Performs a single iteration of model training.

    Args:
    - model: The neural network model.
    - criterion: The loss function used to compute the training loss.
    - optimizer: The optimizer for updating model parameters.
    - Samples: Input samples for training.
    - Labels: Ground truth labels.

    Returns:
    - Sigma: Sigmoid output of the model.
    - Correct: Number of correctly predicted samples.
    - Loss: Current loss value.
    - protein1_attention_list: protein1 attention list
    - protein2_attention_list: protein2 attention list
    """
    
    zhat, protein1_attention_list, protein2_attention_list = model.forward(*Samples)
    
    Loss = criterion(zhat, Labels)
    
    Loss.backward()         
    optimizer.step()
    optimizer.zero_grad()
    
    Sigma   = torch.softmax(zhat, dim=1)[:,1]
    yhat    = torch.max(zhat, dim=1)[1]
    Correct = torch.sum(yhat == Labels[:,1])
    
    return Sigma, Correct, Loss, protein1_attention_list, protein2_attention_list


def TestOnce(model, criterion, Samples, Labels):
    """
    Perform a single forward pass and evaluation using the given model and samples.

    Args:
    - model: The neural network model to be evaluated.
    - criterion: The loss function used to compute the evaluating loss.
    - Samples: Input samples for evaluation.
    - Labels: Ground truth labels for evaluation.

    Returns:
    - Sigma: Sigmoid output of the model.
    - Correct: Number of correctly predicted samples.
    - Loss: Loss value for the given samples and labels.
    - protein1_attention_list: protein1 attention list
    - protein2_attention_list: protein2 attention list
    """
    zhat, protein1_attention_list, protein2_attention_list = model.forward(*Samples)
    Loss = criterion(zhat,Labels)
    
    Sigma   = torch.softmax(zhat, dim=1)[:,1]
    yhat    = torch.max(zhat, dim=1)[1]
    Correct = torch.sum(yhat == Labels[:,1])
    
    return Sigma, Correct, Loss, protein1_attention_list, protein2_attention_list


def ROC_curve(Score_ThisEpoch, Label_ThisEpoch):
    """
    Compute the ROC (Receiver Operating Characteristic) and PR (Precision-Recall) curves related values.

    Args:
    - Score_ThisEpoch: Predicted scores for the current epoch.
    - Label_ThisEpoch: True labels for the current epoch.

    Returns:
    - Auroc_Area_ThisEpoch: Area under the ROC curve for the current epoch.
    - Aupr_Area_ThisEpoch:  Area under the Precision-Recall curve for the current epoch.
    - Fpr_ThisEpoch:        False Positive Rate values for the current epoch.
    - Tpr_ThisEpoch:        True Positive Rate values for the current epoch.
    - Precision_ThisEpoch:  Precision values for the current epoch.
    - Recall_ThisEpoch:     Recall values for the current epoch.
    """
    Fpr_ThisEpoch, Tpr_ThisEpoch, _          = roc_curve(np.array(Label_ThisEpoch), np.array(Score_ThisEpoch))
    Precision_ThisEpoch, Recall_ThisEpoch, _ = precision_recall_curve(np.array(Label_ThisEpoch), np.array(Score_ThisEpoch))
    Auroc_Area_ThisEpoch = auc(Fpr_ThisEpoch, Tpr_ThisEpoch)
    Aupr_Area_ThisEpoch  = auc(Recall_ThisEpoch, Precision_ThisEpoch)
    return Auroc_Area_ThisEpoch, Aupr_Area_ThisEpoch, Fpr_ThisEpoch, Tpr_ThisEpoch, Precision_ThisEpoch, Recall_ThisEpoch

def Graph_Collect_Func(Samples):
    """
    Collect and prepare data for graph-based models.

    Args:
    - Samples: Protein_Pair_idx, Protein1_length, Protein2_length, Protein1_Embedding_graphs, Protein2_Embedding_graphs, Labels

    Returns:
    - Protein_Pair_idx: Tensor containing protein pair indices.
    - Protein1_length: Tensor containing lengths of protein 1 embeddings.
    - Protein2_length: Tensor containing lengths of protein 2 embeddings.
    - Protein1_Embedding_graphs: Batch of graph data for protein 1.
    - Protein2_Embedding_graphs: Batch of graph data for protein 2.
    - Labels: Tensor containing labels.
    """
    Protein_Pair_idx, Protein1_length, Protein2_length, Protein1_Embedding_graphs, Protein2_Embedding_graphs, Labels = map(list,zip(*Samples))
    return torch.tensor(Protein_Pair_idx), \
            torch.tensor(Protein1_length), torch.tensor(Protein2_length), \
                Batch.from_data_list(Protein1_Embedding_graphs), Batch.from_data_list(Protein2_Embedding_graphs), torch.tensor(Labels)