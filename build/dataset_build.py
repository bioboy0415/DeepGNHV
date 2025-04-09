import os
import sys
import random
import torch

import numpy as np
from pathlib import Path
from torchvision import datasets, transforms
from torch_scatter import segment_csr,scatter
from torch_cluster import radius
from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from build.Pair_dataset import Pair_dataset
from module.utils import Graph_Collect_Func


def build_loader(config, logger):
    logger.info(f"Building Training Datasets ...")
    logger.info(f'Train protein1 dir:      {config.DATA.Train.Protein1_Dir}')
    logger.info(f'Train protein2 dir:      {config.DATA.Train.Protein2_Dir}')
    logger.info(f'Train positive datasets: {config.DATA.Train.Positive_Catalog}')
    logger.info(f'Train negative datasets: {config.DATA.Train.Negative_Catalog}')
    
    logger.info(f'Loaded protein suffix:   {config.DATA.Processing.File_Suffix}')
    
    Datasets_Train = build_dataset(is_Train = True, config = config, logger = logger)
    
    logger.info(f"Building Training Datasets Successfully !!!")
    
    logger.info(f"Building Testing Datasets ...")
    logger.info(f'Test_Protein1_Dir: {config.DATA.Test.Protein1_Dir}')
    logger.info(f'Test_Protein2_Dir: {config.DATA.Test.Protein2_Dir}')
    Datasets_Test  = build_dataset(is_Train = False, config = config, logger = logger)
    logger.info(f"Building Testing Datasets Successfully !!!")


    DataLoader_Train = DataLoader(Datasets_Train
                                ,batch_size = config.Train.Batch_Size
                                ,shuffle    = True
                                ,drop_last  = False
                                ,pin_memory = True
                                ,collate_fn = Graph_Collect_Func
                                )

    DataLoader_Test  = DataLoader(Datasets_Test
                                ,batch_size = config.Train.Batch_Size
                                ,shuffle    = False
                                ,drop_last  = False
                                ,pin_memory = True
                                ,collate_fn = Graph_Collect_Func)
    
    logger.info(f'------------------------------------------------------------------------------------')
    return DataLoader_Train, DataLoader_Test, Datasets_Train, Datasets_Test

def build_dataset(is_Train, config, logger):
    if is_Train:
        File_Suffix      = config.DATA.Processing.File_Suffix
        # transform = transforms.Compose([transforms.Resize((256,256),antialias=True)
        #                                ,transforms.RandomCrop((224,224))])
        Data_Process     = config.DATA.Processing.Data_Process
        transform        = transforms.RandomCrop((224,224))
        Positive_Catalog = Path(config.DATA.Train.Positive_Catalog)
        Negative_Catalog = Path(config.DATA.Train.Negative_Catalog)
        Protein1_Dir     = Path(config.DATA.Train.Protein1_Dir)
        Protein2_Dir     = Path(config.DATA.Train.Protein2_Dir)
        Numbers_Positive = config.DATA.Train.Numbers_Positive
        Numbers_Negative = config.DATA.Train.Numbers_Negative
        Fix_Length       = config.DATA.Processing.Fix_Length
    else:
        File_Suffix      = config.DATA.Processing.File_Suffix
        # transform = transforms.Compose([transforms.Resize((256,256),antialias=True)
        #                                 ,transforms.CenterCrop((224,224))])
        Data_Process     = config.DATA.Processing.Data_Process
        transform        = transforms.CenterCrop((224,224))
        Positive_Catalog = Path(config.DATA.Test.Positive_Catalog)
        Negative_Catalog = Path(config.DATA.Test.Negative_Catalog)
        Protein1_Dir     = Path(config.DATA.Test.Protein1_Dir)
        Protein2_Dir     = Path(config.DATA.Test.Protein2_Dir)
        Numbers_Positive = config.DATA.Test.Numbers_Positive
        Numbers_Negative = config.DATA.Test.Numbers_Negative
        Fix_Length       = config.DATA.Processing.Fix_Length
        
    dataset = Pair_dataset(Positive_Catalog = Positive_Catalog
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
                         )
    # dataset.__info__()
    return dataset
