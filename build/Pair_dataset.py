import os
import random
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from torch_geometric.data import Data
from torch.utils.data import Dataset

class Pair_dataset(Dataset):
    def __init__(self
                , Positive_Catalog
                , Negative_Catalog
                , Protein1_Dir
                , Protein2_Dir
                , logger
                , Data_Process
                , transform
                , Numbers_Positive
                , Numbers_Negative
                , File_Suffix
                , Fix_Length
            ):
        
        super(Pair_dataset,self).__init__()
        self.Positive_Catalog  = Positive_Catalog
        self.Negative_Catalog  = Negative_Catalog
        self.logger            = logger
        self.Protein1_Dir      = Protein1_Dir
        self.Protein2_Dir      = Protein2_Dir
        self.Data_Process      = Data_Process
        self.transform         = transform
        self.Numbers_Positive  = Numbers_Positive
        self.Numbers_Negative  = Numbers_Negative
        self.File_Suffix       = File_Suffix
        self.Fix_Length        = Fix_Length
        
        self.Positive_list    = self.Load_Pair(self.Positive_Catalog, self.Numbers_Positive, self.Protein1_Dir, self.Protein2_Dir, 1, self.File_Suffix)   
        self.Negative_list    = self.Load_Pair(self.Negative_Catalog, self.Numbers_Negative, self.Protein1_Dir, self.Protein2_Dir, 0, self.File_Suffix)
        
        self.Protein_dict     = self.Load_Dict_Path(self.Positive_list, self.Negative_list, self.Protein1_Dir, self.Protein2_Dir) 
        self.Protein_dict     = self.Append_message(self.Protein_dict, self.File_Suffix, Data_Process)

        self.Sample_List      = [*self.Positive_list, *self.Negative_list]

    def Load_Pair(self, Catalog, Numbers, Protein1_dir, Protein2_dir, Label_Type, File_Suffix):
        pair_list = []
        with open(Catalog,'r') as h:
            content = h.readlines()
        if Numbers != None:
            content = content[:Numbers]
        for item in content:
            Protein1_Name, Protein2_Name, _ = item.strip("\n").split("\t")
            Protein1_file_path = Protein1_dir/Protein1_Name/f'{Protein1_Name}.{File_Suffix}'
            Protein2_file_path = Protein2_dir/Protein2_Name/f'{Protein2_Name}.{File_Suffix}'
            if Protein1_file_path.exists() and Protein2_file_path.exists():
                # pair_list.append((Protein1_Name, Protein2_Name, Label_Type))
                if Label_Type == 0:
                    pair_list.append((Protein1_Name, Protein2_Name, (1, 0)))
                else:
                    pair_list.append((Protein1_Name, Protein2_Name, (0, 1)))
        Labedl_Type_Str = 'Positive' if Label_Type == 1 else 'Negative'
        self.logger.info(f'{Labedl_Type_Str} datasets have {len(pair_list)}\tvalid protein pairs.') 
        return pair_list
    
    def Load_Dict_Path(self, Positive_list, Negative_list, Protein1_dir, Protein2_dir):
        Protein_dict = dict()
        for item in Positive_list:
            Protein1_Name, Protein2_Name, interaction_type = item
            Protein_dict[Protein1_Name] = dict(Protein_path = Protein1_dir/Protein1_Name)
            Protein_dict[Protein2_Name] = dict(Protein_path = Protein2_dir/Protein2_Name)
        for item in Negative_list:
            Protein1_Name, Protein2_Name, interaction_type = item
            Protein_dict[Protein1_Name] = dict(Protein_path = Protein1_dir/Protein1_Name)
            Protein_dict[Protein2_Name] = dict(Protein_path = Protein2_dir/Protein2_Name)
        self.logger.info(f'The datasets contain {len(Protein_dict)} valid proteins.') 
        return Protein_dict
    
    def Append_message(self, Protein_dict, File_Suffix, Data_process):
        self.logger.info(f"Loading ...")
        count = 0
        for This_Protein_Name, This_Protein_dict in list(Protein_dict.items()):
            File_path = This_Protein_dict["Protein_path"]/f'{This_Protein_Name}.{File_Suffix}'
            if File_path.exists():
                This_Protein_message = torch.load(File_path)
                if Data_process == "None":
                    Protein_dict[This_Protein_Name]["Protein_Length"] = torch.tensor(This_Protein_message.x.size(0), dtype=torch.float)
                    Protein_dict[This_Protein_Name][File_Suffix]      = This_Protein_message
                elif Data_process == "transform":
                    Protein_dict[This_Protein_Name]["Protein_Length"] = torch.tensor(This_Protein_message.size(0), dtype=torch.float)
                    Protein_dict[This_Protein_Name][File_Suffix]      = self.transform(This_Protein_message.unsqueeze(0))
                elif Data_process == "fix_length":
                    Protein_dict[This_Protein_Name]["Protein_Length"] = torch.tensor(This_Protein_message.size(0), dtype=torch.float)
                    This_Protein_message = This_Protein_message[:self.fix_length]
                    tensor_matrix = torch.zeros([self.fix_length, This_Protein_message.size(-1)], dtype = torch.float)
                    tensor_matrix[:This_Protein_message.size(0),] = This_Protein_message
                    Protein_dict[This_Protein_Name][File_Suffix]    = tensor_matrix
                count += 1
                if count % 500 == 0:
                    self.logger.info(f"Loaded {count} proteins.")
            else:
                self.logger.info(f'{This_Protein_Name} not exists')

        self.logger.info(f"Loaded num {count} {File_Suffix} Successfully !!!") 
        return Protein_dict
   
    def __info__(self):
        self.logger.info(f'The actual number of all samples loaded.:     {len(self.Sample_List)}')
        self.logger.info(f'The actual number of positive samples loaded: {len(self.Positive_list)}')
        self.logger.info(f'The actual number of negative samples loaded: {len(self.Negative_list)}')
        self.logger.info(f'The actual number of proteins loaded:         {len(self.Protein_dict)}')
        
    def __getitem__(self,idx):
        Protein1_Name, Protein2_Name, Label = self.Sample_List[idx]
        
        Protein1_Length = self.Protein_dict[Protein1_Name]['Protein_Length']
        Protein2_Length = self.Protein_dict[Protein2_Name]['Protein_Length']
        Protein1_Embedding_Graph = self.Protein_dict[Protein1_Name][self.File_Suffix]
        Protein2_Embedding_Graph = self.Protein_dict[Protein2_Name][self.File_Suffix]

        return (idx, Protein1_Length, Protein2_Length, Protein1_Embedding_Graph, Protein2_Embedding_Graph, Label)

    def __len__(self):
        return len(self.Sample_List)
    