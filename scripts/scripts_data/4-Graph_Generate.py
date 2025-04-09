import os
import argparse
from pathlib import Path

import torch
torch.set_num_threads(10)
from torch_geometric.data  import Data
from torch_geometric.utils import add_self_loops
from torch_cluster import radius

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFreeSASA, MolFromSmiles

import random
random.seed(100)

ATOMS = {'CA': 0,'N': 1,'C': 2,'CB': 3,'O': 4,'NX': 5,'CX': 6,'OX': 7, 'SX': 8,'HX': 9,'X': 10}
ATOM_ONES = torch.eye(len(ATOMS.items()), dtype=torch.float32)

amino_acid_sequence = 'ACDEFGHIKLMNPQRSTVWY'
amino_acid_smiles   = {
    'A': 'C[C@H](N)C(O)=O',
    'C': 'O=C(O)[C@H](CS)N',
    'D': 'O=C(O)[C@H](CC(O)=O)N',
    'E': 'O=C(O)[C@H](CCC(O)=O)N',
    'F': 'O=C(O)[C@@H](N)CC1=CC=CC=C1',
    'G': 'O=C(O)CN',
    'H': 'O=C(O)[C@H](CC1=CNC=N1)N',
    'I': 'CC[C@H](C)[C@H](N)C(O)=O',
    'K': 'O=C([C@@H](N)CCCCN)O',
    'L': 'CC(C)C[C@H](N)C(O)=O',
    'M': 'O=C(O)[C@@H](N)CCSC',
    'N': 'O=C(O)[C@H](CC(N)=O)N',
    'P': 'O=C([C@@H]1CCCN1)O',
    'Q': 'O=C(O)[C@H](CCC(N)=O)N',
    'R': 'O=C(O)[C@H](CCCNC(N)=N)N',
    'S': 'O=C(O)[C@H](CO)N',
    'T': 'O=C(O)[C@H]([C@H](O)C)N',
    'V': 'CC(C)[C@H](N)C(O)=O',
    'W': 'O=C(O)[C@@H](N)CC1=CNC2=CC=CC=C12',
    'Y': 'O=C(O)[C@H](CC1=CC=C(O)C=C1)N'
}

CA_IDX = ATOMS['CA']
N_IDX = ATOMS['N']
C_IDX = ATOMS['C']
H_IDX = ATOMS['HX']


def embedding_batch_normalize(unprocessed_tokens):
    assert len(unprocessed_tokens.shape) == 2
    
    min_values = unprocessed_tokens.min(dim=0)[0]
    max_values = unprocessed_tokens.max(dim=0)[0]
    scaled_tensor = 2 * (unprocessed_tokens - min_values) / (max_values - min_values) - 1

    return scaled_tensor

def physchem_AA_dict_generate():
    '''
    1.  TPSA
    2.  MolLogP
    3.  MolMR
    4.  CalcNumLipinskiHBA
    5.  CalcNumLipinskiHBD
    6.  ExactMolWt
    7.  CalcNumAtoms
    8.  CalcNumHeteroatoms
    9.  NumValenceElectrons
    10. RSA
    '''
    physchem_list = ['TPSA', 'MolLogP', 'MolMR', 'CalcNumLipinskiHBA', 'CalcNumLipinskiHBD', 'ExactMolWt', 'CalcNumAtoms', 'CalcNumHeteroatoms', 'NumValenceElectrons']
    physchem_dict = dict()
    for _AA in amino_acid_sequence:
        smiles_amino = amino_acid_smiles[_AA]
        smiles_mol   = MolFromSmiles(smiles_amino)
        
        physchem_dict[_AA] = dict(amino_smiles = smiles_amino)
        
        amino_TPSA                = round(Descriptors.TPSA(smiles_mol), 2)
        amino_MolLogP             = round(Descriptors.MolLogP(smiles_mol), 2)
        amino_MolMR               = round(Descriptors.MolMR(smiles_mol), 2)
        amino_ExactMolWt          = round(Descriptors.ExactMolWt(smiles_mol), 2)
        amino_NumValenceElectrons = round(Descriptors.NumValenceElectrons(smiles_mol), 2)
        amino_CalcNumAtoms        = round(rdMolDescriptors.CalcNumAtoms(smiles_mol), 2)
        amino_CalcNumHeteroatoms  = round(rdMolDescriptors.CalcNumHeteroatoms(smiles_mol), 2)
        amino_CalcNumLipinskiHBA  = round(rdMolDescriptors.CalcNumLipinskiHBA(smiles_mol), 2)
        amino_CalcNumLipinskiHBD  = round(rdMolDescriptors.CalcNumLipinskiHBD(smiles_mol), 2)
        
        amino_physchem_list = [amino_TPSA, amino_MolLogP, amino_MolMR, amino_ExactMolWt, amino_NumValenceElectrons, amino_CalcNumAtoms, amino_CalcNumHeteroatoms, amino_CalcNumLipinskiHBA, amino_CalcNumLipinskiHBD]

        physchem_dict[_AA]['amino_physchem_list'] = amino_physchem_list

    return physchem_dict, physchem_list
    
def coord_from_pdb(_pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', _pdb_path)
    model = structure[0]
    coordinates_list = []
    for chain in model:
        for residue in chain:
            # 获取氨基酸的三维坐标
            coordinates = residue['CA'].get_coord()
            coordinates_list.append(coordinates)
    coordinates_tokens = torch.tensor(coordinates_list, dtype = torch.float32)
    return coordinates_tokens

def rsa_from_pdb(_pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', _pdb_path)
    dssp = DSSP(structure[0], _pdb_path)
    rsa_list = []
    for item_dssp in dssp:
        residue_id = (item_dssp[0], item_dssp[1])
        rsa = item_dssp[3]
        rsa_list.append(rsa)
    rsa_tokens = torch.tensor(rsa_list, dtype = torch.float32)
    return rsa_tokens


def physchem_from_fasta(_fasta_path, physchem_dict):
    with open(_fasta_path, 'r') as h:
        contents = h.readlines()
    # protein_name = contents[0].strip('\n')[1:]
    protein_seq  = contents[1].strip('\n')
    AA_physchem_list = [physchem_dict[AA]['amino_physchem_list'] for AA in protein_seq]
    AA_physchem_tokens = torch.tensor(AA_physchem_list, dtype = torch.float32)
    
    return AA_physchem_tokens
    
def pssm_load(pssm_path):
    pssm_list = []
    with open(pssm_path, 'r') as h:
        for _item in h.readlines():
            pssm_list.append([int(num) for num in _item.strip('\n').split(' ')])
    pssm_array = torch.tensor(pssm_list, dtype = torch.float)
    return pssm_array


def DeepGNHV_graphtype_generate(item, use_protT5, use_esm2, use_SaProt, use_onehot, use_foldseek, use_pssm, use_coord, use_physchem, distance_cutoff, graph_type, rsa_cutoff = 0.3):
    assert graph_type in ['Undirectedgraph', 'Surfacegraph', 'Semidirectedgraph']
    
    suffix = f'.DeepGNHV_{distance_cutoff}A_rsacutoff{rsa_cutoff}'
    Path_this_fasta_tokens = monomoer_dir/item/f'{item}.fasta'
    Path_this_pdb_file     = monomoer_dir/item/f'{item}.pdb'
    
    tokens_list = []
    if use_protT5:
        suffix += f'_protT5'
        Path_this_protT5_tokens = monomoer_dir/item/f'{item}.protT5_tokens'
        protT5_tokens = torch.load(Path_this_protT5_tokens).to(torch.float32)
        tokens_list.append(protT5_tokens)
    if use_esm2:
        suffix += f'_esm2'
        Path_this_esm2_tokens = monomoer_dir/item/f'{item}.esm2_tokens'
        esm2_tokens = torch.load(Path_this_esm2_tokens).to(torch.float32)
        tokens_list.append(esm2_tokens)
    if use_SaProt:
        suffix += f'_SaProt'
        Path_this_SaProt_tokens = monomoer_dir/item/f'{item}.SaProt_tokens'
        SaProt_tokens = torch.load(Path_this_SaProt_tokens).to(torch.float32)
        tokens_list.append(SaProt_tokens)
    if use_onehot:
        suffix += f'_onehot'
        Path_this_onehot_tokens = monomoer_dir/item/f'{item}.onehot'
        onehot_tokens = torch.load(Path_this_onehot_tokens).to(torch.float32)
        tokens_list.append(onehot_tokens)
    if use_foldseek:
        suffix += f'_foldseek'
        Path_this_foldseek_tokens = monomoer_dir/item/f'{item}.foldseek'
        foldseek_tokens = torch.load(Path_this_foldseek_tokens).to(torch.float32)
        tokens_list.append(foldseek_tokens)
    if use_pssm:
        suffix += f'_pssm'
        Path_this_pssm_tokens = monomoer_dir/item/f'{item}.pssm'
        pssm_tokens = pssm_load(Path_this_pssm_tokens).to(torch.float32)
        tokens_list.append(embedding_batch_normalize(pssm_tokens))
    if use_coord:
        suffix += f'_coord'
        coordinates_tokens = coord_from_pdb(Path_this_pdb_file)
        tokens_list.append(embedding_batch_normalize(coordinates_tokens))
    if use_physchem:
        suffix += f'_physchem'

        AA_physchem_tokens = physchem_from_fasta(Path_this_fasta_tokens, physchem_dict)
        tokens_list.append(embedding_batch_normalize(AA_physchem_tokens))

        rsa_tokens = rsa_from_pdb(Path_this_pdb_file).reshape(-1,1)
        tokens_list.append(embedding_batch_normalize(rsa_tokens))

    cat_tokens = torch.cat(tokens_list, dim=1)
    
    if graph_type in ['Surfacegraph']:
        rsa_tokens               = rsa_from_pdb(Path_this_pdb_file)
        surface_node_indices     = (rsa_tokens >= rsa_cutoff).nonzero(as_tuple=True)[0]
        final_CA_coord           = coord_from_pdb(Path_this_pdb_file)[surface_node_indices]
        final_tokens             = cat_tokens[surface_node_indices]
        target_node, source_node = radius(final_CA_coord, final_CA_coord, distance_cutoff, max_num_neighbors=final_CA_coord.size(0))
        edge_index               = torch.stack((source_node, target_node), dim=0).long()
        graph                    = Data(x = final_tokens, edge_index = edge_index, surface_node_indices = surface_node_indices)

    elif graph_type == 'Undirectedgraph':
        final_CA_coord           = coord_from_pdb(Path_this_pdb_file)
        final_tokens             = cat_tokens
        target_node, source_node = radius(final_CA_coord, final_CA_coord, distance_cutoff, max_num_neighbors=final_CA_coord.size(0))
        edge_index               = torch.stack((source_node, target_node), dim=0).long()
        graph                    = Data(x = final_tokens, edge_index = edge_index)
        
    elif graph_type == 'Semidirectedgraph':
        rsa_tokens               = rsa_from_pdb(Path_this_pdb_file)
        surface_node_indices     = (rsa_tokens >= rsa_cutoff).nonzero(as_tuple=True)[0]
        final_CA_coord           = coord_from_pdb(Path_this_pdb_file)
        final_tokens             = cat_tokens
        target_node, source_node = radius(final_CA_coord, final_CA_coord, distance_cutoff, max_num_neighbors=final_CA_coord.size(0))
        edge_index_third         = (target_node.unsqueeze(1) == surface_node_indices).any(dim=1).int().to(torch.float32)
        edge_index               = torch.stack((source_node, target_node, edge_index_third), dim=0).long()
        graph                    = Data(x = final_tokens, edge_index = edge_index, surface_node_indices = surface_node_indices)
    
    suffix += f'_{graph_type}'
    
    return graph, suffix

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--monomoer_dir',    type = str, default = './DeepGNHV/data/monomer_data/human_processed/human_monomer', help = 'Path for protein monomer directory.')
    parser.add_argument('--graph_type',      type = str, default = 'Undirectedgraph', choices = ['Undirectedgraph', 'Semidirectedgraph', 'Surfacegraph'], help = 'Path for protein graph type.')
    parser.add_argument('--distance_cutoff', type = str, default = '8', help = 'Path for protein edge index distance cutoff.')

    args = parser.parse_args()

    monomoer_dir    = Path(args.monomoer_dir)
    graph_type      = args.graph_type
    distance_cutoff = int(args.distance_cutoff)


    use_protT5      = True
    use_esm2        = False
    use_SaProt      = False
    use_onehot      = False
    use_foldseek    = False
    use_pssm        = False
    use_coord       = False
    use_physchem    = False
    rsa_cutoff      = 0.3

    physchem_dict, physchem_list = physchem_AA_dict_generate()


    for item in os.listdir(monomoer_dir):
        try:
            graph, suffix = DeepGNHV_graphtype_generate(item
                                                    , use_protT5      = use_protT5
                                                    , use_esm2        = use_esm2
                                                    , use_SaProt      = use_SaProt
                                                    , use_onehot      = use_onehot
                                                    , use_foldseek    = use_foldseek
                                                    , use_pssm        = use_pssm
                                                    , use_coord       = use_coord
                                                    , use_physchem    = use_physchem
                                                    , distance_cutoff = distance_cutoff
                                                    , graph_type      = graph_type
                                                    , rsa_cutoff      = rsa_cutoff)
            graph_save_path = monomoer_dir/item/f'{item}{suffix}'
            torch.save(graph, graph_save_path)
            print(f'Successfully: {graph_save_path}')
        except:
            print(f'Something wrong in {item}')
            pass
