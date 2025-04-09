# Extract the sequence of the human protein and virus protein from the PDB files

import os
import shutil
import argparse
from pathlib import Path
from Bio.PDB import *
from concurrent.futures import ThreadPoolExecutor

amino_acid_three = [
    'ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS', 'LEU',
    'MET', 'ASN', 'PRO', 'GLN', 'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'X',
]
amino_acid_one = 'ACDEFGHIKLMNPQRSTVWYX'

d3_to_d1 = {}
d1_to_d3 = {}

for a1, a3 in zip(amino_acid_one, amino_acid_three):
    d3_to_d1[a3] = a1
    d1_to_d3[a1] = a3


def Sequence_from_pdb(item):
    protein_item       = item.split('.')[0]
    monomer_protein_dir        = monomoer_dir/f'{protein_item}'
    monomer_protein_dir.mkdir(exist_ok=True)
    
    af2_pdb_path       = af2_dir/f'{protein_item}.pdb'
    monomer_pdb_path   = monomer_protein_dir/f'{protein_item}.pdb'
    monomer_fasta_path = monomer_protein_dir/f'{protein_item}.fasta'
    
    if not monomer_pdb_path.exists():
        shutil.copy(af2_pdb_path, monomer_protein_dir)
    
    if monomer_pdb_path.exists() and not monomer_fasta_path.exists():
        parser = PDBParser()
        chain = list(parser.get_structure(protein_item, monomer_pdb_path).get_models())[0].get_list()
        assert len(chain) == 1, f'{protein_item} contain more than one chain'
        
        residues = []

        for residue in chain[0].get_list(): 
            amino_name = residue.get_resname()
            atom_type = set(atom.get_id() for atom in residue.get_list())
            if sum([_ in atom_type for _ in ('N', 'CA', 'C')]) != 3:
                print(str(protein_item) + ' not complete')
                break
            residues.append(residue)

        protein_seq, protein_seq_position = [],[]
        for residue in residues:
            protein_seq.append(d3_to_d1.get(residue.get_resname(),'X'))
            protein_seq_position.append(residue.id[1])

        with open(monomer_fasta_path, 'wt') as h:
            h.write(f'>{protein_item}\n{"".join(protein_seq)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--monomoer_dir', type = str, help = 'Path for protein monomer directory.')
    parser.add_argument('--af2_dir',      type = str, help = 'Path for unpacked pdb files directory..')
    
    args = parser.parse_args()
    
    monomoer_dir = Path(args.monomoer_dir)
    af2_dir      = Path(args.af2_dir)
    monomoer_dir.mkdir(exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        req = [executor.submit(Sequence_from_pdb, item) for item in os.listdir(af2_dir)]
        