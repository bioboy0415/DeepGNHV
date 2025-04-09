from transformers import T5Tokenizer, T5EncoderModel
import torch
import re
import os
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Start', add_help=False)
    parser.add_argument('--monomoer_dir', type = str, help = 'Path for protein monomer directory.')
    args = parser.parse_args()
    monomoer_dir = Path(args.monomoer_dir)

    tokenizer = T5Tokenizer.from_pretrained('/home/lyjiang/.cache/huggingface/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73', do_lower_case=False)
    model     = T5EncoderModel.from_pretrained('/home/lyjiang/.cache/huggingface/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73').eval()


    for item in os.listdir(monomoer_dir):
        fasta_path                = monomoer_dir/item/f'{item}.fasta'
        token_representation_path = monomoer_dir/item/f'{item}.protT5_tokens'
        if fasta_path.exists() and not token_representation_path.exists():
            print(f'Treating {item}')
            with open(fasta_path,'r') as h:
                fasta_sequence = h.readlines()[1].strip('\n')
            sequence_list = [fasta_sequence]
            sequence_processed = [' '.join(list(re.sub(r'[UZOB]', 'X', sequence))) for sequence in sequence_list]
            ids = tokenizer.batch_encode_plus(sequence_processed, add_special_tokens=True, padding='longest')
            input_ids = torch.tensor(ids['input_ids'])
            attention_mask = torch.tensor(ids['attention_mask'])
            with torch.no_grad():
                embedding_rpr = model(input_ids=input_ids,attention_mask=attention_mask)
            protein_protT5_embedding = embedding_rpr.last_hidden_state[0,:-1].cpu()
            torch.save(protein_protT5_embedding,token_representation_path)
            print(f'Successfully treated {item}')

'''
If the model cannot be successfully loaded, please replace the loading path (Rostlab/prot_t5_xl_uniref50) with the path to the locally downloaded model located in the /home/Username/.cache/huggingface/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/xxxxxx subdirectory.

/home/lyjiang/.cache/huggingface/hub/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73
'''

