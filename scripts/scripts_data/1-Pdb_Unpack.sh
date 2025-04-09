#!/bin/bash

DeepGNHV_Dir=$1
mkdir -p ${DeepGNHV_Dir}/data/monomer_data/human_processed/UP000005640_9606_HUMAN_v4/
tar -xvf ${DeepGNHV_Dir}/data/monomer_data/human_processed/UP000005640_9606_HUMAN_v4.tar -C ${DeepGNHV_Dir}/data/monomer_data/human_processed/UP000005640_9606_HUMAN_v4/
unzip ${DeepGNHV_Dir}/data/monomer_data/virus_processed/alphafold2_virus_pdb.zip -d ${DeepGNHV_Dir}/data/monomer_data/virus_processed/alphafold2_virus_pdb

params="python -u        ${DeepGNHV_Dir}/scripts/scripts_data/1-Pdb_Unpack.py
        --compressed_dir ${DeepGNHV_Dir}/data/monomer_data/human_processed/UP000005640_9606_HUMAN_v4
        --af2_dir        ${DeepGNHV_Dir}/data/monomer_data/human_processed/alphafold2_human_pdb
        "
echo -e "Running script:\n${params}"
eval $params
echo ""--------------------------------------------------------------------------------------------------------""
echo ""--------------------------------------------------------------------------------------------------------""