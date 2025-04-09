#!/bin/bash

DeepGNHV_Dir=$1
processed_target_list=("human" "virus")

for processed_target in ${processed_target_list[@]}
do
    params="python -u     ${DeepGNHV_Dir}/scripts/scripts_data/2-Seq_from_Pdb.py
            --monomoer_dir ${DeepGNHV_Dir}/data/monomer_data/${processed_target}_processed/${processed_target}_monomer
            --af2_dir      ${DeepGNHV_Dir}/data/monomer_data/${processed_target}_processed/alphafold2_${processed_target}_pdb
            "
    
    echo -e "Running script:\n${params}"
    eval $params
    echo ""--------------------------------------------------------------------------------------------------------""
    echo ""--------------------------------------------------------------------------------------------------------""
done
