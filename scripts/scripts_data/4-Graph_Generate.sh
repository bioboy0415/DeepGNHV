#!/bin/bash

DeepGNHV_Dir=$1
processed_target_list=("human" "virus")

for processed_target in ${processed_target_list[@]}
do
    params="python -u      ${DeepGNHV_Dir}/scripts/scripts_data/4-Graph_Generate.py
            --monomoer_dir ${DeepGNHV_Dir}/data/monomer_data/${processed_target}_processed/${processed_target}_monomer
            --graph_type   Undirectedgraph
            --distance_cutoff 8
            "
    echo -e "Running script:\n${params}"
    eval $params
done
