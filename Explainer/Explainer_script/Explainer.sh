# #!/bin/bash

DeepGNHV_Dir=$1

is_true=1
is_false=0

GPU_Id=0
Model_Name="DeepGNHV"
GNN_Type="GATConv"
Graph_Type="Undirectedgraph"
Dropout_Rates="0.3"
Graph_Distance_Threshold="8"
Lr=0.0003

Max_Epochs="100"
Graph_RSA_Threshold="0.3"
num_GNNLayer="2"
num_MLPLayer="1"
MLPDecrease="2"

Eval_Explain_LR=0.05
Eval_Explain_Epochs=5000


declare -A Batch_Size_Map
Batch_Size_Map[4]="64"
Batch_Size_Map[6]="64"
Batch_Size_Map[8]="64"
Batch_Size_Map[10]="64"
Batch_Size_Map[12]="32"
Batch_Size_Map[14]="32"

use_protT5=is_true
use_nnembedding=is_false
use_proteinbert=is_false
use_esm2=is_false
use_SaProt=is_false
use_onehot=is_false
use_foldseek=is_false
use_pssm=is_false
use_coord=is_false
use_physchem=is_false

Input_dim=$((use_protT5 * 1024 + use_nnembedding * 1024 + use_proteinbert * 1562 + use_esm2 * 1280 + use_SaProt * 1280 + use_onehot * 21 + use_foldseek * 21 + use_pssm * 20 + use_coord * 3 + use_physchem * 10))

declare -A Dimensions_Map
Dimensions_Map[2]="${Input_dim},720,360"
Dimensions_Map[3]="${Input_dim},720,360,360"
Dimensions_Map[4]="${Input_dim},720,360,360,360"

declare -A Attention_Heads
Attention_Heads_Map[2]="12,12"
Attention_Heads_Map[3]="12,12,12"
Attention_Heads_Map[4]="12,12,12,12"


Batch_Size=${Batch_Size_Map[$Graph_Distance_Threshold]:-64}
Dimensions=${Dimensions_Map[$num_GNNLayer]}
Attention_Heads=${Attention_Heads_Map[$num_GNNLayer]}
File_Suffix="${Model_Name}_${Graph_Distance_Threshold}A_rsacutoff${Graph_RSA_Threshold}"

[ "${use_protT5}"      = is_true ] && File_Suffix+="_protT5"
[ "${use_nnembedding}" = is_true ] && File_Suffix+="_nnembedding"
[ "${use_proteinbert}" = is_true ] && File_Suffix+="_proteinbert"
[ "${use_esm2}"        = is_true ] && File_Suffix+="_esm2"
[ "${use_SaProt}"      = is_true ] && File_Suffix+="_SaProt"
[ "${use_onehot}"      = is_true ] && File_Suffix+="_onehot"
[ "${use_foldseek}"    = is_true ] && File_Suffix+="_foldseek"
[ "${use_pssm}"        = is_true ] && File_Suffix+="_pssm"
[ "${use_coord}"       = is_true ] && File_Suffix+="_coord"
[ "${use_physchem}"    = is_true ] && File_Suffix+="_physchem"
File_Suffix+="_${Graph_Type}"



params="CUDA_VISIBLE_DEVICES=${GPU_Id} python -u ${DeepGNHV_Dir}/DeepGNHV_main.py
        --Eval                      True
        --Eval_Type                 Highest
        --Explain                   True
        --Mission_Name              DeepGNHV_model_Trained_on_Whole_Datasets
        --Model_Name                ${Model_Name}
        --GNN_Type                  ${GNN_Type}
        --Graph_Type                ${Graph_Type}
        --File_Suffix               ${File_Suffix}
        --Attention_Heads           ${Attention_Heads}
        --Dropout_Rates             ${Dropout_Rates}
        --Dimensions                ${Dimensions}
        --Max_Epochs                ${Max_Epochs}
        --Lr                        ${Lr}
        --num_GNN_Layers            ${num_GNNLayer}
        --num_MLP_Layers            ${num_MLPLayer}
        --Graph_RSA_Threshold       ${Graph_RSA_Threshold}
        --Graph_Distance_Threshold  ${Graph_Distance_Threshold}
        --MLPDecrease               ${MLPDecrease}
        --Eval_Explain_Dir          ${DeepGNHV_Dir}/Explainer/Explainer_Result/${Graph_Type}_ExplainEpochs${Eval_Explain_Epochs}_ExplainLR${Eval_Explain_LR}
        --Eval_Explain_LR           ${Eval_Explain_LR}
        --Eval_Explain_Epochs       ${Eval_Explain_Epochs}
        --Eval_Output_Dir           ${DeepGNHV_Dir}/output
        --Eval_Positive_Catalog     ${DeepGNHV_Dir}/Explainer/Explainer_Data/Explain_sample.txt
        --Eval_Negative_Catalog     ${DeepGNHV_Dir}/Explainer/Explainer_Data/Explain_sample.txt
        --Eval_Protein1_Dir         ${DeepGNHV_Dir}/data/monomer_data/human_processed/human_monomer
        --Eval_Protein2_Dir         ${DeepGNHV_Dir}/data/monomer_data/virus_processed/virus_monomer
        --GPU_Id                    ${GPU_Id}"

echo "Running:          ${params}"
echo "use_protT5:       ${use_protT5}"
echo "use_nnembedding:  ${use_nnembedding}"
echo "use_proteinbert:  ${use_proteinbert}"
echo "use_esm2:         ${use_esm2}"
echo "use_SaProt:       ${use_SaProt}"
echo "use_onehot:       ${use_onehot}"
echo "use_foldseek:     ${use_foldseek}"
echo "use_pssm:         ${use_pssm}"
echo "use_coord:        ${use_coord}"
echo "use_physchem:     ${use_physchem}"

eval $params

sleep 10

echo ""--------------------------------------------------------------------------------------------------------""
echo ""--------------------------------------------------------------------------------------------------------""
                                        
