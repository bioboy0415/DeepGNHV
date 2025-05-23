# DeepGNHV
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0.0-blue.svg)
![PyG](https://img.shields.io/badge/pyg-2.3.0-blue.svg)
![Platform](https://img.shields.io/badge/platform-linux-blue.svg)

Graph neural network integrated with pre-trained protein language model for predicting human–virus protein–protein interactions.

## ⚙️ Environment Setup
- Please follow the steps in the [requirements.txt](https://github.com/bioboy0415/DeepGNHV/blob/master/requirements.txt) to set up the environment.


## 📊 Prepare Data
1. Download the predicted human protein structure models from the [Alphafold2 Database](https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v4.tar) and place the compressed file in the directory ".DeepGNHV/data/monomer_data/human_processed".
2. Download the virus protein structure models predicted using the locally deployed AlphaFold2 from [Zenodo](https://zenodo.org/records/15180938) and place the compressed file in the directory ".DeepGNHV/data/monomer_data/virus_processed".
3. Execute the script below to preprocess the source data from pdb files and [protT5](https://github.com/agemagician/ProtTrans).

```bash
cd ~/DeepGNHV-main
bash ./scripts/scripts_data/1-Pdb_Unpack.sh .
bash ./scripts/scripts_data/2-Seq_from_Pdb.sh .
bash ./scripts/scripts_data/3-ProtT5_Embedding.sh .
bash ./scripts/scripts_data/4-Graph_Generate.sh .
```

## 🚀 Training & Evaluating
- Train the DeepGNHV model using the prepared dataset:
```bash
nohup bash ./scripts/scripts_training/DeepGNHV_TrainTest.sh . &
```
- Evaluate the performance on the test dataset using the trained DeepGNHV model.
```bash
nohup bash ./scripts/scripts_training/DeepGNHV_Testonly.sh . &
```

## 💡 Analyzing the key residues
- Attempt to extract the amino acid residue sites that play a key role in interactions prediction.
```bash
cd ～/DeepGNHV
bash ./Explainer/Explainer_script/Explainer.sh .
```