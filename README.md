# GearBind

For the latest version of GearBind code and the link to the datasets, please refer to https://github.com/DeepGraphLearning/GearBind.

## Overview

GearBind is a pretrainable geometric graph neural network for protein-protein binding affinity change (ddG_bind) prediction.
It is pretrained on CATH using contrastive learning and fine-tuned on SKEMPI with a regression loss.
Here we provide the inference code of GearBind.

This codebase is based on PyTorch and [TorchDrug]. It supports training and inference with multiple GPUs or multiple machines.

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug

## Installation

You may install the dependencies via either conda or pip. Generally, GearBind works
with Python 3.8/3.9 and PyTorch version >= 1.8.0.

Windows, Mac OS X and Linux should all be supported.

### From Conda

If internet connection is smooth, the installation should be completed within 15 minutes.
[Using mamba as the conda solver](https://www.anaconda.com/blog/a-faster-conda-for-a-growing-community) can potentially speed up the installation process.

```bash
conda install pyg pytorch=1.8.0 cudatoolkit=11.1 torchdrug -c pyg -c pytorch -c conda-forge
conda install rdkit easydict pyyaml biopython gdown -c conda-forge
```


## Inference on HER2 and CR3022

Now we show how to use our (pre-)trained models for inference on new wild-type proteins.
Here we take the HER2 and CR3022 proteins used in the paper as examples.
First, you need to download the checkpoints to the `./checkpoints` directory.
Note that we can not provide FoldX-generated HER2 and CR3022 mutant structures due to license restrictions.
Please prepare the wild-type and mutant structures yourself.
The prepared dataset should have the following file structure:

- `data.csv`: a csv file with columns "pdb_id", "mutation", "chain_a", "chain_b", "wt_protein", "mt_protein", where
    - "pdb_id" is the stem of the protein complex structure file name
    - "mutation" is the comma-separated mutation list
    - "chain_a" and "chain_b" are interacting chains in the complex (e.g., HL and C),
    - "wt_protein" and "mt_protein" are the file names of the wild-type and mutant structures, respectively.
- `data`: folder storing the wild-type and mutant structures.

The PDB structures used to prepare the HER2 (`1n8z.pdb`) and CR3022 dataset (`6xc3_wt.pdb`) are provided in the `data` directory.
`6xc3_ba4.pdb` and `6xc3_ba11.pdb` are the PDB structures of CR3022 against the RBD of BA.4 and BA.1.1 strains of SARS-CoV-2, respectively, modelled by SWISS-MODEL.

```bash
# Downloading model checkpoints
cd checkpoints
gdown 1nFEjbjdlRWFwYz7LUNv_D6oLnEsZ5beJ
unzip new-gearbind-model-weights.zip
mv new-gearbind-model-weights/*.pth ./
rm -rf new-gearbind-model-weights
cd ..
```

We have prepared the config file in the `./config/predict` directory.
To get the prediction results of the pre-trained models on different variants, you can run the following commands.

```bash
# Run GearBind-P models on CR3022 datasets
python script/predict.py -c config/predict/CR3022_GearBindP.yaml

# Run GearBind models on HER2 datasets
python script/predict.py -c config/predict/HER2_GearBind.yaml
```

The inference should take about 2 minutes on a single A100 GPU. The expected output for HER2 binders are stored in `results/GearBind_HER2_1n8z_renum.pdb_HL_C.csv`.
After finishing the prediction, you are expected to get an output file called `<model_class>_<dataset_class>_<test_split>.csv`.
For the second case, the name of the output file is `GearBind_HER2_1n8z_renum.pdb_HL_C.csv`.
You can compare this output with the results we provide in `./results`.

To run the model on your own protein complexes, you need to
1. prepare the dataset with FoldX
2. write a customized dataset class following `dataset.HER2` and `dataset.CR3022`
3. add a `.yaml` file by modifying the configuration of the dataset class


## SKEMPI preprocesssing

The following commands process SKEMPI from raw data, including downloading the raw data, processing the data so that it is ready for FoldX mutagenesis.

```bash
python script/process_skempi.py --csv-path $SKEMPI_CSV_PATH --pdb-dir $SKEMPI_PDB_DIR --output-csv-path $PROCESSED_SKEMPI_CSV_PATH --output-pdb-dir $PROCESSED_SKEMPI_PDB_DIR --no-repair
```

where
- `SKEMPI_CSV_PATH`: the path to the raw SKEMPI csv file.
- `SKEMPI_PDB_DIR`: the directory containing the raw SKEMPI pdb files.
- `PROCESSED_SKEMPI_CSV_PATH`: the path to the processed SKEMPI csv file.
- `PROCESSED_SKEMPI_PDB_DIR`: the directory to store the processed SKEMPI pdb files.

The processed SKEMPI dataset and all model predictions can be found in `data/skempi_v2_with_all_results_0415.csv`.
