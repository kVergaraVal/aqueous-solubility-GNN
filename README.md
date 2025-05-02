# aqueous-solubility-GNN

## Overview

This repository corresponds to one of the solubility models of the ongoing study of Kevin Vergara, Dr. Pedro Saa and Dr. Nicolás Gajardo. The present model is based on the architecture of Qin et al., (2023), Rittig & Mitsos, (2024), and Abranches et al., (2024). The proposed architecture is the following:



## Datasets
This part of the study uses five main datasets, obtained from Ali et al., 2024 (https://github.com/ComPlat/water-solubility-prediction). All dataset files are found in `data/solubility_water/`:
- **BNNlab_water_solubility_data.csv** (900 samples)
- **Gihan_water_solubility.csv** (6,154 samples)
- **Sorkun_water_solubility.csv** (9,943 samples)
- **Xian_Zeng_water_solubility.csv** (11,862 samples)
- **Huuskonen_water_solubility.csv** (1,282 Test set - samples)

The combined dataset results in a total of 28,859 datapoints.

## Preprocessing
The datasets processing was done in the notebook `merge_water_solubility_datasets.ipynb`. This included (as done in Ali et al., 2024):
- **Removing duplicates**: Ensuring that no duplicate entries exist within the combined dataset.
- **Canonical SMILES**: Generating canonical SMILES strings to identify and remove matching data points between the training and test datasets.
- **Resulting Datasets**: After preprocessing, the datasets files are 
  - Training data: **unique_train_water_final.csv** (17,737 samples)
  - Test data: **unique_test_water_final.csv** (1,282 samples)

Further processing was done, which was not done in the Ali et al. (2024) study. This was done in `compatibilize_water_solubility_with_rittig_architecture.ipynb` notebook. This includes:
- **Generating solute and solvent lists**: in order to use the datasets with the Rittig et al. (2024) based model, a generation of files containing the solute and solvent (only water) information was necessary. The files are:
  - **solute_list.csv**
  - **solvent_list_water.csv**
- **Compatibilizig the train/test datasets with the solute and solvent list**: through adding columns for solute id and solvent id. This was done in `compatibilize_water_solubility_with_rittig_architecture.ipynb` notebook. The resulting files are:
  - **COMPATIBLE_unique_train_water_final.csv**
  - **COMPATIBLE_unique_test_water_final.csv**
- **Adding polarity and molecular size information**: To apply stratified sampling based on these parameters since they are deemed important for solubility. This was done in `add_molecular_size_and_polarity.ipynb` notebook. The resulting **final** files:
  - **solute_list_with_polarity_and_size.csv**
  - **solvent_list_water_with_polarity_and_size.csv**
  - **COMPATIBLE_unique_train_water_with_hybrid_classes.csv**
  - **COMPATIBLE_unique_test_water_with_hybrid_classes.csv**
 
## Model training
The model architecture is found in `model/model_GNN_water.py` (the proposed architecture is under the class `Rittig_and_Abranches`). The 5-fold cross-validation training and evaluation was done in `train_water_solubility.py`, and the retraining of the model through the whole training dataset was done in `retrain_water_solubility_whole_dataset.py`. The model testing was done in the notebook `test_models.ipynb`.

## Repository architecture



