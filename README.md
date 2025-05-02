# aqueous-solubility-GNN

## Overview

This repository corresponds to one of the solubility models of the ongoing study of Kevin Vergara, Dr. Pedro Saa and Dr. Nicolás Gajardo. The present model is based on the architecture of [Qin et al., (2023)](https://pubs.rsc.org/en/content/articlelanding/2023/dd/d2dd00045h), [Rittig & Mitsos, (2024)](https://pubs.rsc.org/en/content/articlelanding/2024/sc/d4sc04554h), and [Abranches et al., (2023)](https://pubs.acs.org/doi/10.1021/acs.jctc.3c01003). The proposed architecture is the following (this image does **not** show the optimized parameters):

![Rittig_Abranches](doc/Rittig-Abranches.png)

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
- **Resulting Datasets**: After preprocessing, the datasets files are:
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

### Architecture purely based on Rittig & Mitsos (2024) version:
![Pure_Rittig](doc/Pure_Rittig.png)

### Architecture purely based on Abranches et al. (2024) version:
![Pure_Abranchs](doc/Pure_Abranches.png)

### Architecture based on a combination of both Rittig and Abranches:
![Rittig_Abranches](doc/Rittig-Abranches.png)

### A simplified version of the Rittig-Abranches proposed model:
![Rittig_Abranches_variant](doc/Rittig-Abranches_variant.png)

## Hyperparameter optimization
The hyperparameter optimization was done with Optuna library and through the CENIA cluster, the files are:
- **BO_RittigPure.py**
- **BO_Rittig_Abranches.py**
- **BO_Rittig_Abranches_variant.py**

The Abranches_Pure version of the model was not optimized since it showed significantly bad evaluation results. The optimization results are as presented:
![Hyperparameters_results](doc/Hyperparameter_results.png)

## Repository structure
- `SigmaProfileModel`:  Obtained from the Abranches et. al (2023) study [repository](https://github.com/MaginnGroup/GSP). This module generates a Sigma Profile from a given SMILES.
  
- `data`: contains mainly the datasets used in the aqueous solubility module of the study.
  - `data/Rittig`: contains dataset originally used in the Rittig & Mitsos (2024) study, which was focused on activity coeficient prediction and not solubility prediction.
  - `data/solubility_water`: contains all the datasets and their processed version as stated in the "Dataset" segment of this README. Here, it is important to consider that the **final** files are:
    - **solute_list_with_polarity_and_size.csv**
    - **solvent_list_water_with_polarity_and_size.csv**
    - **COMPATIBLE_unique_train_water_with_hybrid_classes.csv**
    - **COMPATIBLE_unique_test_water_with_hybrid_classes.csv**

Additionally, in the `data/solubility_water` folder, there are 2 files not previously referenced:
  - **MORGAN_train_water_with_hybrid_classes.csv**: training dataset with Morgan fingerprint features (a Morgan fingerprint is a binary vector of variable length that gives information about structure and functional groups of the molecule).
  - **MACCS_train_water_with_hybrid_classes.csv**: training dataset with MACCS fingerprint features (a MACCS fingerprint is a binary vector of a fixed length that gives information about functional groups of the molecule).
These were made in a failed attempt of increasing model performance and/or replacing the MLP by an XGB machine.
 
- `doc`: images used in this README.

- `model`: contains the model classes used
  - `model_GNN.py`: contains model classes used for the activity coeficient module (not relevant for the present module).
  - `model_GNN_water.py`: contains model classes used for aqueous solubility prediction.

- `notebooks`: contains the jupyter notebooks used.
  - `add_molecular_size_and_polarity.ipynb`: adds columns of molecular weight and TPSA to the solute and solvent lists datasets.
  - `compatibilize_water_solubility_with_rittig_architecture.ipynb`: adds columns of solute_id and solvent_id to training and testing datasets 
  - `merge_water_solubility_datasets.ipynb`: merges all four training datasets, adds canonical smiles and removes duplicates present in the resulting training/testing datasets.
  - `morgan_fingerprint_dataset_generator.ipynb`: adds columns of Morgan and MACCS fingerprints to training/testing datasets in a failed attempt of increasing model performance.

- `results_BO`:

- `results_hybrids`:

- `results_water`:

- `util_water`:
  -**atom_feat_encoding_water.py**:
  -**data_splitting_water.py**:
  -**generate_dataset_for_training.py**:

- **

