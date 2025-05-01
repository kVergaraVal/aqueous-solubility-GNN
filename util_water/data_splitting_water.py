import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def data_split_standard(dataset, n_splits=5, seed=2021):
    train_indices_splits = []
    val_indices_splits = []
    
    dataset_size = len(dataset)
    all_ind = np.arange(dataset_size)
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_indices, valid_indices in kf.split(all_ind):
        train_indices_splits.append(train_indices)
        val_indices_splits.append(valid_indices)
    return train_indices_splits, val_indices_splits

def data_split_stratified_hybrid(dataset, n_splits=5, seed=2021):
    train_indices_splits = []
    val_indices_splits = []
    
    dataset_size = len(dataset)
    all_ind = np.arange(dataset_size)
    hybrid_class = dataset.dataset['hybrid_class']
    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_indices, valid_indices in kf.split(all_ind,hybrid_class):
        train_indices_splits.append(train_indices)
        val_indices_splits.append(valid_indices)
    return train_indices_splits, val_indices_splits