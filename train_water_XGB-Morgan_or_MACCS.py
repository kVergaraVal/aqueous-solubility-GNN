import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.model_selection import StratifiedKFold
import xgboost
import optuna
import torch

FINGERPRINT_TYPE = 'MACCS'

def data_split(dataframe, n_splits=5, seed=2025):
    train_indices_splits = []
    val_indices_splits = []
    
    dataframe_size = len(dataframe)
    all_ind = np.arange(dataframe_size)
    hybrid_class = dataframe['hybrid_class']
    kf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    for train_indices, valid_indices in kf.split(all_ind,hybrid_class):
        train_indices_splits.append(train_indices)
        val_indices_splits.append(valid_indices)
    return train_indices_splits, val_indices_splits

seed = 2025

def train_and_evaluate(trial, cv_splits, device, fingerprint_type, n_estimators, eta, max_depth, n_jobs):

    if fingerprint_type == 'Morgan':
        dataset = pd.read_csv('C:/Users/kverg/GDI-NN/data/solubility_water/MORGAN_train_water_with_hybrid_classes.csv')

    elif fingerprint_type == 'MACCS':
        dataset = pd.read_csv('C:/Users/kverg/GDI-NN/data/solubility_water/MACCS_train_water_with_hybrid_classes.csv')

    X = dataset[dataset.columns[7:]]
    y = dataset['LogS']

    train_indices_splits, val_indices_splits = data_split(dataset, n_splits=cv_splits, seed=seed)

    cv_index = 0
    val_losses = []
    for train_indices, val_indices in zip(train_indices_splits, val_indices_splits):
        X_train = X.iloc[train_indices].to_numpy()
        X_val = X.iloc[val_indices].to_numpy()
        y_train = y.iloc[train_indices].to_numpy()
        y_val = y.iloc[val_indices].to_numpy()

        model_xgb = xgboost.XGBRegressor(n_estimators=n_estimators,
                                         max_depth=max_depth,
                                         learning_rate=eta,
                                         n_jobs=n_jobs,
                                         device=device)
    
        # Training
        model_xgb.fit(X_train, y_train)
        y_train_pred = model_xgb.predict(X_train)
        xgb_mse_train = ((y_train - y_train_pred)**2).mean()
        print(f'Training MSE for split {cv_index} in trial {trial.number} is: {xgb_mse_train}')

        # Validation
        y_val_pred = model_xgb.predict(X_val)
        xgb_mse_val = ((y_val - y_val_pred)**2).mean()
        val_losses.append(xgb_mse_val)
        print(f'Validation MSE for split {cv_index} in trial {trial.number} is: {xgb_mse_val}')

        cv_index += 1

    val_losses_mean = np.mean(val_losses)

    print(f'Validation MSE mean for trial {trial.number} is: {val_losses_mean}')
    return val_losses_mean


def objective(trial):
    #hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 20, 500)
    eta = trial.suggest_float("eta", 0.01, 0.2)
    max_depth = trial.suggest_int("max_depth", 2, 20)

    cv_splits = 5

    # Assign GPU for this trial
    device = torch.device('cpu')
    n_jobs = 4

    # Fingerprint type
    fingerprint_type = FINGERPRINT_TYPE # "Morgan" or MACCS

    loss = train_and_evaluate(trial, cv_splits, device, fingerprint_type, n_estimators, eta, max_depth, n_jobs)

    return loss

# Set up Optuna study with SQLite storage for checkpointing
sampler = optuna.samplers.TPESampler()

study_name = 'BO_' + FINGERPRINT_TYPE + '_XGB'
storage = 'sqlite:///TRIALS_BO_' + FINGERPRINT_TYPE + '_XGB.db'
study = optuna.create_study(
    study_name=study_name,
    direction="minimize",
    storage=storage,  # Checkpointing database
    load_if_exists=True,
    sampler=sampler
)

study.optimize(objective, n_trials=100, n_jobs=1)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
