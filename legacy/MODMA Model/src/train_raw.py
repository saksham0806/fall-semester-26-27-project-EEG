import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_loader_128 import MODMADataLoader128
from models_raw import EEGNet

# Configuration
BASE_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA"
REST_DIR = os.path.join(BASE_DIR, "EEG_128channels_resting_lanzhou_2015")
ERP_DIR = os.path.join(BASE_DIR, "EEG_128channels_ERP_lanzhou_2015")
EXCEL_REST = os.path.join(REST_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
EXCEL_ERP = os.path.join(ERP_DIR, "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")

MODALITY = 'rest' # 'rest' or 'erp'
WINDOW_SIZE = 4 # seconds
TARGET_RATE = 128
EPOCHS = 20
BATCH_SIZE = 32
N_SPLITS = 5

def train_dl_model(X_train, y_train, X_val, y_val):
    """Train EEGNet"""
    # X shape: (N, Chans, Samples)
    # EEGNet expects (N, Chans, Samples, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    
    n_samples = X_train.shape[2]
    n_chans = X_train.shape[1]
    
    model = EEGNet(nb_classes=1, Chans=n_chans, Samples=n_samples)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    return acc, auc, model

def train_ml_baseline(X_train, y_train, X_val, y_val):
    """Train PCA + SVM Baseline"""
    # Flatten: (N, Chans*Samples)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # Pipeline: Scale -> PCA -> SVM
    # PCA to reduce 128*512=65k dimensions to something manageable
    # Keeping 0.95 variance or fixed components
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=50)), # Fixed for speed/memory
        ('svm', SVC(probability=True))
    ])
    
    pipeline.fit(X_train_flat, y_train)
    
    y_pred = pipeline.predict(X_val_flat)
    y_pred_prob = pipeline.predict_proba(X_val_flat)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    return acc, auc, pipeline

def main():
    print(f"Loading {MODALITY} data...")
    loader = MODMADataLoader128(REST_DIR, ERP_DIR, EXCEL_REST, EXCEL_ERP, target_rate=TARGET_RATE)
    X, y, groups = loader.create_dataset(MODALITY, window_size=WINDOW_SIZE)
    
    print(f"Data Loaded: {X.shape}, {y.shape}")
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    dl_results = {'acc': [], 'auc': []}
    ml_results = {'acc': [], 'auc': []}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 1. Deep Learning (EEGNet)
        print("Training EEGNet...")
        acc_dl, auc_dl, _ = train_dl_model(X_train, y_train, X_val, y_val)
        dl_results['acc'].append(acc_dl)
        dl_results['auc'].append(auc_dl)
        print(f"EEGNet: Acc={acc_dl:.4f}, AUC={auc_dl:.4f}")
        
        # 2. ML Baseline (PCA+SVM)
        print("Training PCA+SVM...")
        acc_ml, auc_ml, _ = train_ml_baseline(X_train, y_train, X_val, y_val)
        ml_results['acc'].append(acc_ml)
        ml_results['auc'].append(auc_ml)
        print(f"PCA+SVM: Acc={acc_ml:.4f}, AUC={auc_ml:.4f}")

    print("\n=== Final Results ===")
    print(f"EEGNet  -> Avg Acc: {np.mean(dl_results['acc']):.4f}, Avg AUC: {np.mean(dl_results['auc']):.4f}")
    print(f"PCA+SVM -> Avg Acc: {np.mean(ml_results['acc']):.4f}, Avg AUC: {np.mean(ml_results['auc']):.4f}")

if __name__ == "__main__":
    main()
