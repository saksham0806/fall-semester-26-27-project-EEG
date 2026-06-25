import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_loader_128 import MODMADataLoader128
from models_raw import EEGNet, TSception

# Configuration
BASE_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA"
REST_DIR = os.path.join(BASE_DIR, "EEG_128channels_resting_lanzhou_2015")
ERP_DIR = os.path.join(BASE_DIR, "EEG_128channels_ERP_lanzhou_2015")
EXCEL_REST = os.path.join(REST_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
EXCEL_ERP = os.path.join(ERP_DIR, "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")

MODALITY = 'rest' 
WINDOW_SIZE = 4 
STRIDE = 2 # 50% overlap (Data Augmentation)
TARGET_RATE = 128
EPOCHS = 50
BATCH_SIZE = 32
N_SPLITS = 5
MODEL_TYPE = 'EEGNet' # 'EEGNet' or 'TSception'

def plot_history(history, fold_idx, model_name):
    if not os.path.exists('results_phase3'):
        os.makedirs('results_phase3')
        
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title(f'{model_name} Fold {fold_idx} Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'{model_name} Fold {fold_idx} Loss')
    plt.legend()
    
    plt.savefig(f'results_phase3/{model_name}_fold_{fold_idx}_history.png')
    plt.close()

def train_model(X_train, y_train, X_val, y_val, model_type=MODEL_TYPE):
    # X shape: (N, Chans, Samples)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    
    n_samples = X_train.shape[2]
    n_chans = X_train.shape[1]
    
    if model_type == 'EEGNet':
        model = EEGNet(nb_classes=1, Chans=n_chans, Samples=n_samples, dropoutRate=0.75)
    elif model_type == 'TSception':
        model = TSception(nb_classes=1, Chans=n_chans, Samples=n_samples, sampling_rate=TARGET_RATE)
        
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_prob)
    
    return acc, auc, y_pred, history

def main():
    print(f"Loading {MODALITY} data with stride={STRIDE}...")
    loader = MODMADataLoader128(REST_DIR, ERP_DIR, EXCEL_REST, EXCEL_ERP, target_rate=TARGET_RATE)
    X, y, groups = loader.create_dataset(MODALITY, window_size=WINDOW_SIZE, stride=STRIDE)
    
    print(f"Data Loaded: {X.shape}, {y.shape}")
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        acc, auc, y_pred, history = train_model(X_train, y_train, X_val, y_val, MODEL_TYPE)
        results.append(acc)
        
        print(f"{MODEL_TYPE} Fold {fold+1}: Acc={acc:.4f}, AUC={auc:.4f}")
        plot_history(history, fold+1, MODEL_TYPE)
        
        # Save CM
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f'Fold {fold+1} CM')
        plt.savefig(f'results_phase3/{MODEL_TYPE}_fold_{fold+1}_cm.png')
        plt.close()

    print(f"\n=== Final Results ({MODEL_TYPE}) ===")
    print(f"Avg Accuracy: {np.mean(results):.4f} (+/- {np.std(results):.4f})")

if __name__ == "__main__":
    main()
