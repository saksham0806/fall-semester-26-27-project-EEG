import os
import glob
import pickle
import json
import gc
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

# Configurations
DATA_DIR = "/run/media/Saksham/Saksham/Code/Projects/fall-semester-26-27-project-EEG/datasets/WESAD"
OUTPUT_DIR = "/run/media/Saksham/Saksham/Code/Projects/fall-semester-26-27-project-EEG/WESAD model"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

WINDOW_SIZE = 500  # 5 seconds @ 100Hz
STRIDE = 200      # 2 seconds stride (60% overlap)
EPOCHS = 10
BATCH_SIZE = 256
N_SPLITS = 5

# WESAD Chest physiological channel names: ACC (3), ECG (1), EMG (1), EDA (1), Temp (1), Resp (1)
CHANNELS = ['ACC_x', 'ACC_y', 'ACC_z', 'ECG', 'EMG', 'EDA', 'Temp', 'Resp']

# 1. Memory-Efficient Subject Loader & Window Segmenter
def load_and_process_subject(subject_dir):
    subject_name = os.path.basename(subject_dir)
    pkl_file = os.path.join(subject_dir, f"{subject_name}.pkl")
    
    if not os.path.exists(pkl_file):
        print(f"Warning: Pickle file not found for {subject_name}")
        return None, None, None, None
        
    print(f"Loading and processing {subject_name}...")
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error reading {pkl_file}: {e}")
        return None, None, None, None
        
    # Extract chest data
    chest_data = data['signal']['chest']
    labels = data['label']
    
    acc = chest_data['ACC']
    ecg = chest_data['ECG']
    emg = chest_data['EMG']
    eda = chest_data['EDA']
    temp = chest_data['Temp']
    resp = chest_data['Resp']
    
    X_chest = np.hstack([acc, ecg, emg, eda, temp, resp]) # Shape: (N, 8)
    
    # Separate Baseline (label 1) and Stress (label 2)
    # We map Baseline -> 0, Stress -> 1
    base_idx = np.where(labels == 1)[0]
    stress_idx = np.where(labels == 2)[0]
    
    X_base = X_chest[base_idx]
    X_stress = X_chest[stress_idx]
    
    # Downsample by factor of 7 (from 700Hz to 100Hz)
    X_base_down = X_base[::7]
    X_stress_down = X_stress[::7]
    
    raw_segments = []
    features = []
    labels_list = []
    
    def window_block(block_data, block_label):
        n_samples = block_data.shape[0]
        for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            seg = block_data[start:start+WINDOW_SIZE]
            
            # Segment Z-score Standardization
            mean = np.mean(seg, axis=0)
            std = np.std(seg, axis=0) + 1e-8
            seg_norm = (seg - mean) / std
            
            # Extract features for SVM/RF
            rms = np.sqrt(np.mean(seg**2, axis=0))
            feat = np.hstack([mean, std, np.min(seg, axis=0), np.max(seg, axis=0), rms])
            
            raw_segments.append(seg_norm)
            features.append(feat)
            labels_list.append(block_label)
            
    window_block(X_base_down, 0)
    window_block(X_stress_down, 1)
    
    # Free memory
    del data, chest_data, labels, X_chest, X_base, X_stress, X_base_down, X_stress_down
    gc.collect()
    
    if not raw_segments:
        return None, None, None, None
        
    return np.array(raw_segments), np.array(features), np.array(labels_list), subject_name

# 2. Model Builders
def build_1d_cnn(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Conv1D(128, kernel_size=5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_shape):
    model = models.Sequential([
        layers.Conv1D(16, kernel_size=5, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(pool_size=5), # Compresses 500 length to 100
        
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 3. Helpers to Plot/Save
def save_history_plot(history, model_name, fold, path):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b-', label='Train Acc')
    plt.plot(epochs, val_acc, 'r-', label='Val Acc')
    plt.title(f'{model_name} Fold {fold} Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.title(f'{model_name} Fold {fold} Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{model_name}_fold{fold}_history.png"))
    plt.close()

def save_confusion_matrix(y_true, y_pred, model_name, fold, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Baseline', 'Stress'], yticklabels=['Baseline', 'Stress'])
    plt.title(f'{model_name} Fold {fold} Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(path, f"{model_name}_fold{fold}_cm.png"))
    plt.close()

# Main Training Pipeline
def main():
    # Setup folders
    for folder in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR, METRICS_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder)
        
    subject_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "S*")))
    
    X_raw_all = []
    X_feat_all = []
    y_all = []
    groups_all = []
    
    print("Step 1: Sequentially loading WESAD subjects physiological data...")
    for sub_dir in subject_dirs:
        sub_name = os.path.basename(sub_dir)
        if sub_name == "S12":
            continue
            
        raw_seg, feat, labels, name = load_and_process_subject(sub_dir)
        if raw_seg is not None:
            X_raw_all.append(raw_seg)
            X_feat_all.append(feat)
            y_all.append(labels)
            groups_all.extend([name] * len(labels))
            print(f"Processed {name}: segments={len(labels)}")
            
    X_raw = np.vstack(X_raw_all)
    X_feats = np.vstack(X_feat_all)
    y = np.concatenate(y_all)
    groups = np.array(groups_all)
    
    print(f"\nTotal Dataset Created:")
    print(f"Raw shape: {X_raw.shape}")
    print(f"Feature shape: {X_feats.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique subjects: {np.unique(groups)}")
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    metrics = {
        'SVM': {'acc': [], 'prec': [], 'rec': [], 'auc': []},
        'Random Forest': {'acc': [], 'prec': [], 'rec': [], 'auc': []},
        '1D CNN': {'acc': [], 'prec': [], 'rec': [], 'auc': []},
        'LSTM': {'acc': [], 'prec': [], 'rec': [], 'auc': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups)):
        print(f"\n==================== FOLD {fold+1}/{N_SPLITS} ====================")
        
        # Prepare CV data for ML
        X_train_f, X_val_f = X_feats[train_idx], X_feats[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_f = scaler.fit_transform(X_train_f)
        X_val_f = scaler.transform(X_val_f)
        
        # Prepare CV data for DL
        X_train_raw, X_val_raw = X_raw[train_idx], X_raw[val_idx]
        
        # --- A. Train SVM ---
        print("Training SVM...")
        svm_model = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
        svm_model.fit(X_train_f, y_train)
        
        y_pred_svm = svm_model.predict(X_val_f)
        y_prob_svm = svm_model.predict_proba(X_val_f)[:, 1]
        
        metrics['SVM']['acc'].append(accuracy_score(y_val, y_pred_svm))
        metrics['SVM']['prec'].append(precision_score(y_val, y_pred_svm, zero_division=0))
        metrics['SVM']['rec'].append(recall_score(y_val, y_pred_svm, zero_division=0))
        metrics['SVM']['auc'].append(roc_auc_score(y_val, y_prob_svm))
        save_confusion_matrix(y_val, y_pred_svm, 'SVM', fold+1, PLOTS_DIR)
        joblib.dump(svm_model, os.path.join(MODELS_DIR, f"svm_fold{fold+1}.joblib"))
        
        # --- B. Train Random Forest ---
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_f, y_train)
        
        y_pred_rf = rf_model.predict(X_val_f)
        y_prob_rf = rf_model.predict_proba(X_val_f)[:, 1]
        
        metrics['Random Forest']['acc'].append(accuracy_score(y_val, y_pred_rf))
        metrics['Random Forest']['prec'].append(precision_score(y_val, y_pred_rf, zero_division=0))
        metrics['Random Forest']['rec'].append(recall_score(y_val, y_pred_rf, zero_division=0))
        metrics['Random Forest']['auc'].append(roc_auc_score(y_val, y_prob_rf))
        save_confusion_matrix(y_val, y_pred_rf, 'RF', fold+1, PLOTS_DIR)
        joblib.dump(rf_model, os.path.join(MODELS_DIR, f"rf_fold{fold+1}.joblib"))
        
        # --- C. Train 1D CNN ---
        print("Training 1D CNN...")
        cnn = build_1d_cnn(input_shape=(WINDOW_SIZE, 8))
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]
        
        history_cnn = cnn.fit(
            X_train_raw, y_train,
            validation_data=(X_val_raw, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        y_prob_cnn = cnn.predict(X_val_raw).flatten()
        y_pred_cnn = (y_prob_cnn > 0.5).astype(int)
        
        metrics['1D CNN']['acc'].append(accuracy_score(y_val, y_pred_cnn))
        metrics['1D CNN']['prec'].append(precision_score(y_val, y_pred_cnn, zero_division=0))
        metrics['1D CNN']['rec'].append(recall_score(y_val, y_pred_cnn, zero_division=0))
        metrics['1D CNN']['auc'].append(roc_auc_score(y_val, y_prob_cnn))
        
        save_history_plot(history_cnn, '1D_CNN', fold+1, PLOTS_DIR)
        save_confusion_matrix(y_val, y_pred_cnn, '1D_CNN', fold+1, PLOTS_DIR)
        cnn.save(os.path.join(MODELS_DIR, f"cnn_fold{fold+1}.keras"))
        
        # --- D. Train LSTM ---
        print("Training LSTM...")
        lstm = build_lstm(input_shape=(WINDOW_SIZE, 8))
        
        history_lstm = lstm.fit(
            X_train_raw, y_train,
            validation_data=(X_val_raw, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        y_prob_lstm = lstm.predict(X_val_raw).flatten()
        y_pred_lstm = (y_prob_lstm > 0.5).astype(int)
        
        metrics['LSTM']['acc'].append(accuracy_score(y_val, y_pred_lstm))
        metrics['LSTM']['prec'].append(precision_score(y_val, y_pred_lstm, zero_division=0))
        metrics['LSTM']['rec'].append(recall_score(y_val, y_pred_lstm, zero_division=0))
        metrics['LSTM']['auc'].append(roc_auc_score(y_val, y_prob_lstm))
        
        save_history_plot(history_lstm, 'LSTM', fold+1, PLOTS_DIR)
        save_confusion_matrix(y_val, y_pred_lstm, 'LSTM', fold+1, PLOTS_DIR)
        lstm.save(os.path.join(MODELS_DIR, f"lstm_fold{fold+1}.keras"))
        
        print(f"\nFold {fold+1} Summary:")
        print(f"SVM Acc: {metrics['SVM']['acc'][-1]:.4f} | RF Acc: {metrics['Random Forest']['acc'][-1]:.4f}")
        print(f"CNN Acc: {metrics['1D CNN']['acc'][-1]:.4f} | LSTM Acc: {metrics['LSTM']['acc'][-1]:.4f}")

    # Generate final results
    print("\n==================== TRAINING COMPLETED ====================")
    results_summary = "# WESAD Model Training Results (Physiological Chest Data)\n\n"
    results_summary += "This file contains the cross-validation performance metrics of SVM, Random Forest, 1D CNN, and LSTM trained on the WESAD dataset (Chest ACC, ECG, EMG, EDA, Temp, and Resp downsampled to 100Hz).\n\n"
    results_summary += "| Model | Accuracy (Mean ± SD) | Precision (Mean) | Recall (Mean) | ROC-AUC (Mean) |\n"
    results_summary += "|---|---|---|---|---|\n"
    
    for model_name, data in metrics.items():
        acc_mean, acc_std = np.mean(data['acc']), np.std(data['acc'])
        prec_mean = np.mean(data['prec'])
        rec_mean = np.mean(data['rec'])
        auc_mean = np.mean(data['auc'])
        
        results_summary += f"| {model_name} | {acc_mean:.4f} ± {acc_std:.4f} | {prec_mean:.4f} | {rec_mean:.4f} | {auc_mean:.4f} |\n"
        
    with open(os.path.join(OUTPUT_DIR, "wesad_results.md"), "w") as f:
        f.write(results_summary)
        
    with open(os.path.join(METRICS_DIR, "wesad_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nResults and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
