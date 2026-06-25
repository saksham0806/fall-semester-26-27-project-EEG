import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models

# Configurations
DATA_DIR = "/run/media/Saksham/Saksham/Code/Projects/fall-semester-26-27-project-EEG/datasets/MODMA/EEG_3channels_resting_lanzhou_2015"
EXCEL_FILE = os.path.join(DATA_DIR, "subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx")
OUTPUT_DIR = "/run/media/Saksham/Saksham/Code/Projects/fall-semester-26-27-project-EEG/modma Model new"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")

SEGMENT_LENGTH = 500  # 4 seconds @ 125Hz (downsampled from 250Hz)
SAMPLING_RATE = 125
EPOCHS = 10
BATCH_SIZE = 128
N_SPLITS = 5

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 45)
}

# 1. DataLoader
class MODMA3ChannelLoader:
    def __init__(self, dataset_path, excel_path):
        self.dataset_path = dataset_path
        self.excel_path = excel_path
        self.subjects_df = None
        self.file_map = {}
        
        self._load_metadata()
        self._map_files()

    def _load_metadata(self):
        if not os.path.exists(self.excel_path):
            raise FileNotFoundError(f"Excel file not found at {self.excel_path}")
        self.subjects_df = pd.read_excel(self.excel_path)
        self.subjects_df.columns = self.subjects_df.columns.str.strip()
        if 'subject id' in self.subjects_df.columns:
            self.subjects_df['subject id'] = self.subjects_df['subject id'].astype(str)
        print(f"Loaded metadata for {len(self.subjects_df)} subjects.")

    def _map_files(self):
        files = glob.glob(os.path.join(self.dataset_path, "*_still.txt"))
        for file_path in files:
            basename = os.path.basename(file_path)
            file_id = basename.split('_')[0]
            self.file_map[file_id] = file_path
        print(f"Mapped {len(self.file_map)} data files.")

    def get_subject_data(self, subject_id):
        subject_id_str = str(subject_id)
        if subject_id_str in self.file_map:
            path = self.file_map[subject_id_str]
        elif f"0{subject_id_str}" in self.file_map:
            path = self.file_map[f"0{subject_id_str}"]
        else:
            return None
        
        try:
            data = pd.read_csv(path, sep=r'\s+', header=None)
            # Downsample from 250Hz to 125Hz by taking every 2nd sample
            return data.values[::2]  # Shape: (n_samples, 3)
        except Exception as e:
            print(f"Error loading file for {subject_id}: {e}")
            return None

    def load_dataset(self, segment_length=500):
        X_list = []
        y_list = []
        groups_list = []
        
        for _, row in self.subjects_df.iterrows():
            sub_id = row['subject id']
            label_str = row['type']
            
            if label_str not in ['MDD', 'HC']:
                continue
                
            label = 1 if label_str == 'MDD' else 0
            raw_data = self.get_subject_data(sub_id)
            if raw_data is None:
                continue
                
            n_samples = raw_data.shape[0]
            step = segment_length
            
            for start in range(0, n_samples - segment_length + 1, step):
                segment = raw_data[start:start+segment_length, :]
                
                # Check for flat line or NaN artifacts
                if np.isnan(segment).any() or np.std(segment) < 1e-4:
                    continue
                
                X_list.append(segment)
                y_list.append(label)
                groups_list.append(sub_id)
                
        X = np.array(X_list)
        y = np.array(y_list)
        groups = np.array(groups_list)
        
        print(f"Created dataset: X shape={X.shape}, y shape={y.shape}, subjects count={len(np.unique(groups))}")
        return X, y, groups

# 2. PSD Feature Extraction
def extract_psd_features(X, fs=125):
    n_segments, segment_length, n_channels = X.shape
    features_list = []
    
    for i in range(n_segments):
        seg = X[i].T  # Shape: (n_channels, segment_length)
        freqs, psd = welch(seg, fs=fs, nperseg=fs)
        
        feat = []
        for _, (low, high) in BANDS.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            band_power = np.mean(psd[:, idx_band], axis=1)  # (n_channels,)
            feat.append(band_power)
            
        features_list.append(np.concatenate(feat))
        
    return np.array(features_list)

# 3. Model Builders
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
        layers.MaxPooling1D(pool_size=4), # Compresses length 500 to 125
        
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 4. Helpers to Plot/Save
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
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HC', 'MDD'], yticklabels=['HC', 'MDD'])
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
        
    print("Step 1: Loading MODMA 3-channel resting state EEG dataset...")
    loader = MODMA3ChannelLoader(DATA_DIR, EXCEL_FILE)
    X, y, groups = loader.load_dataset(segment_length=SEGMENT_LENGTH)
    
    print("\nStep 2: Normalizing raw EEG signals (Segment-wise Z-score)...")
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        mean = np.mean(X[i], axis=0)
        std = np.std(X[i], axis=0) + 1e-8
        X_normalized[i] = (X[i] - mean) / std
        
    print("Step 3: Extracting PSD features for SVM/Random Forest...")
    X_feats = extract_psd_features(X, fs=SAMPLING_RATE)
    if np.isnan(X_feats).any():
        X_feats = np.nan_to_num(X_feats)
        
    print(f"PSD Feature shape: {X_feats.shape}")
    
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    metrics = {
        'SVM': {'acc': [], 'prec': [], 'rec': [], 'auc': []},
        'Random Forest': {'acc': [], 'prec': [], 'rec': [], 'auc': []},
        '1D CNN': {'acc': [], 'prec': [], 'rec': [], 'auc': []},
        'LSTM': {'acc': [], 'prec': [], 'rec': [], 'auc': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n==================== FOLD {fold+1}/{N_SPLITS} ====================")
        
        # Prepare CV data for SVM & RF
        X_train_f, X_val_f = X_feats[train_idx], X_feats[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_f = scaler.fit_transform(X_train_f)
        X_val_f = scaler.transform(X_val_f)
        
        # Prepare CV data for CNN & LSTM
        X_train_raw, X_val_raw = X_normalized[train_idx], X_normalized[val_idx]
        
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
        cnn = build_1d_cnn(input_shape=(SEGMENT_LENGTH, 3))
        
        # Early stopping with patience=3
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
        lstm = build_lstm(input_shape=(SEGMENT_LENGTH, 3))
        
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
    results_summary = "# MODMA Model Training Results (3-Channel EEG)\n\n"
    results_summary += "This file contains the cross-validation performance metrics of SVM, Random Forest, 1D CNN, and LSTM trained on the MODMA dataset (3-channel resting state EEG downsampled to 125Hz).\n\n"
    results_summary += "| Model | Accuracy (Mean ± SD) | Precision (Mean) | Recall (Mean) | ROC-AUC (Mean) |\n"
    results_summary += "|---|---|---|---|---|\n"
    
    for model_name, data in metrics.items():
        acc_mean, acc_std = np.mean(data['acc']), np.std(data['acc'])
        prec_mean = np.mean(data['prec'])
        rec_mean = np.mean(data['rec'])
        auc_mean = np.mean(data['auc'])
        
        results_summary += f"| {model_name} | {acc_mean:.4f} ± {acc_std:.4f} | {prec_mean:.4f} | {rec_mean:.4f} | {auc_mean:.4f} |\n"
        
    with open(os.path.join(OUTPUT_DIR, "modma_results.md"), "w") as f:
        f.write(results_summary)
        
    with open(os.path.join(METRICS_DIR, "modma_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nResults and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
