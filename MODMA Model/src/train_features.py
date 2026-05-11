import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from data_loader_128 import MODMADataLoader128
from features import extract_features

# Configuration
BASE_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA"
REST_DIR = os.path.join(BASE_DIR, "EEG_128channels_resting_lanzhou_2015")
ERP_DIR = os.path.join(BASE_DIR, "EEG_128channels_ERP_lanzhou_2015")
EXCEL_REST = os.path.join(REST_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
EXCEL_ERP = os.path.join(ERP_DIR, "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")

MODALITY = 'rest'
WINDOW_SIZE = 4
TARGET_RATE = 128
N_SPLITS = 5

def build_dnn_model(input_dim):
    """Simple MLP for feature classification"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print(f"Loading {MODALITY} data...")
    loader = MODMADataLoader128(REST_DIR, ERP_DIR, EXCEL_REST, EXCEL_ERP, target_rate=TARGET_RATE)
    X, y, groups = loader.create_dataset(MODALITY, window_size=WINDOW_SIZE)
    
    print(f"Data Loaded: {X.shape}, {y.shape}")
    
    print("Extracting Features (PSD)...")
    # This might take time
    X_feats = extract_features(X, fs=TARGET_RATE, feature_type='psd')
    print(f"Features Extracted: {X_feats.shape}")
    
    # Check for NaNs
    if np.isnan(X_feats).any():
        print("Warning: NaNs found in features. Replacing with 0.")
        X_feats = np.nan_to_num(X_feats)

    gkf = GroupKFold(n_splits=N_SPLITS)
    
    results = {
        'svm': {'acc': [], 'auc': []},
        'rf': {'acc': [], 'auc': []},
        'xgb': {'acc': [], 'auc': []},
        'dnn': {'acc': [], 'auc': []}
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_feats, y, groups)):
        print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
        X_train, y_train = X_feats[train_idx], y[train_idx]
        X_val, y_val = X_feats[val_idx], y[val_idx]
        
        # Scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 1. SVM
        clf = SVC(probability=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        results['svm']['acc'].append(accuracy_score(y_val, y_pred))
        results['svm']['auc'].append(roc_auc_score(y_val, y_prob))
        print(f"SVM: {results['svm']['acc'][-1]:.4f}")
        
        # 2. RF
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        results['rf']['acc'].append(accuracy_score(y_val, y_pred))
        results['rf']['auc'].append(roc_auc_score(y_val, y_prob))
        print(f"RF: {results['rf']['acc'][-1]:.4f}")
        
        # 3. XGBoost
        clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]
        results['xgb']['acc'].append(accuracy_score(y_val, y_pred))
        results['xgb']['auc'].append(roc_auc_score(y_val, y_prob))
        print(f"XGB: {results['xgb']['acc'][-1]:.4f}")
        
        # 4. DNN
        model = build_dnn_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        y_prob = model.predict(X_val, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        results['dnn']['acc'].append(accuracy_score(y_val, y_pred))
        results['dnn']['auc'].append(roc_auc_score(y_val, y_prob))
        print(f"DNN: {results['dnn']['acc'][-1]:.4f}")

    print("\n=== Final Results (Avg Acc) ===")
    for model_name, metrics in results.items():
        print(f"{model_name.upper()}: {np.mean(metrics['acc']):.4f} (AUC: {np.mean(metrics['auc']):.4f})")

if __name__ == "__main__":
    main()
