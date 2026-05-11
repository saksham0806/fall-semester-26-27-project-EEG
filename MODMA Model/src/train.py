import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import MODMADataLoader
from models import build_cnn_model

# Constants
DATA_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA/EEG_3channels_resting_lanzhou_2015"
EXCEL_FILE = os.path.join(DATA_DIR, "subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx")
SEGMENT_LENGTH = 1000 # 4 seconds @ 250Hz
BATCH_SIZE = 32
EPOCHS = 2
N_SPLITS = 5

def plot_history(history, fold_idx):
    """Plot training and validation accuracy/loss."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title(f'Fold {fold_idx} Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title(f'Fold {fold_idx} Loss')
    plt.savefig(f'results/history_fold_{fold_idx}.png')
    plt.close()

def main():
    if not os.path.exists('results'):
        os.makedirs('results')
        
    # 1. Load Data
    print("Loading Data...")
    loader = MODMADataLoader(DATA_DIR, EXCEL_FILE)
    X, y, groups = loader.load_dataset(segment_length=SEGMENT_LENGTH, overlap=0)
    
    # 2. Cross Validation
    gkf = GroupKFold(n_splits=N_SPLITS)
    
    fold_accs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\nTraining Fold {fold_idx + 1}/{N_SPLITS}")
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Verify subject independence
        train_subs = set(groups[train_idx])
        val_subs = set(groups[val_idx])
        assert len(train_subs.intersection(val_subs)) == 0, "Data Leakage detected!"
        
        print(f"Train shapes: {X_train.shape}, Val shapes: {X_val.shape}")
        
        # 3. Build Model
        model = build_cnn_model(input_shape=(SEGMENT_LENGTH, 3))
        
        # 4. Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1
        )
        
        # 5. Evaluate and Plot
        plot_history(history, fold_idx + 1)
        
        scores = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold_idx + 1} Accuracy: {scores[1]:.4f}")
        fold_accs.append(scores[1])
        
        # Confusion Matrix
        y_pred = (model.predict(X_val) > 0.5).astype("int32")
        cm = confusion_matrix(y_val, y_pred)
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HC', 'MDD'], yticklabels=['HC', 'MDD'])
        plt.title(f'Confusion Matrix Fold {fold_idx + 1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'results/cm_fold_{fold_idx + 1}.png')
        plt.close()
        
    print(f"\nAverage Accuracy: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

if __name__ == "__main__":
    main()
