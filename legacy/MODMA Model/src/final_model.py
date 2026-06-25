import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Import local modules
from data_loader_128 import MODMADataLoader128
from models_raw import EEGNet, TSception

# Default Configuration
BASE_DIR = "/run/media/Saksham/Saksham/Code/Projects/stress_analysis_research/datasets/MODMA"
REST_DIR = os.path.join(BASE_DIR, "EEG_128channels_resting_lanzhou_2015")
ERP_DIR = os.path.join(BASE_DIR, "EEG_128channels_ERP_lanzhou_2015")
EXCEL_REST = os.path.join(REST_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
EXCEL_ERP = os.path.join(ERP_DIR, "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Final Depression Detection Models")
    parser.add_argument('--model', type=str, default='EEGNet', choices=['EEGNet', 'TSception'], help='Model architecture to use')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation (overlapping windows)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the model')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    return parser.parse_args()

def plot_history(history, fold_idx, model_name, save_dir):
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
    
    plt.savefig(os.path.join(save_dir, f'{model_name}_fold_{fold_idx}_history.png'))
    plt.close()

def train_fold(model, X_train, y_train, X_val, y_val, fold_idx, args, save_dir):
    # Callbacks
    checkpoint_path = os.path.join(save_dir, f'{args.model}_fold_{fold_idx}_best.keras')
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=args.patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=0)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Load best weights
    # Note: restore_best_weights=True in EarlyStopping usually handles this, 
    # but Checkpoint ensures we have the file on disk.
    model.load_weights(checkpoint_path)
    
    return model, history

def main():
    args = parse_args()
    
    # Setup directories
    save_dir = f"results_final_{args.model}"
    if args.augment:
        save_dir += "_aug"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"=== Starting Training: {args.model} ===")
    print(f"Epochs: {args.epochs}, Batch: {args.batch_size}, Augment: {args.augment}, Dropout: {args.dropout}")
    
    # Load Data
    stride = 2 if args.augment else None # 50% overlap if augment is True
    print(f"Loading data with stride={stride}...")
    
    loader = MODMADataLoader128(REST_DIR, ERP_DIR, EXCEL_REST, EXCEL_ERP, target_rate=128)
    X, y, groups = loader.create_dataset('rest', window_size=4, stride=stride)
    
    print(f"Data Loaded: {X.shape}, labels: {y.shape}")
    
    # 5-Fold CV
    gkf = GroupKFold(n_splits=5)
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Add channel dim
        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        
        n_samples = X_train.shape[2]
        n_chans = X_train.shape[1]
        
        # Build Model
        if args.model == 'EEGNet':
            model = EEGNet(nb_classes=1, Chans=n_chans, Samples=n_samples, dropoutRate=args.dropout)
        elif args.model == 'TSception':
            # TSception doesn't strictly take a dropout arg in our current implementation, 
            # but we hardcoded it. For rigorous fine-tuning we might want to pass it.
            # Assuming TSception signature update or using default.
            model = TSception(nb_classes=1, Chans=n_chans, Samples=n_samples, sampling_rate=128, dropout_rate=args.dropout)
        else:
            raise ValueError(f"Unknown model: {args.model}")
            
        # Train
        model, history = train_fold(model, X_train, y_train, X_val, y_val, fold+1, args, save_dir)
        
        # Evaluate
        y_pred_prob = model.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        
        # Plot & Save
        plot_history(history, fold+1, args.model, save_dir)
        
        cm = confusion_matrix(y_val, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Fold {fold+1} Confusion Matrix')
        plt.savefig(os.path.join(save_dir, f'{args.model}_fold_{fold+1}_cm.png'))
        plt.close()
        
    print("\n=== Final Results ===")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

if __name__ == "__main__":
    main()
