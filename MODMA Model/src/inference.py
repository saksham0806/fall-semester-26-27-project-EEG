import os
import argparse
import numpy as np
import tensorflow as tf
from models_raw import EEGNet, TSception

def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference on External Data")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .keras model file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the .npy data file (shape: N, 128, 512, 1)')
    parser.add_argument('--model_type', type=str, default='TSception', choices=['EEGNet', 'TSception'], help='Model architecture')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    print(f"Loading data from {args.data_path}...")
    data = np.load(args.data_path)
    
    # Data Validation
    if data.ndim == 3:
        # Add channel dim if missing
        print("Adding channel dimension...")
        data = data[..., np.newaxis]
        
    if data.shape[1:] != (128, 512, 1):
        raise ValueError(f"Expected data shape (N, 128, 512, 1), but got {data.shape}")
        
    print(f"Data loaded: {data.shape}")
    
    # 2. Build Model
    # We need to rebuild the architecture to load weights
    n_chans = 128
    n_samples = 512
    
    if args.model_type == 'EEGNet':
        model = EEGNet(nb_classes=1, Chans=n_chans, Samples=n_samples)
    elif args.model_type == 'TSception':
        model = TSception(nb_classes=1, Chans=n_chans, Samples=n_samples, sampling_rate=128)
        
    # 3. Load Weights
    print(f"Loading weights from {args.model_path}...")
    model.load_weights(args.model_path)
    
    # 4. Predict
    print("Running prediction...")
    probs = model.predict(data)
    preds = (probs > 0.5).astype(int)
    
    # 5. Output
    print("\n=== Inference Results ===")
    for i in range(len(preds)):
        status = "Depressed" if preds[i][0] == 1 else "Healthy"
        print(f"Sample {i+1}: {status} (Probability: {probs[i][0]:.4f})")
        
if __name__ == "__main__":
    main()
