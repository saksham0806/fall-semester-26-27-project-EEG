# Project Instructions: Depression Analysis Model

This project contains tools to train and evaluate deep learning models (TSception and EEGNet) for depression analysis using EEG data.

## 1. Environment Setup

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` doesn't exist, here are the main packages: `tensorflow`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `openpyxl`)*

## 2. Training the Model

The main training script is `src/final_model.py`. You can train either **TSception** (recommended) or **EEGNet**.

### Train TSception (Recommended)
Acc: ~75-80%
```bash
python src/final_model.py --model TSception --epochs 50 --augment --dropout 0.6
```

### Train EEGNet
Acc: ~60-68%
```bash
python src/final_model.py --model EEGNet --epochs 50 --augment --dropout 0.5
```

### Key Arguments
*   `--model`: Model architecture (`TSception` or `EEGNet`).
*   `--epochs`: Number of training epochs (default: 50).
*   `--batch_size`: Batch size (default: 32).
*   `--augment`: Enable data augmentation (50% overlap).
*   `--dropout`: Dropout rate (TSception uses 0.6, EEGNet uses 0.5).

**Output:**
*   Trained model weights will be saved in `results_final_<MODEL>_aug/` as `.keras` files (e.g., `TSception_fold_3_best.keras`).
*   Training history plots and confusion matrices will also be saved there.

## 3. Running Inference on External Data

To use a trained model (`.keras` file) on new, unseen EEG data, use the `src/inference.py` script.

### Input Data Format
The external data must be a NumPy array with the shape:
`(n_samples, 128, 512, 1)`
*   **128**: Number of Channels.
*   **512**: Time points (4 seconds @ 128Hz).
*   **1**: Channel dimension.

### Usage Example

```bash
python src/inference.py --model_path results_final_TSception_aug/TSception_fold_3_best.keras --data_path path/to/your/new_data.npy
```

### Python Code Snippet
If you want to run inference inside your own Python script:

```python
import numpy as np
import tensorflow as tf
from src.models_raw import TSception # or EEGNet

# 1. Load the Model Architecture
# Note: You must re-instantiate the model with the same parameters used during training
model = TSception(nb_classes=1, Chans=128, Samples=512, sampling_rate=128, dropout_rate=0.6)

# 2. Load Weights
model.load_weights('results_final_TSception_aug/TSception_fold_3_best.keras')

# 3. Load Data
# Data should be shape (N, 128, 512, 1)
new_data = np.load('path/to/data.npy') 

# 4. Predict
predictions = model.predict(new_data)
predicted_labels = (predictions > 0.5).astype(int)

print(predicted_labels)
```
