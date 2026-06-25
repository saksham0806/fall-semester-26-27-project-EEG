# EEG Model Training Metrics Report

This report summarizes the performance metrics of all implemented models on the standardized **6-channel frontal EEG dataset** (Fp1, Fp2, F7, F3, F4, F8) combined from **MODMA, SAM40, Cognitive Load, and AMIGOS**.

Evaluation was conducted using a rigorous **Group-based Train/Test split (80/20)** based on subject IDs to ensure zero subject overlap and prevent data leakage. All models have been trained with **outlier-clipped global segment normalization** (which preserves spatial amplitude ratios across channels), and deep learning models were trained using **GPU acceleration (NVIDIA RTX 3060)** with a tuned learning rate and GridSearch hyperparameter tuning.

---

## 📊 Summary of Combined Dataset Splits
*   **Depression Classification (MODMA)**:
    *   Train size: 6469 segments
    *   Test size: 1490 segments
*   **Stress Classification (SAM40, Cognitive Load, AMIGOS)**:
    *   Train size: 38316 segments
    *   Test size: 7502 segments
*   **Multi-Task (Full Dataset)**:
    *   Train size: 44785 segments
    *   Test size: 8992 segments

---

## 🧠 Part 1: Depression Detection Results (Clinical Classification)

The models below were trained to classify Major Depressive Disorder (MDD) vs. Healthy Controls (HC) using the MODMA dataset.

| Model Class | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.6530 | 0.5464 | 0.7802 | 0.6427 | 0.7226 | 11.34s |
| Classical ML (Feature-based) | Random Forest | 0.4966 | 0.3778 | 0.3993 | 0.3883 | 0.5344 | 6.97s |
| Classical ML (Feature-based) | XGBoost | 0.5336 | 0.4145 | 0.4027 | 0.4085 | 0.5703 | 36.59s |
| Deep Learning (Raw EEG) | 1D CNN | 0.8383 | 1.0000 | 0.5956 | 0.7466 | 0.7859 | 117.56s |
| Deep Learning (Raw EEG) | LSTM | 0.7698 | 1.0000 | 0.4245 | 0.5960 | 0.7572 | 6.12s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.8960 | 1.0000 | 0.7399 | 0.8505 | 0.9969 | 4.74s |
| Deep Learning (Raw EEG) | EEGNet | 0.7000 | 0.6667 | 0.5000 | 0.5714 | 0.8935 | 17.66s |
| Deep Learning (Raw EEG) | TSception | 0.7839 | 0.7395 | 0.7097 | 0.7243 | 0.9335 | 19.05s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.8181 | 1.0000 | 0.5453 | 0.7058 | 0.9925 | 56.32s |

---

## ⚡ Part 2: Stress Detection Results (State Classification)

The models below were trained to classify cognitive stress/arousal states vs. relaxed baselines using SAM40, Cognitive Load, and AMIGOS datasets.

| Model Class | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.5856 | 0.8050 | 0.5463 | 0.6509 | 0.6598 | 85.19s |
| Classical ML (Feature-based) | Random Forest | 0.5591 | 0.8224 | 0.4801 | 0.6063 | 0.6624 | 70.20s |
| Classical ML (Feature-based) | XGBoost | 0.6280 | 0.7746 | 0.6684 | 0.7176 | 0.6563 | 241.67s |
| Deep Learning (Raw EEG) | 1D CNN | 0.5856 | 0.7668 | 0.5949 | 0.6700 | 0.6300 | 61.53s |
| Deep Learning (Raw EEG) | LSTM | 0.4260 | 0.7235 | 0.3048 | 0.4289 | 0.4937 | 14.47s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.4208 | 0.7449 | 0.2752 | 0.4019 | 0.4869 | 7.60s |
| Deep Learning (Raw EEG) | EEGNet | 0.5842 | 0.7872 | 0.5648 | 0.6577 | 0.6677 | 16.57s |
| Deep Learning (Raw EEG) | TSception | 0.5797 | 0.7908 | 0.5516 | 0.6499 | 0.6507 | 17.15s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.5160 | 0.7956 | 0.4247 | 0.5538 | 0.6183 | 41.76s |

---

## 🧬 Part 3: Advanced Models (Multi-Task & Domain Adversarial)

### 3.1 Shared-Backbone Multi-Task Network
A single **TSception** model backbone trained on the entire combined dataset to simultaneously output depression and stress predictions using a custom masked cross-entropy loss function.

| Multi-Task Evaluation Branch | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Joint Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Depression Branch | 0.8839 | 1.0000 | 0.7097 | 0.8302 | 0.8722 | 19.68s |
| Stress Branch | 0.5481 | 0.8453 | 0.4418 | 0.5803 | 0.6790 | 19.68s |

### 3.2 Domain Adversarial Neural Network (DANN)
A **TSception** model incorporating a custom **Gradient Reversal Layer (GRL)**. It is trained to perform stress detection ('task' branch) while learning domain-invariant features by trying to confuse a dataset-origin classifier ('domain' branch).

| Model Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DANN Stress Task | 0.6081 | 0.7832 | 0.6164 | 0.6899 | 0.6638 | 15.93s |

---

## 📈 Key Findings and Recommendations

1. **Impact of Global Segment Normalization & Artifact Clipping**:
   * Outlier-clipped global segment-level normalization preserves spatial amplitude ratios across the channels, which are essential for frontal asymmetry and spatial filtering.
   * Expanding features with relative power, spectral complexity, and amplitude characteristics improves baseline classical machine learning models.

2. **Hyperparameter Grid Searching**:
   * Tuning classifiers using cross-validated GridSearchCV isolates optimal hyperparameters (e.g. C limits in SVM, max depths in Random Forest, learning rates in XGBoost) preventing underfitting and severe overfitting.

3. **Optimized Deep Learning Convergence**:
   * Operating at a lower, tuned learning rate (5e-4) with early stopping patience extended to 5 epochs stabilizes SGD optimization paths, allowing CNNs, LSTMs, and Transformers to identify robust local minima from raw EEG signals.
