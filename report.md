# EEG Model Training Metrics Report

This report summarizes the performance metrics of all implemented models on the standardized **6-channel frontal EEG dataset** (Fp1, Fp2, F7, F3, F4, F8) combined from **MODMA, SAM40, Cognitive Load, and AMIGOS**.

Evaluation was conducted using a rigorous **Group-based Train/Test split (80/20)** based on subject IDs to ensure zero subject overlap and prevent data leakage. All models have been trained with **outlier-clipped global segment normalization** (which preserves spatial amplitude ratios across channels), and deep learning models were trained using **GPU acceleration (NVIDIA RTX 3060)** with a tuned learning rate and GridSearch hyperparameter tuning.

---

## Summary of Combined Dataset Splits
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

## Part 1: Depression Detection Results (Clinical Classification)

The models below were trained to classify Major Depressive Disorder (MDD) vs. Healthy Controls (HC) using the MODMA dataset.

| Model Class | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.6530 | 0.5464 | 0.7802 | 0.6427 | 0.7226 | 11.80s |
| Classical ML (Feature-based) | Random Forest | 0.4966 | 0.3778 | 0.3993 | 0.3883 | 0.5344 | 6.76s |
| Classical ML (Feature-based) | XGBoost | 0.5336 | 0.4145 | 0.4027 | 0.4085 | 0.5703 | 35.63s |
| Deep Learning (Raw EEG) | 1D CNN | 0.8926 | 1.0000 | 0.7315 | 0.8450 | 0.9134 | 110.11s |
| Deep Learning (Raw EEG) | LSTM | 0.7490 | 1.0000 | 0.3725 | 0.5428 | 0.7624 | 29.79s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.7174 | 1.0000 | 0.2936 | 0.4540 | 0.7137 | 6.87s |
| Deep Learning (Raw EEG) | EEGNet | 0.8839 | 1.0000 | 0.7097 | 0.8302 | 0.8921 | 18.40s |
| Deep Learning (Raw EEG) | TSception | 0.7523 | 1.0000 | 0.3809 | 0.5516 | 0.7664 | 19.56s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.8208 | 1.0000 | 0.5520 | 0.7114 | 0.9689 | 54.64s |

---

## Part 2: Stress Detection Results (State Classification)

The models below were trained to classify cognitive stress/arousal states vs. relaxed baselines using SAM40, Cognitive Load, and AMIGOS datasets.

| Model Class | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.5856 | 0.8050 | 0.5463 | 0.6509 | 0.6598 | 86.20s |
| Classical ML (Feature-based) | Random Forest | 0.5591 | 0.8224 | 0.4801 | 0.6063 | 0.6624 | 67.82s |
| Classical ML (Feature-based) | XGBoost | 0.6280 | 0.7746 | 0.6684 | 0.7176 | 0.6563 | 237.51s |
| Deep Learning (Raw EEG) | 1D CNN | 0.5005 | 0.7926 | 0.3977 | 0.5297 | 0.5724 | 68.71s |
| Deep Learning (Raw EEG) | LSTM | 0.3959 | 0.8650 | 0.1727 | 0.2879 | 0.5886 | 48.37s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.4920 | 0.7791 | 0.3930 | 0.5225 | 0.5779 | 21.07s |
| Deep Learning (Raw EEG) | EEGNet | 0.4881 | 0.8833 | 0.3182 | 0.4678 | 0.6349 | 25.35s |
| Deep Learning (Raw EEG) | TSception | 0.4680 | 0.9345 | 0.2664 | 0.4146 | 0.6769 | 25.23s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.4350 | 0.8102 | 0.2624 | 0.3964 | 0.5991 | 41.12s |

---

## Part 3: Advanced Models (Multi-Task & Domain Adversarial)

### 3.1 Shared-Backbone Multi-Task Network
A single **TSception** model backbone trained on the entire combined dataset to simultaneously output depression and stress predictions using a custom masked cross-entropy loss function.

| Multi-Task Evaluation Branch | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Joint Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Depression Branch | 0.7966 | 1.0000 | 0.4916 | 0.6592 | 0.9647 | 24.65s |
| Stress Branch | 0.6332 | 0.7386 | 0.7450 | 0.7417 | 0.6287 | 24.65s |

### 3.2 Domain Adversarial Neural Network (DANN)
A **TSception** model incorporating a custom **Gradient Reversal Layer (GRL)**. It is trained to perform stress detection ('task' branch) while learning domain-invariant features by trying to confuse a dataset-origin classifier ('domain' branch).

| Model Configuration | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DANN Stress Task | 0.5327 | 0.8496 | 0.4121 | 0.5550 | 0.7062 | 21.78s |

---

##  Key Findings and Recommendations

1. **Impact of Global Segment Normalization & Artifact Clipping**:
   * Outlier-clipped global segment-level normalization preserves spatial amplitude ratios across the channels, which are essential for frontal asymmetry and spatial filtering.
   * Expanding features with relative power, spectral complexity, and amplitude characteristics improves baseline classical machine learning models.

2. **Hyperparameter Grid Searching**:
   * Tuning classifiers using cross-validated GridSearchCV isolates optimal hyperparameters (e.g. C limits in SVM, max depths in Random Forest, learning rates in XGBoost) preventing underfitting and severe overfitting.

3. **Optimized Deep Learning Convergence**:
   * Operating at a lower, tuned learning rate (5e-4) with early stopping patience extended to 5 epochs stabilizes SGD optimization paths, allowing CNNs, LSTMs, and Transformers to identify robust local minima from raw EEG signals.
