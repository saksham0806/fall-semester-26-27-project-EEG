# Consolidated Research Report: Depression Analysis using MODMA Dataset

## Executive Summary

This research project aimed to classify depression using Resting-state EEG data from the MODMA dataset. We progressed through four phases of analysis, starting from basic 3-channel analysis to advanced 128-channel deep learning models.

The final phase (Phase 4) focused on fine-tuning and comparing two state-of-the-art architectures: **TSception** and **EEGNet**.

**Key Findings:**
*   **TSception** significantly outperformed **EEGNet** in our 5-fold cross-validation setup.
*   **TSception Average Accuracy**: **74.87%** (+/- 4.89%)
*   **EEGNet Average Accuracy**: **60.83%** (+/- 13.51%)
*   **Best Single Model**: TSception (Fold 3) achieved **81.75%** validation accuracy.
*   **Challenges**: High inter-subject variability led to significant performance fluctuations across folds, particularly for EEGNet which collapsed in Fold 4.

---

## Methodology Overview

### Dataset
*   **Source**: MODMA Dataset (128-channel EEG)
*   **Subjects**: 53 participants (Matched Depressed & Healthy Controls)
*   **Preprocessing**: Bandpass filtering (0.5-50Hz), Artifact Removal (ICA/ASR), Segmentation (4s windows).

### Models Evaluated
1.  **TSception**: Temporal-Spatial Conception model. Deep learning architecture designed for EEG emotion/state recognition.
2.  **EEGNet**: Compact Convolutional Neural Network specialized for EEG.

### Training Configuration (Phase 4)
*   **Cross-Validation**: 5-Fold Group Stratified (ensuring subjects are not split between train/val).
*   **Epochs**: 50 (with Early Stopping, Patience=10).
*   **Optimization**: Adam (lr=0.001), ReduceLROnPlateau.
*   **Augmentation**: 50% Overlap sliding windows.

---

## Detailed Results

### 1. TSception Performance
TSception showed robust performance and better generalization capabilities.

| Fold | Accuracy | Evaluation |
| :--- | :--- | :--- |
| Fold 1 | **79.11%** | Strong start, early stopping effective. |
| Fold 2 | **72.46%** | Consistent with average. |
| Fold 3 | **81.75%** | **Best performance observed.** |
| Fold 4 | **68.25%** | Lowest for TSception, but still usable. |
| Fold 5 | **72.78%** | Stable. |
| **Average** | **74.87%** | **std: +/- 4.89%** |

### 2. EEGNet Performance
EEGNet struggled with stability and convergence on this specific dataset partition, showing high variance.

| Fold | Accuracy | Evaluation |
| :--- | :--- | :--- |
| Fold 1 | **68.57%** | Acceptable baseline. |
| Fold 2 | **60.45%** | Underperformed. |
| Fold 3 | **76.50%** | Comparable to TSception. |
| Fold 4 | **36.27%** | **Failed to converge.** Model collapsed. |
| Fold 5 | **62.36%** | Below average. |
| **Average** | **60.83%** | **std: +/- 13.51%** |

---

## Comparative Analysis

*   **Stability**: TSception was far more stable across folds (std 4.89%) compared to EEGNet (std 13.51%).
*   **Converge**: TSception converged faster and more reliably. EEGNet required more sensitivity to initialization and hyperparameter tuning, failing completely in Fold 4.
*   **Feature Extraction**: TSception's dual temporary-spatial kernels likely captured the complex depressive patterns in 128-channel data better than EEGNet's compact depthwise-separable convolutions.

## Conclusion & Recommendations

Based on the extensive experimentation, **TSception** is the recommended model for depression detection on the MODMA dataset. It achieves a reliable ~75% accuracy with potential to reach >80% on specific subject groups.

**Future Work:**
*   **Hyperparameter Optimization**: Further tuning of TSception's temporal kernels.
*   **Subject-Adaptive Transfer Learning**: To address the high inter-subject variability.
*   **Multimodal Integration**: Combining EEG with other physiological signals if available.
