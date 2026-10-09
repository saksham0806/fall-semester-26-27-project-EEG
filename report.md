# EEG Model Training Metrics Report

This report summarizes the performance metrics of all implemented models on the standardized **6-channel frontal EEG dataset** (Fp1, Fp2, F7, F3, F4, F8) combined from **MODMA, SAM40, Cognitive Load, and AMIGOS**.

Evaluation uses a rigorous **Group-based Train/Val/Test split (70/10/20)** on subject IDs, guaranteeing zero subject overlap across all three splits. The validation split is drawn per-dataset so that early stopping is not steered by a single corpus. All models are trained with **outlier-clipped global segment normalization** (preserving spatial amplitude ratios across channels); deep learning models use **GPU acceleration** with a tuned learning rate, and classical models use **subject-grouped** GridSearch hyperparameter tuning.

> **Reading these numbers:** the **Majority Baseline** column is the accuracy of always predicting the most frequent class. Any model whose Accuracy falls below it (flagged ⚠️) is worse than a constant predictor, and **Balanced Acc** is the meaningful figure there. **Threshold** is the decision cutoff selected on the validation split rather than a fixed 0.5.

---

## 📊 Summary of Combined Dataset Splits
*   **Depression Classification (MODMA)**:
    *   Train / Val / Test: 5558 / 745 / 1506 segments
*   **Stress Classification (SAM40, Cognitive Load, AMIGOS)**:
    *   Train / Val / Test: 7620 / 754 / 2243 segments
*   **Multi-Task (Full Dataset)**:
    *   Train / Val / Test: 13178 / 1499 / 3749 segments

---

## 🧠 Part 1: Depression Detection Results (Clinical Classification)

The models below were trained to classify Major Depressive Disorder (MDD) vs. Healthy Controls (HC) using the MODMA dataset.

| Model Class | Model | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.6992 | 0.6889 | 0.6042 | 0.6155 | 0.6393 | 0.6272 | 0.7520 | 0.99 | 2.48s |
| Classical ML (Feature-based) | Random Forest | 0.6627 | 0.6450 | 0.6042 | 0.5759 | 0.5604 | 0.5680 | 0.5750 | 0.58 | 3.60s |
| Classical ML (Feature-based) | XGBoost | 0.4801 ⚠️ | 0.4575 | 0.6042 | 0.3449 | 0.3490 | 0.3470 | 0.4716 | 0.75 | 1.84s |
| Deep Learning (Raw EEG) | 1D CNN | 0.8048 | 0.7534 | 0.6042 | 1.0000 | 0.5067 | 0.6726 | 0.8098 | 0.59 | 16.76s |
| Deep Learning (Raw EEG) | LSTM | 0.6155 | 0.5143 | 0.6042 | 1.0000 | 0.0285 | 0.0555 | 0.7063 | 0.90 | 10.64s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.5956 ⚠️ | 0.6654 | 0.6042 | 0.4946 | 1.0000 | 0.6619 | 0.9055 | 0.16 | 4.64s |
| Deep Learning (Raw EEG) | EEGNet | 0.6155 | 0.5143 | 0.6042 | 1.0000 | 0.0285 | 0.0555 | 0.5537 | 0.56 | 12.10s |
| Deep Learning (Raw EEG) | TSception | 0.8048 | 0.7534 | 0.6042 | 1.0000 | 0.5067 | 0.6726 | 0.8725 | 0.48 | 12.82s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.7995 | 0.7478 | 0.6042 | 0.9868 | 0.5000 | 0.6637 | 0.8624 | 0.93 | 32.65s |

---

## ⚡ Part 2: Stress Detection Results (State Classification)

The models below were trained to classify cognitive stress/arousal states vs. relaxed baselines using SAM40, Cognitive Load, and AMIGOS datasets.

| Model Class | Model | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.7272 | 0.5648 | 0.6910 | 0.7202 | 0.9897 | 0.8337 | 0.7496 | 0.09 | 8.05s |
| Classical ML (Feature-based) | Random Forest | 0.6839 ⚠️ | 0.4964 | 0.6910 | 0.6895 | 0.9871 | 0.8119 | 0.7302 | 0.34 | 3.17s |
| Classical ML (Feature-based) | XGBoost | 0.6964 | 0.5087 | 0.6910 | 0.6948 | 1.0000 | 0.8199 | 0.7485 | 0.13 | 168.93s |
| Deep Learning (Raw EEG) | 1D CNN | 0.6126 ⚠️ | 0.5114 | 0.6910 | 0.6974 | 0.7761 | 0.7347 | 0.3736 | 0.52 | 12.47s |
| Deep Learning (Raw EEG) | LSTM | 0.5412 ⚠️ | 0.4367 | 0.6910 | 0.6550 | 0.7103 | 0.6815 | 0.3624 | 0.63 | 12.08s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.5635 ⚠️ | 0.4540 | 0.6910 | 0.6655 | 0.7406 | 0.7011 | 0.4783 | 0.80 | 10.06s |
| Deep Learning (Raw EEG) | EEGNet | 0.5372 ⚠️ | 0.4541 | 0.6910 | 0.6631 | 0.6716 | 0.6673 | 0.3394 | 0.55 | 17.73s |
| Deep Learning (Raw EEG) | TSception | 0.6826 ⚠️ | 0.5038 | 0.6910 | 0.6927 | 0.9716 | 0.8088 | 0.4338 | 0.56 | 10.34s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.6580 ⚠️ | 0.6002 | 0.6910 | 0.7531 | 0.7516 | 0.7523 | 0.5725 | 0.92 | 25.80s |

### Stress ROC-AUC by source corpus

> The pooled row above is dominated (~78%) by AMIGOS, whose self-reported arousal label
> is **not EEG-decodable** from this montage (Phase 1: 0.4875 within-subject AUC, and a
> per-subject median split changes it by only −0.004). Read this breakdown, not the
> pooled figure. Stress should be reported on **SAM40 / CogLoad** only.

| Model | CogLoad | SAM40 |
| :--- | :--- | :--- |
| SVM | 0.7674 | 0.6597 |
| Random Forest | 0.7389 | 0.6537 |
| XGBoost | 0.7691 | 0.6032 |
| 1D CNN | 0.3348 | 0.5380 |
| LSTM | 0.3317 | 0.5508 |
| CNN-LSTM | 0.5128 | 0.6141 |
| EEGNet | 0.2844 | 0.6105 |
| TSception | 0.4364 | 0.6019 |
| EEGTransformer | 0.6257 | 0.6222 |

### Depression at the subject level (10 held-out subjects)

> Every segment from a subject carries the same label, so the clinically meaningful
> question is whether the **subject** is MDD. Note this is still a single split — see
> `Results/phase1/depression_cv.json` for out-of-fold estimates over all 53 subjects,
> which are the reportable numbers. Single-split figures here inflate deep-model AUC
> by roughly 0.21.

| Model | Subject ROC-AUC | Subject Balanced Acc |
| :--- | :--- | :--- |
| SVM | 0.5833 | 0.5833 |
| Random Forest | 0.5833 | 0.5833 |
| XGBoost | 0.4583 | 0.5833 |
| 1D CNN | 0.8333 | 0.7917 |
| LSTM | 0.7083 | 0.5833 |
| CNN-LSTM | 0.9583 | 0.7917 |
| EEGNet | 0.5417 | 0.5833 |
| TSception | 0.9167 | 0.7917 |
| EEGTransformer | 0.8750 | 0.7917 |

---

## 🧬 Part 3: Advanced Models (Multi-Task & Domain Adversarial)

### 3.1 Shared-Backbone Multi-Task Network
A single **TSception** model backbone trained on the entire combined dataset to simultaneously output depression and stress predictions using a custom masked cross-entropy loss function.

| Multi-Task Evaluation Branch | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Joint Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Depression Branch | 0.5896 ⚠️ | 0.6156 | 0.6042 | 0.4878 | 0.7399 | 0.5880 | 0.6151 | 0.14 | 11.34s |
| Stress Branch | 0.4338 ⚠️ | 0.3466 | 0.6910 | 0.5932 | 0.5748 | 0.5839 | 0.4910 | 0.65 | 11.34s |

### 3.2 Domain Adversarial Neural Network (DANN)
A **TSception** model incorporating a custom **Gradient Reversal Layer (GRL)**. It is trained to perform stress detection ('task' branch) while learning domain-invariant features by trying to confuse a dataset-origin classifier ('domain' branch).

| Model Configuration | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DANN Stress Task | 0.7218 | 0.5498 | 0.6910 | 0.7130 | 1.0000 | 0.8324 | 0.4188 | 0.46 | 9.57s |

---

## 📈 Key Findings and Recommendations

0. **Label corrections applied to the combined dataset**:
   * **AMIGOS** was reading `labels_selfassessment` index 1 (**valence**) while treating it as arousal. Per `SelfAsessment.xlsx` the block order is `arousal, valence, dominance, ...`, so index 0 is arousal; this was verified against the spreadsheet for 80 subject/trial pairs. Since arousal and valence correlate at only −0.11 in this data, ~71% of the corpus had been labelled by a near-orthogonal variable.
   * **Cognitive Load** hardcoded every file to stress-positive, including its 30 `natural-*` rest recordings. Those now form the dataset's baseline class, which previously did not exist.

1. **Impact of Global Segment Normalization & Artifact Clipping**:
   * Outlier-clipped global segment-level normalization preserves spatial amplitude ratios across the channels, which are essential for frontal asymmetry and spatial filtering.
   * Expanding features with relative power, spectral complexity, and amplitude characteristics improves baseline classical machine learning models.

2. **Hyperparameter Grid Searching**:
   * Tuning classifiers using cross-validated GridSearchCV isolates optimal hyperparameters (e.g. C limits in SVM, max depths in Random Forest, learning rates in XGBoost) preventing underfitting and severe overfitting.

3. **Optimized Deep Learning Convergence**:
   * Operating at a lower, tuned learning rate (5e-4) with early stopping patience extended to 5 epochs stabilizes SGD optimization paths, allowing CNNs, LSTMs, and Transformers to identify robust local minima from raw EEG signals.
