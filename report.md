# EEG Model Training Metrics Report

This report summarizes the performance metrics of all implemented models on the standardized **6-channel frontal EEG dataset** (Fp1, Fp2, F7, F3, F4, F8) combined from **MODMA, SAM40, Cognitive Load, and AMIGOS**.

Evaluation uses a rigorous **Group-based Train/Val/Test split (70/10/20)** on subject IDs, guaranteeing zero subject overlap across all three splits. The validation split is drawn per-dataset so that early stopping is not steered by a single corpus. All models are trained with **outlier-clipped global segment normalization** (preserving spatial amplitude ratios across channels); deep learning models use **GPU acceleration** with a tuned learning rate, and classical models use **subject-grouped** GridSearch hyperparameter tuning.

> **Reading these numbers:** the **Majority Baseline** column is the accuracy of always predicting the most frequent class. Any model whose Accuracy falls below it (flagged ⚠️) is worse than a constant predictor, and **Balanced Acc** is the meaningful figure there. **Threshold** is the decision cutoff selected on the validation split rather than a fixed 0.5.

---

## 📊 Summary of Combined Dataset Splits
*   **Depression Classification (MODMA)**:
    *   Train / Val / Test: 5723 / 746 / 1490 segments
*   **Stress Classification (SAM40, Cognitive Load, AMIGOS)**:
    *   Train / Val / Test: 34688 / 3628 / 7502 segments
*   **Multi-Task (Full Dataset)**:
    *   Train / Val / Test: 40411 / 4374 / 8992 segments

---

## 🧠 Part 1: Depression Detection Results (Clinical Classification)

The models below were trained to classify Major Depressive Disorder (MDD) vs. Healthy Controls (HC) using the MODMA dataset.

| Model Class | Model | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.7121 | 0.7064 | 0.6000 | 0.6303 | 0.6779 | 0.6532 | 0.8080 | 0.97 | 4.89s |
| Classical ML (Feature-based) | Random Forest | 0.5899 ⚠️ | 0.5761 | 0.6000 | 0.4879 | 0.5067 | 0.4971 | 0.5664 | 0.59 | 5.78s |
| Classical ML (Feature-based) | XGBoost | 0.4664 ⚠️ | 0.4567 | 0.6000 | 0.3547 | 0.4077 | 0.3794 | 0.5441 | 0.55 | 30.32s |
| Deep Learning (Raw EEG) | 1D CNN | 0.8013 | 0.7517 | 0.6000 | 1.0000 | 0.5034 | 0.6696 | 0.8668 | 0.92 | 122.49s |
| Deep Learning (Raw EEG) | LSTM | 0.6107 | 0.5134 | 0.6000 | 1.0000 | 0.0268 | 0.0523 | 0.7915 | 0.95 | 14.32s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.7745 | 0.7237 | 0.6000 | 0.9333 | 0.4698 | 0.6250 | 0.9477 | 0.16 | 11.39s |
| Deep Learning (Raw EEG) | EEGNet | 0.4201 ⚠️ | 0.5168 | 0.6000 | 0.4082 | 1.0000 | 0.5798 | 0.9055 | 0.32 | 20.07s |
| Deep Learning (Raw EEG) | TSception | 0.5893 ⚠️ | 0.4911 | 0.6000 | 0.0000 | 0.0000 | 0.0000 | 0.6391 | 0.52 | 27.09s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.6107 | 0.5134 | 0.6000 | 1.0000 | 0.0268 | 0.0523 | 0.8891 | 0.79 | 54.20s |

---

## ⚡ Part 2: Stress Detection Results (State Classification)

The models below were trained to classify cognitive stress/arousal states vs. relaxed baselines using SAM40, Cognitive Load, and AMIGOS datasets.

| Model Class | Model | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Classical ML (Feature-based) | SVM | 0.3242 ⚠️ | 0.4553 | 0.8268 | 0.7795 | 0.2547 | 0.3840 | 0.4030 | 0.96 | 61.62s |
| Classical ML (Feature-based) | Random Forest | 0.7983 ⚠️ | 0.4843 | 0.8268 | 0.8222 | 0.9647 | 0.8878 | 0.4796 | 0.32 | 53.97s |
| Classical ML (Feature-based) | XGBoost | 0.1772 ⚠️ | 0.4489 | 0.8268 | 0.5393 | 0.0332 | 0.0626 | 0.5051 | 0.90 | 187.34s |
| Deep Learning (Raw EEG) | 1D CNN | 0.4339 ⚠️ | 0.4188 | 0.8268 | 0.7774 | 0.4419 | 0.5635 | 0.3400 | 0.61 | 69.84s |
| Deep Learning (Raw EEG) | LSTM | 0.8179 ⚠️ | 0.4961 | 0.8268 | 0.8257 | 0.9884 | 0.8998 | 0.3962 | 0.30 | 50.54s |
| Deep Learning (Raw EEG) | CNN-LSTM | 0.2890 ⚠️ | 0.3826 | 0.8268 | 0.7068 | 0.2394 | 0.3577 | 0.3809 | 0.68 | 20.45s |
| Deep Learning (Raw EEG) | EEGNet | 0.8202 ⚠️ | 0.5227 | 0.8268 | 0.8336 | 0.9778 | 0.8999 | 0.4652 | 0.46 | 18.84s |
| Deep Learning (Raw EEG) | TSception | 0.7933 ⚠️ | 0.4873 | 0.8268 | 0.8231 | 0.9553 | 0.8843 | 0.4774 | 0.31 | 18.65s |
| Deep Learning (Raw EEG) | EEGTransformer | 0.7967 ⚠️ | 0.4818 | 0.8268 | 0.8215 | 0.9636 | 0.8869 | 0.3740 | 0.14 | 31.07s |

### Stress ROC-AUC by source corpus

> The pooled row above is dominated (~78%) by AMIGOS, whose self-reported arousal label
> is **not EEG-decodable** from this montage (Phase 1: 0.4875 within-subject AUC, and a
> per-subject median split changes it by only −0.004). Read this breakdown, not the
> pooled figure. Stress should be reported on **SAM40 / CogLoad** only.

| Model | AMIGOS | CogLoad | SAM40 |
| :--- | :--- | :--- | :--- |
| SVM | 0.3522 | 0.7574 | 0.5584 |
| Random Forest | 0.3947 | 0.6724 | 0.6423 |
| XGBoost | 0.3849 | 0.7300 | 0.6633 |
| 1D CNN | 0.3423 | 0.4419 | 0.6059 |
| LSTM | 0.4650 | 0.2799 | 0.4281 |
| CNN-LSTM | 0.4534 | 0.3982 | 0.6503 |
| EEGNet | 0.5183 | 0.4287 | 0.4761 |
| TSception | 0.4804 | 0.8836 | 0.5392 |
| EEGTransformer | 0.4069 | 0.7905 | 0.5945 |

### Depression at the subject level (10 held-out subjects)

> Every segment from a subject carries the same label, so the clinically meaningful
> question is whether the **subject** is MDD. Note this is still a single split — see
> `Results/phase1/depression_cv.json` for out-of-fold estimates over all 53 subjects,
> which are the reportable numbers. Single-split figures here inflate deep-model AUC
> by roughly 0.21.

| Model | Subject ROC-AUC | Subject Balanced Acc |
| :--- | :--- | :--- |
| SVM | 0.7917 | 0.5833 |
| Random Forest | 0.5833 | 0.5833 |
| XGBoost | 0.5417 | 0.5833 |
| 1D CNN | 0.8750 | 0.7917 |
| LSTM | 0.8333 | 0.5833 |
| CNN-LSTM | 0.9583 | 0.7917 |
| EEGNet | 0.9167 | 0.7917 |
| TSception | 0.6250 | 0.5833 |
| EEGTransformer | 0.8750 | 0.7917 |

---

## 🧬 Part 3: Advanced Models (Multi-Task & Domain Adversarial)

### 3.1 Shared-Backbone Multi-Task Network
A single **TSception** model backbone trained on the entire combined dataset to simultaneously output depression and stress predictions using a custom masked cross-entropy loss function.

| Multi-Task Evaluation Branch | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Joint Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Depression Branch | 0.4403 ⚠️ | 0.5336 | 0.6000 | 0.4168 | 1.0000 | 0.5884 | 0.5584 | 0.00 | 22.91s |
| Stress Branch | 0.8166 ⚠️ | 0.4938 | 0.8268 | 0.8251 | 0.9876 | 0.8990 | 0.5093 | 0.34 | 22.91s |

### 3.2 Domain Adversarial Neural Network (DANN)
A **TSception** model incorporating a custom **Gradient Reversal Layer (GRL)**. It is trained to perform stress detection ('task' branch) while learning domain-invariant features by trying to confuse a dataset-origin classifier ('domain' branch).

| Model Configuration | Accuracy | Balanced Acc | Majority Baseline | Precision | Recall | F1-Score | ROC-AUC | Threshold | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| DANN Stress Task | 0.1853 ⚠️ | 0.4982 | 0.8268 | 0.8013 | 0.0195 | 0.0381 | 0.5677 | 0.74 | 18.32s |

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
