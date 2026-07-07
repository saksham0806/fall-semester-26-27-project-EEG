# Hybrid Model Pipeline Plan for EEG-Based Multi-Label Classification

## 1. Project Analysis Summary

### 1.1 Classification Target
A **unified multi-label model** that simultaneously predicts:
- **Depression** (MDD vs. HC) — from MODMA data
- **Stress** (Stress/Arousal vs. Relaxed/Baseline) — from SAM40, Cognitive Load, AMIGOS data

Using a masked loss strategy where each sample only contributes to the task(s) its dataset supports (label = -1 for non-applicable tasks).

---

### 1.2 Current Dataset Profile

| Dataset | Subjects | Original Channels | Fs (Hz) | Mapped to 6-Ch Frontal | Dataset ID | Task | Segments |
|---|---|---|---|---|---|---|---|
| **MODMA (3-ch)** | 55 (26 MDD, 29 HC) | 3 (Fp1, Fpz, Fp2) | 250 | Fp1, Fp2 native; F7, F3, F4, F8 padded/mapped | 0 | Depression | 7,959 (3,599 MDD + 4,360 HC) |
| **SAM40** | 40 | 32 | 256 | All 6 frontal channels native ✓ | 1 | Stress | 2,400 (1,800 active + 600 relax) |
| **Cognitive Load** | ~120 recordings | 8 (F3, F4, C3, C4, P3, P4, O1, O2) | 250 | F3, F4 native; Fp1, Fp2, F7, F8 padded | 2 | Stress | 5,369 (all active, 0 relax) |
| **AMIGOS** | 40 | 14 (Emotiv EPOC) | 128 | AF3→Fp1, AF4→Fp2, F7, F3, F4, F8 (approximate) | 3 | Stress | 39,429 (23,185 active + 16,244 relax) |

**Combined data**: 6 channels (Fp1, Fp2, F7, F3, F4, F8), **128 Hz**, **4-second segments** (512 samples).

> [!NOTE]
> AMIGOS dominates the combined dataset at **71.5%** of all segments. Cognitive Load has **no relaxation baseline** (100% stress_active). These distributional asymmetries may strongly bias models.

> [!WARNING]
> **Channel mismatch is a critical issue.** Only SAM40 has all 6 frontal channels natively. MODMA 3-ch is missing 4 channels, Cognitive Load is missing 4 channels (only F3, F4 overlap), and AMIGOS uses approximate mappings (AF3→Fp1, AF4→Fp2). This heterogeneity is likely a major contributor to poor stress detection and cross-dataset generalization.

### 1.3 Data Scale

| Split | Depression (MODMA) | Stress (SAM40 + CogLoad + AMIGOS) | Full (Multi-Task) |
|---|---|---|---|
| Train | 6,469 segments | 38,316 segments | 44,785 segments |
| Test | 1,490 segments | 7,502 segments | 8,992 segments |

### 1.4 Extracted Features (81 per segment)

| Feature Group | Count | Description |
|---|---|---|
| Relative Band Power (Welch PSD) | 30 | 5 bands (δ 0.5–4, θ 4–8, α 8–12, β 12–30, γ 30–45 Hz) × 6 channels |
| Differential Entropy (DE) | 30 | 5 bands × 6 channels, via Butterworth bandpass (order 3) |
| Frontal Alpha Asymmetry (FAA) | 3 | ln(α_right) − ln(α_left) for 3 pairs: (Fp2−Fp1), (F4−F3), (F8−F7) |
| Spectral Entropy | 6 | Shannon entropy of normalized PSD per channel |
| RMS Amplitude | 6 | Per channel |
| Standard Deviation | 6 | Per channel |

---

### 1.5 Current Results Analysis

#### Depression Detection — **Strong performance, solved to a reasonable degree**

| Model | Accuracy | F1 | ROC-AUC | Notes |
|---|---|---|---|---|
| CNN-LSTM | **0.8960** | **0.8505** | **0.9969** | Best overall |
| 1D CNN | 0.8383 | 0.7466 | 0.7859 | High precision, low recall |
| EEGTransformer | 0.8181 | 0.7058 | 0.9925 | Excellent AUC, recall bottleneck |
| TSception | 0.7839 | 0.7243 | 0.9335 | Balanced performance |
| SVM (features) | 0.6530 | 0.6427 | 0.7226 | Best classical |
| Multi-Task (TSception backbone) | 0.8839 | 0.8302 | 0.8722 | Competitive with standalone |

#### Stress Detection — **The bottleneck, needs significant improvement**

| Model | Accuracy | F1 | ROC-AUC | Notes |
|---|---|---|---|---|
| XGBoost (features) | **0.6280** | **0.7176** | 0.6563 | Best classical, best F1 |
| DANN (TSception+GRL) | 0.6081 | 0.6899 | 0.6638 | Domain adaptation helps marginally |
| 1D CNN | 0.5856 | 0.6700 | 0.6300 | Matches classical |
| EEGNet | 0.5842 | 0.6577 | 0.6677 | Best AUC among standalone DL |
| Multi-Task (TSception backbone) | 0.5481 | 0.5803 | 0.6790 | Stress benefits from shared features |
| CNN-LSTM | **0.4208** | 0.4019 | 0.4869 | **Collapses** on stress task |
| LSTM | 0.4260 | 0.4289 | 0.4937 | **Collapses** on stress task |

> [!IMPORTANT]
> **Key observation**: Models that excel at depression (CNN-LSTM, LSTM) catastrophically fail on stress. This indicates these architectures overfit to depression-specific temporal patterns that don't transfer. The stress task's difficulty stems from: (1) cross-dataset domain shift, (2) channel mismatch/padding artifacts, (3) diverse stress paradigms (math, Stroop, symmetry, arousal).

---

### 1.6 Key Challenges to Address

1. **Stress detection plateau** (~63% accuracy) — the primary bottleneck
2. **Domain shift** across 3 stress datasets with different paradigms and channel configurations
3. **Channel mismatch** — zero-padded channels introduce artificial features
4. **Scale imbalance** — stress data is ~6× larger than depression data
5. **Precision-recall trade-off** — many DL models achieve high precision but low recall (predicting majority class)
6. **Temporal pattern diversity** — stress manifests differently across paradigms (arithmetic cognitive load ≠ Stroop interference ≠ emotional arousal)

---

## 2. Proposed Hybrid Model Pipelines

### Pipeline 1: Dual-Stream Feature-Fusion Network (DSFFN)

#### Core Idea
Combine **handcrafted domain features** (the 81 features from PSD, DE, FAA, etc.) with **learned deep features** (from raw EEG via a CNN backbone) through a learnable fusion mechanism. This addresses the observation that classical ML (XGBoost: 0.72 F1) and DL (EEGNet: 0.66 F1) capture complementary signal aspects for stress.

#### Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           RAW EEG INPUT                  │
                    │      (batch, 512, 6, 1)                  │
                    └────────────┬────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼                               ▼
    ┌────────────────────┐          ┌────────────────────────┐
    │   Stream 1: Deep    │          │  Stream 2: Handcrafted  │
    │  Feature Extractor  │          │   Feature Extractor     │
    │                    │          │                        │
    │  TSception Backbone │          │   PSD (30) + DE (30)    │
    │  (multi-scale temp  │          │   + FAA (3) + Spectral  │
    │   + spatial conv)   │          │   Entropy (6) + RMS (6) │
    │                    │          │   + STD (6) = 81 feats  │
    │  → Flatten → d₁    │          │                        │
    │    (e.g. 128-dim)   │          │   Dense(64, relu)       │
    └─────────┬──────────┘          │   BatchNorm → d₂        │
              │                     │    (64-dim)              │
              │                     └───────────┬─────────────┘
              │                                 │
              └─────────────┬───────────────────┘
                            ▼
              ┌──────────────────────────┐
              │   Gated Fusion Module     │
              │                          │
              │  concat = [d₁; d₂]       │
              │  gate = σ(W_g · concat)  │
              │  fused = gate ⊙ d₁ +     │
              │     (1-gate) ⊙ d₂        │
              └─────────────┬────────────┘
                            ▼
              ┌──────────────────────────┐
              │    Shared Dense (128)     │
              │    BatchNorm + ReLU       │
              │    Dropout(0.4)          │
              └────────────┬─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼                         ▼
    ┌──────────────────┐     ┌──────────────────┐
    │ Depression Head   │     │   Stress Head     │
    │ Dense(32) → σ(1)  │     │ Dense(32) → σ(1)  │
    │ name='depression' │     │ name='stress'     │
    └──────────────────┘     └──────────────────┘
```

#### Why This Works
- **Complementary representations**: XGBoost on handcrafted features achieves the best stress F1 (0.72), while TSception captures temporal dynamics that handcrafted features miss. The gate learns when to trust which stream.
- **Channel mismatch resilience**: Handcrafted features like FAA and spectral entropy are more robust to zero-padded channels because they compute relative/normalized metrics. The gate can upweight the handcrafted stream for samples where deep features are degraded by padding.
- **Neuroscience grounding**: FAA is a clinically established biomarker for depression, while spectral entropy and band power ratios are validated stress indicators. Embedding this domain knowledge directly gives the model a strong prior.

#### Training Strategy
- Feature extraction is performed as a preprocessing step (non-trainable stream 2)
- The TSception backbone (stream 1) is trained end-to-end
- The handcrafted feature branch uses a trainable Dense projection
- Masked binary cross-entropy for multi-label training
- Class-weighted sample weighting for imbalance

---

### Pipeline 2: Hierarchical Attention-Gated Multi-Scale Network (HAMS-Net)

#### Core Idea
Address the **temporal diversity problem** (stress manifests at different timescales across paradigms) by processing raw EEG at **three temporal resolutions** simultaneously, then using a **cross-scale attention mechanism** to dynamically weight the contribution of each scale for each sample. Additionally, incorporate a **channel reliability mask** that learns to downweight zero-padded channels.

#### Architecture

```
                         RAW EEG INPUT
                        (batch, 512, 6, 1)
                              │
                              ▼
               ┌──────────────────────────────┐
               │   Channel Reliability Gate    │
               │                              │
               │   For each of 6 channels:    │
               │   r_c = σ(W_r · stats(ch_c)) │
               │   where stats = [mean, var,  │
               │   max, min, energy]          │
               │                              │
               │   X_masked = X ⊙ R           │
               │   (R broadcast over time)    │
               └──────────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌────────────┐   ┌────────────┐   ┌────────────┐
     │ Scale 1:   │   │ Scale 2:   │   │ Scale 3:   │
     │ Fine-grain │   │ Mid-grain  │   │ Coarse     │
     │            │   │            │   │            │
     │ Conv1D     │   │ Conv1D     │   │ Conv1D     │
     │ k=16,32   │   │ k=64,128  │   │ k=256     │
     │ (captures  │   │ (captures  │   │ (captures  │
     │  spikes,   │   │  alpha/    │   │  slow      │
     │  HF noise) │   │  beta      │   │  trends,   │
     │            │   │  rhythms)  │   │  delta)    │
     │ + BiLSTM   │   │ + BiLSTM   │   │ + BiLSTM   │
     │ (32 units) │   │ (32 units) │   │ (32 units) │
     └─────┬──────┘   └─────┬──────┘   └─────┬──────┘
           │                │                │
           └────────────────┼────────────────┘
                            ▼
               ┌──────────────────────────┐
               │  Cross-Scale Attention    │
               │                          │
               │  S = stack([s1, s2, s3]) │
               │       (batch, 3, d)      │
               │                          │
               │  Q = Linear(S)           │
               │  K = Linear(S)           │
               │  V = Linear(S)           │
               │  Attn = softmax(QK^T/√d)│
               │  out = Attn · V          │
               │  → Flatten               │
               └─────────────┬────────────┘
                             ▼
               ┌──────────────────────────┐
               │   Dense(128) → BN → ReLU │
               │   Dropout(0.4)          │
               └────────────┬─────────────┘
                            │
               ┌────────────┼────────────┐
               ▼                         ▼
     ┌──────────────────┐     ┌──────────────────┐
     │ Depression Head   │     │   Stress Head     │
     │ Dense(32) → σ(1)  │     │ Dense(32) → σ(1)  │
     └──────────────────┘     └──────────────────┘
```

#### Why This Works
- **Multi-scale temporal capture**: Stress from arithmetic tasks may show fast beta-band bursts (Scale 1), while depression shows slow alpha asymmetry shifts (Scale 3). Different kernel sizes isolate these.
- **Cross-scale attention**: Learns which temporal resolution matters for each sample. An AMIGOS arousal sample might attend more to mid-grain (emotional response rhythms), while a Cognitive Load sample attends to fine-grain (cognitive spike patterns).
- **Channel reliability gate**: Explicitly learns to ignore/downweight zero-padded channels (e.g., for MODMA 3-ch and Cognitive Load where only 2-3 of 6 channels have real data). This is more principled than hoping the model implicitly learns to ignore zeros.
- **BiLSTM per scale**: Captures temporal dependencies within each resolution, addressing the failure of standalone LSTM/CNN-LSTM on stress (which may be due to single-scale temporal modeling).

#### Training Strategy
- End-to-end training with multi-label masked loss
- The channel reliability gate is learned jointly
- Curriculum learning: train on single-dataset batches for first 5 epochs (warm-up), then mix datasets
- Gradient accumulation for effective batch size of 256

---

### Pipeline 3: Contrastive Domain-Adapted Ensemble (CoDAE)

#### Core Idea
Instead of a single model, use an **ensemble of specialized sub-models** that are aligned in a shared feature space via **supervised contrastive learning**. This directly addresses the domain shift problem that limits stress detection. Each sub-model specializes in a different aspect of the signal, and a **meta-learner** combines their predictions.

#### Architecture

```
                         RAW EEG INPUT
                        (batch, 512, 6, 1)
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
  ┌──────────────────┐ ┌──────────────┐ ┌──────────────────┐
  │  Sub-Model A:    │ │ Sub-Model B: │ │  Sub-Model C:    │
  │  EEGNet          │ │ TSception    │ │  EEGTransformer  │
  │  (spatial filter │ │ (multi-scale │ │  (global context │
  │   specialist)    │ │  temporal)   │ │   via attention) │
  │                  │ │              │ │                  │
  │  → Embedding     │ │ → Embedding  │ │  → Embedding     │
  │    e_A (64-dim)  │ │   e_B (64-d) │ │    e_C (64-dim)  │
  └───────┬──────────┘ └──────┬───────┘ └────────┬─────────┘
          │                   │                  │
          └───────────────────┼──────────────────┘
                              │
                              ▼
               ┌──────────────────────────────┐
               │   Contrastive Alignment       │
               │   (during training only)      │
               │                              │
               │   L_contrast = SupCon(       │
               │     [e_A, e_B, e_C],          │
               │     y_task, dataset_id)       │
               │                              │
               │   Pull same-class embeddings  │
               │   together ACROSS datasets,   │
               │   push different classes apart │
               └──────────────────────────────┘
                              │
          ┌───────────────────┼──────────────────┐
          │                   │                  │
          ▼                   ▼                  ▼
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │  Pred_A:     │   │  Pred_B:     │   │  Pred_C:     │
  │  Dense(32)   │   │  Dense(32)   │   │  Dense(32)   │
  │  → σ(1)      │   │  → σ(1)      │   │  → σ(1)      │
  │  (per task)  │   │  (per task)  │   │  (per task)  │
  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
         │                  │                  │
         └──────────────────┼──────────────────┘
                            ▼
               ┌──────────────────────────────┐
               │     Meta-Learner (Stacking)   │
               │                              │
               │  Input: [p_A_dep, p_A_str,   │
               │          p_B_dep, p_B_str,   │
               │          p_C_dep, p_C_str,   │
               │          dataset_id_onehot]  │
               │                              │
               │  Dense(16) → ReLU → Dense(2) │
               │  → sigmoid (dep, stress)     │
               └──────────────────────────────┘
```

#### Why This Works
- **Architecture diversity**: EEGNet excels at spatial filtering (depthwise convolutions learn frontal asymmetry patterns), TSception captures multi-scale temporal dynamics, and EEGTransformer captures long-range dependencies via self-attention. Each sees different aspects of the same signal.
- **Supervised contrastive alignment**: By pulling same-class embeddings together across datasets (e.g., "stressed" from SAM40 near "stressed" from AMIGOS) and pushing different classes apart, the sub-models learn domain-invariant representations. This is more powerful than the current DANN approach because:
  - DANN only has one adversarial signal; contrastive learning uses every pair as a training signal
  - SupCon directly optimizes the feature space geometry rather than indirectly via gradient reversal
- **Dataset-aware meta-learner**: The stacking meta-learner receives dataset_id as input, allowing it to learn dataset-specific weighting of sub-model predictions. For example, it can learn that EEGNet is more reliable for SAM40 (which has all 6 channels) while TSception is better for Cognitive Load (fewer real channels).
- **Robustness through diversity**: Ensemble methods consistently outperform individual models in EEG literature, especially when base learners are architecturally diverse.

#### Training Strategy (3 Phases)

**Phase 1: Pre-train sub-models (independent)**
- Train each sub-model (EEGNet, TSception, EEGTransformer) independently on the combined dataset with multi-label masked loss
- Standard training as currently done in `run_models.py`

**Phase 2: Contrastive fine-tuning (joint)**
- Freeze classification heads, add contrastive loss on embeddings
- Total loss = L_task (masked BCE) + λ · L_contrast (supervised contrastive)
- λ annealed from 0.1 → 1.0 over 10 epochs
- This aligns the embedding spaces of all three sub-models

**Phase 3: Train meta-learner (frozen sub-models)**
- Freeze all sub-models
- Collect predictions on validation set
- Train a lightweight Dense meta-learner on stacked predictions + dataset_id
- Uses 5-fold cross-validation to prevent overfitting

---

### Pipeline 4: Spectro-Temporal Transformer with Adaptive Domain Normalization (ST²-ADN)

#### Core Idea
A novel architecture that first converts raw EEG into a **time-frequency representation** (spectrogram), then processes it with a **Vision Transformer (ViT)-style patch embedding**, combined with **Adaptive Instance Normalization (AdaIN)** conditioned on dataset identity to explicitly handle domain shift at the feature normalization level.

#### Architecture

```
                     RAW EEG INPUT
                    (batch, 512, 6, 1)
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Spectrogram Transform       │
           │  (Non-trainable)             │
           │                              │
           │  Per-channel STFT:           │
           │  window=64, hop=16, nfft=128 │
           │  → (batch, 6, F=65, T=29)   │
           │  (at 128 Hz, window≈0.5s)    │
           │                              │
           │  Stack channels as "image"   │
           │  → (batch, 6, 65, 29)        │
           │  Reshape → (batch, 65, 29, 6)│
           │  (Freq × Time × Channels)    │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Adaptive Domain Norm (AdaIN) │
           │                              │
           │  dataset_id → Embedding(4,d) │
           │  → MLP → (γ, β) per feature  │
           │                              │
           │  X_norm = γ · InstanceNorm(X) │
           │          + β                 │
           │                              │
           │  (Normalizes per-dataset      │
           │   spectral statistics)       │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Patch Embedding (ViT-style)  │
           │                              │
           │  Patch size: (13, 5) →       │
           │  ~5 freq × ~6 time patches   │
           │  ≈ 30 patches total          │
           │                              │
           │  Conv2D(d_model=128,          │
           │    (13,5), stride=(13,5))    │
           │  + Learnable CLS token       │
           │  + Positional embedding      │
           │  → (batch, 31, 128)          │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Transformer Encoder          │
           │  (4 layers, 8 heads, d=128)  │
           │                              │
           │  Each layer:                 │
           │    MHSA → LayerNorm          │
           │    FFN(256) → LayerNorm      │
           │    Dropout(0.3)              │
           │                              │
           │  Use CLS token output        │
           │  → (batch, 128)              │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │  Frequency-Band Attention     │
           │  (interpretability module)   │
           │                              │
           │  Learns attention weights    │
           │  over frequency patches:     │
           │  "which bands matter?"       │
           │                              │
           │  Produces interpretable      │
           │  band importance scores      │
           └──────────────┬───────────────┘
                          │
             ┌────────────┼────────────┐
             ▼                         ▼
   ┌──────────────────┐     ┌──────────────────┐
   │ Depression Head   │     │   Stress Head     │
   │ Dense(64) → ReLU  │     │ Dense(64) → ReLU  │
   │ Dense(1) → σ      │     │ Dense(1) → σ      │
   └──────────────────┘     └──────────────────┘
```

#### Why This Works
- **Time-frequency representation**: Converting raw EEG to spectrograms makes frequency-band information (δ, θ, α, β, γ) spatially explicit. The current models process raw time-series and must implicitly learn frequency decomposition. A spectrogram makes this trivially available.
- **Adaptive Instance Normalization**: The key insight is that different datasets have different spectral statistics due to hardware differences (Emotiv EPOC at 128 Hz vs. research-grade at 256 Hz), different recording conditions, and different subject populations. AdaIN normalizes these statistics per-dataset while preserving discriminative patterns. This is fundamentally different from DANN — DANN tries to make features indistinguishable across domains, but AdaIN instead adapts the normalization to each domain while keeping the same discriminative features.
- **ViT-style processing**: Treating the spectrogram as a 2D "image" and patching it lets the Transformer attend across both time and frequency simultaneously. Each patch represents a specific time-frequency region (e.g., "alpha band at 1.0-1.3 seconds"), making attention patterns directly interpretable.
- **Frequency-band attention**: Provides interpretability — the model can tell us "depression classification relies heavily on alpha-band patches" or "stress classification attends to beta-gamma patches", which aligns with neuroscience literature and adds research value.
- **CLS token**: A learnable classification token (as in BERT/ViT) aggregates information from all patches, providing a fixed-size representation regardless of spectrogram dimensions.

#### Training Strategy
- Spectrogram computation is a fixed (non-trainable) transform, applied as preprocessing
- AdaIN parameters are learned jointly with the rest of the model
- Training uses the masked multi-label loss
- The frequency-band attention weights can be visualized post-training for interpretability analysis
- Label smoothing (0.1) to prevent overconfident predictions
- Cosine annealing learning rate schedule with warm restarts

---

## 3. Pipeline Comparison Summary

| Aspect | Pipeline 1: DSFFN | Pipeline 2: HAMS-Net | Pipeline 3: CoDAE | Pipeline 4: ST²-ADN |
|---|---|---|---|---|
| **Core Innovation** | Gated fusion of handcrafted + deep features | Multi-scale temporal + channel reliability gate | Contrastive-aligned ensemble + meta-learner | Spectrogram ViT + adaptive domain normalization |
| **Addresses Stress Bottleneck** | Leverages strong XGBoost features | Multi-scale captures diverse stress paradigms | Ensemble diversity + contrastive domain alignment | AdaIN handles spectral domain shift |
| **Handles Channel Mismatch** | Handcrafted features are mismatch-robust | Explicit channel reliability gate | Dataset-aware meta-learner weighting | AdaIN normalizes per-dataset statistics |
| **Handles Domain Shift** | Indirect (via robust features) | Indirect (via multi-scale diversity) | Direct (contrastive + meta-learner) | Direct (AdaIN conditioning) |
| **Model Complexity** | Medium (TSception + MLP fusion) | High (3 scales × BiLSTM + attention) | High (3 sub-models + meta-learner) | Medium-High (spectrogram + ViT) |
| **Training Complexity** | Simple (end-to-end) | Medium (curriculum learning) | High (3 phases) | Medium (end-to-end with preprocessing) |
| **Interpretability** | Low-Medium (gate weights) | Medium (scale attention weights) | Low (black-box ensemble) | **High** (frequency-band attention maps) |
| **Novelty** | Medium (fusion is well-studied, but gated fusion of domain features + deep for EEG is less common) | High (channel reliability gate is novel for EEG) | Medium-High (contrastive + ensemble for EEG domain adaptation is novel) | **High** (AdaIN for EEG domain shift is novel; spectrogram ViT for EEG multi-label is novel) |
| **Research Value** | Good | Very Good | Good | **Excellent** (interpretable + novel) |

---

## 4. Recommended Implementation Order

> [!TIP]
> Start with Pipeline 1 (DSFFN) as the baseline — it's the simplest to implement and directly builds on your existing infrastructure (TSception backbone + feature extraction are already done). Then progress to more complex pipelines.

1. **Pipeline 1 (DSFFN)** — Quick win, leverages existing code, validates fusion hypothesis
2. **Pipeline 4 (ST²-ADN)** — Highest novelty and research value, excellent interpretability
3. **Pipeline 2 (HAMS-Net)** — If channel mismatch proves critical, this directly addresses it
4. **Pipeline 3 (CoDAE)** — Most complex training, save for last; use if ensemble gains are needed

---

## 5. Open Questions for User

> [!IMPORTANT]
> Please review and provide feedback on the following:

1. **Pipeline selection**: Which pipeline(s) do you want to implement? All four, or a subset?
2. **Spectrogram preprocessing** (Pipeline 4): Should the STFT be computed on-the-fly or precomputed and cached like the current features?
3. **MODMA 128-channel data**: The 128-channel resting-state MODMA data is available but unused. It has all 6 frontal channels natively (no padding needed). Should we integrate it into the combined dataset to improve depression detection data quality?
4. **Clinical metadata** (MODMA): PHQ-9, GAD-7, and other clinical scores are available. Should any pipeline incorporate these as auxiliary features or targets?
5. **Evaluation protocol**: Should we keep the current 80/20 group-based split, or add k-fold cross-validation for more robust comparison?
