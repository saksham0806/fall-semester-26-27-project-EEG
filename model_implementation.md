# Hybrid Model Implementation Guide

> This document provides step-by-step implementation instructions for each of the 4 proposed hybrid model pipelines from `plan.md`. Every instruction is grounded in the **exact existing codebase** — specific function signatures, layer indices, tensor shapes, and file paths.

---

## 0. Existing Codebase Reference

Before implementing any pipeline, understand these exact interfaces that all pipelines must integrate with.

### 0.1 Data Loading Convention

```python
# File locations (relative to PROJECT_ROOT)
X = np.load("datasets/combined_data/X.npy")                      # shape: (55157, 6, 512)
y_depression = np.load("datasets/combined_data/y_depression.npy") # shape: (55157,) — values: 0, 1, -1
y_stress = np.load("datasets/combined_data/y_stress.npy")         # shape: (55157,) — values: 0, 1, -1
subjects = np.load("datasets/combined_data/subjects.npy")         # shape: (55157,) — subject IDs
datasets = np.load("datasets/combined_data/datasets.npy")         # shape: (55157,) — values: 0, 1, 2, 3

# DL input: add trailing channel dimension
X_dl = X[..., np.newaxis]  # (55157, 6, 512) → (55157, 6, 512, 1)

# Features (precomputed or extracted)
features = extract_all_features(X, fs=128)  # (55157, 81)
```

### 0.2 Signal Specifications

| Parameter | Value |
|---|---|
| Channels | 6 (Fp1, Fp2, F7, F3, F4, F8) |
| Sampling Rate | 128 Hz |
| Segment Duration | 4 seconds |
| Samples per Segment | 512 |
| DL Input Shape | `(batch, 6, 512, 1)` |
| Feature Vector Size | 81 |

### 0.3 Existing Model Backbone Outputs (Flatten/Embedding Dimensions)

These are the embedding dimensions available when extracting backbone features from existing models:

| Model | Flatten Layer Index | Embedding Dim | How to Extract |
|---|---|---|---|
| **TSception** | Layer 10 | **768** | `base.layers[10].output` (after Flatten) |
| **EEGNet** | Layer 13 (Flatten) | **256** | `base.layers[13].output` |
| **EEGTransformer** | GlobalAveragePooling1D | **64** | `base.layers[-5].output` (after GAP) |

### 0.4 Training Constants

```python
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 5e-4
EARLY_STOP = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)
```

### 0.5 Existing Masked Loss

```python
# Already implemented in src/proposed_model/multitask_dann_models.py
def masked_binary_crossentropy(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_safe = tf.where(tf.equal(y_true, -1), tf.zeros_like(y_true), y_true)
    y_true_safe = tf.cast(y_true_safe, tf.float32)
    bce = tf.keras.losses.binary_crossentropy(y_true_safe, y_pred)
    masked_bce = bce * mask
    return tf.reduce_sum(masked_bce) / (tf.reduce_sum(mask) + 1e-7)
```

### 0.6 File Structure Convention

All new model code goes into `src/proposed_model/`. Each pipeline gets its own file:

```
src/proposed_model/
├── __init__.py                    # UPDATE: add new exports
├── features.py                    # EXISTING: 81-feature extraction
├── classical_models.py            # EXISTING: SVM, RF, XGBoost
├── deep_learning_baselines.py     # EXISTING: 1D CNN, LSTM, CNN-LSTM
├── eeg_models.py                  # EXISTING: EEGNet, TSception, EEGTransformer
├── multitask_dann_models.py       # EXISTING: Multi-Task, DANN
├── pipeline1_dsffn.py             # NEW: Dual-Stream Feature-Fusion Network
├── pipeline2_hamsnet.py           # NEW: Hierarchical Attention-Gated Multi-Scale Network
├── pipeline3_codae.py             # NEW: Contrastive Domain-Adapted Ensemble
└── pipeline4_st2adn.py            # NEW: Spectro-Temporal Transformer + AdaIN
```

Each pipeline also gets a runner script:

```
run_pipeline1.py                   # NEW: Training/evaluation for DSFFN
run_pipeline2.py                   # NEW: Training/evaluation for HAMS-Net
run_pipeline3.py                   # NEW: Training/evaluation for CoDAE
run_pipeline4.py                   # NEW: Training/evaluation for ST²-ADN
```

---

## Pipeline 1: Dual-Stream Feature-Fusion Network (DSFFN)

### 1.1 Overview

**File**: `src/proposed_model/pipeline1_dsffn.py`

Two parallel input streams — raw EEG through a TSception backbone and precomputed handcrafted features — fused via a learnable gating mechanism, with dual task heads for depression and stress.

### 1.2 Inputs

| Input | Shape | Source |
|---|---|---|
| `raw_eeg` | `(batch, 6, 512, 1)` | `X[..., np.newaxis]` |
| `handcrafted_features` | `(batch, 81)` | `extract_all_features(X, fs=128)` |

### 1.3 Architecture — Layer by Layer

```
STREAM 1: Deep Feature Extractor
──────────────────────────────────────────────
Input: raw_eeg (batch, 6, 512, 1)

Step 1: Create TSception backbone
    base_model = TSception(nb_classes=1, Chans=6, Samples=512, sampling_rate=128)
    backbone_output = base_model.layers[10].output    # Flatten → (batch, 768)

Step 2: Project to d₁
    d1 = Dense(128, activation='relu', name='deep_proj')(backbone_output)
    d1 = BatchNormalization(name='deep_bn')(d1)       # → (batch, 128)


STREAM 2: Handcrafted Feature Projector
──────────────────────────────────────────────
Input: handcrafted_features (batch, 81)

Step 3: Project to d₂
    d2 = Dense(64, activation='relu', name='hc_proj')(handcrafted_features)
    d2 = BatchNormalization(name='hc_bn')(d2)         # → (batch, 64)


GATED FUSION MODULE
──────────────────────────────────────────────
Step 4: Make d₁ and d₂ same dimension for gating
    d1_reduced = Dense(64, activation='relu', name='d1_reduce')(d1)   # → (batch, 64)

Step 5: Compute gate
    concat = Concatenate()([d1_reduced, d2])           # → (batch, 128)
    gate = Dense(64, activation='sigmoid', name='gate')(concat)  # → (batch, 64)

Step 6: Apply gated fusion
    fused = gate * d1_reduced + (1 - gate) * d2        # → (batch, 64)
    # Use Lambda or custom layer for element-wise ops

Step 7: Also concatenate the full d₁ for extra capacity
    combined = Concatenate()([fused, d1])               # → (batch, 192)


SHARED REPRESENTATION
──────────────────────────────────────────────
Step 8: Shared dense
    shared = Dense(128, activation='relu', name='shared_dense')(combined)
    shared = BatchNormalization(name='shared_bn')(shared)
    shared = Dropout(0.4, name='shared_dropout')(shared)  # → (batch, 128)


TASK HEADS
──────────────────────────────────────────────
Step 9: Depression head
    dep = Dense(32, activation='relu', name='dep_hidden')(shared)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep)

Step 10: Stress head
    str = Dense(32, activation='relu', name='str_hidden')(shared)
    str_out = Dense(1, activation='sigmoid', name='stress')(str)
```

### 1.4 Implementation Steps

#### Step 1: Create `src/proposed_model/pipeline1_dsffn.py`

```python
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout,
    Concatenate, Multiply, Lambda
)
from tensorflow.keras.models import Model
from .eeg_models import TSception


def build_dsffn(Chans=6, Samples=512, sampling_rate=128,
                deep_proj_dim=128, hc_proj_dim=64, shared_dim=128,
                dropout_rate=0.4):
    """
    Dual-Stream Feature-Fusion Network.

    Inputs:
        raw_eeg:              (batch, Chans, Samples, 1)
        handcrafted_features: (batch, 81)

    Outputs:
        depression: (batch, 1) — sigmoid
        stress:     (batch, 1) — sigmoid
    """
    # --- Input layers ---
    raw_input = Input(shape=(Chans, Samples, 1), name='raw_eeg')
    hc_input = Input(shape=(81,), name='handcrafted_features')

    # --- Stream 1: TSception backbone (frozen or trainable) ---
    base_model = TSception(
        nb_classes=1, Chans=Chans, Samples=Samples,
        sampling_rate=sampling_rate
    )
    # Extract up to Flatten (layer index 10) → 768-dim
    backbone = Model(
        inputs=base_model.input,
        outputs=base_model.layers[10].output,
        name='tsception_backbone'
    )
    deep_features = backbone(raw_input)                # (batch, 768)
    d1 = Dense(deep_proj_dim, activation='relu', name='deep_proj')(deep_features)
    d1 = BatchNormalization(name='deep_bn')(d1)        # (batch, 128)

    # --- Stream 2: Handcrafted feature projection ---
    d2 = Dense(hc_proj_dim, activation='relu', name='hc_proj')(hc_input)
    d2 = BatchNormalization(name='hc_bn')(d2)          # (batch, 64)

    # --- Gated Fusion ---
    d1_reduced = Dense(hc_proj_dim, activation='relu', name='d1_reduce')(d1)  # (batch, 64)
    concat_for_gate = Concatenate(name='gate_input')([d1_reduced, d2])        # (batch, 128)
    gate = Dense(hc_proj_dim, activation='sigmoid', name='gate')(concat_for_gate)  # (batch, 64)

    # fused = gate ⊙ d1_reduced + (1-gate) ⊙ d2
    gated_deep = Multiply(name='gated_deep')([gate, d1_reduced])
    inv_gate = Lambda(lambda g: 1.0 - g, name='inv_gate')(gate)
    gated_hc = Multiply(name='gated_hc')([inv_gate, d2])
    fused = tf.keras.layers.Add(name='fused')([gated_deep, gated_hc])  # (batch, 64)

    # Concatenate fused representation with full deep projection for capacity
    combined = Concatenate(name='combined')([fused, d1])  # (batch, 192)

    # --- Shared representation ---
    shared = Dense(shared_dim, activation='relu', name='shared_dense')(combined)
    shared = BatchNormalization(name='shared_bn')(shared)
    shared = Dropout(dropout_rate, name='shared_dropout')(shared)  # (batch, 128)

    # --- Task Heads ---
    dep_hidden = Dense(32, activation='relu', name='dep_hidden')(shared)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep_hidden)

    str_hidden = Dense(32, activation='relu', name='str_hidden')(shared)
    str_out = Dense(1, activation='sigmoid', name='stress')(str_hidden)

    model = Model(
        inputs=[raw_input, hc_input],
        outputs=[dep_out, str_out],
        name='DSFFN'
    )
    return model
```

#### Step 2: Create `run_pipeline1.py`

This script follows the same pattern as `run_models.py` but handles dual inputs:

```python
# Key differences from run_models.py:

# 1. Load both raw data AND features
X, y_depression, y_stress, subjects, datasets = load_data()
features = get_or_extract_features(X)  # (N, 81)

# 2. Prepare dual inputs for training
X_train_dl = X_train_full[..., np.newaxis]     # (N_train, 6, 512, 1)
feat_train = features[train_mask]               # (N_train, 81)

# 3. Build and compile model
model = build_dsffn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={
        'depression': masked_binary_crossentropy,
        'stress': masked_binary_crossentropy
    },
    metrics={
        'depression': 'accuracy',
        'stress': 'accuracy'
    }
)

# 4. Train with dual inputs
model.fit(
    [X_train_dl, feat_train],                      # ← dual inputs
    [y_dep_train_full, y_str_train_full],           # ← dual targets
    validation_split=0.1,
    epochs=25,
    batch_size=128,
    sample_weight=[sample_weight_dep, sample_weight_str],
    callbacks=[early_stop],
    verbose=1
)

# 5. Evaluate
dep_probs, str_probs = model.predict([X_test_dl, feat_test], verbose=0)
```

#### Step 3: Update `src/proposed_model/__init__.py`

Add export:
```python
from .pipeline1_dsffn import build_dsffn
```

### 1.5 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam, lr=5e-4 | Consistent with existing pipeline |
| Loss | `masked_binary_crossentropy` (both heads) | Existing loss, handles -1 labels |
| Epochs | 25 | Consistent with existing pipeline |
| Batch Size | 128 | Consistent with existing pipeline |
| Early Stopping | patience=5, monitor='val_loss' | Existing callback |
| Dropout | 0.4 (shared), 0.5 (backbone internal) | Slightly lower shared to preserve fusion info |
| Class Weights | Via sample_weight (same as multi-task) | Existing imbalance handling |

### 1.6 Evaluation Procedure

Evaluate on the same splits as `run_models.py`:
1. Depression evaluation: filter test set to `y_depression != -1`
2. Stress evaluation: filter test set to `y_stress != -1`
3. Compute: Accuracy, Precision, Recall, F1-Score, ROC-AUC
4. Compare against:
   - Multi-Task TSception baseline (Depression: 0.8839 acc, Stress: 0.5481 acc)
   - Best standalone models (Depression CNN-LSTM: 0.8960, Stress XGBoost: 0.6280)

### 1.7 Gate Analysis (Post-training)

After training, extract gate values to analyze which stream the model trusts per dataset:

```python
gate_model = Model(inputs=model.inputs, outputs=model.get_layer('gate').output)
gate_values = gate_model.predict([X_test_dl, feat_test])  # (N_test, 64)

# Average gate per dataset
for ds_id in [0, 1, 2, 3]:
    mask = datasets[test_mask] == ds_id
    avg_gate = gate_values[mask].mean()
    print(f"Dataset {ds_id}: avg gate = {avg_gate:.3f}")
    # gate → 1.0 means trusting deep stream
    # gate → 0.0 means trusting handcrafted stream
```

---

## Pipeline 2: Hierarchical Attention-Gated Multi-Scale Network (HAMS-Net)

### 2.1 Overview

**File**: `src/proposed_model/pipeline2_hamsnet.py`

A channel reliability gate suppresses zero-padded channels, then three parallel Conv1D+BiLSTM branches at different temporal scales extract multi-resolution features. A cross-scale self-attention mechanism dynamically weights scales, feeding into dual task heads.

### 2.2 Input

| Input | Shape | Source |
|---|---|---|
| `raw_eeg` | `(batch, 6, 512, 1)` | `X[..., np.newaxis]` |

### 2.3 Architecture — Layer by Layer

```
CHANNEL RELIABILITY GATE
──────────────────────────────────────────────
Input: raw_eeg (batch, 6, 512, 1)

Step 1: Reshape to (batch, 512, 6) for per-channel statistics
    x = Reshape((6, 512))(raw_eeg_input)     # strip trailing 1
    x = Permute((2, 1))(x)                    # → (batch, 512, 6)

Step 2: Compute per-channel statistics
    # For each channel c ∈ [0..5], compute 5 stats over time dimension:
    # [mean, variance, max, min, energy(=sum of squares)]
    # → (batch, 6, 5) → flatten to (batch, 30)
    # Use Lambda layer or custom layer

Step 3: Learn reliability scores
    channel_stats = Lambda(compute_channel_stats)(x)  # → (batch, 30)
    reliability = Dense(6, activation='sigmoid',
                        name='channel_reliability')(channel_stats)  # → (batch, 6)

Step 4: Apply channel mask
    # Broadcast reliability over time: (batch, 6) → (batch, 512, 6)
    r_expanded = RepeatVector(512)(reliability)        # → (batch, 512, 6)
    # Note: RepeatVector gives (batch, 512, 6) which aligns with x's (batch, 512, 6)
    x_masked = Multiply()([x, r_expanded])             # → (batch, 512, 6)


MULTI-SCALE TEMPORAL BRANCHES
──────────────────────────────────────────────
Scale 1: Fine-grain (high-frequency events, beta/gamma bursts)

Step 5a: Conv1D blocks with small kernels
    s1 = Conv1D(32, kernel_size=16, activation='relu', padding='same')(x_masked)
    s1 = BatchNormalization()(s1)
    s1 = MaxPooling1D(2)(s1)                           # → (batch, 256, 32)
    s1 = Conv1D(32, kernel_size=32, activation='relu', padding='same')(s1)
    s1 = BatchNormalization()(s1)
    s1 = MaxPooling1D(2)(s1)                           # → (batch, 128, 32)

Step 5b: BiLSTM to capture temporal dependencies
    s1 = Bidirectional(LSTM(32))(s1)                   # → (batch, 64)

Scale 2: Mid-grain (alpha/beta rhythms)

Step 6a: Conv1D blocks with medium kernels
    s2 = Conv1D(32, kernel_size=64, activation='relu', padding='same')(x_masked)
    s2 = BatchNormalization()(s2)
    s2 = MaxPooling1D(2)(s2)                           # → (batch, 256, 32)
    s2 = Conv1D(32, kernel_size=128, activation='relu', padding='same')(s2)
    s2 = BatchNormalization()(s2)
    s2 = MaxPooling1D(2)(s2)                           # → (batch, 128, 32)

Step 6b: BiLSTM
    s2 = Bidirectional(LSTM(32))(s2)                   # → (batch, 64)

Scale 3: Coarse (delta/theta, slow trends)

Step 7a: Conv1D blocks with large kernels
    s3 = Conv1D(32, kernel_size=128, activation='relu', padding='same')(x_masked)
    s3 = BatchNormalization()(s3)
    s3 = MaxPooling1D(4)(s3)                           # → (batch, 128, 32)
    s3 = Conv1D(32, kernel_size=256, activation='relu', padding='same')(s3)
    s3 = BatchNormalization()(s3)
    s3 = MaxPooling1D(2)(s3)                           # → (batch, 64, 32)

Step 7b: BiLSTM
    s3 = Bidirectional(LSTM(32))(s3)                   # → (batch, 64)


CROSS-SCALE ATTENTION
──────────────────────────────────────────────
Step 8: Stack scale embeddings
    S = tf.stack([s1, s2, s3], axis=1)                 # → (batch, 3, 64)

Step 9: Self-attention over scales
    attn_output = MultiHeadAttention(
        num_heads=4, key_dim=16
    )(S, S)                                            # → (batch, 3, 64)
    S = LayerNormalization()(S + attn_output)           # Residual + norm

Step 10: Flatten
    scale_features = Flatten()(S)                      # → (batch, 192)


SHARED REPRESENTATION + TASK HEADS
──────────────────────────────────────────────
Step 11:
    shared = Dense(128, activation='relu')(scale_features)
    shared = BatchNormalization()(shared)
    shared = Dropout(0.4)(shared)                      # → (batch, 128)

Step 12: Depression head
    dep = Dense(32, activation='relu')(shared)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep)

Step 13: Stress head
    str = Dense(32, activation='relu')(shared)
    str_out = Dense(1, activation='sigmoid', name='stress')(str)
```

### 2.4 Implementation Steps

#### Step 1: Create `src/proposed_model/pipeline2_hamsnet.py`

```python
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, BatchNormalization, Dropout, Flatten,
    Conv1D, MaxPooling1D, LSTM, Bidirectional,
    Reshape, Permute, Multiply, RepeatVector,
    Concatenate, Lambda, LayerNormalization
)
from tensorflow.keras.models import Model


def _compute_channel_stats(x):
    """
    x: (batch, 512, 6)
    Returns: (batch, 30) — 5 stats per channel
    """
    mean = tf.reduce_mean(x, axis=1)                      # (batch, 6)
    var = tf.math.reduce_variance(x, axis=1)               # (batch, 6)
    max_val = tf.reduce_max(x, axis=1)                     # (batch, 6)
    min_val = tf.reduce_min(x, axis=1)                     # (batch, 6)
    energy = tf.reduce_sum(x ** 2, axis=1)                 # (batch, 6)
    return tf.concat([mean, var, max_val, min_val, energy], axis=-1)  # (batch, 30)


def _build_scale_branch(x, conv_kernels, pool_sizes, lstm_units, name_prefix):
    """
    Builds one temporal scale branch: Conv1D blocks → BiLSTM.

    Args:
        x: (batch, 512, 6) — channel-masked input
        conv_kernels: list of 2 kernel sizes, e.g. [16, 32]
        pool_sizes: list of 2 pool sizes, e.g. [2, 2]
        lstm_units: int, e.g. 32
        name_prefix: string for unique layer naming
    Returns:
        (batch, lstm_units*2) — BiLSTM output
    """
    h = Conv1D(32, conv_kernels[0], activation='relu', padding='same',
               name=f'{name_prefix}_conv1')(x)
    h = BatchNormalization(name=f'{name_prefix}_bn1')(h)
    h = MaxPooling1D(pool_sizes[0], name=f'{name_prefix}_pool1')(h)

    h = Conv1D(32, conv_kernels[1], activation='relu', padding='same',
               name=f'{name_prefix}_conv2')(h)
    h = BatchNormalization(name=f'{name_prefix}_bn2')(h)
    h = MaxPooling1D(pool_sizes[1], name=f'{name_prefix}_pool2')(h)

    h = Bidirectional(
        LSTM(lstm_units), name=f'{name_prefix}_bilstm'
    )(h)
    return h


def build_hamsnet(Chans=6, Samples=512, lstm_units=32,
                  shared_dim=128, dropout_rate=0.4):
    """
    Hierarchical Attention-Gated Multi-Scale Network (HAMS-Net).

    Input:  raw_eeg — (batch, Chans, Samples, 1)
    Outputs: depression (batch, 1), stress (batch, 1)
    """
    raw_input = Input(shape=(Chans, Samples, 1), name='raw_eeg')

    # --- Reshape: (batch, 6, 512, 1) → (batch, 512, 6) ---
    x = Reshape((Chans, Samples), name='strip_channel_dim')(raw_input)
    x = Permute((2, 1), name='to_time_first')(x)  # (batch, 512, 6)

    # --- Channel Reliability Gate ---
    ch_stats = Lambda(_compute_channel_stats, name='channel_stats')(x)  # (batch, 30)
    reliability = Dense(Chans, activation='sigmoid',
                        name='channel_reliability')(ch_stats)            # (batch, 6)
    r_expanded = RepeatVector(Samples, name='expand_reliability')(reliability)  # (batch, 512, 6)
    x_masked = Multiply(name='apply_channel_mask')([x, r_expanded])     # (batch, 512, 6)

    # --- Multi-Scale Branches ---
    s1 = _build_scale_branch(x_masked, conv_kernels=[16, 32],
                             pool_sizes=[2, 2], lstm_units=lstm_units,
                             name_prefix='fine')      # (batch, 64)

    s2 = _build_scale_branch(x_masked, conv_kernels=[64, 128],
                             pool_sizes=[2, 2], lstm_units=lstm_units,
                             name_prefix='mid')       # (batch, 64)

    s3 = _build_scale_branch(x_masked, conv_kernels=[128, 256],
                             pool_sizes=[4, 2], lstm_units=lstm_units,
                             name_prefix='coarse')    # (batch, 64)

    # --- Cross-Scale Attention ---
    # Stack: (batch, 3, 64)
    S = Lambda(lambda args: tf.stack(args, axis=1),
               name='stack_scales')([s1, s2, s3])

    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=4, key_dim=16, name='cross_scale_attn'
    )(S, S)
    S = LayerNormalization(name='scale_norm')(S + attn_out)
    scale_features = Flatten(name='scale_flatten')(S)  # (batch, 192)

    # --- Shared Representation ---
    shared = Dense(shared_dim, activation='relu', name='shared_dense')(scale_features)
    shared = BatchNormalization(name='shared_bn')(shared)
    shared = Dropout(dropout_rate, name='shared_dropout')(shared)

    # --- Task Heads ---
    dep_h = Dense(32, activation='relu', name='dep_hidden')(shared)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep_h)

    str_h = Dense(32, activation='relu', name='str_hidden')(shared)
    str_out = Dense(1, activation='sigmoid', name='stress')(str_h)

    return Model(inputs=raw_input, outputs=[dep_out, str_out], name='HAMS_Net')
```

#### Step 2: Create `run_pipeline2.py`

```python
# Same structure as run_models.py multi-task section but with single input:

model = build_hamsnet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={
        'depression': masked_binary_crossentropy,
        'stress': masked_binary_crossentropy
    },
    metrics={'depression': 'accuracy', 'stress': 'accuracy'}
)

model.fit(
    X_train_full_dl,                                    # single input
    [y_dep_train_full, y_str_train_full],
    validation_split=0.1,
    epochs=25,
    batch_size=128,
    sample_weight=[sample_weight_dep, sample_weight_str],
    callbacks=[early_stop],
    verbose=1
)
```

#### Step 3: Curriculum Learning (Optional Enhancement)

```python
# Phase 1 (epochs 1-5): Train on single-dataset batches
# Phase 2 (epochs 6-25): Train on mixed batches

class CurriculumCallback(tf.keras.callbacks.Callback):
    def __init__(self, switch_epoch=5):
        self.switch_epoch = switch_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.switch_epoch:
            print(f"Epoch {epoch}: CURRICULUM mode (single-dataset batches)")
        else:
            print(f"Epoch {epoch}: MIXED mode (all datasets)")
```

### 2.5 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam, lr=5e-4 | Consistent |
| Loss | `masked_binary_crossentropy` (both heads) | Handles -1 labels |
| Epochs | 25 (optionally 30 for curriculum) | Allow extra for curriculum warmup |
| Batch Size | 128 | Consistent |
| BiLSTM Units | 32 per direction (64 total) | Balance capacity vs. overfitting |
| Conv1D Filters | 32 per layer | Moderate — 6-channel input doesn't need large filter banks |
| Dropout | 0.4 (shared), BiLSTM has recurrent dropout 0 by default | Legacy showed 0.5+ is critical |

### 2.6 Post-Training Analysis

#### Channel Reliability Visualization

```python
# Extract learned channel reliability scores
reliability_model = Model(
    inputs=model.input,
    outputs=model.get_layer('channel_reliability').output
)
r_scores = reliability_model.predict(X_test_dl)  # (N_test, 6)

# Per-dataset channel reliability
for ds_id, ds_name in enumerate(['MODMA', 'SAM40', 'CogLoad', 'AMIGOS']):
    mask = datasets[test_mask] == ds_id
    avg_r = r_scores[mask].mean(axis=0)
    print(f"{ds_name}: {dict(zip(['Fp1','Fp2','F7','F3','F4','F8'], avg_r.round(3)))}")

# Expected: SAM40 ≈ high across all 6, MODMA ≈ high on Fp1/Fp2, low on F7/F3/F4/F8
```

#### Scale Attention Visualization

```python
# Extract cross-scale attention weights
# Get attention scores from MultiHeadAttention layer
attn_layer = model.get_layer('cross_scale_attn')
# This requires a separate forward pass with return_attention_scores=True
```

---

## Pipeline 3: Contrastive Domain-Adapted Ensemble (CoDAE)

### 3.1 Overview

**Files**:
- `src/proposed_model/pipeline3_codae.py` — Model definitions + contrastive loss
- `run_pipeline3.py` — 3-phase training orchestrator

Three architecturally diverse sub-models (EEGNet, TSception, EEGTransformer) are individually pre-trained, then jointly fine-tuned with a supervised contrastive loss to align their embedding spaces across datasets. Finally, a lightweight meta-learner stacks their predictions.

### 3.2 Training Phases Overview

```
Phase 1: Independent Pre-training (≈3× current training time)
    ┌──────────┐   ┌──────────┐   ┌──────────────┐
    │ EEGNet   │   │TSception │   │EEGTransformer│
    │ train on │   │ train on │   │  train on    │
    │ combined │   │ combined │   │  combined    │
    │ dataset  │   │ dataset  │   │  dataset     │
    └────┬─────┘   └────┬─────┘   └──────┬───────┘
         │              │               │
         ▼              ▼               ▼
    Saved weights  Saved weights   Saved weights

Phase 2: Contrastive Fine-tuning (joint, ≈10 epochs)
    Load Phase 1 weights
    Freeze classification heads
    Add SupCon loss on embeddings
    Train jointly with L_task + λ·L_contrast

Phase 3: Meta-Learner Training (frozen sub-models, ≈seconds)
    Collect Phase 2 model predictions on validation set
    Train Dense(16) → Dense(2) meta-learner
```

### 3.3 Architecture — Sub-Models with Embedding Extraction

Each sub-model needs to expose its **embedding** (the representation before classification heads):

```python
# Sub-Model A: EEGNet
base_a = EEGNet(nb_classes=1, Chans=6, Samples=512)
# Flatten output at layer 13 → 256-dim
embed_a = base_a.layers[13].output                    # (batch, 256)
proj_a = Dense(64, activation='relu', name='proj_a')(embed_a)  # (batch, 64)

# Sub-Model B: TSception
base_b = TSception(nb_classes=1, Chans=6, Samples=512, sampling_rate=128)
# Flatten output at layer 10 → 768-dim
embed_b = base_b.layers[10].output                    # (batch, 768)
proj_b = Dense(64, activation='relu', name='proj_b')(embed_b)  # (batch, 64)

# Sub-Model C: EEGTransformer
base_c = EEGTransformer(nb_classes=1, Chans=6, Samples=512)
# GAP output → 64-dim (already 64, but project for consistency)
embed_c = base_c.layers[-5].output                    # (batch, 64)
proj_c = Dense(64, activation='relu', name='proj_c')(embed_c)  # (batch, 64)
```

### 3.4 Supervised Contrastive Loss

```python
def supervised_contrastive_loss(embeddings, labels, dataset_ids, temperature=0.07):
    """
    SupCon loss that pulls same-class embeddings together
    ACROSS datasets and pushes different classes apart.

    Args:
        embeddings: (batch, embed_dim) — L2-normalized
        labels:     (batch,) — 0 or 1 (task label)
        dataset_ids: (batch,) — 0, 1, 2, or 3
        temperature: float — scaling temperature

    Returns:
        scalar loss

    Key insight: We want stress=1 from SAM40 to be near stress=1 from AMIGOS
    in embedding space, even though their raw signals look very different.
    """
    # Normalize embeddings to unit sphere
    embeddings = tf.math.l2_normalize(embeddings, axis=1)

    # Compute similarity matrix
    sim_matrix = tf.matmul(embeddings, embeddings, transpose_b=True)  # (B, B)
    sim_matrix = sim_matrix / temperature

    # Create label match mask: positive pairs = same class
    labels = tf.cast(labels, tf.int32)
    label_match = tf.equal(
        tf.expand_dims(labels, 1),
        tf.expand_dims(labels, 0)
    )  # (B, B) boolean

    # Cross-dataset bonus: upweight positive pairs from DIFFERENT datasets
    ds_match = tf.equal(
        tf.expand_dims(dataset_ids, 1),
        tf.expand_dims(dataset_ids, 0)
    )  # (B, B) boolean
    cross_ds_weight = tf.where(ds_match, 1.0, 2.0)  # 2× weight for cross-dataset positives

    # Self-mask: exclude diagonal
    batch_size = tf.shape(embeddings)[0]
    self_mask = tf.logical_not(tf.eye(batch_size, dtype=tf.bool))

    # Positive mask: same class AND not self
    pos_mask = tf.logical_and(label_match, self_mask)
    pos_mask_float = tf.cast(pos_mask, tf.float32) * cross_ds_weight

    # Numerator: exp(sim) for positive pairs
    exp_sim = tf.exp(sim_matrix) * tf.cast(self_mask, tf.float32)
    pos_sim = exp_sim * pos_mask_float

    # Denominator: sum of exp(sim) for all non-self pairs
    denom = tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-7

    # Loss: -log(sum_positives / sum_all)
    loss = -tf.math.log(
        tf.reduce_sum(pos_sim, axis=1) / tf.squeeze(denom) + 1e-7
    )

    # Only compute loss for samples with at least one positive pair
    has_pos = tf.cast(tf.reduce_sum(tf.cast(pos_mask, tf.float32), axis=1) > 0, tf.float32)
    loss = tf.reduce_sum(loss * has_pos) / (tf.reduce_sum(has_pos) + 1e-7)

    return loss
```

### 3.5 Implementation Steps

#### Step 1: Create `src/proposed_model/pipeline3_codae.py`

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
from .eeg_models import EEGNet, TSception, EEGTransformer
from .multitask_dann_models import masked_binary_crossentropy


def build_submodel_with_embedding(architecture, embed_dim=64, Chans=6, Samples=512):
    """
    Builds a sub-model that exposes both embedding and task predictions.

    Args:
        architecture: 'EEGNet', 'TSception', or 'EEGTransformer'

    Returns:
        full_model:  Model with outputs [dep_pred, str_pred, embedding]
        embed_model: Model with output [embedding] only
    """
    if architecture == 'EEGNet':
        base = EEGNet(nb_classes=1, Chans=Chans, Samples=Samples)
        flatten_output = base.layers[13].output          # 256-dim
    elif architecture == 'TSception':
        base = TSception(nb_classes=1, Chans=Chans, Samples=Samples, sampling_rate=128)
        flatten_output = base.layers[10].output          # 768-dim
    elif architecture == 'EEGTransformer':
        base = EEGTransformer(nb_classes=1, Chans=Chans, Samples=Samples)
        flatten_output = base.layers[-5].output          # 64-dim

    # Projection to shared embedding space
    proj = Dense(embed_dim, activation='relu',
                 name=f'{architecture}_proj')(flatten_output)
    embedding = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1),
        name=f'{architecture}_embedding'
    )(proj)

    # Task heads
    shared = Dense(32, activation='relu', name=f'{architecture}_shared')(proj)
    dep_out = Dense(1, activation='sigmoid', name=f'{architecture}_depression')(shared)
    str_out = Dense(1, activation='sigmoid', name=f'{architecture}_stress')(shared)

    full_model = Model(
        inputs=base.input,
        outputs=[dep_out, str_out, embedding],
        name=f'{architecture}_submodel'
    )
    return full_model


def build_meta_learner(num_submodels=3, num_datasets=4):
    """
    Meta-learner that combines predictions from all sub-models.

    Input:  [p_A_dep, p_A_str, p_B_dep, p_B_str, p_C_dep, p_C_str, ds_onehot]
    Output: [depression, stress]
    """
    # 6 prediction scores + 4 dataset one-hot = 10
    meta_input = Input(shape=(num_submodels * 2 + num_datasets,), name='meta_input')
    h = Dense(16, activation='relu', name='meta_hidden')(meta_input)
    dep_out = Dense(1, activation='sigmoid', name='depression')(h)
    str_out = Dense(1, activation='sigmoid', name='stress')(h)

    return Model(inputs=meta_input, outputs=[dep_out, str_out], name='MetaLearner')
```

#### Step 2: Create `run_pipeline3.py` — 3-Phase Training

```python
# ============================
# PHASE 1: Independent Pre-training
# ============================

for arch_name in ['EEGNet', 'TSception', 'EEGTransformer']:
    model = build_submodel_with_embedding(arch_name)

    # Compile with task losses only (ignore embedding output during Phase 1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss={
            f'{arch_name}_depression': masked_binary_crossentropy,
            f'{arch_name}_stress': masked_binary_crossentropy,
            f'{arch_name}_embedding': lambda y_true, y_pred: 0.0  # dummy
        },
        loss_weights={
            f'{arch_name}_depression': 1.0,
            f'{arch_name}_stress': 1.0,
            f'{arch_name}_embedding': 0.0
        },
        metrics={
            f'{arch_name}_depression': 'accuracy',
            f'{arch_name}_stress': 'accuracy'
        }
    )

    model.fit(
        X_train_full_dl,
        [y_dep_train_full, y_str_train_full, np.zeros((len(y_dep_train_full), 64))],
        validation_split=0.1,
        epochs=25,
        batch_size=128,
        callbacks=[early_stop],
        verbose=1
    )

    model.save_weights(f'Results/pipeline3/{arch_name}_phase1.weights.h5')


# ============================
# PHASE 2: Contrastive Fine-tuning
# ============================

# Load all 3 models with Phase 1 weights
models = {}
for arch_name in ['EEGNet', 'TSception', 'EEGTransformer']:
    m = build_submodel_with_embedding(arch_name)
    m.load_weights(f'Results/pipeline3/{arch_name}_phase1.weights.h5')
    models[arch_name] = m

# Custom training loop for contrastive loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)  # Lower LR for fine-tuning
lambda_contrast = 0.1  # Anneal from 0.1 → 1.0

for epoch in range(10):
    lambda_contrast = min(0.1 + epoch * 0.1, 1.0)  # Linear anneal

    for batch_X, batch_y_dep, batch_y_str, batch_ds in train_dataset:
        with tf.GradientTape() as tape:
            total_loss = 0.0

            # Forward pass all 3 models
            embeddings = []
            for arch_name, m in models.items():
                dep_pred, str_pred, embed = m(batch_X, training=True)

                # Task losses (with masking)
                task_loss_dep = masked_binary_crossentropy(batch_y_dep, dep_pred)
                task_loss_str = masked_binary_crossentropy(batch_y_str, str_pred)
                total_loss += task_loss_dep + task_loss_str

                embeddings.append(embed)

            # Contrastive loss across all embeddings
            all_embeds = tf.concat(embeddings, axis=0)  # (3*batch, 64)
            all_labels = tf.tile(batch_y_str, [3])       # Use stress labels (larger set)
            all_ds = tf.tile(batch_ds, [3])

            # Only use samples with valid labels for contrastive
            valid = tf.not_equal(all_labels, -1)
            if tf.reduce_sum(tf.cast(valid, tf.float32)) > 2:
                con_loss = supervised_contrastive_loss(
                    all_embeds[valid], all_labels[valid], all_ds[valid]
                )
                total_loss += lambda_contrast * con_loss

        # Update all models jointly
        all_vars = []
        for m in models.values():
            all_vars.extend(m.trainable_variables)
        grads = tape.gradient(total_loss, all_vars)
        optimizer.apply_gradients(zip(grads, all_vars))


# ============================
# PHASE 3: Meta-Learner Training
# ============================

# Collect predictions from all 3 sub-models on validation set
meta_features = []
for arch_name, m in models.items():
    dep_pred, str_pred, _ = m.predict(X_val_dl)
    meta_features.extend([dep_pred.flatten(), str_pred.flatten()])

# Add dataset one-hot encoding
ds_onehot = tf.keras.utils.to_categorical(datasets_val, num_classes=4)  # (N_val, 4)

meta_X = np.column_stack(meta_features + [ds_onehot])  # (N_val, 10)

# Train meta-learner
meta_model = build_meta_learner()
meta_model.compile(
    optimizer='adam',
    loss={'depression': masked_binary_crossentropy,
          'stress': masked_binary_crossentropy}
)
meta_model.fit(meta_X, [y_dep_val, y_str_val], epochs=50, batch_size=64, verbose=0)
```

### 3.6 Training Configuration

| Phase | Optimizer | LR | Epochs | Batch | Notes |
|---|---|---|---|---|---|
| Phase 1 | Adam | 5e-4 | 25 | 128 | Standard pre-training per model |
| Phase 2 | Adam | 1e-4 | 10 | 128 | Lower LR, λ anneals 0.1→1.0 |
| Phase 3 | Adam | 1e-3 | 50 | 64 | Fast convergence, tiny model |

### 3.7 Evaluation

```python
# Final inference pipeline:
# 1. Forward pass through all 3 sub-models
# 2. Collect 6 prediction scores
# 3. Append dataset_id one-hot
# 4. Forward through meta-learner

def predict_codae(models, meta_model, X_dl, dataset_ids):
    preds = []
    for arch_name in ['EEGNet', 'TSception', 'EEGTransformer']:
        dep_p, str_p, _ = models[arch_name].predict(X_dl, verbose=0)
        preds.extend([dep_p.flatten(), str_p.flatten()])
    ds_onehot = tf.keras.utils.to_categorical(dataset_ids, num_classes=4)
    meta_X = np.column_stack(preds + [ds_onehot])
    dep_final, str_final = meta_model.predict(meta_X, verbose=0)
    return dep_final.flatten(), str_final.flatten()
```

---

## Pipeline 4: Spectro-Temporal Transformer with Adaptive Domain Normalization (ST²-ADN)

### 4.1 Overview

**File**: `src/proposed_model/pipeline4_st2adn.py`

Raw EEG is converted to per-channel spectrograms (STFT), normalized via Adaptive Instance Normalization conditioned on dataset identity, then processed by a Vision Transformer with patch embedding. A frequency-band attention module provides interpretability.

### 4.2 Inputs

| Input | Shape | Source |
|---|---|---|
| `raw_eeg` | `(batch, 6, 512, 1)` | `X[..., np.newaxis]` |
| `dataset_id` | `(batch,)` | `datasets` array — integer 0-3 |

### 4.3 STFT Preprocessing — Exact Computation

```python
import tensorflow as tf

def compute_spectrogram(x, frame_length=64, frame_step=16, fft_length=128):
    """
    x: (batch, 6, 512) — raw EEG, 6 channels, 512 samples

    Returns: (batch, 65, 29, 6) — (Freq_bins, Time_frames, Channels)

    Computation per channel:
        tf.signal.stft(channel, frame_length=64, frame_step=16, fft_length=128)
        → (batch, T, F_complex) where T = (512-64)//16 + 1 = 29, F = 128//2+1 = 65
        Take magnitude: tf.abs(stft_result)
        Take log: tf.math.log(magnitude + 1e-6)

    Then stack channels as last dimension.
    """
    # x shape: (batch, 6, 512)
    specs = []
    for ch in range(6):
        channel_data = x[:, ch, :]  # (batch, 512)
        stft_result = tf.signal.stft(
            channel_data,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length
        )  # (batch, 29, 65) complex
        magnitude = tf.abs(stft_result)              # (batch, 29, 65)
        log_mag = tf.math.log(magnitude + 1e-6)      # (batch, 29, 65)
        specs.append(log_mag)

    # Stack: (batch, 6, 29, 65) then transpose to (batch, 65, 29, 6)
    stacked = tf.stack(specs, axis=1)                 # (batch, 6, 29, 65)
    spectrogram = tf.transpose(stacked, perm=[0, 3, 2, 1])  # (batch, 65, 29, 6)
    return spectrogram
```

**Spectrogram dimensions breakdown:**

| Parameter | Value | Derivation |
|---|---|---|
| frame_length | 64 samples | 0.5 seconds at 128 Hz |
| frame_step | 16 samples | 0.125 seconds → 75% overlap |
| fft_length | 128 | Frequency resolution: 128 Hz / 128 = 1 Hz per bin |
| Freq bins (F) | 65 | fft_length/2 + 1 = 65 (0 to 64 Hz) |
| Time frames (T) | 29 | (512 - 64) / 16 + 1 = 29 |
| Output shape | (batch, 65, 29, 6) | Freq × Time × Channels |

### 4.4 Architecture — Layer by Layer

```
SPECTROGRAM TRANSFORM (non-trainable)
──────────────────────────────────────────────
Input: raw_eeg (batch, 6, 512, 1)

Step 1: Strip trailing dim and compute STFT
    x = Reshape((6, 512))(raw_eeg)
    spec = Lambda(compute_spectrogram)(x)              # → (batch, 65, 29, 6)


ADAPTIVE INSTANCE NORMALIZATION (AdaIN)
──────────────────────────────────────────────
Input: spec (batch, 65, 29, 6), dataset_id (batch,)

Step 2: Embed dataset ID
    ds_embed = Embedding(4, 32)(dataset_id)             # → (batch, 32)

Step 3: Generate AdaIN parameters
    gamma = Dense(6, activation=None, name='adain_gamma')(ds_embed)  # → (batch, 6)
    beta = Dense(6, activation=None, name='adain_beta')(ds_embed)    # → (batch, 6)
    # gamma and beta are per-channel scaling/shifting

Step 4: Instance Normalize per channel, then apply γ/β
    # InstanceNorm: normalize each (65, 29) slice per channel
    mean = tf.reduce_mean(spec, axis=[1, 2], keepdims=True)   # (batch, 1, 1, 6)
    var = tf.math.reduce_variance(spec, axis=[1, 2], keepdims=True)
    spec_norm = (spec - mean) / tf.sqrt(var + 1e-5)           # (batch, 65, 29, 6)

    # Apply learned affine transform
    gamma_4d = Reshape((1, 1, 6))(gamma)                       # (batch, 1, 1, 6)
    beta_4d = Reshape((1, 1, 6))(beta)                         # (batch, 1, 1, 6)
    spec_adain = spec_norm * (1 + gamma_4d) + beta_4d          # (batch, 65, 29, 6)
    # Note: (1 + gamma) so that default gamma=0 → identity transform


PATCH EMBEDDING (ViT-style)
──────────────────────────────────────────────
Step 5: Conv2D patch embedding
    patch_size = (13, 5)
    # 65 / 13 = 5 freq patches, 29 / 5 = 5 time patches (with 4 remainder → use valid padding)
    # Actually: floor(65/13) = 5, floor(29/5) = 5 → 25 patches
    # Remainder: 65 - 5*13 = 0 (exact), 29 - 5*5 = 4 (lost)

    patches = Conv2D(128, kernel_size=(13, 5), strides=(13, 5),
                     padding='valid', activation='relu',
                     name='patch_embed')(spec_adain)     # → (batch, 5, 5, 128)
    patches = Reshape((25, 128))(patches)                # → (batch, 25, 128)

Step 6: Add CLS token
    cls_token = tf.Variable(
        tf.random.normal([1, 1, 128]), trainable=True, name='cls_token'
    )
    cls_broadcast = tf.tile(cls_token, [tf.shape(patches)[0], 1, 1])  # (batch, 1, 128)
    tokens = Concatenate(axis=1)([cls_broadcast, patches])             # (batch, 26, 128)

Step 7: Add positional embedding
    pos_embed = Embedding(26, 128)(tf.range(26))        # (26, 128) — learned
    tokens = tokens + pos_embed                          # (batch, 26, 128)


TRANSFORMER ENCODER
──────────────────────────────────────────────
Step 8: 4 Transformer layers
    for i in range(4):
        # Multi-Head Self-Attention
        attn_out = MultiHeadAttention(
            num_heads=8, key_dim=16, name=f'mhsa_{i}'
        )(tokens, tokens)
        attn_out = Dropout(0.3)(attn_out)
        tokens = LayerNormalization()(tokens + attn_out)

        # Feed-Forward Network
        ffn_out = Dense(256, activation='relu')(tokens)
        ffn_out = Dense(128)(ffn_out)
        ffn_out = Dropout(0.3)(ffn_out)
        tokens = LayerNormalization()(tokens + ffn_out)

Step 9: Extract CLS token
    cls_output = tokens[:, 0, :]                        # (batch, 128)


FREQUENCY-BAND ATTENTION (interpretability)
──────────────────────────────────────────────
Step 10: Attention over frequency-axis patches
    # The 25 patches are arranged as 5 freq × 5 time
    # Group by frequency row: patches 0-4 (lowest freq), 5-9 (next), ..., 20-24 (highest)
    # Average attention to each frequency group from CLS token

    # Alternative simpler approach: learn a frequency importance vector
    freq_attention = Dense(5, activation='softmax',
                           name='freq_band_attention')(cls_output)   # (batch, 5)
    # This gives interpretable weights:
    # freq_group 0 (bins 0-12): ~0-12 Hz (delta+theta)
    # freq_group 1 (bins 13-25): ~13-25 Hz (alpha+low beta)
    # freq_group 2 (bins 26-38): ~26-38 Hz (beta)
    # freq_group 3 (bins 39-51): ~39-51 Hz (high beta+gamma)
    # freq_group 4 (bins 52-64): ~52-64 Hz (gamma)


TASK HEADS
──────────────────────────────────────────────
Step 11: Depression head
    dep_h = Dense(64, activation='relu', name='dep_h1')(cls_output)
    dep_h = Dropout(0.4)(dep_h)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep_h)

Step 12: Stress head
    str_h = Dense(64, activation='relu', name='str_h1')(cls_output)
    str_h = Dropout(0.4)(str_h)
    str_out = Dense(1, activation='sigmoid', name='stress')(str_h)
```

### 4.5 Implementation Steps

#### Step 1: Create `src/proposed_model/pipeline4_st2adn.py`

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Reshape, Permute, Concatenate,
    Conv2D, Embedding, LayerNormalization, Lambda, Flatten
)
from tensorflow.keras.models import Model


class CLSToken(tf.keras.layers.Layer):
    """Learnable [CLS] token prepended to patch sequence."""
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='random_normal',
            trainable=True,
            name='cls_token'
        )

    def call(self, patches):
        batch_size = tf.shape(patches)[0]
        cls_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcast, patches], axis=1)


class AdaptiveInstanceNorm(tf.keras.layers.Layer):
    """
    Adaptive Instance Normalization conditioned on dataset_id.
    Normalizes each channel's spectrogram independently, then applies
    learned per-dataset affine transform.
    """
    def __init__(self, num_datasets=4, num_channels=6, style_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.num_datasets = num_datasets
        self.num_channels = num_channels
        self.style_dim = style_dim

    def build(self, input_shape):
        self.ds_embedding = Embedding(self.num_datasets, self.style_dim)
        self.gamma_dense = Dense(self.num_channels, use_bias=True)
        self.beta_dense = Dense(self.num_channels, use_bias=True)

    def call(self, inputs):
        spec, dataset_id = inputs
        # spec: (batch, F, T, C), dataset_id: (batch,)

        # Instance normalize per channel
        mean = tf.reduce_mean(spec, axis=[1, 2], keepdims=True)
        var = tf.math.reduce_variance(spec, axis=[1, 2], keepdims=True)
        spec_norm = (spec - mean) / tf.sqrt(var + 1e-5)

        # Generate style from dataset_id
        style = self.ds_embedding(dataset_id)              # (batch, style_dim)
        gamma = self.gamma_dense(style)                     # (batch, C)
        beta = self.beta_dense(style)                       # (batch, C)

        # Reshape for broadcasting: (batch, 1, 1, C)
        gamma = tf.reshape(gamma, [-1, 1, 1, self.num_channels])
        beta = tf.reshape(beta, [-1, 1, 1, self.num_channels])

        return spec_norm * (1.0 + gamma) + beta


class SpectrogramLayer(tf.keras.layers.Layer):
    """Non-trainable layer that computes per-channel log-magnitude STFT."""
    def __init__(self, frame_length=64, frame_step=16, fft_length=128, **kwargs):
        super().__init__(**kwargs)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

    def call(self, x):
        # x: (batch, 6, 512)
        specs = []
        for ch in range(6):
            channel = x[:, ch, :]                                      # (batch, 512)
            stft = tf.signal.stft(
                channel,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                fft_length=self.fft_length
            )                                                          # (batch, 29, 65)
            magnitude = tf.abs(stft)
            log_mag = tf.math.log(magnitude + 1e-6)
            specs.append(log_mag)

        stacked = tf.stack(specs, axis=1)                              # (batch, 6, 29, 65)
        return tf.transpose(stacked, perm=[0, 3, 2, 1])               # (batch, 65, 29, 6)

    def get_config(self):
        config = super().get_config()
        config.update({
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'fft_length': self.fft_length,
        })
        return config


class TransformerEncoder(tf.keras.layers.Layer):
    """Single Transformer encoder layer with MHSA + FFN."""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.mhsa = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.ln1 = LayerNormalization(epsilon=1e-6)
        self.ln2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        attn = self.mhsa(x, x)
        attn = self.drop1(attn, training=training)
        x = self.ln1(x + attn)

        ffn = self.ffn(x)
        ffn = self.drop2(ffn, training=training)
        return self.ln2(x + ffn)


def build_st2adn(Chans=6, Samples=512, num_datasets=4,
                  embed_dim=128, num_heads=8, ff_dim=256,
                  num_layers=4, dropout_rate=0.3,
                  patch_freq=13, patch_time=5):
    """
    Spectro-Temporal Transformer with Adaptive Domain Normalization.

    Inputs:
        raw_eeg:    (batch, Chans, Samples, 1)
        dataset_id: (batch,) — integer 0-3

    Outputs:
        depression: (batch, 1)
        stress:     (batch, 1)
        freq_attention: (batch, 5) — interpretable frequency band weights
    """
    # --- Inputs ---
    raw_input = Input(shape=(Chans, Samples, 1), name='raw_eeg')
    ds_input = Input(shape=(), dtype=tf.int32, name='dataset_id')

    # --- Spectrogram ---
    x = Reshape((Chans, Samples), name='strip_dim')(raw_input)
    spec = SpectrogramLayer(name='stft')(x)             # (batch, 65, 29, 6)

    # --- Adaptive Instance Normalization ---
    spec_adain = AdaptiveInstanceNorm(
        num_datasets=num_datasets,
        num_channels=Chans,
        name='adain'
    )([spec, ds_input])                                  # (batch, 65, 29, 6)

    # --- Patch Embedding ---
    # (65, 29, 6) with patch (13, 5) → (5, 5, embed_dim) = 25 patches
    # Note: 29 / 5 = 5 remainder 4, use valid padding to drop edge
    patches = Conv2D(
        embed_dim, kernel_size=(patch_freq, patch_time),
        strides=(patch_freq, patch_time), padding='valid',
        activation='relu', name='patch_embed'
    )(spec_adain)                                        # (batch, 5, 5, 128)

    num_patches_f = 65 // patch_freq  # = 5
    num_patches_t = 29 // patch_time  # = 5
    num_patches = num_patches_f * num_patches_t  # = 25

    patches = Reshape((num_patches, embed_dim),
                       name='flatten_patches')(patches)  # (batch, 25, 128)

    # --- CLS Token ---
    tokens = CLSToken(embed_dim, name='cls_token')(patches)  # (batch, 26, 128)

    # --- Positional Embedding ---
    positions = tf.range(num_patches + 1)                # [0, 1, ..., 25]
    pos_embed = Embedding(num_patches + 1, embed_dim,
                          name='pos_embedding')(positions) # (26, 128)
    tokens = tokens + pos_embed                           # (batch, 26, 128)

    # --- Transformer Encoder Stack ---
    for i in range(num_layers):
        tokens = TransformerEncoder(
            embed_dim, num_heads, ff_dim, dropout_rate,
            name=f'transformer_{i}'
        )(tokens)

    # --- Extract CLS token ---
    cls_output = Lambda(
        lambda t: t[:, 0, :], name='extract_cls'
    )(tokens)                                             # (batch, 128)

    # --- Frequency Band Attention (interpretability) ---
    freq_attention = Dense(
        num_patches_f, activation='softmax',
        name='freq_band_attention'
    )(cls_output)                                         # (batch, 5)
    # Maps to frequency groups:
    # 0: 0-12 Hz (delta+theta), 1: 13-25 Hz (alpha+beta_low)
    # 2: 26-38 Hz (beta), 3: 39-51 Hz (beta_high+gamma)
    # 4: 52-64 Hz (gamma_high)

    # --- Task Heads ---
    dep_h = Dense(64, activation='relu', name='dep_h1')(cls_output)
    dep_h = Dropout(0.4)(dep_h)
    dep_out = Dense(1, activation='sigmoid', name='depression')(dep_h)

    str_h = Dense(64, activation='relu', name='str_h1')(cls_output)
    str_h = Dropout(0.4)(str_h)
    str_out = Dense(1, activation='sigmoid', name='stress')(str_h)

    model = Model(
        inputs=[raw_input, ds_input],
        outputs=[dep_out, str_out, freq_attention],
        name='ST2_ADN'
    )
    return model
```

#### Step 2: Create `run_pipeline4.py`

```python
# Key differences from other pipelines:

# 1. Need dataset_id as additional input
datasets_train = datasets[train_mask]  # (N_train,) — values 0,1,2,3

# 2. Build and compile
model = build_st2adn()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={
        'depression': masked_binary_crossentropy,
        'stress': masked_binary_crossentropy,
        'freq_band_attention': lambda y_true, y_pred: 0.0  # No supervision
    },
    loss_weights={
        'depression': 1.0,
        'stress': 1.0,
        'freq_band_attention': 0.0
    },
    metrics={
        'depression': 'accuracy',
        'stress': 'accuracy'
    }
)

# 3. Label smoothing (optional, applied to targets)
y_dep_smooth = np.where(y_dep_train_full == 1, 0.9, 0.0)
y_dep_smooth = np.where(y_dep_train_full == -1, -1, y_dep_smooth)
# Similar for y_str

# 4. Cosine annealing LR schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate=5e-4,
    first_decay_steps=5 * steps_per_epoch,   # Restart every 5 epochs
    t_mul=1.0,
    m_mul=0.9
)

# 5. Train with dual inputs
model.fit(
    [X_train_full_dl, datasets_train],
    [y_dep_train_full, y_str_train_full, np.zeros((len(y_dep_train_full), 5))],
    validation_split=0.1,
    epochs=25,
    batch_size=128,
    sample_weight=[sample_weight_dep, sample_weight_str, None],
    callbacks=[early_stop],
    verbose=1
)

# 6. Evaluate
dep_probs, str_probs, freq_attn = model.predict(
    [X_test_dl, datasets_test], verbose=0
)
```

### 4.6 Training Configuration

| Parameter | Value | Rationale |
|---|---|---|
| Optimizer | Adam with Cosine Annealing + Warm Restarts | Novel architecture benefits from LR restarts |
| Initial LR | 5e-4 | Consistent with project |
| Loss | `masked_binary_crossentropy` + dummy for freq_attention | Multi-label + interpretability |
| Label Smoothing | 0.1 (y=1 → 0.9, y=0 → 0.1) | Prevents overconfident transformer |
| Epochs | 25-30 | Transformer may need more epochs |
| Batch Size | 128 | Consistent (can reduce to 64 if OOM) |
| Transformer Layers | 4 | Balance depth vs. overfitting on ~45K samples |
| Heads | 8 | Rich attention patterns for 25 patches |
| embed_dim | 128 | Sufficient for 25 patches |
| Dropout | 0.3 (transformer), 0.4 (heads) | Moderate regularization |

### 4.7 Interpretability Analysis (Post-training)

```python
# 1. Frequency Band Importance
_, _, freq_attn = model.predict([X_test_dl, datasets_test])
freq_labels = ['δ+θ (0-12Hz)', 'α+β_low (13-25Hz)', 'β (26-38Hz)',
               'β_hi+γ (39-51Hz)', 'γ_hi (52-64Hz)']

# Per-task importance
dep_mask = y_test_dep != -1
str_mask = y_test_str != -1

print("Depression - Frequency Band Importance:")
for i, label in enumerate(freq_labels):
    print(f"  {label}: {freq_attn[dep_mask, i].mean():.4f}")

print("Stress - Frequency Band Importance:")
for i, label in enumerate(freq_labels):
    print(f"  {label}: {freq_attn[str_mask, i].mean():.4f}")

# 2. AdaIN Parameters Analysis
adain_layer = model.get_layer('adain')
for ds_id, ds_name in enumerate(['MODMA', 'SAM40', 'CogLoad', 'AMIGOS']):
    ds_embed = adain_layer.ds_embedding(tf.constant([ds_id]))
    gamma = adain_layer.gamma_dense(ds_embed).numpy().flatten()
    beta = adain_layer.beta_dense(ds_embed).numpy().flatten()
    print(f"{ds_name}: gamma={gamma.round(3)}, beta={beta.round(3)}")
```

---

## Common Implementation Utilities

### Shared Evaluation Function

All pipelines should use the same evaluation function for fair comparison:

```python
# Add to a shared utilities file: src/proposed_model/evaluation.py

def evaluate_pipeline(model, X_test_inputs, y_test_dep, y_test_str,
                      dep_mask, str_mask, pipeline_name):
    """
    Unified evaluation for all pipelines.

    Args:
        model: Keras model or callable that returns (dep_probs, str_probs)
        X_test_inputs: model input(s) — single array or list of arrays
        y_test_dep: (N,) depression labels (0/1/-1)
        y_test_str: (N,) stress labels (0/1/-1)
        dep_mask: boolean mask for depression samples
        str_mask: boolean mask for stress samples
        pipeline_name: string for display

    Returns:
        dict with depression and stress metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score
    )

    predictions = model.predict(X_test_inputs, verbose=0)
    dep_probs = predictions[0].flatten()
    str_probs = predictions[1].flatten()

    results = {}
    for task, probs, y_true, mask in [
        ('Depression', dep_probs, y_test_dep, dep_mask),
        ('Stress', str_probs, y_test_str, str_mask)
    ]:
        y = y_true[mask]
        p = probs[mask]
        preds = (p >= 0.5).astype(int)

        results[task] = {
            'Accuracy': accuracy_score(y, preds),
            'Precision': precision_score(y, preds, zero_division=0),
            'Recall': recall_score(y, preds, zero_division=0),
            'F1-Score': f1_score(y, preds, zero_division=0),
            'ROC-AUC': roc_auc_score(y, p)
        }

    # Print results
    print(f"\n{'='*60}")
    print(f"  {pipeline_name} Results")
    print(f"{'='*60}")
    for task, metrics in results.items():
        print(f"\n  {task}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.4f}")

    return results
```

### Results Directory Structure

```
Results/
├── pipeline1_dsffn/
│   ├── model.weights.h5
│   ├── training_history.npy
│   ├── gate_analysis.png
│   └── metrics.json
├── pipeline2_hamsnet/
│   ├── model.weights.h5
│   ├── training_history.npy
│   ├── channel_reliability.png
│   ├── scale_attention.png
│   └── metrics.json
├── pipeline3_codae/
│   ├── EEGNet_phase1.weights.h5
│   ├── TSception_phase1.weights.h5
│   ├── EEGTransformer_phase1.weights.h5
│   ├── EEGNet_phase2.weights.h5
│   ├── TSception_phase2.weights.h5
│   ├── EEGTransformer_phase2.weights.h5
│   ├── meta_learner.weights.h5
│   ├── embedding_tsne.png
│   └── metrics.json
└── pipeline4_st2adn/
    ├── model.weights.h5
    ├── training_history.npy
    ├── freq_band_attention.png
    ├── adain_parameters.json
    └── metrics.json
```

---

## Implementation Checklist

### Pipeline 1: DSFFN
- [ ] Create `src/proposed_model/pipeline1_dsffn.py` with `build_dsffn()`
- [ ] Update `src/proposed_model/__init__.py` with new export
- [ ] Create `run_pipeline1.py` (based on `run_models.py` multi-task section)
- [ ] Create `Results/pipeline1_dsffn/` directory
- [ ] Train model and evaluate
- [ ] Generate gate analysis visualization
- [ ] Record metrics to `metrics.json`

### Pipeline 2: HAMS-Net
- [ ] Create `src/proposed_model/pipeline2_hamsnet.py` with `build_hamsnet()`
- [ ] Update `src/proposed_model/__init__.py` with new export
- [ ] Create `run_pipeline2.py`
- [ ] Create `Results/pipeline2_hamsnet/` directory
- [ ] Train model and evaluate
- [ ] Generate channel reliability + scale attention visualizations
- [ ] Record metrics to `metrics.json`

### Pipeline 3: CoDAE
- [ ] Create `src/proposed_model/pipeline3_codae.py` with sub-model builders + contrastive loss + meta-learner
- [ ] Update `src/proposed_model/__init__.py` with new exports
- [ ] Create `run_pipeline3.py` (3-phase training loop)
- [ ] Create `Results/pipeline3_codae/` directory
- [ ] Phase 1: Pre-train 3 sub-models independently
- [ ] Phase 2: Contrastive fine-tuning (custom training loop)
- [ ] Phase 3: Train meta-learner on stacked predictions
- [ ] Generate t-SNE of aligned embeddings
- [ ] Record metrics to `metrics.json`

### Pipeline 4: ST²-ADN
- [ ] Create `src/proposed_model/pipeline4_st2adn.py` with custom layers + `build_st2adn()`
- [ ] Update `src/proposed_model/__init__.py` with new export
- [ ] Create `run_pipeline4.py` (with dataset_id input + cosine LR schedule)
- [ ] Create `Results/pipeline4_st2adn/` directory
- [ ] Train model and evaluate
- [ ] Generate frequency band attention heatmap
- [ ] Analyze AdaIN parameters per dataset
- [ ] Record metrics to `metrics.json`

### Shared
- [ ] Create `src/proposed_model/evaluation.py` with `evaluate_pipeline()`
- [ ] Update `__init__.py` with evaluation export
- [ ] Create comparative results table across all pipelines
- [ ] Update `report.md` with hybrid model results
