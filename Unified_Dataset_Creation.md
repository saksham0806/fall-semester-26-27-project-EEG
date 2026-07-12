# 02 - Unified Dataset Creation

## Objective

The objective of this stage is to convert the merged multi-task EEG dataset into a unified multi-class dataset that satisfies the updated project requirements.

The original merged dataset was designed for multi-task learning and contained two independent target vectors:

- `y_depression.npy`
- `y_stress.npy`

However, the proposed model architecture requires a **single output layer** capable of classifying EEG signals into three mutually exclusive classes:

- Healthy
- Depressed
- Stressed

Therefore, a unified target vector was constructed while preserving the original EEG signals, subject information, and dataset metadata.

---

# Input Dataset

The unified dataset was generated from the previously created merged dataset.

### Files Used

| File | Description |
|------|-------------|
| X.npy | EEG signal segments |
| y_depression.npy | Depression labels |
| y_stress.npy | Stress labels |
| subjects.npy | Subject identifiers |
| datasets.npy | Dataset identifiers |
| metadata.json | Dataset metadata |

---

# Original Label Structure

The merged dataset used two independent target variables.

### Depression Labels

| Value | Meaning |
|------:|---------|
| -1 | Label not applicable |
| 0 | Healthy Control |
| 1 | Depressed |

### Stress Labels

| Value | Meaning |
|------:|---------|
| -1 | Label not applicable |
| 0 | Relaxed |
| 1 | Stressed |

The value **-1** indicates that the corresponding sample does not belong to that particular classification task.

For example,

### MODMA sample

```
Depression = 1
Stress = -1
```

### SAM40 sample

```
Depression = -1
Stress = 1
```

This label masking strategy is suitable for multi-task learning but cannot be directly used for a single-output multi-class classifier.

---

# Need for Dataset Unification

According to the revised project requirements, the proposed model should contain

- Shared feature extraction
- One output layer
- Three target classes

Therefore, two independent target vectors cannot be used.

Instead, a single target vector was created for every EEG segment.

---

# Label Mapping Strategy

To construct the unified dataset, the following mapping strategy was adopted.

| Original Dataset | Original Label | Unified Label |
|------------------|---------------|---------------|
| MODMA | Healthy Control | Healthy (0) |
| MODMA | Depressed | Depressed (1) |
| SAM40 | Relax | Healthy (0) |
| SAM40 | Stress | Stressed (2) |
| Cognitive Load | Stress | Stressed (2) |
| AMIGOS | Relax | Healthy (0) |
| AMIGOS | Stress | Stressed (2) |

For this project, relaxed baseline segments from the stress datasets were considered equivalent to the Healthy class in order to create a unified three-class classification problem.

---

# Unified Label Distribution

After conversion, the new target vector contains the following class distribution.

| Class | Label | Samples |
|--------|------:|---------:|
| Healthy | 0 | 21,204 |
| Depressed | 1 | 3,599 |
| Stressed | 2 | 30,354 |

Total samples:

```
55,157
```

The total number of samples remains unchanged after the conversion process.

---

# Validation

Several validation checks were performed after constructing the unified dataset.

### Shape Verification

```
X.shape         = (55157, 6, 512)
y.shape         = (55157,)
subjects.shape  = (55157,)
datasets.shape  = (55157,)
```

All arrays contain the same number of samples.

---

### Label Verification

The unified target vector contains only three valid classes.

```
0
1
2
```

No invalid labels remain after the conversion process.

---

### Dataset Integrity

The following properties were preserved during dataset construction.

- EEG signals
- Subject identifiers
- Dataset identifiers
- Sampling rate
- Window size
- Channel configuration

Only the target labels were modified.

---

# Output Dataset

The newly constructed dataset is stored separately to preserve the original merged dataset.

```
datasets/
└── Unified/
    ├── X.npy
    ├── y.npy
    ├── subjects.npy
    ├── datasets.npy
    └── metadata.json
```

This dataset will be used throughout the remainder of the project.

---

# Conclusion

The multi-task merged dataset was successfully transformed into a unified three-class EEG classification dataset.

The resulting dataset satisfies the revised project requirements by replacing the two task-specific label vectors with a single target vector representing:

- Healthy
- Depressed
- Stressed

The conversion process preserved all EEG recordings, subject information, and dataset metadata while producing a dataset compatible with a single-output classification model.

---

# Next Step

```
03_Data_Preprocessing.ipynb
```

Objective:

- Verify signal quality.
- Detect invalid EEG segments.
- Check for missing or infinite values.
- Apply normalization and preprocessing required before feature extraction.
- Save the final preprocessed dataset for model training.