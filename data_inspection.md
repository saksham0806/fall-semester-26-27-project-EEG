# 01 - Data Inspection

## Objective

The objective of this stage is to understand the structure and composition of the merged EEG dataset before performing any preprocessing or model development.

The combined dataset consists of EEG recordings collected from multiple publicly available datasets and stored in NumPy format (`.npy`). Before building a unified classification dataset, it is necessary to verify the integrity, dimensions, labels, subject information, and dataset composition.

---

# Files Inspected

| File | Description |
|------|-------------|
| X.npy | EEG signal segments |
| y_depression.npy | Depression classification labels |
| y_stress.npy | Stress classification labels |
| subjects.npy | Subject identifiers |
| datasets.npy | Dataset identifiers |
| metadata.json | Dataset metadata and summary |

---

# EEG Signal Inspection

The EEG data was loaded successfully using NumPy.

### Data Type

```
numpy.ndarray
```

### Shape

```
(55157, 6, 512)
```

which represents

- 55,157 EEG segments
- 6 frontal EEG channels
- 512 time samples per segment

Each EEG sample has the shape

```
(6, 512)
```

representing six frontal electrodes:

- Fp1
- Fp2
- F7
- F3
- F4
- F8

---

# Sampling Information

From the metadata file:

| Property | Value |
|----------|------:|
| Sampling Rate | 128 Hz |
| Window Size | 512 samples |
| Segment Duration | 4 seconds |

Each EEG segment therefore represents approximately four seconds of brain activity.

---

# Dataset Composition

The merged dataset consists of four EEG datasets.

| Dataset | Segments |
|----------|---------:|
| MODMA | 7,959 |
| SAM40 | 2,400 |
| Cognitive Load | 5,369 |
| AMIGOS | 39,429 |

Total:

```
55,157 EEG segments
```

---

# Label Inspection

The current dataset is organized for **multi-task learning**.

Two label arrays are provided:

```
y_depression.npy

y_stress.npy
```

Each contains three values:

| Value | Meaning |
|-------:|----------|
| -1 | Label not applicable |
| 0 | Negative class |
| 1 | Positive class |

The value **-1** indicates that the corresponding sample does not belong to that task.

Example:

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

This masking strategy enables a single backbone network to be trained on multiple datasets while computing task-specific losses only where valid labels exist.

---

# Depression Label Distribution

| Label | Samples |
|--------|---------:|
| Not Applicable | 47,198 |
| Healthy | 4,360 |
| Depressed | 3,599 |

---

# Stress Label Distribution

| Label | Samples |
|--------|---------:|
| Not Applicable | 7,959 |
| Relaxed | 16,844 |
| Stressed | 30,354 |

---

# Subject Information

The merged dataset contains

```
123 unique subjects
```

Each subject contributes multiple EEG segments.

Examples include

```
modma_2030020

amigos_P01

...
```

The project uses a **Group-based Train/Test split** based on subject IDs to ensure that EEG segments from the same subject are never present in both training and testing sets, preventing subject-level data leakage.

---

# Metadata Verification

The metadata file confirms:

- total number of EEG segments
- sampling rate
- segment duration
- number of EEG channels
- dataset-wise segment counts
- class distributions

The metadata is internally consistent with the loaded NumPy arrays.

---

# Important Observation

The current dataset is designed for **multi-task learning**.

Current architecture:

```
                X
                │
      Shared Feature Extractor
         ┌──────────────┐
         │              │
 Depression Head    Stress Head
```

However, the project requirements have changed.

The proposed architecture requires a **single output layer** that predicts

```
Healthy

Depressed

Stressed
```

instead of separate depression and stress heads.

Therefore, the existing label structure must be converted into a unified multi-class dataset.

---

# Proposed Label Mapping

To construct a unified three-class dataset, the following mapping will be adopted.

| Original Label | Unified Label |
|----------------|---------------|
| Healthy Control (MODMA) | Healthy |
| Relax (SAM40) | Healthy |
| Relax (AMIGOS) | Healthy |
| Depression | Depressed |
| Stress | Stressed |

This produces a single target vector

```
y
```

with

| Class | Value |
|--------|------:|
| Healthy | 0 |
| Depressed | 1 |
| Stressed | 2 |

---

# Conclusion

The data inspection phase confirms that

- the merged dataset has been constructed correctly,
- all NumPy files are internally consistent,
- EEG segments, subject IDs, dataset IDs, and labels are correctly aligned,
- the metadata accurately describes the dataset,
- the project currently follows a multi-task learning formulation.

Since the final project requires a unified three-class classification problem, the next stage is to construct a new dataset by replacing the two task-specific label arrays with a single unified label vector.

---

# Next Step

```
02_Unified_Dataset_Creation.ipynb
```

Objective:

- Convert the multi-task labels into a unified three-class label.
- Preserve EEG signals, subject IDs, and dataset information.
- Save the new dataset without modifying the original merged dataset.