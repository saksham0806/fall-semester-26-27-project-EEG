# Data Validation and Preprocessing

## Objective

The objective of this stage is to validate the unified EEG dataset and remove any corrupted or redundant samples before feature extraction.

The cleaned dataset produced in this stage serves as the final input for the proposed deep learning architecture.

---

# Input

Location:

datasets/Unified/

Files:

- X.npy
- y.npy
- subjects.npy
- datasets.npy
- metadata.json

---

# Validation Steps

## 1. Dataset Loading

The unified dataset was loaded successfully.

Data loaded:

- EEG signals (X)
- Unified labels (y)
- Subject IDs
- Dataset IDs
- Metadata

---

## 2. Missing Value Detection

The EEG signals were checked for missing values.

Result:

- Total NaN values detected: **4,239,360**
- Samples containing NaN values: **1,380**

Further investigation showed that **all NaN samples belonged to the AMIGOS dataset**.

---

## 3. Removal of Corrupted Samples

Since EEG signals are continuous physiological recordings, replacing missing values could introduce artificial brain activity.

Therefore, corrupted EEG windows were completely removed.

Removed:

- 1,380 EEG windows

Remaining samples:

53,777

---

## 4. Verification

The dataset was checked again after removal.

Result:

- Remaining NaN values: **0**

The dataset is now completely free from missing values.

---

## 5. Class Distribution

After removing corrupted samples:

| Class | Samples | Percentage |
|--------|---------|-----------|
| Healthy | 20,227 | 37.61% |
| Depressed | 3,599 | 6.69% |
| Stressed | 29,951 | 55.69% |

Observation:

The dataset is imbalanced, with the Depressed class being significantly smaller.

No balancing technique was applied during preprocessing.

Class balancing will be handled during model training using appropriate loss weighting or augmentation techniques.

---

## 6. Duplicate Detection

The EEG windows were flattened and checked for exact duplicates.

Result:

- Duplicate samples detected: **882**

Further analysis showed that all duplicate windows belonged to the Cognitive Load dataset.

---

## 7. Removal of Duplicate Samples

Duplicate EEG windows provide no additional information and may bias model training.

Therefore, duplicate windows were removed while keeping the first occurrence.

Removed:

- 882 duplicate windows

Remaining samples:

52,895

---

## 8. Verification

Duplicate detection was repeated.

Result:

- Remaining duplicate windows: **0**

---

# Final Clean Dataset

Location:

datasets/Cleaned/

Files:

- X.npy
- y.npy
- subjects.npy
- datasets.npy
- metadata.json

---

# Final Dataset Summary

| Property | Value |
|-----------|------|
| Total Samples | 52,895 |
| Channels | 6 |
| Window Length | 512 |
| Classes | 3 |
| Healthy | 20,227 |
| Depressed | 3,599 |
| Stressed | 29,951 |
| NaN Samples | 0 |
| Duplicate Samples | 0 |

---

# Outcome

The dataset is now fully validated and cleaned.

This dataset will be used as the input for the feature extraction stage using the proposed hybrid deep learning architecture consisting of:

- TSception
- EEGNet
- Transformer-based model

The output of the feature extraction stage will be used for three-class classification:

- Healthy
- Depressed
- Stressed