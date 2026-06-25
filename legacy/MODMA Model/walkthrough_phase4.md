# Walkthrough: Phase 4 - Final Model & Fine-tuning

In Phase 4, we implemented a robust training pipeline (`src/final_model.py`) and fine-tuned the **TSception** model with **Data Augmentation** and **Dropout (0.6)**.

## Experimental Results (TSception)

**Configuration**:
-   Epochs: 50 (with Early Stopping)
-   Batch Size: 32
-   Augmentation: 50% Overlap
-   Dropout: 0.6

| Fold | Best Validation Accuracy | Best Epoch | Notes |
| :--- | :--- | :--- | :--- |
| **Fold 1** | **79.11%** | 1 | Early Stopping at Epoch 11. |
| **Fold 2** | **72.46%** | 7 | Early Stopping at Epoch 13. |
| **Fold 3** | **81.75%** | 3 | Early Stopping at Epoch 13. |
| **Fold 4** | **68.25%** | 4 | Early Stopping at Epoch 11. |
| **Fold 5** | **72.78%** | 3 | Consistent Performance. |
| **AVERAGE** | **74.87%** | - | **std: +/- 4.89%** |

## Experimental Results (EEGNet)

**Configuration**:
-   Epochs: 50
-   Batch Size: 32
-   Augmentation: 50% Overlap
-   Dropout: 0.5 (Default)

| Fold | Best Validation Accuracy | Best Epoch | Notes |
| :--- | :--- | :--- | :--- |
| **Fold 1** | **68.57%** | 6 | Early Stopping at Epoch 11. |
| **Fold 2** | **60.45%** | 3 | Early Stopping at Epoch 11. |
| **Fold 3** | **76.50%** | 1 | Early Stopping at Epoch 11. |
| **Fold 4** | **36.27%** | 1 | Failed to converge. |
| **Fold 5** | **62.36%** | 1 | Moderate performance. |
| **AVERAGE** | **60.83%** | - | **std: +/- 13.51%** |

## Key Observations

1.  **Fold 1 Performance**:
    -   Achieved a remarkable **79.11%** accuracy in the first epoch.
    -   Quickly overfitted afterwards, but strict early stopping preserved the best weights.

2.  **Fold 2 Performance (Ongoing)**:
    -   Started lower (64%) but showed ability to recover/stabilize around 67% in later epochs, unlike Phase 3 where models collapsed completely.
    -   This suggests the increased **Dropout (0.6)** is helping generalization.

## Next Steps
-   Complete 5-Fold Cross-Validation.
-   Train Final EEGNet for comparison.
-   Generate confusion matrices.
