# MODMA Model Training Results (3-Channel EEG)

This file contains the cross-validation performance metrics of SVM, Random Forest, 1D CNN, and LSTM trained on the MODMA dataset (3-channel resting state EEG downsampled to 125Hz).

| Model | Accuracy (Mean ± SD) | Precision (Mean) | Recall (Mean) | ROC-AUC (Mean) |
|---|---|---|---|---|
| SVM | 0.3747 ± 0.0672 | 0.1248 | 0.4000 | 0.4136 |
| Random Forest | 0.4286 ± 0.1215 | 0.4858 | 0.2008 | 0.4976 |
| 1D CNN | 0.5201 ± 0.0663 | 0.5010 | 0.5054 | 0.5359 |
| LSTM | 0.4084 ± 0.0483 | 0.3373 | 0.3603 | 0.5314 |
