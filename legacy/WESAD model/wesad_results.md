# WESAD Model Training Results (Physiological Chest Data)

This file contains the cross-validation performance metrics of SVM, Random Forest, 1D CNN, and LSTM trained on the WESAD dataset (Chest ACC, ECG, EMG, EDA, Temp, and Resp downsampled to 100Hz).

| Model | Accuracy (Mean ± SD) | Precision (Mean) | Recall (Mean) | ROC-AUC (Mean) |
|---|---|---|---|---|
| SVM | 0.7194 ± 0.1967 | 0.7444 | 0.7022 | 0.7898 |
| Random Forest | 0.7012 ± 0.1964 | 0.6721 | 0.6358 | 0.8109 |
| 1D CNN | 0.7893 ± 0.0852 | 0.8551 | 0.5501 | 0.9032 |
| LSTM | 0.6772 ± 0.0444 | 0.5998 | 0.3262 | 0.7274 |
