# Plan 01 — corrected features (binary)

Dev only, n=233 of 292; test set (n=59) frozen and untouched.
15 folds, nested hyperparameter search on inner folds.

chance 0.5000 · majority 0.6824 · **demographics 0.7598**

| model | balanced acc | macro-F1 | accuracy | macro-AUC | vs demo |
|---|---:|---:|---:|---:|---:|
| DEMO [l2] | **0.7598** | 0.7340 | 0.7511 | 0.8321 | +0.0000 |
| EEG+DEMO [elasticnet] | **0.7440** | 0.7358 | 0.7639 | 0.8435 | -0.0159 |
| RAW_BASELINE [rbf] | **0.7033** | 0.7159 | 0.7725 | 0.8100 | -0.0565 |
| EEG [elasticnet] | **0.7017** | 0.6916 | 0.7210 | 0.7333 | -0.0581 |
| EEG [l2] | **0.6814** | 0.6744 | 0.7082 | 0.7420 | -0.0784 |
| GRAPH [elasticnet] | **0.6413** | 0.6375 | 0.6781 | 0.7081 | -0.1185 |
| EEG [rbf] | **0.6217** | 0.6284 | 0.7253 | 0.7550 | -0.1381 |
| SPECTRAL [elasticnet] | **0.6000** | 0.5918 | 0.6266 | 0.6509 | -0.1598 |

Recall per class (the tell for whether a model predicts every class):

| model | healthy | depression |
|---|---|---|
| DEMO [l2] | 0.784 | 0.736 |
| EEG+DEMO [elasticnet] | 0.689 | 0.799 |
| RAW_BASELINE [rbf] | 0.514 | 0.893 |
| EEG [elasticnet] | 0.649 | 0.755 |
| EEG [l2] | 0.608 | 0.755 |
| GRAPH [elasticnet] | 0.541 | 0.742 |
| EEG [rbf] | 0.338 | 0.906 |
| SPECTRAL [elasticnet] | 0.527 | 0.673 |
