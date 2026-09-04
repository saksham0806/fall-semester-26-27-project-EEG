# Plan 01 — corrected features (4class)

Dev only, n=421 of 527; test set (n=106) frozen and untouched.
15 folds, nested hyperparameter search on inner folds.

chance 0.2500 · majority 0.3777 · **demographics 0.5004**

| model | balanced acc | macro-F1 | accuracy | macro-AUC | vs demo |
|---|---:|---:|---:|---:|---:|
| EEG+DEMO [elasticnet] | **0.5178** | 0.4813 | 0.4703 | 0.7290 | +0.0174 |
| DEMO [l2] | **0.5004** | 0.4272 | 0.4299 | 0.7026 | +0.0000 |
| EEG [l2] | **0.3843** | 0.3682 | 0.3682 | 0.6284 | -0.1161 |
| EEG [elasticnet] | **0.3657** | 0.3466 | 0.3397 | 0.6279 | -0.1346 |
| RAW_BASELINE [rbf] | **0.3480** | 0.2674 | 0.4276 | 0.5993 | -0.1523 |
| EEG [rbf] | **0.3335** | 0.2738 | 0.4181 | 0.6350 | -0.1669 |
| GRAPH [elasticnet] | **0.3305** | 0.3149 | 0.3135 | 0.5886 | -0.1698 |
| SPECTRAL [elasticnet] | **0.3155** | 0.3032 | 0.3017 | 0.5708 | -0.1849 |

Recall per class (the tell for whether a model predicts every class):

| model | healthy | depression | anxiety | stress |
|---|---|---|---|---|
| EEG+DEMO [elasticnet] | 0.784 | 0.283 | 0.465 | 0.539 |
| DEMO [l2] | 0.743 | 0.132 | 0.616 | 0.510 |
| EEG [l2] | 0.635 | 0.346 | 0.233 | 0.324 |
| EEG [elasticnet] | 0.581 | 0.252 | 0.267 | 0.363 |
| RAW_BASELINE [rbf] | 0.486 | 0.906 | 0.000 | 0.000 |
| EEG [rbf] | 0.392 | 0.893 | 0.000 | 0.049 |
| GRAPH [elasticnet] | 0.514 | 0.264 | 0.221 | 0.324 |
| SPECTRAL [elasticnet] | 0.432 | 0.270 | 0.314 | 0.245 |
