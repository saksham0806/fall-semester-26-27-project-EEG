# Explainable AI for the saved screening models

From the repository root, using the existing `.venv`:

```bash
# Numerical checks; no project models or datasets are loaded.
.venv/bin/python explainable_ai/test_explain.py

# Generate explanations for all three existing screening models, CPU only.
.venv/bin/python -u explainable_ai/explain.py

# Regenerate one protocol only.
.venv/bin/python -u explainable_ai/explain.py --protocol resting
```

Reads `models/{resting_healthy_vs_depressed,screen_healthy_vs_ill,task_healthy_vs_stressed}.joblib`
and `datasets/all3/{X,y,subjects,corpus}.npy`. Run on trusted project artifacts only:
joblib and object-dtype subject metadata are executable serialization formats.
Uses existing NumPy, SciPy, scikit-learn, joblib and Matplotlib dependencies. No install,
GPU, training, dataset rebuild, or external service is required. The feature extractor
is imported directly to avoid the parent package's TensorFlow and torch imports.

Outputs stay in `explainable_ai/results/`:

- `REPORT.md`: generated report with figures and limitations.
- Each protocol subfolder has three figures, each as 300-DPI PNG and editable SVG:
  `global_importance`, `local_feature_effects`, `explained_window`.
- `global_importance.csv`: channel, family and band permutation importance.
- `local_feature_effects.csv`: all 81 single-feature replacement effects.
- `sample_predictions.csv`: selected source rows, identities, labels and predictions.
- `explanation.json`: numerical results, individual shuffle drops, column mappings,
  local reference values, model/data SHA-256 hashes and existing CV metrics.

The run samples at most four windows per (corpus, subject), with seed 42. It performs
five shuffles per group **within each corpus**. Group columns share the same donor
row to preserve dependencies inside the group. Importance is the change in sample
ROC AUC; mean absolute probability changes are exported alongside it. The local
example is the first sampled row, selected without looking at labels or predictions.
Its reference is the feature-wise median of other sampled subjects from that corpus.

**Interpretation:** these explain the final fitted models on a training subset. They
are not held-out explanations or new validation results. Local effects are sensitivity
measurements, not additive SHAP attributions or causal effects. Channel groups overlap
at FAA pairs; band groups include PSD and DE only. Correlation across groups can affect
importance. The saved models inherit the documented legacy SAM40 rate error and
device confounds. The task model remains below its CV majority baseline. Read
`results/REPORT.md` before interpreting its plots as evidence about EEG physiology.

Rerunning replaces this folder's generated files for the selected protocol. A partial
run keeps outputs already completed; the report includes all protocol results present
on disk, with their individual hashes in JSON. Nothing is written to `models/` or
`datasets/`.
