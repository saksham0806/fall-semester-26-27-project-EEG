# Rebuilt EEG pipeline — execution status

Run date: 2026-08-17.  All new code and artifacts are isolated under `rebuilt_pipeline/`; no legacy dataset, script, model, or result was edited or deleted.

## Continuation update

The Cognitive Load source was re-inspected file-by-file.  Fifteen genuine `natural-*.txt` recordings were found in `Stroop_Data`; condition is encoded in the filename rather than directory.  They are now assigned stress label 0 exactly as the plan requires.  The earlier `artifacts/` dataset is preserved as a superseded implementation error; the audited corrected dataset is in `rebuilt_pipeline/artifacts_v2/`.

## Source inspection

- MODMA: all six supplied ZIP archives opened successfully.  They contain the 128-channel resting-state MATLAB recordings and the resting-state subject workbook.  The ZIPs are readable.
- SAM40: raw `datasets/SAM40/raw_data/*.mat` files were used (the raw `Data` variable, 128 Hz).
- Cognitive Load: raw text files under `datasets/COGLoad/raw_data-20260817T114751Z-1-001/raw_data/` were used (45 Arithmetic and 60 Stroop recordings).
- AMIGOS and WESAD were excluded.

## Implemented and run

`rebuilt_pipeline/run_rebuilt_preprocessing.py` implements raw-only ingestion, the specified six-channel mappings, 0.5–45 Hz zero-phase filtering, 250→128 Hz polyphase resampling for MODMA/Cognitive Load, explicit 384-sample SAM40 reflection padding, 512-sample windows with 256-sample stride, raw zero-run flatline gating, variance gating, and whole-segment MAD normalization.

Successful corrected generated artifacts:

- `rebuilt_pipeline/artifacts_v2/X_rebuilt.npy`: `(16131, 6, 512)`, 198,217,856 bytes.
- `rebuilt_pipeline/artifacts_v2/metadata.csv`: segment-level subject, trial, corpus, depression/stress labels, domain, and anchor mask.
- `rebuilt_pipeline/artifacts_v2/preprocessing_report.json`: complete audit, labels, environment, and normalization diagnostics.
- `rebuilt_pipeline/artifacts_v2/*_rejection_by_subject.csv`: per-subject rejection counts for all three corpora.

Genuine preprocessing/rejection audit:

| Corpus | Raw Fs | Final Fs | Initial segments | Flatline rejected | Variance rejected | Final segments |
|---|---:|---:|---:|---:|---:|---:|
| MODMA | 250 | 128 | 7,959 | 0 | 698 | 7,261 |
| SAM40 | 128 | 128 | 5,280 | 0 | 165 | 5,115 |
| Cognitive Load | 250 | 128 | 4,533 | 61 | 717 | 3,755 |
| **Total** | — | — | **17,772** | **61** | **1,580** | **16,131** |

Observed retained-data diagnostics:

| Corpus | Final segments | Unique subjects | Unique trials/recordings | Depression labels | Stress labels | Anchor segments |
|---|---:|---:|---:|---|---|---:|
| MODMA | 7,261 | 50 | 50 | 0: 3,953; 1: 3,308 | -1: 7,261 | 3,953 |
| SAM40 | 5,115 | 40 | 479 | -1: 5,115 | 0: 1,286; 1: 3,829 | 1,286 |
| Cognitive Load | 3,755 | 14 | 98 | -1: 3,755 | 0: 726; 1: 3,029 | 726 |
| Total | 16,131 | — | — | — | — | 5,965 |

The observed metadata has no subject assigned to more than one corpus.

## Incomplete items and reasons

1. The original audit-runtime failure is resolved in `artifacts_v2`; all required preprocessing and per-subject rejection reports are now present.
2. Subject-grouped splits, model training (baseline and anchor-GRL), grouped evaluation, cross-corpus evaluation, and bootstrap confidence intervals are not yet run.  No performance metrics, predictions, checkpoints, or CIs have been generated.
3. Creating the Phase 0 branch was attempted but blocked by this execution environment's read-only `.git` access: Git could not create `.git/refs/heads/codex/rebuilt-eeg-pipeline`.  No branch was created.

## Execution conclusion

The raw preprocessing and target-space portions are now complete and scientifically valid; Cognitive Load Natural recordings are available.  The remaining model/evaluation phases still require execution.
