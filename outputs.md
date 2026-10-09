# Rebuilt EEG pipeline — execution results

Run date: 2026-08-18. This run stops at the Phase 16 stopping point in `plan-2.md`. Legacy data, scripts, and results were not modified. Raw data and generated arrays remain untracked.

## Environment, source audit, and reproducibility

- Python 3.10.7; NumPy 2.2.6; SciPy 1.15.3; scikit-learn 1.7.2; PyTorch 2.13.0+cpu; CUDA unavailable.
- `openpyxl` 3.1.5 was installed to read the native MODMA workbook.
- Raw source inputs: 53 usable MODMA 128-channel resting recordings (250 Hz), 480 SAM40 trials (128 Hz), and 120 Cognitive Load text recordings (250 Hz). AMIGOS and WESAD were excluded.
- MODMA is supplied extracted rather than as ZIPs, and SAM40 is supplied in `Data/raw_data`; the raw reader was adjusted only for those locations.
- The raw-only preprocessing output `artifacts_v2/X_rebuilt.npy` has SHA-256 `9F5AFA985A90D967B0F4D496350AAD96413AA8E7208FC288D03BC81E9A137FC5`. It was regenerated after repairing the per-subject rejection-report writer; the hash before and after was identical. Thus the model checkpoints/predictions use the same feature array as the final artifact.
- No branch was created because `.git` is read-only in this execution environment.

## Preprocessing and target-space results

All retained segments are `(6, 512)`: Fp1, Fp2, F7, F3, F4, F8; 128 Hz; 4 seconds; 256-sample stride. MODMA/Cognitive Load used the specified 0.5–45 Hz zero-phase filter followed by 64/125 polyphase resampling; SAM40 used the specified 384-sample reflection pad during filtering and remained at 128 Hz. Flatline threshold was 7 samples at 128 Hz. Each retained segment was whole-matrix MAD normalized.

| Corpus | Raw Fs | Final Fs | Initial segments | Flatline rejected | Variance rejected | Final segments |
|---|---:|---:|---:|---:|---:|---:|
| MODMA | 250 | 128 | 7,959 | 0 | 698 | 7,261 |
| SAM40 | 128 | 128 | 5,280 | 0 | 165 | 5,115 |
| Cognitive Load | 250 | 128 | 5,369 | 61 | 827 | 4,481 |
| **Total** | — | — | **18,608** | **61** | **1,690** | **16,857** |

Label audit: MODMA depression 0/1 = 3,953/3,308; SAM40 + Cognitive Load stress 0/1 = 2,738/6,858; anchor/non-anchor = 6,691/10,166. Subject counts are MODMA 50, SAM40 40, Cognitive Load 14. No metadata subject occurs in more than one corpus.

Warning investigated: Cognitive Load rejected 16.5% of candidate segments, higher than MODMA (8.8%) and SAM40 (3.1%). Rejections are concentrated in supplied raw recordings, notably `cogload_4` (32 flatline, 271 variance) and `cogload_15` (204 variance). The uniform gates were not changed.

## Splits, architecture, and training

Five out-of-fold subject-grouped splits were used. Test folds have 20, 21, 21, 21, and 21 subjects, respectively; every train/test, train/validation, and validation/test subject intersection was exactly zero. All segments/trials from each subject remained in its assigned split.

The shared network uses temporal Conv2D (8 filters, `1×64`), depthwise Conv2D (`6×1`, multiplier 2), separable Conv2D (16 filters, `1×16`), flatten/projection to latent dimension 64, and separate Dense(32)→Dense(1) depression and stress heads. A dummy `(2, 1, 6, 512)` input produced task logits `(2,)` and latent features `(2, 64)`. Depression loss was masked to MODMA and stress loss to SAM40/Cognitive Load.

Fixed configuration, with no search: seed 20260817, Adam learning rate 0.001, batch size 64, 10 epochs, lambda=0 baseline then lambda=1 anchor-GRL. The GRL sign check passed (`dL/dx = -1`). Anchor domain counts and inverse weights were calculated from each training fold only; counts (MODMA/SAM40/Cognitive Load) were `[2754,776,894]`, `[2520,791,1108]`, `[2300,840,804]`, `[2915,805,838]`, and `[2465,842,1120]`.

The CPU host terminated the first long command after checkpoints/predictions had completed but before in-memory per-epoch histories could be serialized for the within-corpus baseline and anchor-GRL runs. This is a logging failure only: all five saved checkpoints and the complete out-of-fold prediction files were recovered and evaluated. Training/validation losses for Cognitive Load→SAM40 are retained in `results_v2/results.json`; histories for the other completed models are unavailable and are not reconstructed or fabricated.

## Grouped within-corpus evaluation

Segment logits were median-aggregated at the specified independent unit. Confidence intervals are percentile 95% CIs from 1,000 bootstrap resamples of subjects/trials/recordings—not segments.

| Model | Experiment | Evaluation unit | N | ROC-AUC | 95% CI |
|---|---|---|---:|---:|---:|
| λ = 0 | MODMA Depression | Subject | 50 | 0.443 | 0.282–0.608 |
| λ = 0 | SAM40 Stress | Trial | 479 | 0.733 | 0.681–0.783 |
| λ = 0 | Cognitive Load Stress | Recording (`trial_id`) | 112 | 0.669 | 0.537–0.782 |
| Anchor-GRL, λ = 1 | MODMA Depression | Subject | 50 | 0.469 | 0.311–0.631 |
| Anchor-GRL, λ = 1 | SAM40 Stress | Trial | 479 | 0.497 | 0.435–0.565 |
| Anchor-GRL, λ = 1 | Cognitive Load Stress | Recording (`trial_id`) | 112 | 0.522 | 0.404–0.637 |

SAM40 secondary mean within-subject AUC (40 eligible subjects): 0.648 for λ=0 and 0.504 for anchor-GRL. Anchor-domain silhouette diagnostics: −0.126 for λ=0 and −0.065 for anchor-GRL.

## Zero-shot cross-corpus stress evaluation

Each model used only source-corpus samples for training/validation; target-domain rows used in training = 0. CIs are 1,000-bootstrap CIs over target recordings/trials.

| Experiment | Model | Evaluation unit | N | ROC-AUC | 95% CI |
|---|---|---|---:|---:|---:|
| SAM40 → Cognitive Load | λ = 0 baseline | Recording (`trial_id`) | 112 | 0.310 | 0.187–0.442 |
| Cognitive Load → SAM40 | λ = 0 baseline | Trial | 479 | 0.531 | 0.466–0.591 |

Anchor-GRL zero-shot: **NOT RUN**. Under the plan, zero-shot forbids target-domain training samples, whereas inverse anchor weighting requires all three anchor-domain counts. With a single source domain, two required counts are zero and the mandated weights are undefined. It was not replaced with a different objective.

## Final comparison

| Model | MODMA AUC | SAM40 AUC | CogLoad AUC | Cross-corpus stress |
|---|---:|---:|---:|---|
| λ = 0 baseline | 0.443 | 0.733 | 0.669 | SAM40→CogLoad 0.310; CogLoad→SAM40 0.531 |
| Anchor-GRL, λ = 1 | 0.469 | 0.497 | 0.522 | NOT RUN (methodologically undefined under strict zero-shot) |

## Files and final status

Generated results include `rebuilt_pipeline/results_v2/results.json`, five checkpoints per within-corpus model, segment-level out-of-fold prediction CSVs, grouped prediction CSVs, both zero-shot checkpoints/predictions, and grouped split indices. All AUCs and CIs above were computed from these files.

- **COMPLETED:** raw-only preprocessing; uniform artifact gating and normalization; target validation; subject-grouped split/leakage checks; shared backbone and multi-task heads; λ=0 baseline; anchor-only GRL within-corpus model; grouped evaluation; both zero-shot baseline stress experiments; 1,000-bootstrap CIs; final tables.
- **FAILED (logging only):** per-epoch training/validation histories for the completed within-corpus baseline/GRL and SAM40→Cognitive Load run were not persisted before the host timeout. No metric was fabricated to fill this gap.
- **NOT RUN:** anchor-GRL zero-shot transfer, because it cannot satisfy both strict target exclusion and the plan-mandated three-domain inverse anchor weighting. No work beyond the plan’s stopping point was performed.
