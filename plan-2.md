# EEG Signal — Rebuilt Pipeline Execution Plan

## Objective

Rebuild the EEG pipeline from raw recordings using the unified preprocessing and artifact-harmonization protocol, then train and evaluate the proposed multi-task EEG model **only through the initial within-corpus and cross-corpus stress evaluation stage**.

The immediate goal is **not** to implement the entire research pipeline in the specification. We stop after obtaining honest subject/trial/recording-grouped performance from the rebuilt pipeline and comparing it against the expected ranges.

---

# Phase 0 — Freeze the Existing Pipeline

- [ ] Create a new branch for the rebuilt pipeline.
- [ ] Do not modify or overwrite the existing/legacy dataset.
- [ ] Preserve all existing scripts and results for historical comparison.
- [ ] Create a separate output/artifact directory for the rebuilt pipeline.
- [ ] Record package versions, Python version, CUDA/PyTorch version, and GPU information.
- [ ] Confirm the raw source files for:
  - MODMA
  - SAM40
  - Cognitive Load
- [ ] Confirm AMIGOS and WESAD are excluded from the rebuilt dataset.

### Required principle

The rebuilt pipeline must be reproducible from raw recordings. No previously windowed/normalized arrays should be reused for the new dataset.

---

# Phase 1 — Unified Preprocessing

## 1.1 Common preprocessing specification

Every retained corpus must ultimately produce:

- Sampling rate: **128 Hz**
- Window length: **4.0 s**
- Samples per window: **512**
- Window stride: **256 samples**
- Overlap: **50%**
- Channels:

```text
Fp1, Fp2, F7, F3, F4, F8
```

- Final segment shape:

```text
(6, 512)
```

## 1.2 MODMA

- [ ] Load native 250 Hz recordings.
- [ ] Map the native electrodes:

```text
E22 → Fp1
E9  → Fp2
E33 → F7
E24 → F3
E124 → F4
E122 → F8
```

- [ ] Apply 0.5–45 Hz zero-phase bandpass using `scipy.signal.filtfilt`.
- [ ] Ensure high-pass filtering removes the large DC pedestal before resampling.
- [ ] Resample 250 → 128 Hz using polyphase filtering:

```text
up = 64
down = 125
```

- [ ] Window into 4-second / 512-sample segments with stride 256.
- [ ] Track subject ID and trial/recording ID for every segment.
- [ ] Expected rough yield: ~7,959 segments before final artifact rejection.

## 1.3 SAM40

- [ ] Load native `raw_data/` recordings at 128 Hz.
- [ ] Do **not** decimate.
- [ ] For each 25-second trial, apply explicit 3-second reflection padding.
- [ ] Use:

```text
padlen = 384
```

- [ ] Apply the same 0.5–45 Hz zero-phase bandpass.
- [ ] Remove the padding after filtering.
- [ ] Window into 4-second / 512-sample segments with stride 256.
- [ ] Preserve subject ID and trial ID.
- [ ] Expected rough yield: ~5,280 segments before final artifact rejection.

## 1.4 Cognitive Load

- [ ] Load native 250 Hz recordings.
- [ ] Map:

```text
Column 1 → Fp1
Column 2 → Fp2
Column 3 → F7
Column 4 → F3
Column 6 → F4
Column 7 → F8
```

- [ ] Apply 0.5–45 Hz zero-phase bandpass.
- [ ] Resample 250 → 128 Hz using:

```text
up = 64
down = 125
```

- [ ] Window into 4-second / 512-sample segments with stride 256.
- [ ] Preserve subject ID and trial ID.
- [ ] Expected rough yield: ~4,800–5,300 segments before final artifact rejection.

---

# Phase 2 — Uniform Artifact Gating

The exact same rejection logic must be applied to all three retained corpora **after filtering/windowing**.

## 2.1 Flatline masking

- [ ] Detect zero-runs in the raw continuous signal.
- [ ] Mark flatline spans ≥ 50 ms.
- [ ] Reject a segment if it overlaps any such span.

At 128 Hz, 50 ms corresponds to approximately:

```text
6.4 samples
```

Therefore the implementation must define the integer sample threshold consistently and document it.

## 2.2 Channel variance gate

For each `(6, 512)` segment:

```text
variance[c] = Var(channel_c)
median_variance = median(variance)
```

Reject if:

```text
max(variance) > 5 × median_variance
```

This is intended to eliminate single-electrode blowouts/spikes without introducing corpus-specific thresholds.

## 2.3 Rejection report

Generate a table containing at minimum:

| Corpus | Initial segments | Flatline rejected | Variance rejected | Final segments |
|---|---:|---:|---:|---:|
| MODMA | | | | |
| SAM40 | | | | |
| Cognitive Load | | | | |
| **Total** | | | | |

- [ ] Save the table.
- [ ] Save rejection counts by subject.
- [ ] Verify no corpus has an unexpectedly extreme rejection rate.
- [ ] Investigate, rather than silently changing thresholds, if a corpus has unusual rejection behavior.

---

# Phase 3 — Robust Segment Normalization

Normalize each retained segment independently.

For each `(6, 512)` matrix:

```text
median = median(X)
MAD = median(|X - median|)
```

Then apply robust scaling using the chosen implementation convention.

Important:

- [ ] Compute statistics across the entire 6 × 512 matrix.
- [ ] Do not independently normalize each channel.
- [ ] Preserve inter-channel amplitude relationships and frontal asymmetry.
- [ ] Handle MAD = 0 safely.
- [ ] Verify the resulting distributions numerically.

Save:

```text
X_rebuilt.npy
metadata.csv / metadata.npy
```

Metadata must retain:

```text
subject_id
trial_id
corpus/domain
depression_label
stress_label
anchor_mask
```

---

# Phase 4 — Target Space

Construct every segment as:

```text
(
    X[i],
    y_dep,
    y_stress,
    y_domain,
    anchor_mask,
    subject_id,
    trial_id
)
```

## Depression target

```text
MODMA HC  → 0
MODMA MDD → 1
SAM40     → -1
CogLoad   → -1
```

## Stress target

```text
SAM40 Relax        → 0
SAM40 Task        → 1

CogLoad Natural   → 0
CogLoad Load/Task  → 1

MODMA             → -1
```

## Domain target

```text
MODMA      → 0
SAM40      → 1
CogLoad    → 2
```

## Anchor mask

```text
anchor_mask = True
if y_dep == 0 OR y_stress == 0
else False
```

The anchor mask must identify only non-pathological baseline recordings.

- [ ] Validate label counts.
- [ ] Validate subject counts.
- [ ] Validate that masked labels are never used by the corresponding supervised loss.
- [ ] Verify that no subject appears under the wrong corpus/domain.

---

# Phase 5 — Subject-Level Dataset Splitting

This is critical because windows from the same subject are highly correlated.

## Required rule

**Never randomly split individual EEG windows between train and test.**

- [ ] Build all splits using subject IDs.
- [ ] Ensure zero subject overlap between train and test.
- [ ] Stratify where feasible.
- [ ] Keep all windows/trials from a subject inside the same split.

Before training, run an automated leakage check:

```text
intersection(train_subjects, test_subjects) == ∅
```

Do the same for every validation/test split.

---

# Phase 6 — Shared EEG Feature Backbone

Implement the proposed shared feature extractor.

Architecture:

```text
Input: (6 × 512)

        ↓

Temporal Conv2D
kernel = 1 × 64
filters = 8

        ↓

Depthwise Conv2D
kernel = 6 × 1
depth multiplier = 2

        ↓

Separable Conv2D
kernel = 1 × 16
filters = 16

        ↓

Flatten / projection

        ↓

Latent feature vector z
```

- [ ] Implement the backbone independently first.
- [ ] Pass dummy `(batch, 6, 512)` input through it.
- [ ] Verify tensor dimensions at every layer.
- [ ] Confirm no accidental channel/time dimension reversal.
- [ ] Record latent dimension `d`.

---

# Phase 7 — Multi-Task Heads

Attach two task-specific heads.

## Depression head

```text
z
 ↓
Dense(32)
 ↓
Dense(1)
```

Used only for MODMA segments.

## Stress head

```text
z
 ↓
Dense(32)
 ↓
Dense(1)
```

Used only for SAM40 + Cognitive Load segments.

The model therefore has:

```text
Shared backbone
       │
       ├── Depression head
       │
       └── Stress head
```

- [ ] Confirm depression loss receives only MODMA samples.
- [ ] Confirm stress loss receives only SAM40/CogLoad samples.
- [ ] Confirm masked labels never contribute to supervised losses.

---

# Phase 8 — Anchor-Only Domain Adaptation

Add the domain adversarial branch only after the base multi-task model works.

```text
z
 │
 ├──────────────→ Depression Head
 │
 ├──────────────→ Stress Head
 │
 └→ Gradient Reversal Layer
          ↓
      Dense(64)
          ↓
      Domain logits (3)
```

The domain branch receives only:

```text
anchor_mask == True
```

Domain classes:

```text
0 = MODMA
1 = SAM40
2 = Cognitive Load
```

Use inverse anchor-domain weighting:

```text
w_c = N_anchor / (3 × N_anchor,c)
```

- [ ] Verify the GRL sign mathematically/experimentally.
- [ ] Verify non-anchor pathological samples receive no domain-alignment gradient.
- [ ] Verify class weights are calculated from the training fold only.
- [ ] Do not leak validation/test domain statistics into training.

---

# Phase 9 — Training Loss

Implement the combined objective:

```text
L_total =
    L_dep
  + L_stress
  + λ × L_domain
```

where the GRL causes the backbone to receive the reversed domain gradient.

The exact loss weighting should be configurable.

Suggested initial configuration:

```text
λ = 0
```

first, to establish the non-adversarial baseline.

Then:

```text
λ > 0
```

for the anchor-GRL model.

Do not begin with extensive hyperparameter search.

---

# Phase 10 — Baseline Before Domain Adaptation

Train the model with:

```text
λ = 0
```

This establishes whether the rebuilt preprocessing + multi-task architecture works before adding domain adaptation.

Record:

- training loss
- validation loss
- depression AUC
- stress AUC
- subject/trial/recording-level predictions
- random seeds
- model checkpoint
- training configuration

This baseline is essential for determining whether any later improvement comes from domain adaptation rather than the preprocessing changes.

---

# Phase 11 — Anchor-Only GRL Model

After the baseline is stable:

- [ ] Enable anchor-only GRL.
- [ ] Use inverse domain weighting.
- [ ] Train using subject-grouped folds.
- [ ] Compare against the λ=0 baseline.
- [ ] Record all predictions at the original segment level before aggregation.

Also calculate the anchor-domain silhouette diagnostic at:

```text
λ = 0
λ = chosen GRL setting
```

This is a diagnostic, not the primary success criterion.

---

# Phase 12 — Grouped Evaluation

Evaluation must happen at the appropriate independent unit rather than treating every overlapping window as an independent sample.

## MODMA

Primary:

```text
Subject-level ROC-AUC
N = 53 subjects
```

Procedure:

1. Generate segment logits.
2. Group segments by subject.
3. Aggregate logits using the median.
4. Convert aggregated logits to predictions.
5. Calculate subject-level ROC-AUC.

## SAM40

Primary:

```text
Trial-level ROC-AUC
N = 480 trials
```

Procedure:

1. Generate segment logits.
2. Group by trial.
3. Median-aggregate logits.
4. Calculate trial-level ROC-AUC.

Secondary:

```text
Mean within-subject AUC
```

## Cognitive Load

Primary:

```text
Recording-level ROC-AUC
N ≈ 120 recordings
```

Procedure:

1. Generate segment logits.
2. Group by recording.
3. Median-aggregate logits.
4. Calculate recording-level ROC-AUC.

---

# Phase 13 — Cross-Corpus Stress Test

This is the headline experiment.

## Experiment A

```text
Train: SAM40
Test:  Cognitive Load
```

No target-domain training samples are used.

## Experiment B

```text
Train: Cognitive Load
Test:  SAM40
```

Again, evaluate zero-shot transfer.

Primary metric:

```text
ROC-AUC
```

This experiment tests whether learned stress-related representations survive a corpus/hardware/recording-environment shift.

---

# Phase 14 — Expected Result Ranges

These are **expectations, not targets to manufacture or tune toward**.

### Depression — MODMA

Subject-level AUC:

```text
Expected: 0.65–0.80
Most likely: ~0.70–0.75
```

Expected 95% CI:

```text
roughly ±0.13
```

Balanced accuracy:

```text
roughly 0.65–0.72
```

### Stress — SAM40

Trial-level AUC:

```text
Expected: 0.75–0.85
```

### Stress — Cognitive Load

Recording-level AUC:

```text
Expected: 0.78–0.88
```

### Cross-Corpus Stress

Expected zero-shot AUC:

```text
Expected: 0.55–0.70
```

This is the most important result to watch.

---

# Phase 15 — Bootstrap Confidence Intervals

For the primary metrics:

- [ ] Bootstrap at the **evaluation unit**, not at the segment level.
- [ ] Use 1,000 bootstrap iterations.
- [ ] MODMA: bootstrap subjects.
- [ ] SAM40: bootstrap trials.
- [ ] Cognitive Load: bootstrap recordings.
- [ ] Cross-corpus: bootstrap the relevant target-domain evaluation units.

Report:

```text
AUC
95% CI
N
```

Do not treat overlapping windows as independent bootstrap observations.

---

# Phase 16 — Final Output Tables

Produce one consolidated results table:

| Experiment | Evaluation Unit | N | ROC-AUC | 95% CI |
|---|---|---:|---:|---:|
| MODMA Depression | Subject | 53 | | |
| SAM40 Stress | Trial | 480 | | |
| Cognitive Load Stress | Recording | ~120 | | |
| SAM40 → CogLoad | Recording/Trial | | | |
| CogLoad → SAM40 | Trial/Recording | | | |

Also produce:

### Preprocessing table

| Corpus | Raw Fs | Final Fs | Initial Segments | Rejected | Final Segments |
|---|---:|---:|---:|---:|---:|
| MODMA | 250 | 128 | | | |
| SAM40 | 128 | 128 | | | |
| Cognitive Load | 250 | 128 | | | |

### Model comparison

| Model | MODMA AUC | SAM40 AUC | CogLoad AUC | Cross-Corpus |
|---|---:|---:|---:|---:|
| λ = 0 baseline | | | | |
| Anchor-GRL | | | | |

---

# STOPPING POINT

**Stop implementation here.**

Do **not** proceed yet to:

- Dual frozen domain probes
- Full-set probe accuracy
- Cross-head extrapolation histograms
- Extensive hyperparameter optimization
- External MODMA/depression transfer
- PRED+CT / Mumtaz external evaluation
- Federated learning
- Additional architecture experimentation

Those are later research stages.

The current milestone is:

> **Build the rebuilt dataset → verify preprocessing → train the shared multi-task model → add anchor-only GRL → perform honest grouped within-corpus evaluation → perform cross-corpus stress transfer → report AUC + bootstrap CIs.**

---

# Historical Comparison — Keep Separate

The following numbers are historical diagnostics only and must not be presented as directly comparable to the rebuilt results:

```text
Legacy random segment split:
~0.939 3-class accuracy

Rebuilt-corpus shortcut baseline:
~0.645 majority accuracy
```

The old ~0.939 result is expected to be inflated because overlapping windows from the same subjects can appear in both train and test.

The rebuilt subject-grouped results should therefore be judged on methodological validity rather than whether they exceed the old number.

---

# Success Criteria

The pipeline is considered successfully rebuilt if:

- [ ] All retained corpora use the same final 128 Hz / 6-channel / 512-sample representation.
- [ ] Filtering occurs before windowing.
- [ ] MODMA and Cognitive Load are correctly resampled.
- [ ] SAM40 remains at native 128 Hz.
- [ ] Artifact rejection is identical across corpora.
- [ ] Segment normalization is robust and corpus-independent.
- [ ] No subject appears in both train and test.
- [ ] Depression supervision is restricted to MODMA.
- [ ] Stress supervision is restricted to SAM40/Cognitive Load.
- [ ] Anchor GRL is restricted to baseline/anchor segments.
- [ ] Evaluation is performed at subject/trial/recording level.
- [ ] Cross-corpus stress transfer is evaluated zero-shot.
- [ ] Primary metrics have 1,000-bootstrap 95% CIs.
- [ ] All preprocessing and rejection counts are logged.

## Scientific Interpretation Rule

Do not optimize the model until it reaches the expected numbers.

The expected ranges are there to provide a sanity check.

A result such as:

```text
Depression AUC = 0.73
```

with honest subject-level evaluation can be more scientifically valuable than:

```text
Accuracy = 0.94
```

from a leaky random-window split.

The central question of this stage is:

> **After removing subject leakage, corpus shortcuts, spectral offsets, and inconsistent preprocessing, does the learned EEG representation still generalize across subjects and recording environments?**
