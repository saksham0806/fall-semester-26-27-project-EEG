# OSF_Park_model — the four-class track

Models on `datasets/OSF_park/EEG.machinelearing_data_BRMH.csv`, the Brain Research & Metrics
Hospital psychiatric cohort. Standalone: pure sklearn + torch, **does not import `src/`**
(that would drag TensorFlow in via `src/proposed_model/__init__.py`).

## Why this dataset

`CLAUDE.md` states the project's goal is 4-class Stress / Anxiety / Healthy / Depression and
that no anxiety label exists anywhere in the repo. This file has all four in **one cohort
with one feature schema** — so unlike `datasets/all3/`, class is not confounded with
recording corpus. That is the only reason to care about it.

| class | source | n |
|---|---|---:|
| healthy | `main.disorder == 'Healthy control'` | 93 |
| depression | `specific.disorder == 'Depressive disorder'` | 199 |
| anxiety | `main.disorder == 'Anxiety disorder'` (panic + social) | 107 |
| stress | `main.disorder == 'Trauma and stress related disorder'` (PTSD + acute + adjustment) | 128 |
| | | **527** |

## Three things that will mislead you

**1. It is not raw EEG.** 114 absolute band powers (6 bands × 19 channels) + 1026 coherences,
precomputed. No time series. Nothing from `combine_datasets.py`, the `(6, 512)` pipeline,
EEGNet or `DL_Without_Features/` applies. "DL" and "Transformer" here mean tabular/graph
models over channel tokens.

**2. The bar is demographics, not chance.** Age, sex, IQ and education alone score ~0.43,
and healthy controls are the youngest and highest-IQ group while the trauma group is oldest
and 34% male. `run.py` computes chance / majority / demographics before any model and refuses
to write a table without them. **Never quote an accuracy from this folder alone.**

**3. There are no subject IDs.** `no.` indexes rows, not people, across 2011–2018.
`StratifiedKFold` is the only option — this is grouping *by necessity*, not by proof. If any
person was recorded twice their rows can straddle a fold. `run.py` therefore also runs a
**recording-year probe**: if year turns out highly recoverable, this cohort has its own
version of the all3 corpus shortcut and every row needs that caveat.

Two data-integrity fixes applied in `data.py`, both silent if skipped:

- `Unnamed: 122` is a trailing-comma artifact and the **only** all-NaN column. Drop it or a
  naive `dropna()` deletes all 945 rows.
- Rows `no. == 206` and `no. == 944` are dropped. Six rows share byte-identical band-power
  blocks (205↔206, 226↔944, 264↔265). 206 goes because 205↔206 is a pair *within* the subset,
  so identical features would land in train and test. 944 goes because its AB block is
  byte-identical to a *different person's* (226, Behavioral addiction), so its band powers are
  probably not its own. Cost falls on the smallest class: healthy 95 → 93.

## Models

| # | Family | Input |
|---|---|---|
| 1 | SVM · RandomForest · LogisticRegression · GradientBoosting | 1140 flat features |
| 2 | MLP (torch) | 1140 flat features |
| 3 | ChannelTransformer | 19 channel tokens × 6 bands |
| 4 | CohTransformer | as above + per-head attention bias from the 6 coherence matrices |
| 5 | **GatedHybrid** (proposed) | branch A = 4 · branch B = 1026 COH projection · branch C = demographics, combined by a learned per-sample softmax gate |

Family 5 is the point. Coherence enters branch A as **graph structure over channels** rather
than 1026 flat numbers, and branch C is **ablatable** — running `[+demographics]` against
`[no demographics]` is the direct measurement of whether EEG contributes anything beyond
age/sex/IQ on this cohort. Its gate weights double as the explanation.

Sizing is deliberately small (d=32, 2 layers, ~50k params against 528 rows). This repo has
already measured twice — v2 features, M5 channel density — that added capacity on ~100
subjects gets consumed by noise.

Boosting is labelled **GradientBoosting, not XGBoost**: `xgboost` is in `requirements.txt`
but is *not* installed in `.venv`.

## Explainability

No new dependency — `shap` is not installed. Everything is aggregated to **19 channels and
6 bands**, which is the readable question, not 1140 raw columns:

- **Grouped permutation importance** — model-agnostic, works on all five families. Permutes
  the raw AB block and rebuilds every view through `data.to_blocks()`, so the flat,
  channel-tensor and coherence-bias inputs move together. Permuting only the flat copy would
  leave the transformer reading unshuffled data and report zero importance.
- **Native** `feature_importances_` / `|coef_|` for the tree and linear models.
- **Gradient × input** for MLP and the transformers.
- **Mean attention received per channel** for families 3–5 (these mean-pool, so there is no
  CLS token to read).
- **Gate weights** for family 5.

XAI runs on a **held-out** fold. On training rows it reports memorisation, not evidence.

## Plan modules (implemented)

`plans/` holds the design docs; these are the implementations.

| module | plan | what |
|---|---|---|
| `make_holdout.py` | 00 | freezes the dev/test split to `splits/*.json`; refuses to overwrite |
| `cv.py` | shared | dev-only nested CV, metrics, paired bootstrap, tables |
| `features_corrected.py` | 01 | CLR / log-abs / FAA / TBR / Fisher-z / graph reduction |
| `plan01.py` | 01 | the 8-row ablation grid |
| `normative.py` | 02 | residualization + ΔAUC with paired bootstrap |
| `hierarchical.py` | 03 | hierarchy, stacking, auxiliary 850, subgroup confusion |
| `final_test.py` | 00 | scores the frozen test set — append-only, needs `--yes` |

**The test set is scored once per plan.** `final_test.py` appends to `Results/OSF_Park/FINAL.md`
and refuses to run without `--yes`. `cv.dev_folds` asserts no test index reaches a CV fold, so
the guarantee is enforced by code rather than discipline.

Two results worth knowing before reading the code:

- **MST-union thresholding, not proportional.** Plain top-*d* thresholding fragments 94% of
  subject-band graphs at `d=0.20` and 22% even at `d=0.50`, making characteristic path length
  infinite. MST-union gives 0 fragmented at `d=0.20`.
- **CLR is gain-invariant by construction.** Scale a channel's six band powers by any constant
  — skull thickness, impedance, electrode contact — and the CLR output is unchanged. That is
  what makes it the right transform for absolute µV², and it is asserted in the self-check
  (in float64; `AB` is float32 and the invariance only holds to ~7e-7 there).

## Commands

```bash
# plan modules
.venv/bin/python -m OSF_Park_model.make_holdout          # freeze the split (once)
.venv/bin/python -m OSF_Park_model.features_corrected    # self-check + build graph cache
.venv/bin/python -m OSF_Park_model.normative --selfcheck
.venv/bin/python -u -m OSF_Park_model.plan01 --task 4class
.venv/bin/python -u -m OSF_Park_model.normative --task 4class
.venv/bin/python -u -m OSF_Park_model.hierarchical
.venv/bin/python -u -m OSF_Park_model.final_test --plan 01 --view EEG+DEMO --yes   # ONCE

# original pipeline
.venv/bin/python -m OSF_Park_model.data       # 527/292 rows, band-major round-trip, symmetry
.venv/bin/python -m OSF_Park_model.models     # all 8 models forward; coherence bias wired in
.venv/bin/python -u -m OSF_Park_model.run --task 4class --smoke   # 1 seed, ~40 s
.venv/bin/python -u -m OSF_Park_model.run --task 4class           # 3 seeds x 5 folds
.venv/bin/python -u -m OSF_Park_model.run --task binary           # healthy vs depressed, n=293
.venv/bin/python -u -m OSF_Park_model.run --task 4class --models SVM Hybrid   # label filter
```

Output: `Results/OSF_Park/<task>.json` (written after every model, so a killed run keeps
finished rows), out-of-fold probabilities in `Results/OSF_Park/preds/`, and
`OSF_Park_model/RESULTS_<task>.md`.

CPU only, no GPU path — 527 × 1140 is trivial and this box is memory-tight.

## Sanity anchors

SVM on the 1140 EEG features should land near **0.43** accuracy, demographics-only near **0.43**.
Under plan 01's dev split and balanced-accuracy ranking the demographics bar is **~0.50**.
Far off means the label mapping or the split is wrong. **Above ~0.75 means the split broke** —
with n=528 and 1140 features that is the failure mode to suspect first, not a discovery.

`--smoke` trains the torch models for 8 epochs, so their rows are undertrained by design and
are not comparable to a full run.

## Not used

`Data1`–`Data6` (the Park et al. biofeedback RCT in the same folder) are skipped: 30 MDD
patients, no healthy control arm, so the contrast is treatment not diagnosis and it cannot
supply a class label.
