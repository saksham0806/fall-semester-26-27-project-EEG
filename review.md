# Review Notes — all3 3-Class Track

**Status: code and reporting infrastructure written. No new experimental results.**
Everything below that carries a number is a *previously measured* result; everything new is
written, syntax-checked, and unrun. That distinction is the most important thing to keep
straight when presenting this.

---

## 1. Lead with this

> A classifier that knows **only which amplifier recorded the segment** — with zero EEG
> understanding — scores **0.6267** on the 3-class task. The best of our 18 models scores
> **0.6256**.

That single comparison is the finding. The 3-class problem as currently constituted is
largely a device-identification problem, and reporting a raw 3-class accuracy without that
baseline next to it would be misleading.

**The supporting evidence, all already measured:**

| Measurement | Value | What it shows |
|---|---|---|
| Chance | 0.3333 | floor |
| Majority class | 0.4217 | constant predictor |
| **Corpus-ID only** | **0.6267** | **the real bar** |
| Best model (SVM, 81 features) | 0.6256 | −0.0011 vs the bar |
| Corpus probe (predict the device) | 0.9161 | the shortcut is fully available |
| Riemannian, device signature removed | 0.3255 | ≈ chance — nothing left |

The last row is the clincher. Strip the amplifier signature via per-corpus Riemannian
re-centering and performance collapses to chance. **The signature was worth +0.18, and
without it nothing remains.**

### Why the confound exists (structural, not fixable by modelling)

| Corpus | healthy | stress | depression |
|---|:---:|:---:|:---:|
| MODMA | ✅ | — | ✅ |
| SAM40 | ✅ | ✅ | — |
| CogLoad | ✅ | ✅ | — |

**No corpus contains all three classes.** Each used a different amplifier. So "which class"
and "which device" are nearly the same question, and no architecture can separate them from
this data. This is a data-collection limitation.

---

## 2. The honest measure: within-corpus accuracy

Pooled accuracy includes the free corpus split. Accuracy *inside* each corpus does not.

| Corpus | majority baseline | best model | margin |
|---|---|---|---|
| MODMA | 0.5478 | 0.6586 | **+0.111** ✅ |
| SAM40 | 0.7500 | 0.7504 | +0.000 |
| CogLoad | 0.6886 | 0.5765 | **−0.112** ❌ |

**In 2 of 3 corpora no model beats a constant predictor.** CogLoad is the open problem and
the target for any new work.

Contrast with the *same data* framed as two binary tasks:

| Task | Best model | Score |
|---|---|---|
| Depression (MODMA) | SVM on 81 handcrafted features | **0.7716** subject-level AUC |
| Stress (SAM40) | Riemannian tangent space | **0.7560** within-subject AUC |

Forcing one model to do both tasks makes it worse at each.

---

## 3. What was built this session

### 3.1 A model that attacks the confound — `scripts/all3_dann.py` *(written, unrun)*

Domain-adversarial network: a gradient reversal layer sits between the shared backbone and a
corpus classifier, pushing the backbone toward features that predict **class** but not
**corpus**. Every prior model exploited the device signature; this one is built to destroy it.

- Reuses `build_dann_eeg_model` in `src/proposed_model/multitask_dann_models.py`, which
  already existed and had never been run on all3.
- Added an `EEGTransformer` backbone branch to it (reusing the existing
  `TransformerEncoderBlock`), so ML / CNN / transformer are all covered under one protocol.
- **8 configs**: 2 backbones × alpha ∈ {0.0, 0.1, 0.3, 1.0}.
- **The alpha = 0.0 runs are controls** — adversary disabled. Without them, a change cannot be
  attributed to the adversary rather than to the backbone.

**State the expected outcome before running it:** pooled accuracy should *fall* toward 0.33,
because the model is deliberately destroying the signature that produces 0.6267. That is the
measurement, not a failure. Success is judged only on CogLoad within-corpus > 0.6886.

There is also a built-in check that the method worked: the domain head's own accuracy should
fall from ~0.92 toward 0.33 as alpha rises. If it stays flat, the gradient reversal isn't
training and the task numbers mean nothing.

### 3.2 Reporting infrastructure — `scripts/all3_diagnostic.py`, `all3_tracks.py`

A real gap: across 18 models, `Results/all3/` contained **no macro-F1, no per-class metrics,
and exactly one confusion matrix**. A 3-class accuracy can hide a class that is never
predicted once.

`report()` — shared by every all3 script, so one edit upgrades all of them — now emits
macro-F1, per-class precision/recall/F1, a confusion matrix for every model, within-corpus
margins, and runtime. It also **saves out-of-fold predictions to disk**, so any metric we
think of later is a recompute from arrays rather than another training run.

`all3_tracks.py` produces `Results/all3/TRACKS.md` with two tables that must be read together:
- **Track A** — within-corpus, the honest measure.
- **Track B** — pooled, plus a **signature delta**: the same model with the device signature
  preserved vs removed. That gap *is* the amplifier's contribution.

### 3.3 Feature extraction — `scripts/extract_all3_features.py` *(written, unrun)*

Materialises `features/all3/`: 81 handcrafted features, both Riemannian views (105-dim each),
labels, corpus, subject IDs, and a JSON with feature names. Reuses the existing extractors and
the cached Riemannian arrays rather than recomputing.

### 3.4 Two efficiency fixes

- `all3_diagnostic.py` was recomputing Riemannian features on every run while
  `all3_models.py` had already cached the identical array. Now reads the cache: **~15 min → ~2 min.**
- No script recorded its own runtime, so time estimates in the docs had been reconstructed
  from file timestamps. Now recorded properly. (This also corrected `all3_deep.py`'s
  documented runtime from ~40 min to ~19 min.)

### 3.5 A documentation correction worth mentioning

`xgboost` is listed in `requirements.txt` but is **not installed** in the environment. Every
row labelled "XGBoost" anywhere in `Results/` or `report.md` was actually produced by
sklearn's `GradientBoostingClassifier` via a silent fallback. **Those rows should be relabelled
or xgboost should be installed before any of them is quoted as an XGBoost result.**

---

## 4. What has NOT been run — say this plainly

Nothing in section 3 has been executed. No new results exist. Specifically:

- ❌ `all3_dann.py` — never run, no DANN numbers exist
- ❌ `extract_all3_features.py` — `features/all3/` does not exist yet
- ❌ `all3_tracks.py` — `TRACKS.md` not yet generated
- ❌ The extended metrics — existing JSONs still carry only the old scalar keys; the scripts
  must be re-run to backfill macro-F1 and per-class numbers

Estimated ~60 minutes total, sequential, on the current machine. A GPU box is being
provisioned, and a larger plan (foundation-model fine-tuning, a nested-CV hyperparameter
sweep, and self-supervised pretraining on the 39,429 unused AMIGOS segments) is designed but
not started.

---

## 5. Anticipated questions

**"Why not just add a bigger/newer model?"**
Three independent lines of evidence say the deep family loses on this data: all 6 deep
architectures land 0.10–0.20 *below* the corpus shortcut; a CBraMod foundation-model probe
scored 0.5716, i.e. 0.18 below a 105-dim linear model; and self-supervised pretraining scored
*below* its own random-init control. The dataset is also small — 15,728 segments, 108
subjects, ~193 MB — so more capacity overfits harder. Capacity was never the constraint.

**"Isn't 0.6256 close to 0.6267? That's nearly the baseline."**
It is *below* it. And the baseline requires no EEG understanding whatsoever. Matching a
device-ID lookup is not evidence of learning anything about mental state.

**"Have you tried combining the models?"**
Yes — three fusion strategies (early, late, and a learned gated fusion) were run. All three
landed 0.03–0.04 *below* the best single view. A fourth (attention over four view embeddings)
was planned but deliberately not run, because the original plan gated it behind "only if the
first three show fusion helps at all." They didn't.

**"So is this a failure?"**
No — it's a well-supported negative result on the 3-class framing, plus two genuinely working
binary models (0.7716 and 0.7560 AUC). The honest conclusion is that the 3-class question
cannot be answered with these three corpora, because no corpus supervises more than two of the
three classes. Answering it properly needs a dataset where one recording setup captures all
three.

**"What's the caveat on the negative result?"**
Every deep result here comes from a **single hardcoded configuration** (25 epochs, batch 128,
LR 5e-4). Nothing has been tuned. "Deep models lose here" is currently an *untuned* finding —
well supported, but a proper hyperparameter sweep is what would make it airtight. That's
planned for the GPU box. Similarly, the CBraMod number is a *frozen probe measured on stress*,
not a fine-tune and not an all3 number.

---

## 6. Proposed next steps

1. **Run the written code** (~60 min) → populates `TRACKS.md` with complete metrics for all
   models plus the 8 DANN configs.
2. **On the GPU box**: fine-tune foundation models (CBraMod / Labram / BIOT, all available in
   braindecode), each against a random-init control; run the nested-CV hyperparameter sweep to
   convert the untuned negative into a tuned one; SSL on the unused AMIGOS segments.
3. **If CogLoad within-corpus stays below 0.6886 across all of it**, write the negative result
   up formally and close the 3-class track — the blocker is data collection, not modelling.

**MODMA also ships PHQ-9 and GAD-7 scores for all 53 subjects, currently unused.** GAD-7 is
the most plausible route to an actual anxiety label, which the original 4-class research goal
requires and which no current dataset provides.

---

## 7. One-paragraph summary

*We asked whether one model could separate healthy, stress, and depression. It can't — not
because the models are weak, but because class identity is confounded with recording device:
each of the three corpora used a different amplifier and none contains more than two of the
three classes. A classifier using only device identity scores 0.6267; our best of 18 models
scores 0.6256, and removing the device signature drops performance to chance. In two of three
corpora, no model beats a constant predictor. The same data split into two binary tasks gives
0.77 and 0.76 AUC. This session added a domain-adversarial model designed to attack that
confound directly, along with the reporting infrastructure needed to tell a real result apart
from device detection — both written and ready to run.*
