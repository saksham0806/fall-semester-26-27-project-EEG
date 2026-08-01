# Review Notes — all3 3-Class Track

**Status: everything described here has now been RUN. 26 models, none beats the baseline.**

This document was originally written when the new code was unrun; it has been updated with
measured results. The headline is unchanged and now much better supported: the 3-class task
is not solvable with these three corpora, and we have a purpose-built model plus a corrected
feature pipeline to show it isn't for want of trying.

**Read §3.6 first if short on time** — that is the new experimental result.

---

## 1. Lead with this

> A classifier that knows **only which amplifier recorded the segment** — with zero EEG
> understanding — scores **0.6267** on the 3-class task. The best of our 26 models scores
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

### 3.1 A model that attacks the confound — `scripts/all3_dann.py` *(run — results in §3.6)*

Domain-adversarial network: a gradient reversal layer sits between the shared backbone and a
corpus classifier, pushing the backbone toward features that predict **class** but not
**corpus**. Every prior model exploited the device signature; this one is built to destroy it.

- Reuses `build_dann_eeg_model` in `src/proposed_model/multitask_dann_models.py`, which
  already existed but had never been run on all3 until now.
- Added an `EEGTransformer` backbone branch to it (reusing the existing
  `TransformerEncoderBlock`), so ML / CNN / transformer are all covered under one protocol.
- **8 configs**: 2 backbones × alpha ∈ {0.0, 0.1, 0.3, 1.0}.
- **The alpha = 0.0 runs are controls** — adversary disabled. Without them, a change cannot be
  attributed to the adversary rather than to the backbone.

**The expected outcome was stated before running it:** pooled accuracy should *fall*, because
the model is deliberately destroying the signature that produces 0.6267. Success was judged
only on CogLoad within-corpus > 0.6886. Both predictions were registered in advance, and §3.6
reports what actually happened.

There is also a built-in check that the method worked: the domain head's own accuracy should
fall as alpha rises. The right reference is **each backbone's own α=0 control** (≈0.50), not
the 0.9161 corpus probe — that figure is an SVM on 81 handcrafted features, whereas the domain
head reads a 128-d learned representation. They are different quantities and were never
comparable.

### 3.2 Reporting infrastructure — `scripts/all3_diagnostic.py`, `all3_tracks.py`

A real gap: across the first 18 models, `Results/all3/` contained **no macro-F1, no per-class metrics,
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

### 3.3 Feature extraction v2 — `scripts/extract_features_v2.py` *(run)*

`features_v2/all3/`, **15,728 × 273** in 168 s: corrected preprocessing (50 Hz notch +
zero-phase 0.5–45 Hz bandpass) plus four families the repo never had — Hjorth, 1/f aperiodic
slope, permutation entropy, and connectivity (coherence + PLV).

It was built to test a specific hypothesis, and **the hypothesis was refuted.** The 45–64 Hz
gap — excluded from every feature band but included in the relative-power denominator — holds
84% of MODMA's total power (50 Hz mains), 35% of SAM40's, 23% of CogLoad's. Removing all of it
changed corpus recoverability by **−0.010** (0.9161 → 0.9057 on the same 81 features). The
amplifier signature lives *inside* the 0.5–45 Hz band, so no filter reaches it.

Two by-products worth stating: going 81 → 273 features *raised* corpus recoverability by
**+0.065** (more features make the shortcut easier, not harder), and a 600-segment pilot gave
the **opposite sign** (−0.124) purely because 273 features are starved at that sample size —
a caution about trusting probes measured on subsets.

`scripts/extract_all3_features.py` (the v1 equivalent) remains unrun and is redundant: every
script recomputes the v1 81 inline.

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

## 3.6 The DANN result — the new experiment

Gradient reversal between the shared backbone and a corpus classifier: 2 backbones ×
α ∈ {0, 0.1, 0.3, 1.0}, where **α=0 disables the adversary and is the control**. All eight
configs ran under the same 5-fold subject-grouped protocol as every other all3 model.

**Primary criterion — CogLoad within-corpus > 0.6886: MISSED by all 8.** Best was
EEGTransformer α=1.0 at **0.5040**. Best pooled accuracy 0.5808, still −0.046 under the
0.6267 shortcut.

The domain-head accuracy is the diagnostic for *why* — it measures whether the adversary
actually removed corpus information from the representation:

| backbone | α=0.0 | α=0.1 | α=0.3 | α=1.0 |
|---|---|---|---|---|
| TSception | 0.5134 | 0.5212 | 0.4896 | **0.5938** |
| EEGTransformer | 0.4962 | 0.4557 | 0.4763 | 0.5071 |

TSception is **non-monotonic and worst at full strength**: corpus was *most* recoverable
exactly where task accuracy was *lowest* (0.4521). That signature is training instability, not
domain removal. EEGTransformer behaved better but traded ~0.04 of domain accuracy for ~0.04 of
task accuracy — a 1:1 exchange that gains nothing.

**Identified cause, if asked:** α is fixed at construction. Standard DANN anneals α from 0→1
so the adversary joins gradually; here it fights the task objective from the first step. That
is a known DANN failure mode and the first thing to change. So the honest claim is that *this
configuration* fails, not that domain-adversarial training cannot work here.

## 4. Status of every piece — all now run

| item | status |
|---|---|
| `all3_diagnostic.py` | ✅ re-run, all metrics backfilled |
| `all3_models.py` | ✅ 15 configs, none clear the shortcut |
| `all3_deep.py` | ✅ 7 deep models, none clear the shortcut |
| `all3_dann.py` | ✅ 8 configs, none clear CogLoad |
| `all3_tracks.py` | ✅ `Results/all3/TRACKS.md` generated |
| `extract_features_v2.py` | ✅ `features_v2/all3/`, 15,728 × 273 |
| `extract_all3_features.py` (v1) | ❌ still unrun — redundant, v1 features are recomputed inline |

**26 models total. None beats 0.6267. None beats CogLoad's 0.6886.**

Still not started: the GPU-box plan — foundation-model fine-tuning (CBraMod/LaBraM/BIOT), a
nested-CV hyperparameter sweep, and SSL on the 39,429 unused AMIGOS segments.

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
three classes. A classifier using only device identity scores 0.6267; our best of 26 models
scores 0.6256, and removing the device signature drops performance to chance. In two of three
corpora, no model beats a constant predictor. The same data split into two binary tasks gives
0.77 and 0.76 AUC. This session added a domain-adversarial model designed to attack that
confound directly, along with the reporting infrastructure needed to tell a real result apart
from device detection — both written and ready to run.*
