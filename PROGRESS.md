# PROGRESS — every approach taken on this project

Whole-project record: what was built, why it was built, what it achieved, and what it measured.
Chronological by track. Last updated **2026-09-06**.

Scope note: this file is the *narrative*. The authoritative numbers live in
`Results/SUMMARY.md` (binary tasks), `Results/all3/TRACKS.md` (3-class), `Results/OSF_Park/`
(4-class cohort), and `arch_metd/REPORT.md` (protocol-gated build). `arch_metd/PROGRESS.md`
is the same kind of log scoped to that one track and goes deeper than the section here.

---

## 0. The one-paragraph summary

The stated goal was 4-class **Stress / Anxiety / Healthy / Depression** from EEG. What the
data supports is **two independent binary tasks** — depression (MDD vs HC, resting state) and
stress (task vs baseline) — because no public corpus on one amplifier contains more than two
of the target classes. Nine tracks were built over five months. The recurring finding, arrived
at five separate ways, is that **corpus/device identity is the dominant signal** in any pooled
multi-class framing, and that **explicit spectral/covariance features beat every learned
representation tried** on this data. The best honest numbers are depression **0.7716** subject
AUC (SVM on 81 handcrafted features, MODMA) and stress **0.7560** within-subject AUC
(Riemannian tangent space, SAM40). Cross-device deployment was measured directly and fails at
~chance.

---

## 1. Timeline of tracks

| # | Track | Dates | State | Headline |
|---|---|---|---|---|
| 1 | Legacy MODMA 3-channel | May–Jun 2026 | superseded | 0.37–0.52 acc, at/below chance |
| 2 | Legacy MODMA 128-channel (TSception/EEGNet) | Jun 2026 | superseded | 74.87% TSception — later shown split-inflated |
| 3 | SAM40 notebooks | Jun–Jul 2026 | disconnected | separate machine, random split, wrong fs |
| 4 | Unified 4-corpus pipeline + 11 models (`run_models.py`) | Jul 2026 | complete | `report.md`; deep models look strong — they aren't |
| 5 | Phase 1 diagnostics + Phase 2 models | Jul–Aug 2026 | complete | the honest numbers: 0.7716 / 0.7560 |
| 6 | `all3/` 3-class track (36 models) | Jul–Aug 2026 | complete | 0.6267 shortcut is the bar; only hierarchical clears it (0.6820) |
| 7 | `OSF_Park_model/` 4-class cohort | Aug 2026 | complete | anxiety finally exists; EEG adds ~nothing over demographics |
| 8 | `distilled/` + `DL_Without_Features/` | Aug 2026 | partial | channel-density inverted U; raw-DL harness built, real runs pending |
| 9 | `trial_cc_pre/` → `arch_metd/` protocol-gated build | Sep 2026 | complete | mechanism works; the outcome it protects has no signal |

---

## 2. Track 1–3 — the legacy attempts (May–Jul 2026)

### 2.1 MODMA 3-channel (`legacy/modma Model new/`)

**Done:** SVM, Random Forest, 1D CNN, LSTM on MODMA's 3-channel resting variant at 125 Hz,
cross-validated.

**Why:** cheapest entry point — the 3-channel folder is small and pre-extracted.

**Result:**

| Model | Accuracy | ROC-AUC |
|---|---|---|
| SVM | 0.3747 ±0.067 | 0.4136 |
| Random Forest | 0.4286 ±0.122 | 0.4976 |
| 1D CNN | 0.5201 ±0.066 | 0.5359 |
| LSTM | 0.4084 ±0.048 | 0.5314 |

**Achieved:** established that 3 channels carry nothing usable — every AUC is within noise of
0.5. The 3-channel variant was abandoned and never re-entered the pipeline. (Older warnings in
`plans/plan.md` about zero-padded channels describe *this* abandoned variant, not current code.)

### 2.2 MODMA 128-channel, TSception vs EEGNet (`legacy/MODMA Model/`, `Results/MODMA/`)

**Done:** full preprocessing (0.5–50 Hz bandpass, ICA/ASR artifact removal, 4 s windows with
50% overlap), then 5-fold group-stratified CV of TSception and EEGNet, 50 epochs, early
stopping patience 10, Adam lr=1e-3.

**Why:** if 3 channels fail, use all 128 and a purpose-built EEG architecture.

**Result:**

| Model | Mean accuracy | Std | Notes |
|---|---|---|---|
| **TSception** | **74.87%** | ±4.89% | best fold 81.75% |
| EEGNet | 60.83% | ±13.51% | collapsed to 36.27% in fold 4 |

**Achieved:** the project's first "working" depression model, and its first lesson in variance
— EEGNet failing to converge in one fold out of five is not a hyperparameter problem, it is
what a 53-subject dataset does to a deep model. **These figures are superseded.** Track 5
re-ran depression under out-of-fold subject-grouped CV and TSception fell to 0.7385 subject
AUC, EEGNet to 0.6250. Kept for provenance only.

### 2.3 SAM40 notebooks (`notebooks/SAM40/`)

**Done:** exploration, feature extraction and classical ML notebooks for a separate 3-class
SAM40 experiment, run on a different machine.

**Why:** parallel exploration of the stress corpus before the unified pipeline existed.

**Achieved:** nothing that carried forward. It is **disconnected** from `src/` and
`run_models.py`, hardcodes `fs = 128`, and uses a **random rather than subject-grouped split**
— so its numbers are leakage-inflated by construction. Retained as history; do not quote.

---

## 3. Track 4 — the unified 4-corpus pipeline (Jul 2026)

### 3.1 What was built

`scripts/combine_datasets.py` — one script that projects four corpora onto a common
representation, and the single source of truth for how they become one training set.

**The common representation:** 6 frontal channels (Fp1, Fp2, F7, F3, F4, F8) at 128 Hz, 4 s
windows (512 samples), 50% overlap.

| Corpus | ID | Native | Fs | Note |
|---|---|---|---|---|
| MODMA (128-ch resting) | 0 | real EGI E22/E9/E33/E24/E124/E122 | 250 | no padding |
| SAM40 | 1 | 6 native from a 32-ch 10-20 cap | 128 (⚠ code assumes 256) | see §3.5 |
| Cognitive Load | 2 | 6 native from OpenBCI 8-ch | 250 | raw counts ~10⁴, unfiltered |
| AMIGOS | 3 | AF3→Fp1, AF4→Fp2 approximated | 128 | only approximation in the pipeline |

**Why 6 channels:** it is the lowest common denominator across four corpora. That choice was
later measured (§6.4, §8.1) and found *not* to be the binding constraint.

**Normalization:** global per segment — median/MAD clipped at ±5 MAD, then z-scored across the
whole `(6, 512)` window rather than per channel. Deliberate: preserves inter-channel amplitude
ratios, which frontal alpha asymmetry depends on. No bandpass or notch in the combine pipeline.

### 3.2 The `-1` masking convention

Every segment carries *both* `y_depression` and `y_stress`; `-1` means "this corpus cannot
supervise this task."

| Corpus | y_depression | y_stress |
|---|---|---|
| MODMA | MDD=1 / HC=0 | −1 |
| SAM40 | −1 | relax=0 / task=1 |
| Cognitive Load | −1 | `natural-*`=0 / task=1 |
| AMIGOS | −1 | arousal ≥ 5.0 → 1 |

`masked_binary_crossentropy` zeroes samples where `y_true == -1`. **Why:** it lets one shared
backbone train on all four corpora at once. **Consequence:** no sample carries both labels, so
the multi-task trunk gets its depression gradient exclusively from MODMA and its stress
gradient exclusively from the other three.

### 3.3 Features and models

**81 handcrafted features** = 30 relative band powers + 30 differential entropy (δθαβγ × 6 ch)
+ 3 frontal alpha asymmetry pairs + 6 spectral entropy + 6 RMS + 6 STD. Layout is **band-major,
not channel-major**.

**11 models** wired into `run_models.py`: SVM / RandomForest / XGBoost on the 81 features; 1D
CNN, LSTM, CNN-LSTM, EEGNet, TSception, EEGTransformer on raw EEG; plus a shared-backbone
multi-task model and a DANN with a real gradient-reversal layer.

**Protocol:** subject-disjoint 70/10/20 train/val/test, stratified per dataset, with assertions
enforcing zero subject overlap and that validation spans every training dataset.

### 3.4 Three threshold-tuning bugs, each found by ROC-AUC

All three produced plausible-looking numbers and failed silently. Each is load-bearing and
must not be reverted.

| Bug | Symptom | Fix |
|---|---|---|
| Tuned threshold on **F1** | F1 ignores true negatives, so "predict everything positive" wins under imbalance. Four depression models collapsed to threshold 0.05 with recall exactly 1.0 — including one with ROC-AUC 1.0000 whose perfect ranking was discarded | maximize **balanced accuracy** |
| Degenerate cutoffs allowed | balanced accuracy scores all-positive *and* all-negative at exactly 0.5. A 1D CNN pinned at threshold 0.98 scoring 0.5000 despite test AUC 0.9460 | skip any threshold predicting a single class |
| Threshold transferred as an **absolute value** | models emit different probability ranges per subject, so a valid val cutoff put every test subject on one side | transfer the cutoff's **rank position** (transductive on scores, never labels) |

### 3.5 Label bugs found and fixed — 45.6% of stress labels were wrong

- **AMIGOS** was reading `labels_selfassessment` index **1 (valence)** as if it were arousal.
  Index 0 is arousal per `SelfAsessment.xlsx`.
- **Cognitive Load** hardcoded every file stress-positive, including its 30 `natural-*` rest
  recordings.

**What this explains:** before the fix, stress AUC was 0.62–0.67 — the models were decoding
**valence** above chance. After the fix, pooled stress AUC dropped to ~0.40, which triggered
Phase 1. So the models were never broken; they were being graded against the wrong target.

### 3.6 The SAM40 sampling-rate defect — found, quantified, deliberately NOT patched

`combine_datasets.py:171` does `data[:, ::2]` with the comment "Since native is 256 Hz". The
SAM 40 paper states **"EEG was sampled at 128 SPS"**, and 128 × 25 s = 3200 = exactly the array
length on disk. The 256 Hz claim traces only to a *generated* metadata report, not a source.

Measured on 40 `Relax_*` trials, same channels:

| | spectral peak | theta share | beta share |
|---|---|---|---|
| raw @ true 128 Hz | 3.00 Hz | 0.138 | 0.329 |
| after `::2`, read as 128 Hz | **5.50 Hz** | 0.181 | 0.204 |

A 1.83× frequency shift (theory says 2×). **Every SAM40 segment in `combined_data/` and
`all3/` has its spectral content at roughly double its true frequency** — 2,400 of 15,728 all3
segments. All 30 relative band powers, all 30 DE features, spectral entropy and the per-band
Riemannian covariances are wrong for that corpus, and it plausibly *inflates* the corpus probe.

**Decision: not patched.** Fixing it invalidates `combined_data/`, `all3/`, all 36 all3
results and both phase directories. That is a full rebuild, not a drive-by edit. `distilled/`
and `arch_metd/` read raw at the correct rate so new work is not built on the error.

Two related loader traps found at the same time, both silent:

- MODMA filenames are inconsistent (`02010002rest 20150416 1017..mat`, `02010008_rest ....mat`,
  `02010021 20150805 1730.mat.mat`). Splitting on `rest` or `_` matches **37 of 53** subjects;
  the leading zero-padded digit run matches **53/53**.
- MODMA `.mat` key detection can match `Impedances_0`, also `(129, 1)`. It survives only
  because dict insertion order favours the data key.

### 3.7 What Track 4 achieved

`report.md` — 11 models × 2 tasks with majority baselines, tuned thresholds, per-corpus stress
AUC breakdown and subject-level depression metrics. Deep models appeared to dominate
depression (CNN-LSTM 0.9891, EEGNet 0.9015 subject AUC).

**That appearance was false**, and Track 5 was built to test it. `report.md` is a single
70/10/20 split; its deep-model figures run **~0.21 AUC high**. Do not quote it.

---

## 4. Track 5 — Phase 1 diagnostics and Phase 2 models (Jul–Aug 2026)

**Why:** pooled stress AUC of ~0.40 after the label fix, and deep depression numbers that
looked too good. Both needed out-of-fold subject-grouped CV rather than one split.

### 4.1 Phase 1 — four diagnostics

| Script | Question | Answer |
|---|---|---|
| `phase1_sam40_ceiling.py` | is stress decodable at all on one clean corpus? | **yes** — SVM 0.6663 ±0.054 within-subject AUC. Every model crosses from below chance to above it |
| `phase1_corpus_ablation.py` | what does each corpus contribute? | SAM40+CogLoad 0.7231 ±0.051 is the best feature-based config; AMIGOS adds +0.001 |
| `phase1_amigos_relabel.py` | is AMIGOS's arousal label the problem? | **yes** — 0.4875 alone (chance). A per-subject median split instead of the global `arousal ≥ 5.0` cut changes it by **−0.004**, so it is *not* rating bias — self-reported arousal from video watching simply is not recoverable from 6 frontal channels at 4 s resolution |
| `phase1_depression_cv.py` | are the deep depression numbers real? | **no** — see below |

**The pooled-stress explanation:** the Phase 0 pooled AUC of 0.40 is explained entirely by a
test set that was ~78% AMIGOS. The models were learning; they were graded against an
unpredictable target.

**Practical rules produced:** report stress on SAM40/CogLoad, **never pooled with AMIGOS**;
keep AMIGOS as training data and for label-free SSL (it is harmless, +0.001), drop it as an
evaluation target; do not spend more effort on relabelling AMIGOS.

**Depression under honest CV** (5-fold subject-grouped over all 53 MODMA subjects, subject-level AUC):

| SVM | TSception | 1D CNN | CNN-LSTM | RF / EEGNet | XGBoost |
|---|---|---|---|---|---|
| **0.7716** | 0.7385 | 0.6897 | 0.6667 | 0.6250 | 0.5546 |

Against `report.md`'s single 10-subject split, deep models lose **0.21 AUC on average**
(CNN-LSTM 0.9891 → 0.6571, EEGNet 0.9015 → 0.6239) while classical models lose **0.001**. The
deep models' dominance was an artifact of which 10 subjects landed in the test split.

### 4.2 Phase 2 — four modelling attempts, each with a control

| Model | What it is | Result |
|---|---|---|
| **M1 Riemannian** | per-band covariance → re-centre → tangent space (105 dims) → logistic regression | **0.7560 ±0.050** stress AUC, **+0.090** over SVM — the best stress model. Loses on depression (0.6595) |
| **M2 CBraMod probe** | frozen foundation-model probe, 211/211 tensors loaded | 0.5716, **+0.076 over its own random-init control** — pretraining genuinely works, and still lands **0.18 below** a 105-dim linear model |
| **M3 self-supervised** | masked reconstruction pretraining | 0.5549, **−0.050 versus its own random-init control**. Masked reconstruction optimises waveform fidelity, not discriminative structure |
| **M5 channel density** | MODMA sweep 6/16/32/64 channels | peaks at **16 ch (0.6825)**, declines to 0.6580 at 64. Tangent dims grow O(ch²) to 10,400 against 53 subjects |

**Achieved — three independent lines of evidence that deep architectures lose on this data.**
Consequence: `src/dsffn/`, `src/hamsnet/`, `src/codae/`, `src/st2adn/` are fully written and
**deliberately never run**. Wiring them up would be building experiments the evidence says not
to run.

**The honest bound on that conclusion:** every deep result here comes from a single hardcoded
config (`EPOCHS=25, BATCH=128, LR=5e-4`, patience 5). Nothing was ever tuned. "Deep models lose
here" is an *untuned* finding — well supported, not airtight. Likewise CBraMod's 0.5716 is a
frozen probe on stress, not a fine-tune.

### 4.3 One methodological result worth not re-deriving

**Re-centering group choice is task-dependent and fails silently.** Stress → group by subject
(the contrast is within-subject). Depression → group **globally** (every segment of a subject
shares one label, so the signal *is* between-subject). Per-subject centering on depression
whitens each subject's mean to identity and deletes the label outright — measured as AUC
exactly 0.5000, Brier exactly 0.2500.

---

## 5. Track 6 — the `all3/` 3-class track (Jul–Aug 2026)

**Why:** to answer directly "can one model predict healthy / stress / depression?"

**Built:** `scripts/build_all3.py` derives `datasets/all3/` from `combined_data/` — segments
unchanged, labels only remapped. 15,728 segments (healthy 6,632 · stress 5,497 · depression
3,599), 108 subjects, zero NaNs. AMIGOS excluded from labels (chance arousal), kept as
`X_unlabelled.npy` for SSL. Labels verified against raw source: 53/53 MODMA subjects match the
clinical spreadsheet; CogLoad/SAM40 segment counts are exactly predicted by raw file lengths.

### 5.1 The confound, established first

MODMA holds only {healthy, depression}; SAM40/CogLoad only {healthy, stress}. Each used a
different amplifier, so **corpus identity is ~92% recoverable from the 81 features alone**.

| Baseline | 3-class accuracy |
|---|---|
| Chance | 0.3333 |
| Majority class | 0.4217 |
| **Corpus ID only — zero EEG understanding** | **0.6267** |

**0.6267 is the bar, not chance.** Also verified as a boolean identity, not a correlation:
`depression` ≡ `corpus == MODMA` — all 3,599 depression segments are MODMA, all 5,497 stress
segments are not. So a stress-vs-depression classifier is a MODMA-detector with a clinical
label; it scores ~0.97 and cannot be evaluated within any corpus.

### 5.2 What was tried against it — 36 models

| Approach | Script | Best result | vs 0.6267 |
|---|---|---|---|
| Tier-1 classical (SVM/LogReg/RF/XGB on 81) | `all3_models.py` | SVM 0.6256 | −0.001 |
| Early / late / gated fusion | `all3_models.py`, `all3_deep.py` | 0.5989 | −0.028 (all three fusions land 0.03–0.04 *below* the best single view) |
| Six deep architectures at `nb_classes=3` | `all3_deep.py` | LSTM 0.5631 | −0.064 (all six 0.10–0.20 below) |
| Riemannian, device signature **removed** | `all3_diagnostic.py` | **0.3255** | −0.301 — chance |
| DANN, 2 backbones × 4 alphas | `all3_dann.py` | 0.5808 | −0.046, all 8 configs failed |
| **Hierarchical two-stage** | `all3_hierarchical.py` | **0.6820** | **+0.055** |

**No FLAT 3-class model beats 0.6267 in 26 attempts.** Removing the device signature drops to
chance — the signature was worth **+0.18**, and without it nothing remains. Within-corpus
accuracy is *below* a constant predictor in 2 of 3 corpora: forcing one model to do both tasks
makes it worse at each.

### 5.3 The DANN attempt and why it failed

Gradient reversal between the shared backbone and a corpus classifier, α ∈ {0, 0.1, 0.3, 1.0}
with α=0 as the adversary-off control. Best pooled 0.5808; **all 8 configs missed the CogLoad
target** (best 0.5040 against a 0.6886 majority).

The domain-head diagnostic says the method itself did not work:

| backbone | α=0.0 | α=0.1 | α=0.3 | α=1.0 |
|---|---|---|---|---|
| TSception | 0.5134 | 0.5212 | 0.4896 | **0.5938** |
| EEGTransformer | 0.4962 | 0.4557 | 0.4763 | 0.5071 |

TSception is **non-monotonic and worst at full strength** — domain recoverability is highest
exactly where task accuracy was lowest (0.4521). That is training instability, not domain
removal. EEGTransformer traded ~0.04 of domain accuracy for ~0.04 of task accuracy: nothing
gained. **Prime suspect:** α is fixed, never ramped — standard DANN anneals 0→1, here the
adversary fights the task objective from step one.

### 5.4 The hierarchical model — the only thing that beats the shortcut

    stage 1  predict corpus              (0.9703 with v2 features)
    stage 2  MODMA → healthy vs depression | SAM40, CogLoad → healthy vs stress
    guard    a specialist is used only if it beats its corpus majority in INNER CV

**Why it works:** one flat decision boundary cannot suit all three corpora. CogLoad is 34% of
the data and every flat model scores below its majority there, cancelling MODMA's genuine +0.11.

| model | 3-class | vs 0.6267 |
|---|---|---|
| best flat (SVM on 81) | 0.6256 | −0.001 |
| hierarchical, v1 router | 0.6486 | +0.022 |
| **hierarchical, v2 router + v1 specialists** | **0.6820** | **+0.055** |
| hierarchical, oracle router (upper bound) | 0.6963 | +0.070 |

Four things it established, each measured:

- **Router and specialists want opposite feature sets.** v2's 273 features route at 0.9703 vs
  v1's 0.9161, but collapse the MODMA specialist to 0.5280 (below its own majority) — 273 dims
  against 53 subjects. v2 for stage 1, v1 for stage 2. Worth **+0.033**.
- **The guard is load-bearing and must be decided on inner CV.** MODMA's specialist wins 5/5
  folds; SAM40 and CogLoad never beat guessing and fall back. Deciding on outer-fold accuracy
  would be test-set selection.
- **Tuning C/gamma is worthless here** (+0.0002).
- **Letting the model choose its feature view made things worse** (−0.025): 3-fold inner CV
  over ~40 subjects cannot tell v1 from PLV, so selection thrashes per fold.

Report the corpus-free row (0.6820), never the oracle row — the oracle reads the true corpus
label at inference. Remaining headroom is small and known: 0.0143 of routing loss, and CogLoad's
best specialist (PLV, 0.6770) still 0.012 under its 0.6886 bar.

### 5.5 Features v2 — 273 columns, and the line-noise hypothesis it refuted

**Why built:** relative band power divides by total power over the **full 0–64 Hz** while the
bands span only 0.5–45 Hz, so mains hum lands in the unused gap and inflates the denominator
of all 30 relative-power features. There is no notch filter anywhere upstream. Measured, the
gap holds **84% of MODMA's total power** (sharp 50 Hz spike), 35% of SAM40's, 23% of CogLoad's.
That looked like a strong candidate for the 92% corpus recoverability.

**Built:** `features_v2.py` — 50 Hz notch + zero-phase 0.5–45 Hz bandpass (`sosfiltfilt`
throughout, since v1's `lfilter` is causal and phase-distorting), plus four families v1 lacks:
Hjorth, 1/f aperiodic slope, permutation entropy, coherence and PLV. 273 columns, materialised
to `features_v2/all3/` (15,728 × 273, 168 s).

**Result — the hypothesis is REFUTED:**

| features | preprocessing | corpus probe |
|---|---|---|
| v1's 81 | none (0–64 Hz denominator) | 0.9161 |
| **the same 81** | **notch + bandpass + matched denominator** | **0.9057** |
| all 273 v2 | notch + bandpass | 0.9703 |

Discarding 23–84% of total spectral power changed device recoverability by **−0.010**. The
amplifier signature lives *inside* the 0.5–45 Hz band; no filter reaches it.

Two consequences that generalise:

- **Adding features makes the shortcut easier.** 81 → 273 raised the probe **+0.065**. Do not
  expect a richer feature set to weaken the confound; it strengthens it.
- **Never trust a probe measured on a data subset when dimensionality also changes.** A
  600-segment pilot gave **−0.124**, the opposite sign, purely because 273 features are starved
  at that sample size while 81 are not.

One filter-design result worth not rediscovering: a Butterworth bandpass cornered at 45 Hz
**cannot** reject 50 Hz mains — they are 11% apart. Order 4 leaves 9.2% of MODMA's power above
45 Hz; order 8 plus an explicit `iirnotch` at 50 Hz leaves 0.01%.

### 5.6 The accuracy ceiling — every lever tried

| framing | accuracy |
|---|---|
| 3-class, hierarchical (best of 36) | 0.6820 |
| 3-class, flat SVM | 0.6256 |
| 3-class, 62 s recordings (30-window vote) | 0.6495 |
| 3-class, whole recording per subject | 0.6792 |
| healthy vs ill (binary) | 0.6174 |
| **binary healthy-vs-depressed, MODMA only** | **0.6949** ← best honest single number |
| binary healthy-vs-stressed, SAM40 only | 0.6367 |
| on unseen hardware | ~0.32 |

Exhausted without success: more features (v2's 273), deep models, transformers, adversarial
training, hierarchical routing, hyperparameter tuning, per-corpus feature selection, longer
windows, subject-level aggregation, binary reduction. **Aggregation specifically does not work**
— errors are correlated within subject, so averaging over a longer recording buys ~0.02 then
declines.

**The one well-posed relabelling is healthy vs ill**, because both classes appear in all three
corpora (MODMA 4360/3599, SAM40 600/1800, CogLoad 1672/3697) — the only framing that learns
from within-device contrast. It beats majority in MODMA (+0.082) and SAM40 (+0.002), fails on
CogLoad (−0.149), and scores 0.6174 pooled. Better posed, not more accurate.

### 5.7 The 0.94 that is not real

A **random** (non-subject-grouped) split reports **0.9393** on the 3-class task; the same model
under `GroupKFold` on subjects reports **0.6256**. Pure leakage — 4 s windows at 50% overlap
put overlapping windows from the same person in both train and test.

This matters because it is a one-word change (`StratifiedKFold` for `GroupKFold`) and it is the
obvious way to "hit" an accuracy target. Much of the published EEG literature reports numbers
this way. **Every split in this repo must be grouped on `subjects`. If a number jumps above
~0.85, suspect the split before believing the model.**

---

## 6. Deployment — and the cross-device failure that bounds everything

**Built:** `scripts/train_deployable.py` (three approaches → `models/*.joblib`),
`scripts/train_screening.py` (three protocol-selected binary models), `predict.py` (runs any
of them against a raw `(6, n_samples)` recording at any rate — resamples to 128 Hz, windows at
4 s / 50% overlap, aggregates per-window votes).

### 6.1 Screening results (5-fold subject-grouped, threshold tuned by nested inner CV)

| model | accuracy | majority | AUC | note |
|---|---|---|---|---|
| `resting_healthy_vs_depressed` | 0.6542 | 0.5478 | **0.7536** | the only one that works |
| `screen_healthy_vs_ill` | 0.5997 | 0.5783 | 0.6527 | +0.02, marginal |
| `task_healthy_vs_stressed` | 0.6590 | 0.7076 | 0.5982 | **below majority** |

`resting_healthy_vs_depressed` has sensitivity 0.44 / specificity 0.83 — it misses 56% of
depressed subjects because the threshold was tuned for raw accuracy. For screening use, move
the threshold; AUC 0.7536 bounds the achievable trade-offs.

### 6.2 Cross-device transfer FAILS — measured directly

Train the stress model on SAM40+CogLoad, apply to MODMA — a device it never saw — and it labels
**90% of MODMA as "stressed"** at mean confidence **0.839**, *higher* than on its own corpora
(0.712). MODMA contains **zero** stress segments, so all of those are confident false positives.
Composed 3-class accuracy collapses to **0.3995**, near chance.

That agrees with the independent estimate from removing the device signature (0.3255). Two
different routes, same answer: **on an unseen amplifier these models are worth about chance.**

**Consequences:** every CV number in this repo is an upper bound, not a deployment estimate —
all results are subject-disjoint but never device-disjoint. `models/A_binary_pair.joblib` must
not be deployed; it is retained only because its failure *is* this measurement. A genuine fix
needs training data spanning multiple devices per class, or leave-one-corpus-out as the
headline protocol. **Track 9 was built to do exactly that.**

---

## 7. Track 7 — `OSF_Park_model/`, the 4-class cohort (Aug 2026)

**Why:** the stated goal needs an **anxiety** class, and no corpus in the main pipeline has one.
`datasets/OSF_park/EEG.machinelearing_data_BRMH.csv` (Brain Research & Metrics Hospital) has
all four classes **in one cohort with one feature schema** — so unlike `all3/`, class is *not*
confounded with recording corpus. That is the only reason to care about it.

| class | source | n |
|---|---|---|
| healthy | `main.disorder == 'Healthy control'` | 93 |
| depression | `specific.disorder == 'Depressive disorder'` | 199 |
| anxiety | `main.disorder == 'Anxiety disorder'` (panic + social) | 107 |
| stress | `main.disorder == 'Trauma and stress related disorder'` | 128 |
| | | **527** |

**Three things about this data that mislead:** it is not raw EEG (114 absolute band powers +
1026 coherences, precomputed — nothing from the `(6, 512)` pipeline applies); **the bar is
demographics, not chance**; and **there are no subject IDs**, so `StratifiedKFold` is grouping
by necessity, not by proof. A **recording-year probe** runs alongside every result to check for
this cohort's own version of the corpus shortcut — it came back clean (0.2179 vs 0.2443
majority over 8 year classes).

Two data-integrity fixes, both silent if skipped: `Unnamed: 122` is a trailing-comma artifact
and the only all-NaN column (a naive `dropna()` deletes all 945 rows); rows `no. == 206` and
`no. == 944` are dropped because six rows share byte-identical band-power blocks.

### 7.1 Round 1 — five model families, 4-class and binary

Models: classical (SVM/RF/LogReg/GradientBoosting on 1140 flat features) · MLP · Channel-
Transformer (19 channel tokens × 6 bands) · CohTransformer (+ per-head attention bias from the
6 coherence matrices) · **GatedHybrid** (branch A = channel graph · branch B = 1026 COH
projection · branch C = demographics, combined by a learned per-sample softmax gate, branch C
ablatable).

**4-class** (n=528, 15 folds; chance 0.2500, majority 0.3769, **demographics 0.3939**):

| model | accuracy | vs demo |
|---|---|---|
| SVM [demographics only] | 0.4299 | +0.0360 |
| SVM [EEG] | 0.4261 | +0.0322 |
| SVM [EEG+demographics] | 0.4261 | +0.0322 |
| GatedHybrid [no demographics] | 0.3447 | −0.0492 |
| GatedHybrid [+demographics] | 0.3239 | −0.0701 |
| CohTransformer | 0.2917 | −0.1023 |
| ChannelTransformer | 0.2879 | −0.1060 |

**Binary** (healthy vs depression, n=293; majority 0.6792, **demographics 0.7576**):

| model | accuracy | vs demo |
|---|---|---|
| SVM [demographics only] | 0.7952 | +0.0376 |
| SVM [EEG+demographics] | 0.7645 | +0.0069 |
| SVM [EEG] | 0.7577 | +0.0001 |
| ChannelTransformer | 0.5154 | −0.2423 |

**Achieved:** the same lesson as Track 5, on completely different data — **the transformers
lose badly, and EEG barely moves the needle over age/sex/IQ/education.** Permutation importance
on the demographics-only SVM puts the entire effect in the DEMO block (+0.2976 balanced-acc
drop, every EEG channel exactly 0.0).

### 7.2 Plan 01 — corrected features and a frozen test set

**Why:** absolute band powers are gain-sensitive (skull thickness, impedance, contact).
Rebuilt with CLR (gain-invariant by construction — scale a channel's six band powers by any
constant and CLR output is unchanged), log-abs, FAA, TBR, Fisher-z, and graph reduction.

**Also built:** `make_holdout.py` freezes a dev/test split to `splits/*.json` and refuses to
overwrite; `cv.dev_folds` **asserts no test index reaches a CV fold**; `final_test.py` appends
to `Results/OSF_Park/FINAL.md` and refuses to run without `--yes`. The test set is scored
**once per plan**, enforced by code rather than discipline.

**4-class dev** (n=421; demographics bar 0.5004 balanced acc):

| model | balanced acc | macro-AUC | vs demo |
|---|---|---|---|
| EEG+DEMO [elasticnet] | **0.5178** | 0.7290 | +0.0174 |
| DEMO [l2] | 0.5004 | 0.7026 | 0.0000 |
| EEG [l2] | 0.3843 | 0.6284 | −0.1161 |
| GRAPH [elasticnet] | 0.3305 | 0.5886 | −0.1698 |
| SPECTRAL [elasticnet] | 0.3155 | 0.5708 | −0.1849 |

**Binary dev** (n=233; demographics bar 0.7598): DEMO [l2] wins at 0.7598; EEG+DEMO 0.7440.

One graph-construction result: **MST-union thresholding, not proportional.** Plain top-*d*
thresholding fragments 94% of subject-band graphs at `d=0.20` and 22% even at `d=0.50`, making
characteristic path length infinite. MST-union gives 0 fragmented at `d=0.20`.

### 7.3 Plan 02 — demographic residualization

**Why:** healthy controls are the youngest and highest-IQ group; the trauma group is oldest and
34% male. Any EEG "signal" could be an age proxy. So: fit a normative model on training-fold
healthy controls only, residualize EEG against it, and measure ΔAUC over demographics alone
with a paired bootstrap.

**Primary analysis is the *no IQ* arm** — IQ was measured after diagnosis (healthy 116.2 vs
89.6–107.8 for patient groups), so it is partly a consequence of illness and medication: a
mediator, not an independent predictor.

| task | arm | ΔAUC | 95% CI | verdict |
|---|---|---|---|---|
| binary | **no IQ** | **+0.0937** | [+0.0222, +0.1650] | **increment** |
| binary | with IQ | +0.0233 | [−0.0347, +0.0789] | null |
| 4class | no IQ | +0.0242 | [−0.0144, +0.0604] | null |
| 4class | with IQ | −0.0094 | [−0.0455, +0.0262] | null |

**Achieved: the single clearest positive EEG result on this cohort** — residualized EEG adds
+0.094 AUC over demographics for healthy-vs-depression, CI excluding zero. It does **not**
replicate on the 4-class task. Normative R² on held-out healthy controls is ~−0.29 in every
arm, i.e. the normative model itself is weak. Residualizing on age/sex/education does not
remove medication status, symptom severity or time of day — none are in this file.

### 7.4 Plan 03 — hierarchy, stacking, auxiliary data

**Why:** the hierarchical routing that worked on `all3/` (§5.4), applied to a cohort without a
device confound.

| approach | balanced acc | vs demo bar (0.5004) |
|---|---|---|
| **stacked (spectral + graph + demo)** | **0.5031** | +0.003 |
| hierarchical (composed) | 0.3491 | −0.151 |

Stage 1 (healthy vs ill) reaches 0.7171 balanced acc against an 0.8242 majority; **stage 2
(mood/anxiety/trauma) is 0.3489 against its own 0.4582 majority** and its guard passes only
10/15 folds. So the hierarchy fails at exactly the step the 4-class goal needs: **separating
depression from anxiety from trauma is where the signal isn't.** Meta-weights in the stack
confirm it — demo 0.501, graph 0.259, spectral 0.240.

Adding the other 417 in-cohort ill subjects moved stage 2 by **+0.0083**. They add no healthy
controls, so they cannot relieve the stage-1 bottleneck.

Per-disorder recall shows where the 4-class problem actually lives:

| specific disorder | pooled as | n | recall |
|---|---|---|---|
| Healthy control | healthy | 74 | 0.622 |
| Acute stress disorder | stress | 27 | 0.481 |
| Depressive disorder | depression | 159 | 0.396 |
| Panic disorder | anxiety | 45 | 0.178 |
| Social anxiety disorder | anxiety | 41 | 0.146 |
| Adjustment disorder | stress | 33 | 0.121 |
| PTSD | stress | 42 | 0.119 |

### 7.5 The frozen test set — scored once

| date | task | model | dev acc | **test acc** | test majority | AUC | n |
|---|---|---|---|---|---|---|---|
| 2026-08-21 | hc_vs_scz | demographics [GB] | 0.8932 | **0.767 ±0.064** | 0.558 | 0.919 | 43 |

A side experiment (healthy vs schizophrenia) confirmed the pattern one more time: demographics
alone reach 0.870 dev accuracy against a 0.550 majority; **age alone** gets 0.775, IQ alone
0.738, education alone 0.728, sex alone 0.550 (nothing). EEG-only top-20 features: 0.683.
Adding EEG to demographics: 0.831 — i.e. **worse than demographics alone**.

---

## 8. Track 8 — channel density and raw-EEG DL (Aug 2026)

### 8.1 `distilled/` — is the 6-channel bottleneck the problem?

**Why:** `combine_datasets.py` discards 123 of MODMA's 129 channels and 26 of SAM40's 32
*before any model sees the data*. The plan was to read each corpus at native resolution and
distill a high-density teacher into a deployable 6-channel student. **The premise was gated
first**, which turned out to be the right call.

`00_validate_premise.py` — MODMA depression, EEGNet, 5-fold subject-grouped, identical
windows/folds/seed with only `n_chans` varying and the frontal 6 nested inside every larger set:

| chans | subject AUC | segment AUC | time |
|---|---|---|---|
| 6 (baseline) | 0.6667 | 0.6641 | 3.7 min |
| **32** | **0.7371** | 0.7195 | 14.2 min |
| 128 | **0.6279** ← worse than 6 | 0.6172 | 46.0 min |

Gain +0.0704 vs the required +0.05 → **gate PASSED**, but the *shape* matters more than the
verdict. Performance is an **inverted U**: 32 channels beats 6, full density is worse than 6.
With 53 subjects, 128 channels overfits. This independently reproduces
`Results/phase2/channel_density.json` (peak at 16 ch, decline by 64) under a completely
different model family — **treat it as a property of the dataset, not of either method.**

**The open caveat:** 0.7371 is still **below the classical SVM on 81 handcrafted 6-channel
features (0.7716)**. Density helps *within* the deep family; it has not beaten the classical
baseline. `01_teacher_bakeoff.py` was written to settle whether the classical model also
improves with density — if it peaks at 6 channels, distillation is unjustified for this task.
**It has not been run.** `data.py`, `models.py`, `distill.py`, `evaluate.py` are not written.

This track is also where the SAM40 sampling defect (§3.6) and the MODMA filename/key traps
were found.

Also documented here: **two of four montages are guesses.** MODMA and SAM40 have real electrode
positions; AMIGOS's channel *order* is inferred; **CogLoad has no placement documented anywhere
on disk** and was reverse-engineered — if wrong, spatial interpolation for that corpus is
meaningless. `montages.py` carries `assumed: bool` per corpus rather than a comment.

Two untapped assets noted: MODMA's **audio modality was never downloaded** (so the published
97.53% EEG+audio result is not reproducible here), and **6.2 GB of MODMA 128-channel ERP task
data** sits unused — the largest untapped asset on disk.

### 8.2 `DL_Without_Features/` — raw-EEG DL with a corpus probe

**Why two things nothing else had done:** (1) braindecode 1.6.1 ships 61 architectures and not
one had been tried — 17 verified to construct and forward at `n_chans=6, n_times=512,
n_outputs=3`; (2) **a corpus probe on the model's own penultimate embedding**. Every other all3
number is graded under subject-grouped CV where corpus is ~97% recoverable *in the test fold
too*, so a model can ride the device signature and the accuracy column will never say so.

Read every row on this grid:

    high acc + high probe  ->  riding the device signature, not real
    high acc + low  probe  ->  the goal
    low  acc + low  probe  ->  invariance removed the signal (the known 0.3255 outcome)

**Four design decisions that are easy to undo by accident:**

- **The KD distillation term is MASKED to the teacher's two classes**, student renormalised
  over the same pair. An unmasked KL would force `P(depression) = 0` on every SAM40/CogLoad
  segment — literally teaching *"not-MODMA ⇒ not depressed"*, i.e. re-deriving the corpus
  shortcut from the loss function. `core.py` asserts that a teacher never seeing class 2 puts
  **exactly zero gradient** on the class-2 logit.
- **Teacher soft targets are out-of-fold** (inner subject-grouped CV inside each outer training
  split). A teacher scoring its own training data emits near-one-hot targets and distillation
  collapses to the hard labels.
- **λ = 0 runs as the method-off control**, per the DANN lesson.
- **`core._inner_split` refuses to hold out a validation subject when a split has fewer than 3.**
  Early stopping normally reserves 1/8 of training subjects; with 1–2 it consumes the whole
  training set and `compute_class_weight` dies on empty input. Found via the per-corpus
  distillation teachers (CogLoad has only 15 subjects). Guard added in the shared function, not
  at the call site.

**State: only `--smoke` results exist** (`Results/DL_Without_Features/_smoke/`) — local torch
is `2.13.0+cpu`. Milestones M0–M6 complete (loader, trainer, benchmark, distiller, docs);
**M7–M9 (the real GPU runs) are pending.** Smoke artifacts are segregated on purpose:
prediction slugs are identical between smoke and real runs, so sharing a directory would
silently overwrite a real 15,728-length array with a 1,128-length one.

**One known measurement flaw, flagged not fixed:** the embedding hook attaches to braindecode's
`final_layer`; ATCNet's is a `ModuleList` that is never called, so it silently falls back to
the **3 output logits**. A 3-d probe reads systematically lower than the 1120-d ShallowFBCSPNet
one, so ATCNet can look corpus-free purely from dimensionality — the same trap as the v2 pilot
that flipped sign at 600 segments. Compare probes only at equal width.

Anchor row: braindecode `EEGNet [segment]` should land near **0.4306**, matching the Keras
EEGNet on identical folds. Above ~0.85 means the split broke.

---

## 9. Track 9 — `trial_cc_pre/` → `arch_metd/`, the protocol-gated build (Sep 2026)

**Why:** §6.2 established that cross-device transfer is what actually matters and had never
been the headline protocol. This track rebuilds the whole thing around it, with five corpora
(MODMA, SAM40, CogLoad + **Mumtaz 2016** and **EEGMAT** downloaded for this purpose), a
pre-registered gate, and leave-one-corpus-out as the primary measurement.

### 9.1 The constraint that shapes the architecture

**Never train a 3-way softmax over {healthy, depression, stress}.** The system emits one label
with three values by combining two learned binary boundaries with **one bit of acquisition
metadata**, which is always known at inference time:

    protocol == "resting" ->  p_dep >= theta_dep  ?  "depression" : "healthy"
    protocol == "task"    ->  p_str >= theta_str  ?  "stress"     : "healthy"

No cross-head argmax — the heads train on disjoint data and their outputs share no scale.
TDBRAIN could not be obtained, so depression external validation is a **reciprocal two-corpus
transfer** (MODMA↔Mumtaz) rather than a clean held-out third; the stress head is unaffected
(SAM40+CogLoad train, EEGMAT fully held out).

### 9.2 `trial_cc_pre/` — the first build

Ran the full pipeline with subject-grouped nested CV, LOSO, cross-corpus transfer and dual
frozen domain probes. Key rows:

| target | AUC |
|---|---|
| MODMA depression (subject) | **0.3434** — inverted |
| SAM40 stress (within-subject) | 0.7546 |
| CogLoad stress (within-subject) | 1.0000 (quantised — 1 baseline vs ~6 load per subject) |
| SAM40→CogLoad transfer, λ=0 | 0.6437 |
| SAM40→CogLoad transfer, λ=1 | 0.4580 |

Frozen probe, λ=0 → λ=1: full-set 0.7629 → 0.4470; anchor 0.7496 → 0.5761; silhouette 0.0133 →
−0.0240. **Manifold gate passed.** Also re-confirmed the leakage result on its own data: leaky
random segment split gives dep AUC 0.7203, the same model subject-grouped gives 0.3874.

**Achieved:** proof the protocol-gated design runs end to end, and the first sign that the
adversary might be *hurting* stress transfer. That sign drove the rebuild.

### 9.3 `arch_metd/` — the rebuild, run twice

Eight phases (P0 contract & registry → P7 transfer + report), each with a pass gate. Two
independent split draws exist: **Run A** (`results_split_A/`, hash `22810e2548e07f90`) and
**Run B** (`results/`, hash `317d4e6107d8e065`). A result that holds on both is real; a result
that moves between them was never a result.

**The finding: the mechanism works, and the outcome it was built to protect has no signal to
protect.** The pre-registered frozen-probe gate — does adversarial training remove domain
information relative to its own method-off control — **passes on both draws**. On the held-out
amplifier that mechanism exists to serve, a model trained on **shuffled labels transfers as
well as either real arm.**

**Cross-corpus transfer** — the only device-disjoint measurements in the entire project:

| direction | Run A | Run B | permuted null (B) | verdict |
|---|---|---|---|---|
| SAM40+CogLoad → EEGMAT (λ=1) | 0.4167 | 0.5000 | **0.5556** | **inert** |
| SAM40+CogLoad → EEGMAT (λ=0 locked) | — | 0.5833 | 0.5556 | inert |
| **Mumtaz → MODMA** | **0.6638** | **0.6667** | 0.5431 | **real, survives both** |
| **MODMA → Mumtaz** | **0.8675** | **0.8438** | 0.3998 | **real**, montage confound attached |
| EEGMAT → SAM40 | 0.3548 | 0.3539 | 0.5925 | inverted, below its own null |
| EEGMAT → CogLoad | 0.7406 | 0.8276 | 0.4803 | real |

**Retracted claim:** an earlier version reported that the adversary *destroyed* stress transfer
(McNemar p = 0.0225 on split A). It did not survive split B (p = 0.5811), and the
transfer-level permutation control — run for the first time with split B — shows why: **there
was no signal to destroy.** Two arbitrary draws plus a label-permutation null are what
separated a finding from a fluctuation. Neither existed before they were asked for.

**Within-corpus CV, both draws:**

| metric | A main | B main | B λ=0 locked | B random backbone | B permuted |
|---|---|---|---|---|---|
| depression / MODMA [subject] | 0.4784 | 0.5862 | 0.5575 | 0.5445 | 0.5934 |
| depression / Mumtaz [subject] | 0.9375 | 0.9364 | 0.9138 | 0.8912 | 0.5787 |
| stress / SAM40 [trial] | 0.5884 | 0.5717 | 0.6002 | **0.6390** | 0.5053 |
| stress / CogLoad [within-subject] | 0.8690 | 0.8405 | 0.9048 | **0.9381** | 0.5992 |
| stress / CogLoad [recording] | 0.7906 | 0.7332 | 0.7348 | **0.8415** | 0.5131 |

Two things to read off this table. **MODMA moved 0.108 between two valid draws** — larger than
most effects under discussion; it is *underpowered*, not "at chance". And **a never-trained
random backbone matches or beats the trained one on every stress metric, on both draws** —
33,287 of the model's 34,487 parameters are in the heads; the convolutional trunk is 1,200.

**The gate and what it does and does not say:**

| | main (λ=1) | λ=0 locked | λ=0 | random |
|---|---|---|---|---|
| anchor probe (chance 0.250) | **0.4019** | 0.4580 | 0.4613 | 0.4669 |
| full-set probe (majority 0.4943) | 0.5575 | 0.5928 | 0.5928 | 0.5998 |

**PASS on both draws** — A by 0.0010 (2/5 folds), B by 0.0094 (3/5 folds), per-fold margin
0.0594 ± 0.0356, i.e. the 0.05 bar sits on the mean. **It passes and the representation is not
invariant**: +0.1519 above balanced chance, +0.0632 above the majority domain. Those coexist
because the gate tests *relative* removal against the method-off control, not invariance.
Pre-registration obliges reporting the verdict as specified; it does not oblige letting a
0.0094 margin carry narrative weight.

**§6.5 item 2 gate: FAILED on Run B**, both criteria. Class weighting fixed the stress-head
calibration on split A and did not hold on B — **calibration on a head with no discrimination
is calibration to chance.**

### 9.4 Data findings from this track

- **Mumtaz is confounded with its acquisition setup.** EDF channel count predicts diagnosis at
  **0.8889** (baseline 0.5238): 23 signals is 24 H / 1 MDD, 21 signals is 6 H / 32 MDD. Montage
  alone scores 0.8685 against the model's 0.9418. Within one montage the model still reaches
  0.9107, so it is not reading setup alone — but no adequately sized subgroup holds both classes
  on one montage. **Never quote a Mumtaz figure without this.**
- **CogLoad is 99.9999% DC** — a ~31,000-count pedestal with tiny AC on top. After a 1 Hz
  high-pass the residue is wideband amplifier and EMG noise, leaving **8× EEGMAT's above-45 Hz
  power after gating** (29× before). No filter reaches it; per-corpus filtering would make
  preprocessing itself the signature.
- **Duplicate recordings, three distinct cases** — same bytes under one subject (keep one),
  same bytes across subjects whose recording sets are identical throughout (merge the ids), same
  bytes across subjects who otherwise hold distinct recordings (drop both). A subject kept under
  two ids defeats grouped CV silently.
- Depression N = 114 subjects (MODMA 53 + Mumtaz 61).

### 9.5 Bugs found and fixed in this track

| what | how it would have failed |
|---|---|
| `GroupKFold` tie-breaking is **sklearn-version dependent** | every SAM40 subject has 12 trials, so ties are everywhere; two machines produced splits differing on 17 of 169 subjects running identical code. Replaced with a self-contained greedy assignment, verified identical across sklearn 1.8.0 and 1.9.0 |
| λ=0 confounded the adversary with the early-stopping lock | λ=0 could select epoch 0 where λ=1 could not select before 11. Added `lambda0_locked` |
| adversary ran against a **single domain** | cross-entropy over 5 logits with one class still has a gradient, and the GRL reverses it — the backbone was pushed to make a trivial prediction harder. Now returns zero below two domains |
| checkpoint selected before the adversary was on | λ at selection was 0.20–0.38 in 3/5 folds while `lambda_at_stop` read 0.74–0.82 |
| predictions carried no identity | subject-level aggregation had to reconstruct it from fold indices |
| `load_oof` globbed `preds_fold*.npz` | also matched `_lambda0`, `_randbb`, `_smoke` — would have pooled every mode and reported the average as the model |
| bandpass order 4 could not meet the mains test | a 45 Hz corner is 11% below 50 Hz; MODMA kept a 2.3× mains peak. Order 8 |

### 9.6 Why P8 was not run

P8 is aux heads, the `filtered_data` ablation, and TDBRAIN. None addresses what the results say.
Its one item with a claim here — a CogLoad ordinal head — is a probe of *why* the stress head
failed, and the permutation control says the cross-device stress target is inert, so a better
stress representation has nothing to be measured against. **The work worth doing is the
writeup:** depression transfer as the positive result, stress transfer as a *demonstrated*
null, and the probe/transfer dissociation as the methodological contribution.

---

## 10. Current best numbers

| Task | Model | Metric | Score | Source |
|---|---|---|---|---|
| Depression (MODMA) | **SVM** on 81 handcrafted features | subject AUC | **0.7716** | `phase1/depression_cv` |
| Stress (SAM40) | **Riemannian tangent space** | within-subject AUC | **0.7560 ±0.050** | `phase2/riemannian` |
| 3-class (all3) | **Hierarchical v2-router / v1-specialists** | accuracy | **0.6820** (bar 0.6267) | `all3/hierarchical_v2_v1` |
| 4-class (BRMH) | EEG+DEMO elasticnet | balanced acc | 0.5178 (demo bar 0.5004) | `OSF_Park/plan01_4class` |
| Binary (BRMH) | demographics + residualized EEG, no IQ | ΔAUC | **+0.0937** [+0.022, +0.165] | `OSF_Park/plan02_binary` |
| Cross-device (METD) | Mumtaz → MODMA, both draws | subject acc | 0.6638 / 0.6667 (null 0.5431) | `arch_metd/REPORT.md` |
| Cross-device stress | SAM40+CogLoad → EEGMAT | — | **inert** (below its permutation null) | `arch_metd/REPORT.md` |

---

## 11. What the project has actually established

1. **The stated 4-class goal is not supported by the main pipeline's data.** No corpus holds
   more than two of the target classes, so a pooled multi-class model learns amplifier identity.
   Measured five ways: corpus-ID shortcut 0.6267 · corpus probe 0.9161 · signature removed
   0.3255 · unseen amplifier 0.3995 · DANN failing 8/8.
2. **Explicit spectral/covariance features beat every learned representation tried**, with
   controls: Riemannian +0.090 over SVM on stress; CBraMod pretraining works (+0.076 over its own
   control) and still loses by 0.18 to a 105-dim linear model; SSL pretraining *hurts* (−0.050).
   Same conclusion independently on the BRMH cohort, where both transformers lose by 0.10–0.24.
3. **The 6-channel bottleneck is not the limit.** Two model families, two protocols, same
   inverted-U: 16–32 channels beats 6, full density is worse than 6. With ~53–108 subjects,
   capacity gets consumed by noise. The same lesson killed v2's 273 features for the specialists
   and killed per-fold feature-view selection (−0.025).
4. **Evaluation protocol is the biggest single source of error in this domain.** Random split
   0.9393 vs grouped 0.6256 · single split inflates deep depression AUC by 0.21 · F1-tuned
   thresholds, degenerate cutoffs and absolute-value threshold transfer each produced plausible
   wrong numbers silently · one split draw moved MODMA by 0.108. ROC-AUC and permutation nulls
   are what caught these.
5. **Cross-device deployment fails.** Two independent routes agree at ~chance. Every CV number
   in this repo is an upper bound.
6. **On a cohort where the confound does *not* exist (BRMH), EEG still adds little over
   demographics** — and where it does add (+0.094 AUC, healthy vs depression, residualized), it
   does not replicate to 4 classes. Anxiety recall is 0.146–0.178. Separating the illness classes
   from each other is where the signal is missing, not separating ill from healthy.

## 12. Open work

| item | track | note |
|---|---|---|
| `benchmark.py` / `distill.py` real GPU runs | 8 | M7–M9; only smoke results exist |
| `01_teacher_bakeoff.py` | 8 | written, never run — decides whether distillation is justified at all |
| `combine_datasets.py` SAM40 rate fix | 4 | full rebuild attached; invalidates 36+ results |
| Ramped-α DANN | 6 | α is fixed, never annealed — the prime suspect for the DANN failure |
| Nested-CV hyperparameter sweep for deep models | 5 | would make "deep models lose here" airtight rather than untuned |
| MODMA audio modality + 6.2 GB ERP task data | 8 | never downloaded / unused — largest untapped assets |
| PHQ-9 / GAD-7 in `subjects_information_*.xlsx` | 4 | 53 subjects, most likely route to an anxiety target in the main pipeline; currently unused |
| METD writeup | 9 | the recommended next step over any further modelling |
