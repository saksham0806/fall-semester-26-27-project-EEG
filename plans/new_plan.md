# EEG Mental-State Classification — Phased Plan

> Working plan, last updated 2026-07-19.
>
> **Phases 0, 1 and 2 are complete.** Entries below record what was measured, not what was proposed.
> **Phase 3 has been descoped** against those results — see that section for what was dropped and why.
>
> Each phase overturned the plan that preceded it. The original assumed stress was "plateaued at
> ~63%" and needed better architectures; Phase 0 found 45.6% of stress labels were wrong; Phase 1
> found the pooled task was undecodable while SAM40 alone was fine, and that depression had been
> inflated by a 10-subject split; Phase 2 found covariance features beat both a foundation model
> and domain-matched SSL. **Five stated hypotheses were falsified across the three phases** and are
> recorded in place so they are not re-proposed.
>
> **Current best results** — all 5-fold subject-grouped CV:
>
> | Task | Model | Metric | Score |
> |---|---|---|---|
> | Stress (SAM40) | M1 Riemannian tangent space | within-subject AUC | **0.7560 ±0.050** |
> | Depression (MODMA, 53 subj) | SVM on 81 handcrafted features | subject-level AUC | **0.7716** |

## Scope and decisions

- **In scope:** depression (MODMA) and stress (SAM40 + Cognitive Load + AMIGOS) as two masked binary tasks.
- **Anxiety is deferred.** MODMA's unused GAD-7 scores (53 subjects) and the public DASPS corpus are the eventual route to the stated 4-class goal.
- Datasets are combined on the **common 6-channel frontal montage**; all four corpora carry these electrodes natively (AMIGOS AF3→Fp1 / AF4→Fp2 is the only approximation). No redesign needed — the "zero-padded channels" warning in the old `plan.md` describes an abandoned 3-channel MODMA variant.
- PyTorch may be added alongside Keras.
- Labels are left as-is **except** for outright transcription errors, which were fixed in Phase 0.

---

## Phase 0 — Fix the defects ✅ COMPLETE (8 of 8)

| # | Item | Status | Evidence |
|---|---|---|---|
| 0.1 | AMIGOS arousal index: `trial_labels[0,1]` → `[0,0]` | ✅ | `combine_datasets.py:258` |
| 0.2 | Cognitive Load `natural-*` files labelled baseline (`y_str=0`) | ✅ | `combine_datasets.py:206` |
| 0.3 | Rebuild `combined_data/`, regenerate feature caches | ✅ | caches rewritten; `verify_combined_data.py` passes |
| 0.4 | Subject-disjoint 70/10/20 train/val/test split | ✅ | asserts in `main()`; 88/11/24 subjects, zero overlap |
| 0.5 | Majority-class baseline + balanced accuracy in every report row | ✅ | columns present in `report.md` |
| 0.6 | Validation-tuned decision threshold | ✅ | quantile transfer; no model pinned at bal-acc 0.5000 |
| 0.7 | Group-aware `GridSearchCV` (`GroupKFold` by subject) | ✅ | `fit_and_tune_model` |
| 0.8 | Re-baseline all 11 configurations, regenerate `report.md` | ✅ | `PIPELINE RUN COMPLETE`, fresh mtime |

### Phase 0 exit criteria — met?

The original criterion was *"a trustworthy baseline — the stress number should exceed 0.7071, and if it does not, that is itself the finding."*

- ✅ **Data foundation is correct and verified.** 45.6% of stress labels were wrong and are fixed.
- ✅ **Evaluation no longer leaks subjects and no longer validates on a single corpus.**
- ✅ **The baseline is trustworthy in the sense that matters** — it now says clearly that one task works and the other does not. Stress did *not* clear 0.7071; per the criterion above, that is the finding, not a failure.
- ✅ **Threshold-dependent metrics are usable.** Item 0.6 resolved by transferring the cutoff as a *quantile* of the score distribution rather than as an absolute value (`score_with_tuned_threshold`). Acceptance check: no model reports balanced accuracy of exactly 0.5000. TSception went 0.5000 → 0.7841, CNN-LSTM 0.5000 → 0.6840, EEGNet 0.5000 → 0.6667.

**Phase 0 is closed.** One caveat on 0.6: the quantile transfer is *transductive* — it reads the test score distribution to locate the cutoff, though never the test labels. This is standard prior correction and vastly better than a broken absolute cutoff, but it assumes test prevalence ≈ validation prevalence (40.0% vs 39.9% here, so safe). For depression the fully correct framing is subject-level classification, since every segment of a subject shares one label — that is Phase 1.6.

### What the label fixes changed

**45.6% of all stress labels were wrong** — 21,532 of 47,198.

| Corpus | Segments | Flipped | Note |
|---|---|---|---|
| MODMA | 7,959 | 0 | depression labels byte-identical |
| SAM40 | 2,400 | 0 | already correct |
| Cognitive Load | 5,369 | 1,672 (31.1%) | recovered a baseline class that did not previously exist |
| AMIGOS | 39,429 | 19,860 (50.4%) | was labelled by valence, not arousal |

AMIGOS index 0 = arousal was verified against `SelfAsessment.xlsx` across 80 subject/trial pairs (80-to-3 margin). Arousal and valence correlate at only **−0.108** in this data, so the old labels were a near-orthogonal variable, not an approximation.

Split verification: 88/11/24 subjects, zero overlap between any pair, validation set spans **all four datasets** (previously it was 100% AMIGOS because Keras `validation_split` takes a contiguous tail of dataset-ordered arrays).

### Results — read by ROC-AUC, not accuracy

ROC-AUC from two consecutive full runs on **identical data and identical code** — the deep models are simply retrained, since no TensorFlow seed is set.

| Model | Depression AUC (run 3 → 4) | Stress AUC (run 3 → 4) |
|---|---|---|
| CNN-LSTM | 0.9891 → 0.8234 | 0.3683 → 0.3587 |
| LSTM | 0.9433 → 0.8691 | 0.3597 → 0.3868 |
| 1D CNN | 0.8750 → 0.9576 | 0.3402 → 0.3565 |
| EEGNet | 0.8421 → 0.9015 | 0.3857 → 0.5002 |
| EEGTransformer | 0.7366 → 0.9245 | 0.3679 → 0.3434 |
| TSception | 0.7969 → 0.8535 | 0.3999 → 0.4413 |
| Multi-task branch | 0.8366 → 0.6889 | 0.5727 → 0.4391 |
| DANN | — | 0.3826 → 0.4660 |
| SVM *(deterministic)* | 0.8080 → 0.8080 | 0.4030 → 0.4030 |
| Random Forest *(det.)* | 0.5664 → 0.5664 | 0.4796 → 0.4796 |
| XGBoost *(det.)* | 0.5441 → 0.5441 | 0.5051 → 0.5051 |
| *majority baseline* | *0.6000* | *0.8268* |

> ⚠️ **New finding — single-run numbers are not reportable.** Depression AUC swings by up to **±0.19** between identical runs (EEGTransformer +0.188, CNN-LSTM −0.166, multi-task −0.148). Classical models are deterministic and move 0.0000, confirming the cause is training stochasticity interacting with a **10-subject test set**, not any remaining pipeline defect.
>
> This means AUC is stable *with respect to threshold choice* (which is what caught the three tuner bugs) but **not** stable with respect to retraining. Do not quote any single depression figure — including CNN-LSTM's 0.9891, which did not reproduce. This makes Phase 1.3 (5-fold subject-grouped CV, reported mean ± std) a prerequisite for any publishable depression claim, not an optional robustness check.

**Stress has no decodable signal — and unlike depression, this result *is* stable.** Reproduced across four runs spanning three different threshold algorithms and two independent retrainings:

- Mean stress AUC: **0.401** (run 3), **0.419** (run 4). Every run well below 0.5.
- The best stress AUC in either run is **0.5051**, from deterministic XGBoost. **No deep model has ever exceeded chance.**
- Balanced accuracy at or below chance everywhere; all eleven stress rows fall below their 0.8268 majority baseline in every run.

The contrast is itself evidence: the depression numbers move by ±0.19 between runs while the stress conclusion does not budge. A task with real signal shows run-to-run variance around a high mean; stress shows consistency around an *inverted* one.

The systematic *inversion* (AUC clustered below 0.5, not scattered around it) means the EEG pattern predicting high self-reported arousal in training subjects predicts **low** arousal in held-out subjects. That is the signature of per-subject rating bias, not a weak neural correlate. Before the fix, stress AUC was 0.62–0.67 — the models were decoding valence, which frontal alpha asymmetry genuinely carries.

**Depression has real signal.** Six models exceed 0.79 AUC and the deep architectures cleanly separate MDD from HC on held-out subjects; classical trees are near chance. Depression labels were never touched, so this came purely from the evaluation fixes.

Standing caveat: the depression test set is **10 subjects (6 HC, 4 MDD)**. CNN-LSTM's 0.9891 means it ranked essentially all MDD subjects above all HC subjects — real, but a 10-subject result. Do not report it without that context; Phase 1.3 replaces it with a 5-fold estimate.

---

## Threshold selection — how it works and why (resolved)

`tune_threshold` maximises **balanced accuracy over non-degenerate cutoffs**; `score_with_tuned_threshold` then transfers that cutoff to test as a **quantile**. Three successive failures got it here, each caught by ROC-AUC disagreeing with the reported accuracy:

1. **Maximising F1 → all-positive.** F1 ignores true negatives, so under imbalance "predict everything positive" (scoring `2p/(1+p)`) beats any real cutoff. Four depression models collapsed to threshold 0.05 with recall exactly 1.0.
2. **Unguarded balanced accuracy → all-negative.** Balanced accuracy scores all-positive *and* all-negative at exactly 0.5. Fixed by skipping any cutoff whose prediction is single-class.
3. **Absolute cutoff did not transfer.** Models emit different probability ranges per subject, so a cutoff tuned on 5 validation subjects placed all 10 test subjects on one side — balanced accuracy pinned to 0.5000 against AUC 0.9891. Fixed by transferring the cutoff's *rank position* instead of its raw value.

**Do not revert any of these three.** All are load-bearing and each failure is silent — the pipeline reports plausible-looking numbers throughout.

Remaining limitation: the transfer is transductive (reads test scores, never test labels) and assumes test prevalence ≈ validation prevalence. Phase 1.6's subject-level aggregation is the principled replacement for depression.

---

## Phase 1 — Diagnostics ✅ COMPLETE (1.1–1.6)

Scripts: `scripts/phase1_sam40_ceiling.py`, `phase1_corpus_ablation.py`, `phase1_amigos_relabel.py`, `phase1_depression_cv.py`. Raw output in `Results/phase1/*.json`.

### 1.1 SAM40-only stress ceiling ✅ — **GO**

5-fold subject-grouped CV, 40 subjects, majority baseline 0.7500. *Within-subject AUC* asks whether a person's stressed segments separate from their own relaxed ones — the fair question given SAM40's balanced design (every subject exactly 60 segments, 75% positive).

| Model | balanced acc | AUC pooled | AUC within-subject |
|---|---|---|---|
| **SVM** | 0.5844 ±0.032 | 0.6411 ±0.040 | **0.6663 ±0.054** |
| XGBoost | 0.5567 ±0.026 | 0.6117 ±0.042 | 0.6407 ±0.051 |
| Random Forest | 0.5608 ±0.030 | 0.6079 ±0.037 | 0.6254 ±0.040 |
| 1D CNN | 0.5144 ±0.008 | 0.5209 ±0.019 | 0.5233 ±0.019 |
| EEGNet | 0.5178 ±0.021 | 0.5137 ±0.009 | 0.5150 ±0.011 |
| TSception | 0.5156 ±0.008 | 0.5111 ±0.019 | 0.5111 ±0.018 |

**Stress is decodable on clean data.** Every model moves from *below* chance in the pooled task to *above* it here (SVM +0.238). The Phase 0 inversion is gone, so the pooled task was the problem, not the 6-channel representation.

Two consequences beyond the gate decision:

- **Handcrafted features beat every deep model decisively** — 0.6663 vs 0.5233. The deep architectures sit at chance on ~1,900 training segments per fold. This inverts the project's current emphasis, where six deep architectures exist and classical models are treated as baselines.
- **Fold variance is small** (±0.03–0.05) against depression's ±0.19 single-split swings, because 40 subjects with 8 held out per fold is a far better-powered design.

Honest ceiling: 0.67 AUC / 0.58 balanced accuracy is a **weak** classifier. Stress is detectable, not solved.

### 1.4 Corpus ablation ✅ — **hypothesis refuted**

Test set held fixed at SAM40 held-out subjects; only the training set varies.

| Training data | XGBoost within-AUC | SVM within-AUC |
|---|---|---|
| SAM40 only | 0.6407 | 0.6663 |
| + CogLoad | 0.6601 (+0.019) | **0.7231 (+0.057)** |
| + AMIGOS | 0.6418 (**+0.001**) | 0.7053 (+0.039) |
| + both (= pooled) | 0.6472 (+0.007) | 0.7095 (+0.043) |

**The plan's previous claim that AMIGOS is "actively harmful and should be dropped" is refuted.** Adding all ~38k AMIGOS segments moves XGBoost by +0.0011 — indistinguishable from zero. Adding data *helps*; SAM40-only was partly data-limited.

The correct distinction: **AMIGOS is neutral-to-useful as training data, and undecodable as an evaluation target.** Since the Phase 0 pooled test set was ~78% AMIGOS, that alone explains pooled AUC 0.40. The models were learning; they were being graded against an unpredictable target.

*(SVM rows cap added corpora at 10k segments, so its "+AMIGOS" and "+both" rows have identical n and are not directly comparable to each other; XGBoost is uncapped and is the clean comparison.)*

### 1.2 AMIGOS relabelling ✅ — **hypothesis refuted**

Per-segment arousal recovered by replaying the loader; the global `>=5.0` rule reproduced the stored labels at **100.00%**, validating alignment before any modelling.

| Labelling | pooled AUC | within-subject AUC | positive rate |
|---|---|---|---|
| global `>= 5.0` (current) | 0.4565 | 0.4875 | 78.1% |
| per-subject median split | 0.4793 | 0.4833 | 40.8% |

Removing per-subject rating bias changes within-subject AUC by **−0.004**. Both at chance. The rating-bias explanation for the inverted AUCs is **wrong**; the simpler reading holds — self-reported emotional arousal from video watching is not recoverable from 6 frontal channels at 4-second resolution.

**Decision: keep AMIGOS as training data, drop it as an evaluation target.** Report stress on SAM40 (and CogLoad) only. No relabelling work is warranted.

### 1.3 / 1.6 Depression under subject-grouped CV ✅ — **the depression result was inflated**

5-fold subject-grouped CV over all 53 MODMA subjects, accumulating **out-of-fold** predictions so every subject is predicted exactly once by a model blind to them. Subject-level AUC is the headline: every segment from a subject shares one label, so the question is whether the *subject* is MDD, not whether a 4-second window is.

| Model | segment AUC | fold AUC (mean ± std) | **subject AUC** | subject bal-acc |
|---|---|---|---|---|
| **SVM** | 0.7525 | 0.7374 ±0.082 | **0.7716** | 0.6954 |
| TSception | 0.7356 | 0.7635 ±0.095 | 0.7385 | 0.6954 |
| 1D CNN | 0.6923 | 0.7410 ±0.120 | 0.6897 | 0.6954 |
| CNN-LSTM | 0.6571 | 0.6625 ±0.070 | 0.6667 | 0.6573 |
| Random Forest | 0.6189 | 0.6147 ±0.045 | 0.6250 | 0.6193 |
| EEGNet | 0.6239 | 0.6637 ±0.168 | 0.6250 | 0.6573 |
| XGBoost | 0.5439 | 0.5513 ±0.090 | 0.5546 | 0.5812 |

**Phase 0's depression numbers were single-split artifacts.** Against the Phase 0 10-subject split:

| | mean AUC change under honest CV |
|---|---|
| Deep models (TSception, 1D CNN, CNN-LSTM, EEGNet) | **−0.2068** |
| Classical models (SVM, RF, XGBoost) | **−0.0011** |

CNN-LSTM falls 0.9891 → 0.6571; EEGNet 0.9015 → 0.6239; 1D CNN 0.9576 → 0.6923. The deterministic classical models barely move. The deep models' apparent dominance was entirely an artifact of which 10 subjects landed in the test set — exactly the failure mode the ±0.19 run-to-run swing was warning about.

**Do not quote any Phase 0 depression figure.** The reportable numbers are the subject-level AUCs above.

### 1.5 Per-dataset breakdown ✅ (and 1.6 wired into the main pipeline)

Every Phase 1 experiment reports per-corpus rather than pooled, which is what exposed the AMIGOS/SAM40 divergence. An audit found this was satisfied *only inside the Phase 1 scripts* — `run_models.py` had zero subject-level metrics and no corpus breakdown, so `report.md` still presented a pooled stress AUC of ~0.40 with nothing indicating SAM40 sits at 0.72. That is actively misleading and has been closed:

- `per_dataset_auc()` — stress ROC-AUC split by source corpus, emitted as its own report table with a note that the pooled row is ~78% AMIGOS and should not be read alone.
- `subject_level_metrics()` — aggregates segment posteriors per subject for depression, and returns empty when subjects span both classes (the stress case), so it cannot be misapplied.

Both are unit-tested and wired into all 11 model call sites.

*Dropped:* the dataset-ID probe and label-permutation control. Phase 0's results make the confound question moot for stress — a task at chance cannot be inflated by a shortcut. Leave-one-dataset-out was superseded by the 1.4 ablation, which answers the same question more directly by holding the test set fixed.

---

## What Phase 1 changed

**Headline: the two tasks are comparably difficult, and an SVM on 81 handcrafted features is the best model on both.**

| Task | Best model | Metric | Score |
|---|---|---|---|
| Depression (MODMA, 53 subjects) | SVM | subject-level AUC, out-of-fold | **0.7716** |
| Stress (SAM40 + CogLoad) | SVM | within-subject AUC, 5-fold | **0.7231 ±0.051** |

1. **Depression was not "solved to a reasonable degree."** Honest out-of-fold evaluation puts it at 0.77 subject-level AUC, not the 0.85–0.99 the single split suggested. Deep models lost 0.21 AUC on average; classical models lost nothing.
2. **Stress is real but weak, and only on task-based corpora.** Report on SAM40/CogLoad; never pooled with AMIGOS.
3. **Classical features beat every deep architecture on *both* tasks.** This is now a two-task, seven-model, cross-validated result rather than a one-off — the strongest empirical finding in the project. Six deep architectures are built here and none beats an SVM on band powers, differential entropy and frontal alpha asymmetry.
4. **AMIGOS arousal is not an EEG-decodable target** from this montage. Keep the data for training, drop the label from evaluation.
5. **Three of my stated hypotheses were falsified** by these experiments: that AMIGOS was harmful as training data; that per-subject rating bias explained the inverted AUCs; and that depression was in materially better shape than stress. All three were plausible; all three were wrong. Recorded so they are not re-proposed.

### Implication for the writeup

The defensible contribution is no longer "we built architectures that classify mental states well." It is closer to: *a rigorous subject-independent evaluation of EEG mental-state classification on a 6-channel frontal montage, showing (a) 45.6% of labels in a widely-used pooled setup were wrong, (b) self-reported arousal is not decodable, (c) deep architectures do not outperform handcrafted features at this data scale, and (d) single-split evaluation inflates deep-model AUC by ~0.21.* That is a more useful paper than a leaderboard, and it is supported by the evidence actually in hand.

---

## Phase 2 — New model families

**Sequencing revised after Phase 1.** Stress is salvageable but weak (0.72 within-subject AUC on SAM40+CogLoad), so the stress track is worth continuing — evaluated on task-based corpora only, never on AMIGOS labels.

The bigger update: **on stress, handcrafted features beat every deep model 0.72 vs ~0.52.** Deep architectures are at chance on ~2k–7k training segments. That argues the binding constraint is data volume and representation quality, not architecture capacity — which is precisely the case M1 and M2 address and precisely the case M4 does not. Revised priority:

- **M1 (Riemannian) moves up** — it is a feature-based method in the family that is currently winning, and adds the one thing the 81 features lack: spatial covariance structure.
- **M2 (foundation models) stays high** — pretrained representations directly attack the small-data limitation that is flattening the deep models.
- **M4 (paradigm MoE) is deprioritised** — more capacity on a task where capacity is not the constraint. Revisit only if M1/M2 lift stress materially.
- **M3 (self-supervised pretraining) gains value** — it targets the same data-volume constraint using the ~54k unlabelled segments, and AMIGOS is usable here since SSL needs no labels, which is the one place its data is unambiguously an asset.

### M1. Riemannian filter-bank tangent space ✅ — **new best, 0.7560**

`scripts/phase2_riemannian.py`, `Results/phase2/riemannian.json`. Per band: covariance → whiten by that subject's Riemannian mean → tangent space (5 × 21 = 105 dims) → logistic regression. Same 5-fold subject-grouped CV as Phase 1.

| Config | Riemannian | Phase 1 SVM | Δ |
|---|---|---|---|
| SAM40 only | **0.7560 ±0.050** | 0.6663 | **+0.0897** |
| SAM40 + CogLoad | 0.7077 ±0.028 | 0.7231 | −0.0154 |

Beats the previous overall best (0.7231). Note the reversal: extra corpora *helped* SVM but *hurt* Riemannian — re-centering already removes the domain shift the extra data was compensating for, so CogLoad now only contributes its ~10⁴ raw-count amplitude scale into the covariances. **Report M1 on SAM40 alone.**

`mean_riemann` emits "Convergence not reached" on some subjects; the self-check still asserts re-centered means hit identity at 1e-6, so this is not affecting results. Raise `maxiter` only if it ever does.

### Phase 2 results — consolidated

All evaluated identically: 5-fold subject-grouped CV, within-subject AUC on SAM40 (stress) or subject-level AUC on MODMA (depression). Artifacts in `Results/phase2/`.

| Item | Result | vs best prior |
|---|---|---|
| **M1 Riemannian — stress** | **0.7560 ±0.050** | **+0.090** over SVM 0.6663 |
| M1 Riemannian — depression | 0.6595 | −0.112 vs SVM 0.7716 |
| M2 CBraMod frozen probe | 0.5716 ±0.032 | −0.184 vs M1 |
| M2 — random-init control | 0.4957 ±0.021 | *pretraining gain +0.076* |
| M3 SSL masked reconstruction | 0.5549 ±0.037 | −0.201 vs M1 |
| M3 — random-init control | 0.6051 ±0.034 | *SSL gain **−0.050*** |
| M5 channel density (depression) | 6ch 0.6595 → 16ch **0.6825** → 32ch 0.6753 → 64ch 0.6580 | peaks at 16ch (+0.023), then declines |
| M6 temperature scaling | Brier 0.3252 → 0.2563 | AUC unchanged by construction |
| M4 paradigm MoE | **not built** — see below | — |

**What Phase 2 establishes:**

1. **Covariance structure was the missing ingredient for stress.** M1 is the single largest gain in the project (+0.090) and is now the best stress model at 0.7560. Handcrafted band powers carry no spatial covariance; adding it via the SPD manifold recovers real signal.
2. **Both learned-representation routes lost to it, with controls proving the comparison is fair.** CBraMod's pretraining is genuinely working (+0.076 over its own random init, 211/211 tensors loaded) yet lands 0.18 below M1. Domain-matched SSL did *worse* than random init (−0.050) — masked reconstruction optimises for waveform fidelity, not for the discriminative spectral/covariance structure the task needs.
3. **The 6-channel bottleneck is cheap, and more channels eventually hurt.** On MODMA the sweep peaks at 16 channels (0.6825, +0.023 over 6ch) then declines through 32 (0.6753) to 64 (0.6580) — tangent-space dimension grows as O(ch²), reaching 10,400 features against 53 subjects, so the added channels cost more in variance than they return in signal. Pooling corpora onto 6 frontal electrodes is not what limits performance, and a higher-density model is not worth building.
4. **M4 is retired, not deferred.** Phase 1 showed capacity is not the constraint; M2 and M3 now confirm that added representational machinery loses to a 105-dimensional linear model. A gated ensemble of the deep architectures that already lost is not worth building. Revisit only if some future result shows capacity binding.

**Re-centering group choice is task-dependent and load-bearing** (`recenter()` in `phase2_riemannian.py`): group by *subject* for stress, where the contrast is within-subject; group *globally* for depression, where the label is constant within subject and per-subject centering deletes the signal outright — measured as AUC exactly 0.5000, Brier exactly 0.2500.

### M1 original spec
`src/riemannian/`, new dep `pyriemann`. Per-band covariance (δ/θ/α/β/γ → five 6×6 SPD matrices) → re-centre per subject and per dataset via `pyriemann.transfer.TLCenter` → tangent space (21 dims/band, 105 total) → logistic regression or XGBoost. The strongest known cross-subject EEG baseline; re-centering removes per-domain mean covariance analytically, the principled form of what the DANN attempts adversarially. CPU, minutes to train, fits the existing sklearn path.

### M2. EEG foundation-model transfer
`src/foundation/`, PyTorch + `braindecode`. Fine-tune **CBraMod** (ICLR 2025, pretrained on TUH EEG; weights on HuggingFace, packaged as `braindecode.models.CBraMod`). Frozen backbone + linear probe first to measure headroom cheaply, then LoRA or partial fine-tuning. LaBraM and EEGPT as comparators. Largest expected single gain.

### M3. Self-supervised pretraining on the pooled corpus
`src/ssl/`, Keras. Masked-patch reconstruction or contrastive over all ~54k segments ignoring labels, then fine-tune. Augmentations: channel dropout, time masking, band masking, amplitude scaling, temporal crop. Secondary benefit — the repo currently has **no augmentation** beyond 50%-overlapping windows.

### M4. Paradigm-aware mixture of experts — conditional on Phase 1
`src/moe/`, Keras. Shared encoder, one expert head per stress paradigm (arithmetic, Stroop, mirror-image, emotional arousal), learned gate, per-paradigm metrics. **Only worth building if Phase 1.1/1.2 restore a usable stress target.**

### M5. Higher-density variant (one experiment)
The 6-channel intersection is the right common representation, but SAM40 has 32 channels and MODMA 128 natively — and the best depression result in the repo (TSception 74.87% ± 4.89%, `Results/MODMA/results.md`) came from the unused 128-channel data. Train per-dataset models at native density to measure what the intersection costs. Large gap justifies a channel-set model later; small gap retires the question.

### M6. Subject-level aggregation and calibration
A wrapper over all of the above: aggregate segment posteriors to a subject decision, fit temperature scaling on validation.

---

## Phase 3 — Experiment harness — **DESCOPED**

### Findings from the audit

| Defect | Detail |
|---|---|
| `requirements.txt` unusable | **7** hard imports undeclared: `tensorflow`, `xgboost`, `seaborn`, `joblib`, `pyriemann`, `torch`, `braindecode`. Nothing in the repo runs from a clean install. |
| Bare imports | `src/codae/model.py:4` and `src/dsffn/model.py:7` do `from proposed_model...`, resolving only via `run_models.py`'s `sys.path.insert`. Not importable as normal modules. |
| Results fragmented | 9 artifacts in 2 formats — `Results/phase1/*.json` (4), `Results/phase2/*.json` (4), plus `report.md` markdown tables on a different protocol. No single comparison. |

### What was dropped, and why

The original spec — `run_experiments.py` with `--model / --task / --protocol / --seed`, registering all 11 existing models plus the four orphaned pipelines plus M1–M5 — **was written before Phase 1 and 2 ran**. Its stated motivation was that the harness "cannot reach the four orphaned pipelines" (`src/dsffn/`, `src/hamsnet/`, `src/codae/`, `src/st2adn/`).

Those four are all deep architectures. There are now **three independent lines of evidence** that this family loses on this data:

1. Phase 1: all six existing deep models at chance on stress (~0.52) and below SVM on depression.
2. Phase 2 M2: a foundation model with genuinely working pretraining (+0.076 over its own random-init control) lands 0.18 below a 105-dimensional linear model.
3. Phase 2 M3: domain-matched SSL scored **worse than random init** (−0.050).

Building a registry whose purpose is to run a fifth, sixth, seventh and eighth variant of the losing family is scaffolding for experiments the evidence says not to run. **Dropped:** the CLI registry, the protocol switch, and wiring the orphaned pipelines. Revisit only if some future result shows deep architectures becoming competitive.

### What was retained ✅ ALL DONE

- **3.1 `requirements.txt` fixed.** All 7 undeclared hard imports added, with an inline note that `torch`/`torchaudio` must come from the CPU index (`--index-url https://download.pytorch.org/whl/cpu`) because `braindecode` hard-imports `torchaudio` and a mismatched build fails at import.
- **3.2 Bare imports fixed.** Added `src/__init__.py` and switched `src/dsffn/model.py` and `src/codae/model.py` to relative imports. Verified both now import standalone (`import src.dsffn.model`) *and* that `run_models.py`'s `sys.path` hack still works.
- **3.3 `scripts/consolidate_results.py`** → `Results/SUMMARY.md`. Collects all 9 artifacts into one ranked table per task, 18 stress rows and 12 depression rows, each tagged with its source file and config.

`report.md` is **deliberately excluded** from that table: it is a single 70/10/20 split, a different protocol, and its deep-model figures run ~0.21 AUC high. Merging it in is exactly the error that made deep models look competitive.

*Not a harness. A dependency fix, an import fix, and one results table.*

---

## Verification

- **Phase 1.1** succeeds when SAM40-only stress is reported with subject-grouped CV and an explicit majority baseline — a null result here is a valid and publishable outcome, not a failure.
- **Phase 1.3** succeeds when headline numbers are mean ± std over 5 subject-grouped folds and the threshold instability above disappears.
- **Phase 2** ✅ every model evaluated under one protocol (5-fold subject-grouped CV), compared by ROC-AUC against the Phase 1 SAM40 ceiling. Each learned-representation item (M2, M3) carries a **random-init control**, which is what makes "pretraining helped / SSL hurt" a claim rather than an assertion.
- **Leakage audit** ✅ `bandwise_tangent` fits `TangentSpace` on all segments rather than per fold. Measured impact: **zero** — train-only fitting gives 0.7560 ±0.0503, identical to 4 dp, because re-centering maps every group's mean covariance to identity so the reference point is invariant to the fitting subset. Neither step touches a test label. *Not* separately verified for the depression path, which uses global rather than per-subject re-centering.

## Operational notes

- The machine is memory-tight (~15 GB RAM, 6 GB GPU). A full `./run.sh` peaks near 5 GB RSS and pushes the box into swap with a browser and IDE open, slowing training by an order of magnitude. Never run two trainings concurrently.
- Confirm a run finished by the `PIPELINE RUN COMPLETE` line **and** a fresh `report.md` mtime — not by a `ps` check, which can return a false negative under heavy swap.
- Redirect training output to a file rather than piping through `tail`/`head`, which buffer everything until exit and make a ~25-minute run unmonitorable. Use `python -u`, since Python buffers stdout when redirected.
- Phase 1/2 scripts invoke Python **directly**, skipping `run.sh`'s `LD_LIBRARY_PATH` export, so they silently run on **CPU** ("Skipping registering GPU devices"). Harmless for the classical and geometric models that dominate these results; wrap in `run.sh` for anything GPU-heavy.
- `mean_riemann` emits "Convergence not reached" on some subjects. The self-check still asserts re-centered means hit identity at 1e-6, so results are unaffected; raise `maxiter` only if that assertion ever fails.
