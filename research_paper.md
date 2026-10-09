# Protocol-Gated Three-Class EEG Classification of Mental State: Healthy, Depression, and Stress

**A five-month empirical study across six public EEG corpora, with a measured account of why a flat multi-class model cannot work on this data and what does.**

Last revised: 2026-09-06 · Companion record: `PROGRESS.md` · Authoritative numbers: `Results/SUMMARY.md`, `Results/all3/TRACKS.md`, `Results/OSF_Park/`, `arch_metd/REPORT.md`

---

## Abstract

We present a three-class EEG classifier over {healthy, depression, stress} built as a **protocol-gated dual-head architecture**: a shared EEGNet backbone feeds two binary heads, and the emitted label is selected by one bit of acquisition metadata (resting-state versus cognitive task) that is known before the recording begins. Rather than a flat softmax over three classes, the system factorises into two binary decision boundaries, each trained only on data where its contrast genuinely exists.

We motivate this design empirically rather than stylistically. On a pooled three-class dataset of 15,728 segments from 108 subjects across three corpora, **a classifier that knows only the recording device and nothing about EEG achieves 0.6267 accuracy**, against 0.3333 chance and 0.4217 majority. Twenty-six flat three-class models — classical, deep, fused, and adversarial — fail to beat that shortcut; the best flat model (SVM on 81 handcrafted features) scores **0.6256**. Removing the device signature by per-corpus Riemannian re-centring collapses accuracy to **0.3255**, i.e. chance. The only architecture out of thirty-six that beats the shortcut is a **two-stage hierarchical router-plus-specialists model at 0.6820 (+0.055)**, structurally identical to the protocol-gated system except that it *predicts* the routing variable instead of being told it.

The system's strongest defensible result is device-disjoint: **depression transfer from MODMA to Mumtaz at 0.8675 / 0.8438 subject accuracy across two independent split draws** (permutation null 0.3998), spanning two amplifiers and two countries; the reciprocal direction gives 0.6638 / 0.6667 (null 0.5431). Stress within-subject reaches 0.8690 AUC on Cognitive Load. Cross-device *stress* transfer is demonstrated **inert**: on held-out EEGMAT, a label-permutation null (0.5556) sits above both real arms (0.5000, 0.5833).

Alongside the system we report the full negative and methodological record: a random rather than subject-grouped split inflates the same three-class model from 0.6256 to **0.9393**; a single train/test split inflates deep-model depression AUC by **0.21**; three threshold-tuning bugs each produced plausible wrong numbers silently; an untrained random backbone matches or beats the trained model on every stress metric; and on a cohort where the device confound does *not* exist (BRMH, n=527, four classes including anxiety), EEG adds little over age/sex/education, with anxiety recall of 0.146–0.178. We argue that in this domain **the evaluation protocol, not the architecture, is the dominant source of reported performance**.

---

## 1. Introduction

### 1.1 The stated goal and the goal the data supports

The project began with a four-class target — **Stress / Anxiety / Healthy / Depression** from EEG. Five months and nine experimental tracks later, the defensible statement is narrower and sharper:

> No public EEG corpus recorded on a single amplifier contains more than two of the four target classes. Consequently, any model trained on a pool of corpora to separate all classes has a legal, high-accuracy route to the answer that involves no neurophysiology at all: identify the recording device.

That is not a hypothesis in this report. It is measured five independent ways (§4.2), and every one of them agrees.

What the data *does* support is two binary contrasts, each of which exists inside at least one corpus:

- **Depression** — major depressive disorder versus healthy control, resting state (MODMA, Mumtaz 2016).
- **Stress** — cognitive-load or arithmetic task versus relaxed baseline (SAM40, Cognitive Load, EEGMAT).

The contribution of this work is to show that these two binary contrasts, plus one bit of metadata that is *always available at inference time*, compose into a genuine three-class system — and to publish the measurements that make the composition necessary rather than convenient.

### 1.2 Contributions

1. **A protocol-gated three-class classifier** (§2). One EEG recording in, one of {healthy, depression, stress} out, with a calibrated probability. Implemented, run twice under independent split draws, evaluated device-disjoint.
2. **A hierarchical three-class classifier** (§3) that predicts its own routing variable and reports a single three-class accuracy: **0.6820** against a **0.6267** device-only shortcut baseline — the only model of thirty-six to clear it.
3. **A quantified refutation of the flat-softmax approach** (§4): 26 failed flat models, a 0.6267 shortcut, a 0.3255 collapse when the shortcut is removed, and eight failed domain-adversarial configurations.
4. **The only device-disjoint clinical result in the project** (§7.4): MODMA↔Mumtaz depression transfer, holding across two split draws and clear of its permutation null.
5. **A demonstrated null** (§7.4): cross-device stress transfer is inert, established with a transfer-level label-permutation control rather than inferred from a low score.
6. **A methodological catalogue** (§8) of the specific ways EEG classification results become inflated — leakage, single-split variance, threshold tuning, feature-count-dependent probes, and untrained-backbone controls — each with the measurement that caught it.

### 1.3 What this paper is not

It is not a claim of clinical utility. Every within-corpus cross-validated number reported here is **subject-disjoint but not device-disjoint**, and §7.4 shows what happens when that assumption is removed. Every such number should be read as an upper bound on deployment performance, not an estimate of it.

---

## 2. The system: a protocol-gated dual-head classifier

### 2.1 The idea in one sentence

Instead of one model choosing among three answers, there are **two binary models and a switch**. The switch is "what kind of recording is this?" — a fact known before the electrodes go on.

An analogy: a clinical laboratory does not ask one instrument "is this blood or urine, and what is wrong with it?" The sample type is on the requisition form, so the sample is routed to the correct assay. The routing variable is not inferred; it is recorded.

### 2.2 Architecture

```
        EEG recording  (6 channels x 4 seconds = 6 x 512)
                          |
                  shared backbone (EEGNet)
                          |
                256-dimensional embedding
                          |
              +-----------+-----------+
              |                       |
      depression head            stress head
          -> p_dep                  -> p_str
              |                       |
              +-----------+-----------+
                          |
            SWITCH: which protocol was recorded?
                          |
     resting <------------+------------> task
        |                                 |
   p_dep >= theta_dep?              p_str >= theta_str?
    yes -> "depression"              yes -> "stress"
    no  -> "healthy"                 no  -> "healthy"
```

Formally, with `protocol` supplied as metadata:

```
protocol == "resting"  ->  p_dep >= theta_dep  ?  "depression" : "healthy"
protocol == "task"     ->  p_str >= theta_str  ?  "stress"     : "healthy"
```

**There is no cross-head argmax.** The two heads train on disjoint data and their outputs share no common scale, so comparing `p_dep` to `p_str` would be meaningless arithmetic dressed as a decision. This constraint is enforced in code, not by convention.

The backbone is trained with a **masked loss**: every segment carries both `y_depression` and `y_stress`, and the sentinel `-1` marks "this corpus cannot supervise this task". `masked_binary_crossentropy` zeroes the contribution of masked samples, which is what permits one shared trunk to consume five corpora simultaneously. The consequence is structural: no sample carries both labels, so the depression gradient comes exclusively from MODMA and Mumtaz, and the stress gradient exclusively from SAM40 and Cognitive Load.

An **anchor-gated gradient-reversal adversary** sits on the backbone with corpus as the domain label, disabled automatically when fewer than two training domains are present (a single-domain adversary still has a gradient over five logits, and reversing it pushes the backbone to make a trivial one-class prediction harder — a silent pathology fixed in §8.4).

### 2.3 Datasets

| Head | Trained on | Subjects | Contrast learned |
|---|---|---|---|
| Depression | MODMA + Mumtaz 2016 | 53 + 61 = **114** | MDD vs healthy control, resting state |
| Stress | SAM40 + Cognitive Load | 40 + 15 = **55** | task vs relaxed baseline |
| *(held out)* | EEGMAT | 36 | never trained on — reserved for transfer measurement |

All corpora are projected onto a **common representation**: six frontal electrodes (Fp1, Fp2, F7, F3, F4, F8) at 128 Hz. SAM40 is read at its true 128 Hz in this track (see the sampling-rate defect in §8.2, which this track deliberately does not inherit).

### 2.4 Input

Two things enter the system:

1. **The EEG** — a `(6, 512)` matrix: six channels, 512 samples, i.e. four seconds at 128 Hz. Band-passed 1–45 Hz with an order-8 Butterworth, notched at 50 Hz, variance-gated, and normalised.
2. **The protocol tag** — one word, `"resting"` or `"task"`, taken from the recording log rather than from the signal.

A full recording yields many overlapping four-second windows (50% overlap). Each window receives a prediction, and per-subject predictions are aggregated by the **median of the logits**.

### 2.5 Output

One label with three possible values — `healthy`, `depression`, or `stress` — plus the probability behind it. From outside the box the interface is exactly that of a three-class classifier: one EEG in, one of three labels out.

### 2.6 Why the gate is legitimate

Three arguments, in increasing order of force.

**It is standard practice.** A router plus specialists is a conventional multi-class decomposition. One-vs-rest, one-vs-one, and error-correcting output codes all reduce a K-class problem to a set of binary boundaries plus a combination rule, and none of them is described as "not real classification". The only unusual feature here is that the combination rule reads a variable that is *known* rather than *predicted*.

**The variable genuinely is known.** It is impossible to record EEG without knowing whether a resting-state assessment or a Stroop/arithmetic task was administered. The operator entered it on the form. Using it is not leakage; declining to use it would be discarding information that will be present at deployment.

**Inferring it is actively harmful.** Every depression corpus in this study is resting-state and every stress corpus is task-based. A model that tried to infer protocol from the signal would therefore be solving, and would be rewarded for solving, exactly the corpus-identification problem that §4 shows to be the dominant confound. The gate is not a way around the confound; it is a way of refusing to be paid for it.

---

## 3. The second three-way system: hierarchical router plus specialists

`scripts/all3_hierarchical.py` implements the same shape as §2 with one difference: **it predicts the routing variable instead of being told it.** Because it makes a genuine three-way decision end to end, it yields a single three-class accuracy number.

```
stage 1   predict corpus                       (0.9703 accuracy, v2 features)
stage 2   MODMA         -> healthy vs depression
          SAM40/CogLoad -> healthy vs stress    (v1 features)
guard     a specialist is used only if it beats its corpus majority in INNER CV
```

### 3.1 Result

| Model | 3-class accuracy | vs shortcut 0.6267 |
|---|---|---|
| chance | 0.3333 | −0.293 |
| majority class | 0.4217 | −0.205 |
| **corpus ID only — zero EEG understanding** | **0.6267** | — |
| best flat model (SVM on 81 features) | 0.6256 | −0.001 |
| hierarchical, v1 router | 0.6486 | +0.022 |
| **hierarchical, v2 router + v1 specialists** | **0.6820** | **+0.055** |
| hierarchical, oracle router (upper bound) | 0.6963 | +0.070 |

**Report the corpus-free row (0.6820); never the oracle row**, which reads the true corpus label at inference and exists only to bound remaining routing headroom (0.0143).

### 3.2 Four measured design findings

- **The router and the specialists want opposite feature sets.** The 273-column v2 feature set routes at 0.9703 versus v1's 0.9161, but collapses the MODMA specialist to 0.5280 — below its own majority — because 273 dimensions against 53 subjects is starvation. Using v2 for stage 1 and v1 for stage 2 was worth **+0.033**, the single largest gain in this track.
- **The guard is load-bearing and must be decided on inner cross-validation.** MODMA's specialist beats its corpus majority in 5/5 folds and is used; SAM40's and CogLoad's never do and correctly fall back to the majority. Deciding the guard on outer-fold accuracy would be test-set selection dressed as model selection.
- **Hyperparameter tuning is worthless here** (+0.0002 from a C/gamma sweep). The repository default was already at the ceiling.
- **Letting the model choose its own feature view made things worse** (−0.025). Three-fold inner CV over ~40 subjects cannot distinguish v1 from PLV features, so the selection thrashes per fold. With 108 subjects, every added degree of freedom is consumed by noise.

### 3.3 The caveat that must travel with 0.6820

This model runs on `datasets/all3/`, which contains the unpatched SAM40 sampling-rate defect (§8.2). **2,400 of its 15,728 segments carry spectral content at approximately 1.83× their true frequency.** The defect was found, quantified, and deliberately not patched because fixing it invalidates the dataset and thirty-six downstream results simultaneously. It plausibly *inflates* the corpus probe, which would make the 0.6267 bar itself slightly optimistic — the direction of that bias favours the baseline, not the model.

### 3.4 How the two systems relate

| | protocol-gated (§2) | hierarchical (§3) |
|---|---|---|
| routing variable | supplied as metadata | predicted from EEG (0.9703) |
| single 3-class accuracy | not defined by design | **0.6820** |
| evaluation | per-branch, device-disjoint transfer | pooled subject-grouped CV |
| purpose | deployable system | headline three-class number |

They are the same architecture under two assumptions about the routing variable. The hierarchical result demonstrates that the decomposition beats the shortcut baseline even when the router has to earn its input; the protocol-gated design is what one would actually deploy, because at deployment the routing variable is free.

---

## 4. Why not a flat softmax

This section is design justification, not apology: the obvious approach was built first, measured, and found to be dominated by a baseline with no neurophysiological content.

### 4.1 The dataset

`datasets/all3/` — 15,728 segments (healthy 6,632 · stress 5,497 · depression 3,599), 108 subjects, zero NaNs, derived from the combined corpora with segments unchanged and labels only remapped. Labels were verified against raw source: 53/53 MODMA subjects match the clinical spreadsheet, and CogLoad/SAM40 segment counts are exactly predicted by raw file lengths. AMIGOS is excluded from the labelled set (its arousal target is at chance, §7.3) and retained as `X_unlabelled.npy` for self-supervised work.

### 4.2 The confound, established before any modelling

MODMA contains only {healthy, depression}; SAM40 and Cognitive Load contain only {healthy, stress}. Each corpus used a different amplifier. Consequently **corpus identity is ~92% recoverable from the 81 handcrafted features alone** (0.9161; 0.9703 with the 273-column v2 set).

Worse, the mapping is exact rather than statistical: **`depression` is boolean-identical to `corpus == MODMA`.** All 3,599 depression segments are MODMA; all 5,497 stress segments are not. A stress-versus-depression classifier is therefore a MODMA detector wearing a clinical label — it scores ~0.97 and cannot be evaluated within any corpus, because no corpus contains both classes.

The device shortcut was measured five independent ways, all agreeing:

| Route | Measurement |
|---|---|
| corpus-ID-only classifier | **0.6267** three-class accuracy |
| corpus probe on v1 features | **0.9161** |
| Riemannian re-centring removes the signature | accuracy falls to **0.3255** (chance) |
| model applied to an unseen amplifier | **0.3995** composed accuracy |
| domain-adversarial training | fails in **8/8** configurations |

### 4.3 Twenty-six flat models, none of which clears the bar

| Approach | Best result | vs 0.6267 |
|---|---|---|
| Tier-1 classical (SVM / LogReg / RF / XGB on 81 features) | SVM **0.6256** | −0.001 |
| Early / late / gated fusion | 0.5989 | −0.028 |
| Six deep architectures at `nb_classes=3` | LSTM 0.5631 | −0.064 |
| DANN, 2 backbones × 4 α values | 0.5808 | −0.046 |
| Riemannian with device signature **removed** | **0.3255** | −0.301 |
| **Hierarchical two-stage (not flat)** | **0.6820** | **+0.055** |

All three fusion strategies land 0.03–0.04 *below* the best single view. All six deep architectures land 0.10–0.20 below the shortcut. Within-corpus accuracy for flat models is *below a constant predictor* in two of three corpora — forcing one model to solve both tasks makes it worse at each of them, while the same data as two binary tasks yields 0.7716 and 0.7560 AUC.

**The one-sentence answer to "why not a single softmax":** a flat model scores 0.6256, below a classifier that knows only the amplifier and nothing whatsoever about EEG.

### 4.4 The adversarial attempt, and why it failed

Gradient reversal was placed between the shared backbone and a corpus classifier, with α ∈ {0, 0.1, 0.3, 1.0} and α=0 serving as the method-off control. Best pooled accuracy 0.5808; **all eight configurations missed the CogLoad target** (best 0.5040 against a 0.6886 majority).

The domain-head diagnostic shows the method itself did not work:

| Backbone | α=0.0 | α=0.1 | α=0.3 | α=1.0 |
|---|---|---|---|---|
| TSception | 0.5134 | 0.5212 | 0.4896 | **0.5938** |
| EEGTransformer | 0.4962 | 0.4557 | 0.4763 | 0.5071 |

TSception is **non-monotonic and worst at full adversarial strength** — domain recoverability is *highest* exactly where task accuracy was *lowest* (0.4521). That is training instability, not domain removal. EEGTransformer traded roughly 0.04 of domain accuracy for 0.04 of task accuracy: a wash.

The prime suspect is that **α is fixed rather than ramped**. Standard DANN anneals α from 0 to 1 so the adversary joins gradually; here it fights the task objective from the first step. That remains untested and is listed as open work (§11).

Two comparisons that must not be made: the domain head reads a 128-dimensional learned representation, so it cannot be compared to the 0.9161 corpus probe (an SVM on 81 handcrafted features). The only valid reference for each backbone is its own α=0 control.

### 4.5 The accuracy ceiling — every lever tried

| Framing | Accuracy |
|---|---|
| 3-class, hierarchical (best of 36) | 0.6820 |
| 3-class, flat SVM | 0.6256 |
| 3-class, 62 s recordings (30-window vote) | 0.6495 |
| 3-class, whole recording per subject | 0.6792 |
| healthy vs ill (binary) | 0.6174 |
| **binary healthy-vs-depressed, MODMA only** | **0.6949** |
| binary healthy-vs-stressed, SAM40 only | 0.6367 |
| on unseen hardware | ~0.32 |

Exhausted without success: richer features (273 v2 columns), deep models, transformers, adversarial training, hierarchical routing, hyperparameter tuning, per-corpus feature selection, longer windows, subject-level aggregation, and binary reduction. **Aggregation specifically does not help** — errors are correlated within subject, so averaging over a longer recording buys roughly 0.02 and then declines.

The one *well-posed* relabelling is **healthy versus ill**, because both classes appear in all three corpora (MODMA 4360/3599, SAM40 600/1800, CogLoad 1672/3697). It is the only framing in the pooled dataset that learns from within-device contrast rather than device signature. It beats majority in MODMA (+0.082) and SAM40 (+0.002), fails on CogLoad (−0.149), and scores 0.6174 pooled — better posed, not more accurate.

---

## 5. Data and preprocessing

### 5.1 Corpora

| Corpus | Role | Subjects | Native channels | Native Fs | Note |
|---|---|---|---|---|---|
| MODMA (128-ch resting) | depression | 53 | EGI E22/E9/E33/E24/E124/E122 | 250 Hz | real positions, no padding |
| Mumtaz 2016 | depression | 61 | EDF 19–23 signals | 256 Hz | montage confound, §7.4 |
| SAM40 | stress | 40 | 6 of 32 (10–20 cap) | **128 Hz** | sampling defect in legacy track, §8.2 |
| Cognitive Load | stress | 15 | 6 of 8 (OpenBCI) | 250 Hz | raw counts ~10⁴, 99.9999% DC |
| EEGMAT | stress (held out) | 36 | 10–20 | 500 Hz | never trained on |
| AMIGOS | training only | 15 of 40 on disk | AF3→Fp1, AF4→Fp2 approximated | 128 Hz | only channel approximation in the pipeline |
| BRMH (OSF Park) | 4-class cohort | 527 | 19-channel precomputed features | — | separate track, §7.6 |

### 5.2 The common representation

Six frontal channels (Fp1, Fp2, F7, F3, F4, F8) at 128 Hz, four-second windows (512 samples) at 50% overlap. Six channels is the lowest common denominator across the corpora — a choice later measured directly and found **not** to be the binding constraint (§7.7).

Normalisation in the legacy pipeline is **global per segment**: median/MAD clipping at ±5 MAD, then z-scoring across the whole `(6, 512)` window rather than per channel. This is deliberate — per-channel normalisation would destroy the inter-channel amplitude ratios on which frontal alpha asymmetry depends.

### 5.3 Features

**v1 — 81 columns** (the set behind every published result here): 30 relative band powers + 30 differential entropy values (δ/θ/α/β/γ × 6 channels) + 3 frontal alpha asymmetry pairs + 6 spectral entropy + 6 RMS + 6 STD. Layout is **band-major, not channel-major**. Relative PSD and DE are near-redundant — DE is `0.5·ln(var)` of the band-passed signal, i.e. log-absolute power, while the PSD block is linear-relative power — two transforms of one quantity across 60 of 81 columns.

**v2 — 273 columns**: zero-phase 0.5–45 Hz bandpass (`sosfiltfilt`) plus a 50 Hz notch, adding Hjorth parameters, 1/f aperiodic slope, permutation entropy, coherence, and phase-locking value. v2 is the only feature set materialised to disk.

The two versions are **not interchangeable**, and §3.2 shows they are optimal for different stages of the same model.

---

## 6. Evaluation protocol

### 6.1 Subject-grouped cross-validation is mandatory

Every split in this work is grouped on subject identity. This is not a stylistic preference:

> A **random** (non-subject-grouped) split reports **0.9393** on the three-class task. The identical model under `GroupKFold` on subjects reports **0.6256**.

The gap is pure leakage. Segments are four-second windows at 50% overlap, so a random split places overlapping windows *from the same person* on both sides of the split, and the model memorises individuals. The change from grouped to random is a single word in the code, and it is the obvious way to "hit" an accuracy target. Much of the published EEG literature reports numbers this way, which is why 95–99% claims are common in the field and why they are not comparable to anything reported here.

**Operational rule adopted throughout: if a number jumps above ~0.85, suspect the split before believing the model.**

### 6.2 A single split is not an evaluation

Under a single subject-disjoint 70/10/20 split, deep models appeared to dominate depression classification (CNN-LSTM 0.9891, EEGNet 0.9015 subject AUC). Re-run as out-of-fold five-fold subject-grouped CV over all 53 MODMA subjects:

| SVM | TSception | 1D CNN | CNN-LSTM | RF / EEGNet | XGBoost |
|---|---|---|---|---|---|
| **0.7716** | 0.7385 | 0.6897 | 0.6667 | 0.6250 | 0.5546 |

Deep models lose **0.21 AUC on average** (CNN-LSTM 0.9891 → 0.6571; EEGNet 0.9015 → 0.6239) while classical models lose **0.001**. The apparent dominance of deep architectures was an artifact of which ten subjects landed in the test split.

### 6.3 Two split draws, and a permutation null

The protocol-gated track was run twice under two independent split draws (Run A, hash `22810e2548e07f90`; Run B, hash `317d4e6107d8e065`). The rule adopted: **a result that holds on both draws is real; a result that moves between them was never a result.** That check has already retracted a headline claim from this project (§7.4).

Additionally, every transfer measurement carries a **label-permutation null** computed on the same protocol. This is what separates "the model scored low" from "there is nothing to score" — a distinction the project got wrong once before the control existed.

### 6.4 Threshold tuning — three bugs, each caught by ROC-AUC

All three produced plausible-looking numbers and failed silently.

| Bug | Symptom | Fix |
|---|---|---|
| Threshold tuned on **F1** | F1 ignores true negatives, so under imbalance "predict everything positive" wins. Four depression models collapsed to threshold 0.05 with recall exactly 1.0 — including one with ROC-AUC 1.0000 whose perfect ranking was discarded | maximise **balanced accuracy** |
| Degenerate cutoffs permitted | balanced accuracy scores all-positive *and* all-negative at exactly 0.5; a 1D CNN pinned at threshold 0.98 scoring 0.5000 despite test AUC 0.9460 | skip any threshold predicting a single class |
| Threshold transferred as an **absolute value** | models emit different probability ranges per subject, so a valid validation cutoff put every test subject on one side | transfer the cutoff's **rank position** (transductive on scores, never on labels) |

A fourth guard was added later for screening models: **a degenerate-threshold check must test both classes' rates**, not merely that both classes are predicted. A cutoff giving 99% sensitivity and 1% specificity satisfies "predicts both classes" while being useless; the guard requires `min(sensitivity, specificity) ≥ 0.15` on inner folds.

**ROC-AUC is what caught all of these**, because it is threshold-independent.

### 6.5 Baselines are mandatory and are not chance

Three-class accuracy is reported against **0.6267**, never against 0.3333. Screening models are reported against their majority-class rate. The BRMH cohort is reported against **demographics**, not chance (§7.6). A model that does not beat its own correct baseline is reported as a failure regardless of how the absolute number reads.

---

## 7. Results

### 7.1 Three-class classification

| System | Metric | Score | Baseline |
|---|---|---|---|
| **Hierarchical (v2 router / v1 specialists)** | 3-class accuracy | **0.6820** | 0.6267 device-only shortcut |
| Best flat model (SVM on 81) | 3-class accuracy | 0.6256 | 0.6267 |
| Device signature removed (Riemannian re-centre) | 3-class accuracy | 0.3255 | 0.3333 chance |

### 7.2 Depression (binary)

| Setting | Model | Metric | Score |
|---|---|---|---|
| MODMA, 5-fold subject-grouped | SVM on 81 handcrafted features | subject AUC | **0.7716** |
| MODMA, 5-fold subject-grouped | TSception | subject AUC | 0.7385 |
| MODMA, screening deployment | `resting_healthy_vs_depressed` | AUC | 0.7536 (acc 0.6542, majority 0.5478) |
| MODMA, EEGNet at 32 channels | deep, `distilled/` track | subject AUC | 0.7371 |

The screening model has **sensitivity 0.44 / specificity 0.83** — it misses 56% of depressed subjects, because the operating point was tuned for raw accuracy. For screening use the threshold must be moved; AUC 0.7536 bounds the achievable trade-offs.

### 7.3 Stress (binary)

| Setting | Model | Metric | Score |
|---|---|---|---|
| SAM40, within-subject | **Riemannian tangent space** | AUC | **0.7560 ±0.050** |
| SAM40, within-subject | SVM on 81 features | AUC | 0.6663 ±0.054 |
| SAM40 + CogLoad | SVM on 81 features | AUC | 0.7231 ±0.051 |
| CogLoad, within-subject (METD, Run A) | protocol-gated stress head | AUC | 0.8690 |
| **AMIGOS alone** | any | AUC | **0.4875 — chance** |

**The pooled stress task has no signal; stress itself is decodable — the AMIGOS label is the problem.** A pooled stress AUC of ~0.40 was explained entirely by a test set that was ~78% AMIGOS. Every model crosses from below chance in the pooled task to above it on SAM40 alone.

The AMIGOS arousal label was tested exhaustively before being abandoned as a target: replacing the global `arousal ≥ 5.0` cut with a per-subject median split changes performance by **−0.004**, so this is not rating bias. Self-reported arousal from video watching simply is not recoverable from six frontal channels at four-second resolution. AMIGOS is harmless as *training* data (+0.001 on SAM40 performance) and is retained for that purpose and for label-free self-supervision.

**Rule adopted: report stress on SAM40/CogLoad, never pooled with AMIGOS.**

### 7.4 Cross-device transfer — the only device-disjoint measurements in the project

| Direction | Run A | Run B | Permutation null (B) | Verdict |
|---|---|---|---|---|
| **MODMA → Mumtaz** | **0.8675** | **0.8438** | 0.3998 | **real**, montage confound attached |
| **Mumtaz → MODMA** | **0.6638** | **0.6667** | 0.5431 | **real, survives both draws** |
| EEGMAT → CogLoad | 0.7406 | 0.8276 | 0.4803 | real |
| SAM40+CogLoad → EEGMAT (λ=1) | 0.4167 | 0.5000 | **0.5556** | **inert** |
| SAM40+CogLoad → EEGMAT (λ=0 locked) | — | 0.5833 | 0.5556 | inert |
| EEGMAT → SAM40 | 0.3548 | 0.3539 | 0.5925 | inverted, below its own null |

**Depression transfer is the result that survives everything.** MODMA→Mumtaz at 0.8675/0.8438 spans two amplifiers and two countries, holds across two independent split draws, and clears a permutation null of 0.3998 by a wide margin. The reciprocal direction is weaker but also survives both draws and its own null.

**Cross-device stress transfer is demonstrated inert.** On held-out EEGMAT the label-permutation null (0.5556) sits *above* both the adversarial arm (0.5000) and the method-off control (0.5833). There is no signal there to improve.

**A retracted claim, reported as such.** An earlier version of this work reported that the adversary *destroyed* stress transfer (McNemar p = 0.0225 on Run A). That did not survive Run B (p = 0.5811), and the transfer-level permutation control — run for the first time with Run B — explains why: there was no signal to destroy. Two arbitrary draws plus a permutation null are what separated a finding from a fluctuation, and neither existed before they were specifically requested.

**Earlier, independent confirmation of the same bound.** Training the stress model on SAM40+CogLoad and applying it to MODMA — a device it never saw — labels **90% of MODMA as "stressed" at mean confidence 0.839**, higher than its confidence on its own corpora (0.712). MODMA contains zero stress segments, so every one of those is a confident false positive. Composed three-class accuracy collapses to **0.3995**. This agrees with the independent estimate from removing the device signature (0.3255): **on an unseen amplifier these models are worth approximately chance.**

### 7.5 The protocol-gated system, measured per branch

Because the system never makes a three-way decision internally, there is no single three-class accuracy for it. It is measured branch by branch, and the draw matters:

| Branch | Metric | Run A | Run B |
|---|---|---|---|
| Depression, MODMA | subject AUC | 0.4784 | 0.5862 |
| Depression, Mumtaz | subject AUC | 0.9375 | 0.9364 |
| **Depression, MODMA → Mumtaz** | **subject accuracy** | **0.8675** | **0.8438** |
| Depression, Mumtaz → MODMA | subject accuracy | 0.6638 | 0.6667 |
| Stress, SAM40 | trial AUC | 0.5884 | 0.5717 |
| Stress, CogLoad | within-subject AUC | 0.8690 | 0.8405 |
| Stress → EEGMAT (held out) | subject accuracy | 0.4167 | 0.5000 |

**Reading this table honestly.** Four of these seven rows require a caveat, and stating them is part of the result:

- **MODMA within-corpus is unusable.** It moved 0.108 between two equally valid split draws — larger than most effects under discussion in this paper. That is *underpowered*, not "at chance", and it should not be quoted in either direction.
- **Mumtaz's 0.9375 must never travel alone.** EDF channel count predicts diagnosis at **0.8889** (baseline 0.5238): the 23-signal montage holds 24 healthy / 1 MDD, the 21-signal montage 6 healthy / 32 MDD. Montage alone scores 0.8685 against the model's 0.9418. Within a single montage the model still reaches 0.9107, so it is not reading setup alone — but no adequately sized subgroup holds both classes on one montage, so the confound cannot be fully removed from this corpus.
- **CogLoad's 0.8690 is worse than it looks.** A **never-trained random backbone scores 0.9381** on the same task. The trained model is losing to noise.
- **Stress cross-device is inert**, per §7.4.

**The random-backbone control is the most uncomfortable result in this track and is reported in full:**

| Metric | A main | B main | B λ=0 locked | B random backbone | B permuted |
|---|---|---|---|---|---|
| depression / MODMA [subject] | 0.4784 | 0.5862 | 0.5575 | 0.5445 | 0.5934 |
| depression / Mumtaz [subject] | 0.9375 | 0.9364 | 0.9138 | 0.8912 | 0.5787 |
| stress / SAM40 [trial] | 0.5884 | 0.5717 | 0.6002 | **0.6390** | 0.5053 |
| stress / CogLoad [within-subject] | 0.8690 | 0.8405 | 0.9048 | **0.9381** | 0.5992 |
| stress / CogLoad [recording] | 0.7906 | 0.7332 | 0.7348 | **0.8415** | 0.5131 |

A never-trained random backbone matches or beats the trained one on **every stress metric, on both draws**. The explanation is architectural: 33,287 of the model's 34,487 parameters live in the heads, and the convolutional trunk holds only 1,200. Any claim that "the backbone learned the representation" has to clear that control first, and on stress it does not.

**The pre-registered frozen-probe gate:**

| | main (λ=1) | λ=0 locked | λ=0 | random |
|---|---|---|---|---|
| anchor probe (chance 0.250) | **0.4019** | 0.4580 | 0.4613 | 0.4669 |
| full-set probe (majority 0.4943) | 0.5575 | 0.5928 | 0.5928 | 0.5998 |

The gate **passes on both draws** — Run A by 0.0010 (2/5 folds), Run B by 0.0094 (3/5 folds), per-fold margin 0.0594 ± 0.0356, i.e. the 0.05 bar sits on the mean. Crucially, **it passes and the representation is still not invariant**: the anchor probe remains +0.1519 above balanced chance and +0.0632 above the majority domain. Those two facts coexist because the gate tests *relative* removal against the method-off control, not absolute invariance. Pre-registration obliges reporting the verdict as specified; it does not oblige letting a 0.0094 margin carry narrative weight. **Quoting this gate as evidence of invariance would be the exact error the report exists to prevent.**

A second pre-registered gate — stress-head calibration — **failed on Run B on both criteria**. Class weighting fixed calibration on Run A and did not hold on Run B. The interpretation is straightforward: **calibration on a head with no discrimination is calibration to chance.** In-domain separation for the stress head is +0.044 against a 0.10 bar, with out-of-domain scores sitting above its own positives.

### 7.6 The four-class cohort — where anxiety finally exists

The stated four-class goal requires an anxiety class, and no corpus in the main pipeline has one. The BRMH cohort (`EEG.machinelearing_data_BRMH.csv`) has all four classes **in one cohort under one feature schema**, so class is *not* confounded with recording device. That is the sole reason it was used.

| Class | Source field | n |
|---|---|---|
| healthy | `main.disorder == 'Healthy control'` | 93 |
| depression | `specific.disorder == 'Depressive disorder'` | 199 |
| anxiety | `main.disorder == 'Anxiety disorder'` (panic + social) | 107 |
| stress | `main.disorder == 'Trauma and stress related disorder'` | 128 |
| | | **527** |

Three properties of this data mislead if not stated: it is **not raw EEG** (114 absolute band powers + 1026 coherences, precomputed); **the correct baseline is demographics, not chance**; and there are **no subject IDs**, so stratified K-fold is grouping by necessity rather than by proof. A **recording-year probe** was run alongside every result to check for this cohort's own version of the corpus shortcut, and came back clean (0.2179 versus a 0.2443 majority over eight year classes).

**Four-class results** (n=421 dev; demographics bar 0.5004 balanced accuracy):

| Model | Balanced accuracy | Macro-AUC | vs demographics |
|---|---|---|---|
| EEG + DEMO [elasticnet] | **0.5178** | 0.7290 | +0.0174 |
| DEMO [l2] | 0.5004 | 0.7026 | 0.0000 |
| EEG [l2] | 0.3843 | 0.6284 | −0.1161 |
| GRAPH [elasticnet] | 0.3305 | 0.5886 | −0.1698 |
| SPECTRAL [elasticnet] | 0.3155 | 0.5708 | −0.1849 |

**The clearest positive EEG result on this cohort** comes from demographic residualisation: fitting a normative model on training-fold healthy controls only, residualising EEG against it, and measuring ΔAUC over demographics with a paired bootstrap. The primary analysis is the *no IQ* arm, because IQ was measured after diagnosis (healthy 116.2 versus 89.6–107.8 for patient groups) and is therefore a mediator rather than an independent predictor.

| Task | Arm | ΔAUC | 95% CI | Verdict |
|---|---|---|---|---|
| binary (healthy vs depression) | **no IQ** | **+0.0937** | [+0.0222, +0.1650] | **increment** |
| binary | with IQ | +0.0233 | [−0.0347, +0.0789] | null |
| 4-class | no IQ | +0.0242 | [−0.0144, +0.0604] | null |
| 4-class | with IQ | −0.0094 | [−0.0455, +0.0262] | null |

**It does not replicate to four classes.** Per-disorder recall shows exactly where the four-class problem lives:

| Specific disorder | Pooled as | n | Recall |
|---|---|---|---|
| Healthy control | healthy | 74 | 0.622 |
| Acute stress disorder | stress | 27 | 0.481 |
| Depressive disorder | depression | 159 | 0.396 |
| Panic disorder | anxiety | 45 | 0.178 |
| Social anxiety disorder | anxiety | 41 | 0.146 |
| Adjustment disorder | stress | 33 | 0.121 |
| PTSD | stress | 42 | 0.119 |

Hierarchical routing — the architecture that worked on the pooled data — **fails here at exactly the step the four-class goal requires**: stage 1 (healthy vs ill) reaches 0.7171 balanced accuracy against an 0.8242 majority, while stage 2 (mood / anxiety / trauma) reaches 0.3489 against its own 0.4582 majority, with its guard passing only 10/15 folds. Stacking (spectral + graph + demographics) reaches 0.5031 against the 0.5004 demographics bar — meta-weights demo 0.501, graph 0.259, spectral 0.240. Adding the other 417 in-cohort ill subjects moved stage 2 by **+0.0083**; they add no healthy controls and therefore cannot relieve the stage-1 bottleneck.

A side experiment (healthy versus schizophrenia, frozen test set scored once, 2026-08-21) reinforced the pattern: demographics alone reach 0.870 dev accuracy against a 0.550 majority; **age alone** reaches 0.775; EEG-only top-20 features 0.683; and **adding EEG to demographics gives 0.831 — worse than demographics alone**. Test accuracy 0.767 ±0.064 against a 0.558 majority, AUC 0.919, n=43.

**Conclusion from this cohort: separating the illness classes from each other is where the signal is missing, not separating ill from healthy.**

### 7.7 Channel density — the six-channel bottleneck is not the limit

MODMA depression, EEGNet, five-fold subject-grouped, identical windows/folds/seed with only `n_chans` varying and the frontal six nested inside every larger set:

| Channels | Subject AUC | Segment AUC | Time |
|---|---|---|---|
| 6 (baseline) | 0.6667 | 0.6641 | 3.7 min |
| **32** | **0.7371** | 0.7195 | 14.2 min |
| 128 | **0.6279** — worse than 6 | 0.6172 | 46.0 min |

Performance is an **inverted U**: 32 channels beats 6, but full density is worse than 6. With 53 subjects, 128 channels overfits. This independently reproduces the Phase-2 channel-density sweep (peak at 16 channels, 0.6825; decline to 0.6580 by 64) under a completely different model family, so it should be treated as **a property of the dataset, not of either method**.

The open caveat: 0.7371 is still **below the classical SVM on 81 handcrafted six-channel features (0.7716)**. Density helps *within* the deep family; it has not beaten the classical baseline.

### 7.8 Learned representations lose to explicit features — three controlled comparisons

| Model | What it is | Result |
|---|---|---|
| **M1 Riemannian** | per-band covariance → re-centre → tangent space (105 dims) → logistic regression | **0.7560 ±0.050** stress AUC, **+0.090 over SVM** — best stress model. Loses on depression (0.6595) |
| **M2 CBraMod frozen probe** | foundation model, 211/211 tensors loaded | 0.5716 — **+0.076 over its own random-init control** (pretraining genuinely works) yet **0.18 below** a 105-dimensional linear model |
| **M3 self-supervised** | masked reconstruction pretraining | 0.5549 — **−0.050 versus its own random-init control**. Masked reconstruction optimises waveform fidelity, not discriminative structure |

The controls are the point. CBraMod's pretraining *does* transfer; it simply transfers less than explicit covariance structure. SSL pretraining actively *hurts* relative to random initialisation.

**Consequence:** four further deep architectures (`src/dsffn/`, `src/hamsnet/`, `src/codae/`, `src/st2adn/`) are fully written and **deliberately never run**, because wiring them up would be building experiments the evidence says not to run.

**The honest bound on that conclusion:** every deep result here comes from a single hardcoded configuration (`EPOCHS=25, BATCH=128, LR=5e-4`, patience 5). Nothing was tuned. "Deep models lose here" is therefore an *untuned* finding — supported by three independent lines of evidence, but not airtight. Likewise CBraMod's 0.5716 is a frozen probe on stress, not a fine-tune, and frozen probes routinely understate what fine-tuning achieves.

One methodological result worth not re-deriving: **the re-centring group choice is task-dependent and fails silently.** Stress must be grouped by subject (the contrast is within-subject); depression must be grouped globally (every segment of a subject shares one label, so the signal *is* between-subject). Per-subject centring on depression whitens each subject's mean to identity and deletes the label outright — measured as AUC exactly 0.5000 and Brier exactly 0.2500.

### 7.9 Deployment and screening

| Model | Accuracy | Majority | AUC | Note |
|---|---|---|---|---|
| `resting_healthy_vs_depressed` | 0.6542 | 0.5478 | **0.7536** | the only one that works |
| `screen_healthy_vs_ill` | 0.5997 | 0.5783 | 0.6527 | +0.02, marginal |
| `task_healthy_vs_stressed` | 0.6590 | 0.7076 | 0.5982 | **below majority** |

`predict.py` accepts a `(6, n_samples)` recording at any sampling rate, resamples to 128 Hz, windows at 4 s / 50% overlap, and aggregates per-window votes. Input validation is deliberately strict: channel order cannot be inferred from the data, so a transposed or misordered array must fail loudly rather than return a confident wrong answer.

`models/A_binary_pair.joblib` scores 0.3995 and **must not be deployed**; it is retained solely because its failure *is* the cross-device measurement of §7.4.

---

## 8. Negative results and methodological findings

This section exists because in this domain the negative results are more transferable than the positive ones.

### 8.1 Label bugs — 45.6% of stress labels were wrong

- **AMIGOS** was reading `labels_selfassessment` index 1 (**valence**) as if it were arousal; index 0 is arousal per `SelfAsessment.xlsx`.
- **Cognitive Load** hardcoded every file stress-positive, including its 30 `natural-*` rest recordings.

**What this explains:** before the fix, stress AUC was 0.62–0.67 — the models were decoding *valence* above chance, which is consistent with frontal alpha asymmetry being a well-established valence marker rather than an arousal marker. After the fix, pooled stress AUC dropped to ~0.40, which triggered the entire Phase 1 diagnostic effort. **The models were never broken; they were being graded against the wrong target.**

### 8.2 The SAM40 sampling-rate defect — found, quantified, deliberately not patched

The legacy combine pipeline decimates SAM40 with `data[:, ::2]` under the comment "since native is 256 Hz". The SAM 40 paper states **"EEG was sampled at 128 SPS"**, and 128 × 25 s = 3200 = exactly the array length on disk. The 256 Hz claim traces only to a *generated* metadata report, not to a source.

| | Spectral peak | Theta share | Beta share |
|---|---|---|---|
| raw at true 128 Hz | 3.00 Hz | 0.138 | 0.329 |
| after `::2`, read as 128 Hz | **5.50 Hz** | 0.181 | 0.204 |

A 1.83× frequency shift (theory predicts 2×). Every SAM40 segment in the legacy datasets carries its spectral content at roughly double its true frequency — 2,400 of 15,728 three-class segments — so all 30 relative band powers, all 30 DE features, spectral entropy, and the per-band Riemannian covariances are wrong for that corpus.

**Decision: not patched.** Fixing it invalidates the combined dataset, the three-class dataset, all 36 three-class results, and both phase directories simultaneously. That is a full rebuild with a decision attached, not a drive-by edit. The two newest tracks read raw data at the correct rate, so no new work is built on the error. The defect is disclosed with every number derived from the affected data.

### 8.3 The line-noise hypothesis — tested and refuted

Relative band power divides by total power over the full 0–64 Hz while the bands span only 0.5–45 Hz, so mains hum lands in the unused gap and inflates the denominator of all 30 relative-power features. The gap holds **84% of MODMA's total power** (sharp 50 Hz spike), 35% of SAM40's, and 23% of CogLoad's. This looked like a strong candidate for the 92% corpus recoverability.

| Features | Preprocessing | Corpus probe |
|---|---|---|
| v1's 81 | none (0–64 Hz denominator) | 0.9161 |
| **the same 81** | **notch + bandpass + matched denominator** | **0.9057** |
| all 273 v2 | notch + bandpass | 0.9703 |

Discarding 23–84% of total spectral power changed device recoverability by **−0.010**. The amplifier signature lives *inside* the 0.5–45 Hz band; no filter reaches it. Two consequences that generalise beyond this study:

- **Adding features makes the shortcut easier.** Going from 81 to 273 features raised the corpus probe by **+0.065**. A richer feature set strengthens the confound rather than weakening it.
- **Never trust a probe measured on a data subset when dimensionality also changes.** A 600-segment pilot of the same comparison gave **−0.124** — the opposite sign — purely because 273 features are starved at that sample size while 81 are not. The paired full-set control is the only valid comparison.

A filter-design result worth not rediscovering: a Butterworth bandpass cornered at 45 Hz **cannot** reject 50 Hz mains, because they are 11% apart. Order 4 leaves 9.2% of MODMA's power above 45 Hz; order 8 plus an explicit notch leaves 0.01%.

### 8.4 Silent failures in the training and evaluation machinery

| What | How it would have failed |
|---|---|
| `GroupKFold` tie-breaking is **sklearn-version dependent** | every SAM40 subject has exactly 12 trials, so ties are everywhere; two machines produced splits differing on 17 of 169 subjects running identical code. Replaced with a self-contained greedy assignment, verified identical across sklearn 1.8.0 and 1.9.0 |
| λ=0 confounded the adversary with the early-stopping lock | λ=0 could select epoch 0 where λ=1 could not select before epoch 11; a separate `lambda0_locked` control was added, and it is the true method-off control |
| adversary ran against a **single domain** | cross-entropy over five logits with one class present still has a gradient, and the GRL reverses it — the backbone was being pushed to make a trivial prediction *harder*. Now returns zero below two domains |
| checkpoint selected before the adversary was active | λ at selection was 0.20–0.38 in 3/5 folds while λ at stop read 0.74–0.82 |
| predictions carried no identity | subject-level aggregation had to reconstruct identity from fold indices, which only ever worked for two corpora of four |
| out-of-fold loader globbed `preds_fold*.npz` | also matched `_lambda0`, `_randbb`, `_smoke` — would have pooled every training mode and reported the average as "the model" |
| `CalibratedClassifierCV(cv=3)` **inverts scores** on corpus-ordered data | the default non-shuffled KFold puts each internal fold on a different amplifier, so the calibrator learns a backwards mapping: AUC 0.4410 calibrated versus 0.6553 raw, negatively correlated (−0.25). It shipped broken artifacts once before being caught |
| the KD distillation term, if unmasked | an unmasked KL would force `P(depression) = 0` on every stress-corpus segment — literally teaching *"not-MODMA ⇒ not depressed"*, re-deriving the corpus shortcut from the loss function. The term is masked to the teacher's two classes and asserted to put exactly zero gradient on the unseen class |
| teacher soft targets computed in-fold | a teacher scoring its own training data emits near-one-hot targets and distillation silently collapses to the hard labels; targets are computed out-of-fold |

### 8.5 Data-integrity findings

- **Mumtaz is confounded with acquisition setup** (§7.5). Never quote a Mumtaz figure without the montage audit.
- **CogLoad is 99.9999% DC** — a ~31,000-count pedestal with tiny AC on top. After a 1 Hz high-pass the residue is wideband amplifier and EMG noise, leaving **8× EEGMAT's above-45 Hz power after gating** (29× before). No filter reaches it, and per-corpus filtering would make preprocessing itself the device signature.
- **Duplicate recordings appear in three distinct forms** — identical bytes under one subject (keep one), identical bytes across subjects whose recording sets match throughout (merge the identities), and identical bytes across subjects who otherwise hold distinct recordings (drop both). A subject retained under two identities defeats grouped cross-validation silently.
- **MODMA filenames are inconsistent.** Splitting on `rest` or `_` recovers 37 of 53 subjects; matching the leading zero-padded digit run recovers 53/53.
- **MODMA `.mat` key detection can match `Impedances_0`**, which shares the data array's shape signature. It works only because dictionary insertion order happens to favour the data key.
- **Two of four montages in the legacy pipeline are guesses.** MODMA and SAM40 have real electrode positions; AMIGOS's channel *order* is inferred; **CogLoad has no placement documented anywhere on disk** and was reverse-engineered. Spatial interpolation for that corpus is meaningless if the reconstruction is wrong, so montage assumption is carried as an explicit flag rather than a comment.
- **Local data is partial.** Only 15 of 40 AMIGOS participants are on disk. WESAD contains no EEG at all (chest ACC/ECG/EMG/EDA/Temp/Resp only). MODMA's audio modality was never downloaded, so the published 97.53% EEG+audio result is not reproducible here, and 6.2 GB of MODMA 128-channel ERP task data sits unused — the largest untapped asset available.

### 8.6 The probe/transfer dissociation — the methodological contribution

The most generalisable finding of the protocol-gated track is a **dissociation between two things commonly treated as equivalent**:

1. A frozen domain probe can show that adversarial training removes domain information *relative to its own method-off control* — the pre-registered gate passes on both split draws.
2. On the held-out amplifier that the mechanism exists to serve, a model trained on **shuffled labels transfers as well as either real arm**.

In other words: **the mechanism works, and the outcome it was built to protect has no signal to protect.** A probe result is evidence about the representation; it is not evidence about transfer. Reporting the first as though it implied the second is an error this project made and then measured its way out of.

The corresponding evaluation rule for raw-EEG deep learning is a two-dimensional grid rather than an accuracy column:

```
high accuracy + high corpus probe  ->  riding the device signature, not real
high accuracy + low  corpus probe  ->  the goal
low  accuracy + low  corpus probe  ->  invariance removed the signal (the 0.3255 outcome)
```

A caveat on that grid, flagged rather than fixed: **probes of different dimensionality are not comparable.** One architecture's embedding hook silently falls back to three output logits, and a 3-dimensional probe reads systematically lower than a 1120-dimensional one — so a model can look corpus-free purely from dimensionality. Compare probes only at equal width, or compare each model against its own control.

---

## 9. Limitations and threats to validity

1. **No result here is device-disjoint except §7.4.** Every cross-validated number is subject-disjoint but shares the amplifier between train and test. Section 7.4 measures what happens when that assumption is dropped, and the answer is approximately chance for the pooled system.
2. **The three-class dataset carries a known preprocessing defect** (§8.2) affecting 15% of its segments. It is disclosed with every number derived from it. The defect plausibly inflates the corpus probe, which biases the 0.6267 baseline upward — against the model, not for it.
3. **Deep-model conclusions are untuned.** A single hardcoded configuration was used throughout. A nested-CV hyperparameter sweep is what would make "deep models lose here" airtight rather than well-supported.
4. **The stress head does not discriminate** in the protocol-gated build: in-domain separation of +0.044 against a 0.10 bar, and an untrained backbone beats it on every stress metric. The stress branch of the deployed system should be treated as unvalidated.
5. **Sample size bounds everything.** 53 MODMA subjects, 61 Mumtaz, 40 SAM40, 15 CogLoad, 36 EEGMAT, 108 subjects in the pooled three-class set. One split draw moved a MODMA figure by 0.108. Effects smaller than that cannot be resolved with this data regardless of method.
6. **The anxiety class exists in only one cohort**, and in that cohort it is the least recoverable class (recall 0.146–0.178). The original four-class goal is not achievable with any data available to this project.
7. **Demographics were not available for the main-pipeline corpora.** The BRMH result — that EEG adds little over age/sex/education — could not be checked against MODMA or SAM40, where no such covariates are on disk. MODMA does ship PHQ-9 and GAD-7 scores for 53 subjects, which remain unused.

---

## 10. Conclusions

Six statements are supported by the measurements in this paper.

1. **A three-class EEG classifier over {healthy, depression, stress} is achievable by factorising into two binary boundaries selected by acquisition context.** The hierarchical form reaches **0.6820** against a **0.6267** device-only baseline — the only architecture of thirty-six to clear it. The protocol-gated form is the deployable version of the same decomposition, and its strongest branch, **depression transfer across two amplifiers and two countries, reaches 0.8675 / 0.8438 subject accuracy across two independent split draws** against a 0.3998 permutation null.

2. **A flat three-class softmax cannot be justified on this data.** The best flat model scores 0.6256, below a classifier that knows only the amplifier. Removing the device signature collapses accuracy to chance (0.3255). The shortcut is worth +0.18 accuracy, and nothing remains without it.

3. **Explicit spectral and covariance features beat every learned representation tried**, and the comparisons carry their own controls: Riemannian tangent space +0.090 over SVM on stress; a foundation model whose pretraining demonstrably works (+0.076 over its own random-init control) still loses by 0.18 to a 105-dimensional linear model; self-supervised pretraining *hurts* by 0.050. The same conclusion arrives independently on a completely different cohort, where both transformers lose by 0.10–0.24.

4. **The six-channel bottleneck is not the limiting factor.** Two model families under two protocols produce the same inverted U: 16–32 channels beats 6, and full density is worse than 6. With 53–108 subjects, additional capacity is consumed by noise — the same lesson that killed the 273-feature set for the specialists and per-fold feature-view selection.

5. **Evaluation protocol is the dominant source of error in this domain, larger than any architectural choice.** Random versus grouped splits: 0.9393 versus 0.6256. Single split versus out-of-fold CV: +0.21 AUC on deep models. Three threshold-tuning bugs, each producing plausible wrong numbers silently. One split draw moving a headline by 0.108. An untrained backbone beating a trained one. ROC-AUC, permutation nulls, method-off controls, and repeated split draws are what caught every one of these.

6. **Cross-device deployment fails, and every within-device number is an upper bound.** Two independent routes agree at approximately chance. A genuine fix requires training data spanning multiple devices per class, or leave-one-corpus-out as the headline protocol rather than an afterthought.

The broader methodological claim: in EEG classification, **a result is only as good as the baseline it is reported against and the control it is compared to.** Chance is almost never the right baseline. On this data the right baselines were the device shortcut (0.6267), the corpus majority, demographics, a permutation null, and an untrained network — and against those, most of what initially looked like performance was not.

---

## 11. Open work

| Item | Note |
|---|---|
| Raw-EEG deep-learning benchmark, real GPU runs | harness complete, only smoke results exist |
| Classical-model channel-density bake-off | decides whether knowledge distillation from a high-density teacher is justified at all |
| SAM40 sampling-rate fix in the legacy pipeline | full rebuild attached; invalidates 36+ results |
| Ramped-α DANN | α is fixed and never annealed — the prime suspect for the adversarial failure |
| Nested-CV hyperparameter sweep for deep models | would make "deep models lose here" airtight rather than untuned |
| MODMA audio modality and 6.2 GB of ERP task data | never downloaded / unused — the largest untapped assets available |
| PHQ-9 / GAD-7 scores for 53 MODMA subjects | the most likely route to an anxiety target inside the main pipeline; currently unused |

---

## Appendix A — Track timeline

| # | Track | Dates | State | Headline |
|---|---|---|---|---|
| 1 | Legacy MODMA 3-channel | May–Jun 2026 | superseded | 0.37–0.52 accuracy, at or below chance |
| 2 | Legacy MODMA 128-channel (TSception / EEGNet) | Jun 2026 | superseded | 74.87% TSception — later shown split-inflated |
| 3 | SAM40 notebooks | Jun–Jul 2026 | disconnected | separate machine, random split, wrong sampling rate |
| 4 | Unified 4-corpus pipeline + 11 models | Jul 2026 | complete | deep models look strong; they are not |
| 5 | Phase 1 diagnostics + Phase 2 models | Jul–Aug 2026 | complete | the honest numbers: 0.7716 / 0.7560 |
| 6 | Three-class track (36 models) | Jul–Aug 2026 | complete | 0.6267 is the bar; only hierarchical clears it (0.6820) |
| 7 | Four-class BRMH cohort | Aug 2026 | complete | anxiety exists; EEG adds ~nothing over demographics |
| 8 | Channel density + raw-EEG DL | Aug 2026 | partial | inverted U; DL harness built, real runs pending |
| 9 | Protocol-gated build (two split draws) | Sep 2026 | complete | mechanism works; the outcome it protects has no signal |

Track 1 (3-channel MODMA) established that three channels carry nothing usable — SVM 0.3747, RF 0.4286, 1D CNN 0.5201, LSTM 0.4084, every AUC within noise of 0.5. Track 2's TSception result (74.87% ±4.89%) was the project's first "working" depression model and its first lesson in variance: EEGNet collapsed to 36.27% in one fold of five, which is what a 53-subject dataset does to a deep model. Both are superseded by Track 5 and retained for provenance only.

## Appendix B — Best numbers, with sources

| Task | Model | Metric | Score | Source |
|---|---|---|---|---|
| Depression (MODMA) | SVM on 81 handcrafted features | subject AUC | **0.7716** | `Results/phase1/depression_cv.json` |
| Stress (SAM40) | Riemannian tangent space | within-subject AUC | **0.7560 ±0.050** | `Results/phase2/riemannian.json` |
| 3-class (pooled) | Hierarchical v2-router / v1-specialists | accuracy | **0.6820** (bar 0.6267) | `Results/all3/hierarchical_v2_v1.json` |
| 4-class (BRMH) | EEG + DEMO elasticnet | balanced accuracy | 0.5178 (demo bar 0.5004) | `Results/OSF_Park/plan01_4class` |
| Binary (BRMH) | demographics + residualised EEG, no IQ | ΔAUC | **+0.0937** [+0.022, +0.165] | `Results/OSF_Park/plan02_binary` |
| Cross-device depression | MODMA → Mumtaz, both draws | subject accuracy | **0.8675 / 0.8438** (null 0.3998) | `arch_metd/REPORT.md` |
| Cross-device depression | Mumtaz → MODMA, both draws | subject accuracy | 0.6638 / 0.6667 (null 0.5431) | `arch_metd/REPORT.md` |
| Cross-device stress | SAM40+CogLoad → EEGMAT | — | **inert** (below its permutation null) | `arch_metd/REPORT.md` |

## Appendix C — Numbers that must never be quoted, and why

| Number | Where it came from | Why not |
|---|---|---|
| **0.9393** three-class accuracy | random rather than subject-grouped split | leakage — overlapping windows from one subject on both sides |
| **0.9891 / 0.9015** depression subject AUC | single 70/10/20 split, deep models | inflated ~0.21 AUC by which ten subjects landed in test |
| **74.87%** TSception depression | legacy 5-fold, superseded protocol | falls to 0.7385 subject AUC under out-of-fold grouped CV |
| **1.0000** CogLoad stress AUC | quantised metric (1 baseline vs ~6 load recordings per subject) | not a real ranking |
| **0.6963** three-class (oracle router) | hierarchical with true corpus label at inference | reads the routing label it is supposed to predict |
| **~0.97** stress-vs-depression | any model on the pooled data | `depression` is boolean-identical to `corpus == MODMA`; it is a device detector |
| **0.9375** Mumtaz depression AUC, unqualified | within-corpus CV | montage predicts diagnosis at 0.8889; an untrained backbone reaches 0.8912 |
| **0.8690** CogLoad stress AUC, unqualified | protocol-gated stress head | an untrained random backbone scores 0.9381 on the same task |
| Any figure from `report.md` | single-split pipeline report | superseded by out-of-fold grouped CV throughout |
