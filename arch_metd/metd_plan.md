# Protocol-Gated EEG Mental-State Classifier — Build Specification

Hand this to Claude Code as-is. It is self-contained: every constraint, path, index,
threshold and acceptance test needed to build the system is here.

> **Revision — TDBRAIN unavailable.** This build uses **five corpora**: MODMA, SAM40,
> Cognitive Load, Mumtaz 2016, EEGMAT. TDBRAIN could not be obtained.
>
> Consequence: the depression head loses its fully held-out corpus and drops from
> ~500 subjects to ~117. External validation for depression is now a **two-corpus
> leave-one-out** (train MODMA, test Mumtaz, and the reverse) instead of a clean
> held-out third. That is a weaker test but still a genuine amplifier, montage,
> country and clinic change — a 128-channel EGI net at 250 Hz in Lanzhou against a
> 19-channel 10-20 cap at 256 Hz in Kelantan. Report it as reciprocal transfer, not
> as a held-out corpus, and say so plainly in the writeup.
>
> The stress head is **unaffected**: SAM40 + CogLoad train, EEGMAT stays fully held
> out. Stress remains the stronger arm of this study.

---

## 0. The one constraint that shapes everything

**Never train a 3-way softmax over {healthy, depression, stress}.**

Depression appears only in resting-state clinical corpora; stress appears only in
task corpora. No public dataset contains both on one amplifier. A 3-way classifier
therefore learns amplifier identity, not neurophysiology. This has been measured on
this exact data: a classifier knowing *only* which corpus a segment came from, with
zero EEG understanding, scored 0.6755 on the 3-class task, and no flat model beat it.

The system still emits **one label with three possible values**. It does so by
combining two learned binary boundaries with one bit of acquisition metadata:

```
protocol == "resting" ->  p_dep >= theta_dep  ?  "depression" : "healthy"
protocol == "task"    ->  p_str >= theta_str  ?  "stress"     : "healthy"
```

Both boundaries are trained on data where the contrast genuinely exists. The
depression-vs-stress distinction comes from the recording protocol, which is always
known at inference time. **Do not add a cross-head argmax** (`p_dep > p_str`); the two
heads are trained on disjoint data and their outputs are not on a common scale.

---

## 1. Data inventory

Five corpora. Three are already on disk; two must be downloaded.

### 1.1 On disk

| Corpus | `domain` | Task | Path | Format | fs | Subjects |
|---|---:|---|---|---|---:|---:|
| MODMA resting | 0 | depression | `datasets/MODMA/EEG_128channels_resting_lanzhou_2015/` | `.mat` v5, big-endian `>f8` | 250 | 53 |
| SAM40 | 1 | stress | `datasets/SAM40/raw_data/` | `.mat` v5, `<f8` | 128 | 40 |
| Cognitive Load | 2 | stress | `datasets/Cognitive Load Assessment Through EEG A Dataset from Arithmetic and Stroop Tasks/raw_data/` | CSV text, no header | 250 | 15 |

### 1.2 To download

| Corpus | `domain` | Task | Source | Format | fs | Subjects |
|---|---:|---|---|---|---:|---|
| Mumtaz 2016 | 3 | depression | figshare `10.6084/m9.figshare.4244171.v2` | `.edf` | 256 | 34 MDD / 30 HC |
| EEGMAT | 4 | stress | PhysioNet `physionet.org/content/eegmat/1.0.0/` | `.edf` | 500 | 36 |

**TDBRAIN is not part of this build.** Leave `domain = 5` unallocated so it can be
added later without renumbering.

**DASPS is out of scope for the label space.** If you have it, note that it is *not* a
clinical anxiety corpus — the 23 participants are healthy subjects with anxious states
induced by face-to-face psychological stimulation, which makes it a state-induction
paradigm closer in kind to SAM40 than to MODMA. It could in principle serve as a
fourth stress-family corpus. Two reasons not to: it needs the AF3→Fp1 / AF4→Fp2
approximation (Emotiv EPOC has no Fp1/Fp2 electrode), and AMIGOS — the same headset,
the same approximation — was measured at AUC 0.4875 for arousal, i.e. chance. Do not
add it without first running it as a standalone binary and confirming it decodes
above chance.

### 1.3 Canonical representation

Every corpus is projected onto the same tensor:

- **6 channels**, fixed positional order: `Fp1, Fp2, F7, F3, F4, F8`
- **128 Hz**
- **4.0 s windows** = 512 samples, **50% overlap**, stride 256
- `float32`, shape `(N, 6, 512)`

---

## 2. Per-corpus loading

Each loader returns a list of `(continuous_array, metadata)` where
`continuous_array` is `(6, n_samples)` at native fs.

### 2.1 MODMA (`domain = 0`)

- 53 `.mat` files. Each holds three variables; the EEG key is **named after the file
  and differs in every file**. Search for it. Matching on `shape[0] in (128, 129)`
  alone also matches `Impedances_0` — **require `shape[1] > 1000`**.
- Channel rows (0-indexed into the `(129, n)` array):
  `Fp1=21 (E22), Fp2=8 (E9), F7=32 (E33), F3=23 (E24), F4=123 (E124), F8=121 (E122)`
- Row 128 (`E129`, Cz reference) is identically zero in every file. Never include it.
- **Subject ID**: filenames are inconsistent (`02010002rest 20150416 1017..mat`,
  `02010008_rest 20150619 1653.mat`, `02010021 20150805 1730.mat.mat`). Splitting on
  `rest` or `_` recovers only 37/53. Use `re.match(r"0*(\d+)", filename)` → 53/53.
- **Labels**: `subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx`,
  join on `subject id`. `type` ∈ {HC, MDD} → 29 HC / 24 MDD.
  `y_dep = 1 if type == "MDD" else 0`, `y_str = -1`.
- Also carry `PHQ-9` and `GAD-7` per subject (unused by the heads; needed for §8.4).

### 2.2 SAM40 (`domain = 1`)

- Read **`raw_data/` (key `Data`)**, not `filtered_data/`. The authors' cleaning is an
  uncontrolled per-corpus filter chain; using it would make preprocessing itself a
  corpus signature. Run `filtered_data/` (`Clean_data`) as a separate ablation only.
- `(32, 3200)` per file, 480 files.
- **fs is 128 Hz. Do NOT decimate.** The legacy pipeline did `data[:, ::2]` with the
  comment "since native is 256 Hz" and then labelled the result 128 Hz, time-stretching
  every SAM40 segment 2x. The paper states 128 SPS and 128 x 25 s = 3200 = the array
  length on disk. This is the single most damaging bug in the old pipeline.
- Channel rows: `[2, 31, 3, 4, 29, 30]` → Fp1, Fp2, F7, F3, F4, F8. All native.
- Filename `<Task>_sub_<N>_trial<T>.mat`. `Relax` → `y_str = 0`; `Arithmetic`,
  `Mirror_image`, `Stroop` → `y_str = 1`. `y_dep = -1`.
- `trial_id = f"sam40_{task}_{sub}_{trial}"`, `subject_id = f"sam40_{N}"`.

### 2.3 Cognitive Load (`domain = 2`)

- Glob `raw_data/Arithmetic_Data*/*.txt` and `raw_data/Stroop_Data/*.txt`.
  **The arithmetic folder name has a trailing space** — never hardcode it.
- 25 columns, no header. EEG is columns `[1, 2, 3, 4, 6, 7]` → Fp1, Fp2, F7, F3, F4, F8
  (skipping col 5 = Fz and col 8 = C2).
- Values are raw ADC counts at the 1e4 scale with a large positive DC pedestal.
- Filename `<level>-<subject>.txt`. `natural-*` → `y_str = 0`;
  `lowlevel/midlevel/highlevel-*` → `y_str = 1`. `y_dep = -1`.
  Also store the ordinal level (`natural`/`low`/`mid`/`high`) — unused by the head,
  needed for §8.5.
- **Subject ID is the first digit run in the basename**, so all four levels across both
  task folders collapse to 15 IDs. Easy to misread as 120 subjects.
- Known: 120 files exist but the design yields ~105 recordings; one cross-subject
  duplicate is unrecoverable and both copies must be dropped. Log the final count.

### 2.4 Mumtaz 2016 (`domain = 3`)

- `.edf`, 19 channels, 10-20 system, 256 Hz, recorded at HUSM Malaysia.
- Sessions per subject: `EC` (eyes closed), `EO` (eyes open), `TASK` (P300).
  **Use EC and EO. Drop TASK** — it is an evoked-potential paradigm, not resting state.
- Filenames encode group (`H` / `MDD`) and session. `y_dep = 1 if MDD else 0`,
  `y_str = -1`.
- Channel names in the EDF may carry prefixes/suffixes (e.g. `EEG Fp1-LE`). Match
  case-insensitively on the bare label; assert all six resolve.
- Not every subject has both EO and EC, and one MDD subject has only TASK — expect
  fewer than 64 usable subjects. Log which.

### 2.5 EEGMAT (`domain = 4`)

- `.edf`, 19 channels, Neurocom 23-channel system, 500 Hz, 36 subjects.
- **Two files per subject**: `Subject%02d_1.edf` = background before the task
  (`y_str = 0`), `Subject%02d_2.edf` = during serial subtraction (`y_str = 1`).
  `y_dep = -1`.
- Channel set includes Fp1, Fp2, F3, F4, Fz, F7, F8 — all six frontals native.
- `subject-info.csv` carries a count-quality group (24 "good" / 12 "poor" counters),
  age, gender. Carry it; do not train on it.
- Paired design: exactly one baseline and one task recording per subject. This
  determines the evaluation metric (§7.3).

### 2.6 (reserved)

`domain = 5` is reserved for TDBRAIN. Do not reuse the index.

Because Mumtaz is now the **only** external depression corpus, its loader carries more
weight than the others. Two things to get right:

- Both `EO` and `EC` sessions are used, and `condition` records which. MODMA is
  eyes-closed resting only, so a MODMA→Mumtaz transfer test on `EO` segments is
  confounded by eye state on top of the amplifier change. Report `EC` transfer as
  primary and `EO` as secondary.
- Subjects missing a session must be logged, not silently dropped. Expect fewer than
  64 usable subjects.

---

## 3. Preprocessing

Applied identically to all five corpora. Order matters; do not reorder.

```
load continuous (6, n) at native fs
  -> mark flatline spans      (mark only, do NOT excise)
  -> reflect-pad 3.0 s each end
  -> 50 Hz notch              (iirnotch, Q=30)
  -> 1-45 Hz bandpass         (Butterworth order 4, SOS, filtfilt)
  -> strip padding
  -> resample_poly to 128 Hz
  -> window 512 / stride 256
  -> reject windows overlapping a flatline span
  -> reject windows failing the variance gate
  -> per-segment normalize
```

### 3.1 Filtering details

- **Zero-phase only** (`scipy.signal.filtfilt`). Design as SOS
  (`butter(4, [1, 45], btype="band", fs=fs, output="sos")`, `sosfiltfilt`).
- **Reflection padding of 3.0 s on every recording, every corpus.** SAM40 trials are
  standalone 25 s files with no continuous context, so their edges would otherwise
  carry a filter transient that no other corpus has — a corpus signature by
  construction. Padding uniformly removes the asymmetry.
- **High-pass at 1 Hz, not 0.5 Hz.** A 0.5 Hz corner has a 2-6 s settling transient;
  1 Hz settles in roughly a third the time and removes the DC pedestals just as well
  (MODMA runs to ~-1e4, CogLoad to ~+1e4). Sub-1 Hz delta is not estimable in a 4 s
  window anyway.
- **Notch before low-pass.** A sharp notch applied to a large DC offset rings badly.
- All five corpora are 50 Hz mains regions (China, India, Malaysia, Ukraine).
  MODMA measured 92.3% of 49-51 Hz band power pre-notch.

### 3.2 Artifact gating — uniform across all corpora

Two rules, identical thresholds everywhere. Applying a gate to one corpus only makes
the *rejection rate* a corpus signature.

**Flatline**: mark zero-runs on the raw continuous signal, per channel, minimum run
length 50 ms. Reject any window overlapping a marked span. Mark before filtering;
reject after windowing. Never excise spans from the continuous signal — cutting
creates step discontinuities that `filtfilt` rings on.

**Channel variance**: reject a window where
`max_c Var(c) > K * median_c Var(c)`, computed **within-segment across the 6 channels**
on **filtered-but-unnormalized** data.

`K` must be **calibrated, not assumed**. Sweep K on MODMA (which has zero dropout and
zero NaN by measurement) and pick the value that rejects 1-2%. A previously-used
K = 5 rejected ~9.7% of MODMA, which is too aggressive. Post-normalization data is
also the wrong input — global MAD scaling forces the whole `(6, 512)` matrix to unit
variance while leaving channels free to differ, and normal frontal EEG routinely has
Fp1 at 3-5x F3 because blinks load on the prefrontal pair.

**Emit a rejection table** (rows: corpus; columns: recordings, windows, kept,
flatline-rejected, variance-rejected, %) and publish it. Divergent rates across
corpora are a finding, not a detail.

### 3.3 Normalization

Per segment, over the whole `(6, 512)` matrix — **not per channel**:

1. `med = median(X)`, `mad = median(|X - med|)`
2. clip to `med +/- 5 * mad`
3. `X = (X - mean(X)) / std(X)` across the whole window

Global rather than per-channel is deliberate: it preserves inter-channel amplitude
ratios, which frontal asymmetry depends on. Post-condition: per-segment mean ~ 0,
std ~ 1.0, values within roughly +/-3.6.

---

## 4. Tensor and label space

```
X          float32 (N, 6, 512)
y_dep      int8    {0: HC, 1: MDD, -1: mask}
y_str      int8    {0: relax/baseline, 1: task/load, -1: mask}
y_domain   int8    {0..4}          # 5 reserved for TDBRAIN
anchor     bool    True iff y_dep == 0 or y_str == 0
subject_id object  "modma_2010002", "sam40_10", "cogload_3",
                   "mumtaz_H12", "eegmat_07"
trial_id   object  unique per recording/trial
protocol   object  {"resting", "task"}
condition  object  {"EC", "EO", "arith", "stroop", "mirror", "relax", "natural", ...}
```

**`-1` means "this corpus does not support this task."** No segment carries both
labels. The masked loss zeroes those samples so one shared backbone trains on all
five corpora.

`protocol` is `"resting"` for domains {0, 3} and `"task"` for {1, 2, 4}. It is an
inference-time input, never a training target.

Store arrays **shuffled**, or ship an explicit shuffled index. Grouping by corpus on
disk has already caused an accidental leave-one-corpus-out split via an unshuffled
`CalibratedClassifierCV(cv=3)`.

---

## 5. Architecture

EEGNet-8,2 backbone, two masked binary heads, anchor-gated domain adversary.

```
Input (1, 6, 512)
  |
  Conv2D(8, (1,64), padding="same", bias=False) -> BatchNorm
  DepthwiseConv2D((6,1), depth_multiplier=2, bias=False, max_norm=1.0)
      -> BatchNorm -> ELU -> AvgPool((1,4)) -> Dropout(0.25)
  SeparableConv2D(16, (1,16), padding="same", bias=False)
      -> BatchNorm -> ELU -> AvgPool((1,8)) -> Dropout(0.25)
  Flatten -> z in R^256                      (16 filters x 16 time steps)
  |
  +-- Dense(32) ReLU -> Dense(1) sigmoid     -> p_dep
  +-- Dense(32) ReLU -> Dense(1) sigmoid     -> p_str
  +-- GRL(-lambda) -> Dense(64) ReLU -> Dense(5) softmax  -> domain
```

Temporal kernel 64 at 128 Hz = 0.5 s receptive field, i.e. down to 2 Hz.

### 5.1 Anchor-gated GRL

The adversary sees **only segments where `anchor == True`** (`y_dep == 0` or
`y_str == 0`).

This is the load-bearing design decision. A marginal adversary over all segments would
be asked to erase the difference between MODMA-MDD and SAM40-task — but those are
different *classes* in different *domains*, so erasing the domain erases the label.
Restricting to the healthy/baseline anchor aligns recording physics without touching
class-conditional structure.

Weight the adversary's cross-entropy by inverse anchor frequency
(`w_c = N_anchor / (5 * N_anchor_c)`). Anchor counts are badly skewed — MODMA HC alone
is several thousand segments against SAM40 Relax's few hundred — and an unweighted
adversary can win by always predicting the largest corpus, applying no alignment
pressure to the rest.

### 5.2 Loss

```
L = L_dep_masked + L_str_masked + lambda_p * L_domain_anchor_weighted
```

Masked BCE: zero the contribution of samples where `y == -1`, divide by the count of
valid samples, not the batch size.

### 5.3 Lambda schedule

`lambda_p = 2 / (1 + exp(-10 * p)) - 1`, `p in [0, 1]` = training progress.

**Lock early stopping until `lambda >= 0.5`**, and log the final lambda at
termination. Without the lock, patience can trigger while lambda is still near zero
and you ship the un-adapted model by accident.

---

## 6. Training protocol

### 6.1 Nested cross-validation

```
Outer: StratifiedGroupKFold(5) on subject_id, per corpus, unioned into joint folds
  Inner: GroupKFold(4) on training subjects
         -> early stopping, lambda scheduling, threshold calibration
  Outer test fold: touched exactly once
```

Everything tuned — stopping epoch, `theta_dep`, `theta_str` — is selected on the
inner fold. The outer fold produces the reported number and nothing else.

**Every split groups on `subject_id`.** Segments are 50%-overlapping windows; a random
split leaks the same person into train and test. Measured on the old pipeline: random
split 0.9393 vs subject-grouped 0.6256 on the same model.

Stratify MODMA and Mumtaz on the class label — plain 5-fold on 53 subjects can hand
you a fold with very few MDD.

CogLoad has 14-15 subjects. Use **outer LOSO with a fixed epoch budget** derived from
the SAM40 inner runs; a 3-subject inner split produces uselessly noisy early stopping.

### 6.2 Held-out corpora

**EEGMAT never appears in training.** It is the leave-one-corpus-out generalization
test for stress — a completely unseen amplifier, lab and montage.

```
Stress head:      train {SAM40, CogLoad}  -> hold out {EEGMAT}
```

Depression has no third corpus, so it gets **reciprocal two-corpus transfer** instead:

```
Depression head:  train {MODMA}   -> test {Mumtaz}
                  train {Mumtaz}  -> test {MODMA}
```

Run both directions, each with its own `lambda = 0` control. Report both; if they
disagree sharply, the smaller or noisier corpus is carrying the result and neither
number should be trusted alone.

This is weaker than a held-out third corpus because the adversary has only two
depression domains to align, and with two domains "align the anchors" can be satisfied
by a single translation rather than by finding genuinely device-invariant structure.
State that limitation explicitly. It is the main methodological cost of losing TDBRAIN.

Within-corpus CV measures decodability. Only the transfer results measure the thing
the architecture claims: that the representation survives a hardware change.

### 6.3 Optimization

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-2 |
| Scheduler | Cosine annealing with warm restarts |
| Batch size | 64, **corpus-balanced sampling** |
| Early stopping | patience 15 on inner-fold sum of valid AUCs, locked until lambda >= 0.5 |
| Normalization | Prefer GroupNorm over BatchNorm, or verify BN running stats behave under single-corpus test batches |

### 6.4 Threshold calibration

Youden's J (`TPR - FPR`) on the inner fold, per head. Thresholds tuned on one corpus
do not transfer to another — recalibrate per held-out corpus and report both the
transferred and recalibrated numbers.

---

## 7. Evaluation

Segment-level metrics are never primary. Aggregate to the level at which the label
actually varies, using **logit medians**, not mean probabilities (means of sigmoid
outputs saturate; the median over a subject's ~150 segments resists a few bad windows).

### 7.1 Depression corpora — subject level

The label is a per-subject trait. Every segment of a subject shares it, so
segment-level N overstates effective sample size by ~150x.

| Corpus | Unit | N |
|---|---|---|
| MODMA | subject | 53 (24 MDD / 29 HC) |
| Mumtaz | subject | up to 64 (34 MDD / 30 HC) |

Total depression N is ~117 subjects. **Every depression number in this build is
underpowered and must ship with an interval.** Primary metric: subject-level ROC-AUC
with 1,000-iteration bootstrap 95% CI. At N = 53 the CI is roughly +/-0.13 even at
AUC 0.80; at N = 64 it is not much better. A point estimate alone is not
interpretable. Use DeLong or bootstrap, always, and never report a depression AUC
without its CI in the same sentence.

Because both corpora are small, also report the **pooled** subject-level AUC across
MODMA + Mumtaz from out-of-fold predictions (N ~ 117) as a secondary figure, with the
caveat that pooling across two amplifiers inflates it if corpus separates the classes.
Check that against the domain probe before believing it.

### 7.2 SAM40 — trial level

| Unit | N |
|---|---|
| trial | 480 (120 relax / 360 task) |

Secondary: mean within-subject AUC over 40 subjects. Note it rests on 3 relax vs 9
task trials per subject — 27 pairs, granularity 0.037.

### 7.3 EEGMAT — paired sign test

Each subject has exactly one baseline and one task recording, so a within-subject AUC
is degenerate. Use the paired form: for each subject, is `p_str(task) > p_str(baseline)`?
Report the proportion over 36 subjects with an exact binomial test against 0.5. This
is well-powered and directly interpretable.

Also report recording-level AUC over N = 72 as secondary.

### 7.4 Cognitive Load — within-subject primary

Every subject has a `natural` baseline and several load recordings, so within-subject
is well posed. **Pooled recording-level AUC there rests on ~14 negative recordings in
total and its CI cannot separate a working model from a broken one** — a previous run
returned exactly 1.0000 on all 14 subjects, which is the corpus design (one baseline
per subject makes "is this the baseline?" identical to the label), not decoding.

Report within-subject as primary, pooled as secondary with its honest interval, and
state the design limitation in the writeup.

### 7.5 Cross-corpus transfer

Both directions for stress, each with a `lambda = 0` control:

```
SAM40 + CogLoad -> EEGMAT           (primary stress generalization test)
EEGMAT          -> SAM40, CogLoad   (diagnostic only; EEGMAT is small)
MODMA           -> Mumtaz           (primary depression transfer, EC segments)
Mumtaz          -> MODMA            (reciprocal direction)
```

Chance floors differ per corpus (SAM40 is 25/75). Report AUC; if you report accuracy
anywhere, state the majority baseline next to it.

---

## 8. Mandatory diagnostics

None of these are optional. The previous iteration of this project shipped a result
that was ~92% amplifier identity, and only these checks surfaced it.

### 8.1 Frozen domain probes

Freeze the backbone. Train an **independent** logistic regression on `z` to predict
`y_domain`. The adversary's own accuracy is not a probe — it is one side of a min-max
game being actively suppressed, and its loss rising means it lost, not that `z` lacks
domain information.

Run **two** probes at both `lambda = 0` and `lambda = 1`:

- **Anchor probe**, balanced anchor subsets. Target: near 1/5 = 0.200.
- **Full-set probe**, all segments. Target: near the majority-domain baseline,
  **computed from the rebuilt arrays**, not hardcoded.

Running only the anchor probe reports invariance exactly where you enforced it. If
anchors align but the full representation stays corpus-separable, that is an honest
result and the protocol-gated inference still stands.

### 8.2 Manifold gate (falsification check)

Before committing to domain adaptation: compute the corpus silhouette score on anchor
embeddings at `lambda = 0` and after warmup. Pre-register the criterion numerically —
e.g. *corpus silhouette on anchors > 0.3 after lambda reaches 1 -> fall back to
independent single-task backbones*. Report the delta, not a single verdict.

### 8.3 Negative controls

- **Permuted labels**: shuffle within corpus. AUC must collapse to 0.50.
- **Random backbone**: freeze random weights, train heads only. Verifies that learned
  representations, not input artifacts, drive performance.
- **Corpus-only dummy**: a classifier knowing only `y_domain`. Report its accuracy
  alongside every headline number, recomputed on the rebuilt arrays.

### 8.4 Cross-head extrapolation

Feed MODMA-MDD segments to the stress head and CogLoad-task segments to the depression
head. Plot output density histograms.

**Compare against in-domain positives from the same head**, not against zero. A
previous run showed every distribution clustered near p ~ 0.4 including the in-domain
case, which cannot distinguish "heads don't extrapolate" from "heads output 0.4 for
everything."

### 8.5 Auxiliary heads (optional, recommended)

Free supervision already in the data, all masked:

- PHQ-9 and GAD-7 regression on MODMA (per-subject continuous severity).
- Ordinal load level on CogLoad (`natural`/`low`/`mid`/`high`). This is the only
  ordinal label in the project and it forces a **within-subject** arousal axis, which
  by construction cannot be device identity.

### 8.6 Sign check on the depression head

A prior run returned MODMA subject-level AUC of 0.3434 with one fold at exactly
0.0000. **That is not chance — it is inversion**, and a model reliably wrong found
signal and assigned it backwards. Before accepting any depression number below 0.5,
verify the label join: MODMA filenames are inconsistent and `int()` of the digit run
must land on the correct spreadsheet row. Add an assertion.

---

## 9. Inference

```python
def predict(segments, protocol):
    z = backbone(segments)
    if protocol == "resting":
        p = logit_median(sigmoid(dep_head(z)))
        return "depression" if p >= theta_dep else "healthy", p
    elif protocol == "task":
        p = logit_median(sigmoid(str_head(z)))
        return "stress" if p >= theta_str else "healthy", p
    raise ValueError("protocol must be 'resting' or 'task'")
```

One label out, three possible values. No cross-head comparison. No comorbidity claim —
no segment in training carries both labels, so the joint-high region of the output
space is entirely unsupervised; the architecture *permits* representing comorbidity
but has no evidence about it.

---

## 10. Repository layout

```
src/
  data/
    loaders/
      modma.py  sam40.py  cogload.py  mumtaz.py  eegmat.py
    preprocess.py        # filter chain, gating, normalization
    build.py             # loaders -> arrays -> disk, emits rejection table
  models/
    eegnet.py            # backbone
    grl.py               # gradient reversal + lambda schedule
    heads.py             # masked binary heads, weighted adversary
    mtdann.py            # assembly
  train/
    folds.py             # nested CV, per-corpus stratification, LOSO
    loop.py              # training loop, early stopping with lambda lock
  eval/
    aggregate.py         # logit medians, per-corpus units
    metrics.py           # AUC + bootstrap CI, paired sign test
    probes.py            # frozen domain probes, silhouette
    controls.py          # permutation, random backbone, corpus dummy
    report.py            # REPORT.md from results/*.json
  infer/
    gate.py              # protocol-gated single output
tests/
configs/
results/
```

`build.py` must be re-runnable and must delete any cached feature arrays on rebuild —
caches keyed on filename only are silently reused and have shipped broken artifacts
before.

---

## 11. Acceptance tests

Assert these. Each one corresponds to a bug that has already occurred.

**Ingestion**
- SAM40 yields **11 windows per trial** (3200 samples, 512/256) -> 5,280 pre-gate.
  If you get 5 windows / 2,400 segments, the decimation bug is present.
- MODMA subject IDs resolve **53/53** against the spreadsheet.
- MODMA row 128 (`E129`) is all-zero in every file, and is never selected.
- MODMA `.mat` key search rejects `Impedances_0` (`shape[1] > 1000` required).
- CogLoad glob matches the trailing-space folder name.
- CogLoad subject IDs collapse to **15**, not 120.
- All six frontal channel names resolve in every EDF/BrainVision file.

**Preprocessing**
- Power in the 49-51 Hz band is `< 0.001` of total, **every corpus**.
- No NaN or Inf anywhere in `X` after gating.
- Per-segment `|mean| < 1e-6` and `|std - 1| < 1e-6`.
- SAM40 mean absolute amplitude across trial time is flat — if the first and last
  seconds sit above the middle, the filter transient survived the padding.
- Rejection rates logged per corpus and per rule.

**Labels**
- `y_dep == -1` for all of domains {1, 2, 4}; `y_str == -1` for all of {0, 3, 5}.
- No segment has both labels valid.
- `anchor == (y_dep == 0) | (y_str == 0)`.
- Class counts match the source spreadsheets exactly.

**Splitting**
- No `subject_id` appears in both train and test of any fold, inner or outer.
- EEGMAT appears in **zero** training folds.
- In the depression transfer runs, MODMA and Mumtaz never appear in the same fold on
  the same side of the split.
- `y_domain` takes values in `{0, 1, 2, 3, 4}` only; index 5 is never emitted.
- Feature-matrix row `i` corresponds to segment `i` — if any stage drops rows, the
  same mask is applied to `y`, `subjects` and `domain`.

**Reporting**
- Every headline AUC ships with a bootstrap CI.
- Every headline number ships with the corpus-dummy baseline alongside.
- The `lambda` value at the stopping epoch is logged for every fold.

---

## 12. Do not reintroduce

Each of these was a real bug in the previous pipeline.

1. `data[:, ::2]` on SAM40 with a "native is 256 Hz" comment.
2. Reading AMIGOS `labels_selfassessment` index 1 (valence) as arousal.
3. Hardcoding every Cognitive Load file to stress-positive (`natural-*` is the baseline).
4. Any unshuffled internal split on corpus-ordered arrays.
5. Feature caches keyed on filename only.
6. Windowing without a finite-check.
7. Segment-level metrics reported as primary.
8. A cross-head argmax (`p_dep > p_str`) at inference.
9. Treating the adversary's own accuracy as a domain probe.
10. Hardcoded baselines — recompute the majority-domain figure from the rebuilt arrays.

---

## 13. Expected results

Calibrate expectations. Every number below is *lower* than the old pipeline produced,
because the old pipeline was measuring leakage and amplifier identity.

| Target | Unit | Expected AUC |
|---|---|---|
| Depression, MODMA within-corpus | subject | 0.65 - 0.80, CI ~ +/-0.13 |
| Depression, Mumtaz within-corpus | subject | 0.70 - 0.85, CI ~ +/-0.12 |
| Depression, MODMA -> Mumtaz (EC) | subject | 0.55 - 0.70 |
| Depression, Mumtaz -> MODMA (EC) | subject | 0.55 - 0.70 |
| Stress, SAM40 within-corpus | trial | 0.75 - 0.85 |
| Stress, held-out EEGMAT | paired | 0.65 - 0.80 |
| Domain probe, full set, lambda = 1 | — | near majority baseline |

Depression transfer is expected to be lower than it would have been against TDBRAIN,
because MODMA and Mumtaz differ on montage as well as amplifier — a 128-channel EGI
geodesic net against a 19-channel 10-20 cap. Six frontal electrodes are nominally the
same positions, but EGI's E22/E9 are not identical placements to a 10-20 Fp1/Fp2.
Expect a real drop and do not read it as model failure without checking a
montage-matched control.

A depression AUC of 0.73 on this pipeline is a **better** result than 0.94 on the old
one. If transfer holds above 0.65 in both directions, that is the paper. If depression
lands near chance with a CI crossing 0.50, the honest conclusion is that six frontal
channels do not carry it — which is also publishable, and more useful than another
irreproducible 0.97.

**Where the study now stands.** Stress is the strong arm: three corpora, a genuine
held-out amplifier, and a previously measured cross-corpus AUC of 0.786 within-subject.
Depression is the weak arm: two small corpora and reciprocal transfer only. Write the
paper around the stress result and report depression honestly as a secondary,
underpowered finding. Do not let the depression numbers set the headline.

---

## 14. Build order

1. `src/data/loaders/` for MODMA, SAM40, CogLoad. Verify against the acceptance tests.
2. `src/data/preprocess.py` + `build.py`. **Calibrate `K` on MODMA.** Emit the
   rejection table and the recomputed corpus-dummy baseline. Stop and read both.
3. Add Mumtaz and EEGMAT loaders (both native 10-20, both immediate downloads).
4. `src/models/` + `src/train/`. Run the sign check on the depression head first.
5. `src/eval/`. Probes and controls before any headline number.
6. If TDBRAIN access later clears, add it at `domain = 5`; §6.2 reverts to a held-out
   depression corpus with no other change.

Steps 1-2 will tell you more about whether this works than any amount of further
architecture work.