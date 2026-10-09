# trial_cc_pre — how to run

Self-contained rebuild of the plan in ` trial_cc_pre_plan.md`. Reads `../datasets/` and nothing
else in the repo — no `src/`, no `scripts/`, no `datasets/combined_data/`.

Three files: `preprocess.py` (section 1), `model.py` (section 3), `experiments.py` (sections 4–5).

```bash
cd " trial_cc_pre"
PY=../.venv/bin/python
```

## 0. Self-checks (seconds, no data)

```bash
$PY preprocess.py  --selfcheck    # flatline mask, filter response, variance gate, normalization
$PY model.py       --selfcheck    # GRL sign flip, lam=0 blocks gradient, -1 mask, anchor weights
$PY experiments.py selfcheck      # median aggregation, Youden guard, bootstrap CI
```

## 1. Build the segment array (~10 min, ~14 GB of reads)

```bash
$PY -u preprocess.py            # --smoke does 3 recordings per corpus in ~20 s
```

Writes `data/`: `X.npy (N, 6, 512) float32`, `y_dep`, `y_stress`, `y_domain`, `anchor`,
`subject`, `unit`, `rejection_table.json`. Rerun after any change to filtering, gating or labels.

As built — **16,063 segments, 108 subjects**:

| corpus | recordings | windows | kept | flatline | variance | blown ch. | subjects | test units |
|---|---|---|---|---|---|---|---|---|
| MODMA | 53 | 7,959 | 7,183 | 0 | 776 | 0 | 53 | 53 subjects (24 MDD / 29 HC) |
| SAM40 | 480 | 5,280 | 5,019 | 0 | 261 | 0 | 40 | 480 trials (120 relax / 360 task) |
| CogLoad | 103 | 4,441 | 3,861 | 61 | 462 | 57 | 15 | 99 recordings (14 natural / 85 task) |

Pre-gate window yields match the plan's predictions exactly (7,959 and 5,280). Majority-domain
baseline **0.4472**. 3-class composition healthy 5,863 / stress 6,934 / depression 3,266.

**Corpus shortcut baseline = 0.6755** — a dummy classifier that knows only which amplifier
produced a segment, recomputed on these arrays. The plan's 0.645 was stale. Grade any 3-class
number against 0.6755, never against chance (0.3333) or the global majority class (0.4317).

### CogLoad: 120 files, 103 recordings

Fully accounted, all 120 parse without error and subject ids resolve to exactly 15 with 8 files
each — no parse failure and no id collision. The corpus simply ships fewer recordings than a
naive file count suggests:

- Each subject's single `natural` baseline is filed in **both** task folders — 15 byte-identical
  pairs. The real design is 1 baseline + 6 task recordings per subject, so 15 × 7 = 105.
- `Stroop/highlevel-14.txt` and `highlevel-15.txt` are byte-identical across **different**
  subject ids — a dataset error whose provenance is unrecoverable. Both copies are dropped;
  filing it under either subject would corrupt the grouping variable. → 103.

`natural-4` is then lost to gating (see below), so subject C4 has load recordings but no
baseline and cannot contribute a within-subject AUC. **14 negative recordings in total** is what
forces the CogLoad reporting choice below.

### What the variance gate actually rejects (MODMA, 776 windows / 9.7%)

Checked rather than assumed, because a gate that fires more on one class is a confound:

| | rejection rate |
|---|---|
| MDD (n=24) | 0.0918 (median 0.0570) |
| HC (n=29) | 0.1012 (median 0.0537) |
| Mann-Whitney | **p = 0.81** |

Not label-correlated, and not predominantly blinks — trigger channel is spread evenly
(F3 166, Fp1 157, F4 140, F8 111, Fp2 104, F7 98) where blinks would pile onto Fp1/Fp2.
Rejected windows run **3.2× peak-to-peak** (297 vs 92) and **89% delta** vs 66% for kept:
large low-frequency movement/drift transients. Threshold sensitivity is smooth, no cliff, and
the class balance holds everywhere — 3× → 16.8% rejected (p=0.65), 5× → 9.8% (p=0.81),
10× → 5.5% (p=0.83). 5× is a defensible default, not a tuned one.

## 2. Experiments (~1–2 h total, CPU)

```bash
$PY -u experiments.py within     # 5-fold nested CV -> OOF logits, domain probes, cross-head   (~25 min)
$PY -u experiments.py loso       # CogLoad leave-one-subject-out, fixed epoch budget           (~45 min)
$PY -u experiments.py cross      # cross-corpus stress transfer, lambda=0 control              (~20 min)
$PY -u experiments.py legacy     # corpus shortcut + leaky random-split diagnostic             (~6 min)
$PY -u experiments.py report     # regenerate REPORT.md from results/*.json                    (instant)
$PY -u experiments.py all
```

`within` must run before `loso` (it supplies the epoch budget) and before `legacy` (it supplies
the grouped-split comparison). Results land in `results/*.json`; `REPORT.md` is generated from
them and goes stale if you add a run without re-running `report`.

Use `-u`: a 45-minute run is unmonitorable otherwise.

## Reading the output

Three things that are properties of the data, not of the model, and that any number here has
to be read against:

**CogLoad's label is confounded with recording identity, and it saturates.** Each subject has
exactly **one** natural baseline recording against ~6 task recordings, so "is this the baseline?"
is the same question as "is this not-stressed?". Any recording-level nuisance — impedance
settling, session drift, block order — answers the label perfectly, and the measured
within-subject AUC is **1.0000 on all 14 evaluable subjects** with per-recording logit gaps of
1–2 (e.g. C14: −0.58 vs +1.46…+1.95). Treat CogLoad within-corpus as an upper bound with an
unmeasurable confound, not as evidence of stress decoding. SAM40 has 3 relax trials per subject,
cannot align this way, and gives a believable 0.6837 — prefer it. The cross-corpus benchmark is
the only stress number here that a recording-level nuisance cannot produce.

**Pooled out-of-fold AUC is not the right summary.** Each fold's model has its own logit offset
(measured spread across MODMA folds: ~1.0), so ranking pooled OOF logits folds calibration
differences between five different models into the metric. Every row reports `per_fold_mean ±
per_fold_std` alongside pooled; quote the per-fold one.

**Depression is at chance in this pipeline.** MODMA per-fold subject AUC 0.412 ± 0.23 — four
folds at noise (0.63, 0.53, 0.56, 0.33) and one fold fully inverted (0.0000, n=10). That is far
below the plan's expected 0.65–0.80. Section 2b of REPORT.md carries the λ=0 control that
separates "the adversary destroyed it" from "the untuned config never learned it"; read it
before concluding anything about the architecture.

- **Every split is grouped on `subject`.** An AUC above ~0.85 means the split broke, not that
  the model got good.
- Report unit-level AUC (MODMA subject / SAM40 trial / CogLoad recording), not segment-level.
  Both are in the tables; segment-level is inflated by 50% window overlap within a unit.
- 3-class accuracy, if you ever compute it, is graded against the corpus-shortcut baseline in
  `results/legacy.json`, never against 0.333.

## Deviations from the plan, and why

| plan | built | reason |
|---|---|---|
| 0.5–45 Hz bandpass only | **50 Hz `iirnotch` (Q=30) then the bandpass**, uniformly on all three corpora | Order 4 at a 45 Hz corner leaves 50 Hz at ~0.3 amplitude. Measured on raw data, the 49–51 Hz band holds **92.3% of MODMA's** 0.5–64 Hz power, 28.1% of CogLoad's and 0.001% of SAM40's (already vendor-notched) — a five-order-of-magnitude corpus signature feeding straight into the domain probe. After notch+bandpass: 0.0004 / 0.0021 / 0.0000. |
| CogLoad primary = pooled recording-level AUC (N=120) | **mean within-subject AUC over 14 subjects**, pooled kept as secondary | Only 14 negative recordings exist corpus-wide, so a pooled interval cannot separate a working model from a broken one. Every subject has both a baseline and load recordings, so within-subject is well posed and yields 14 independent estimates. Quantised to 1/6 per subject (1 baseline vs ~6 load), hence a wide bootstrap — reported honestly rather than hidden. |
| CogLoad N=120 recordings | **N=104** | 16 files are byte-identical duplicates: all 15 `natural-*.txt` appear in both `Arithmetic_Data/` and `Stroop_Data/`, plus `Stroop/highlevel-14 == highlevel-15`. Deduped by content hash; leaving them in puts identical segments in train and test. |
| `filtfilt` (b, a) | `sosfiltfilt` | Same zero-phase operation; a direct-form order-8 with a 0.5 Hz corner at 250 Hz is numerically unstable. |
| variance gate `max_c Var(c) > 5·median_c Var(c)` **within a window** | same 5× threshold, referenced to **each channel's own median across its recording**, plus a 100× recording-level cap for a dead electrode | The within-window form cannot distinguish a transient spike from an electrode that ran hot all session, so it rejected *every* window of 4 subjects and deleted them silently — MODMA 53→50, CogLoad 15→14. The loss was non-random: it removed exactly the subjects with a frontal-electrode imbalance, on the task whose main feature is frontal asymmetry. The plan's stated intent ("eliminates single-electrode spikes") is transient, which is what the fixed form catches. All 108 subjects are retained. |
| — | `--smoke` on `experiments.py` | 2 folds × 2 epochs through every code path, into `results_smoke/`. Caught a NaN-propagation bug in LOSO scoring that would have surfaced ~40 minutes into the real run. |
| inner **4-fold** GroupKFold | one inner fold | Full inner CV costs 4× per outer fold to sharpen a threshold that only moves balanced accuracy — AUC, the primary metric everywhere, is threshold-free. Marked `ponytail:` in `experiments.py`. |
| `λ_p · L_adversary` with `−λ_p` reversal | λ in the GRL only, loss coefficient 1 | Applying λ in both places scales the adversarial gradient by λ². |
| fixed λ | λ ramped `2/(1+e^{−10p})−1` | A fixed λ fights the task objective from step one. |
| — | λ=0 control on every adversarial claim | Without it, "the GRL helped" is unfalsifiable. |

**Open, not fixed:** the 0.5–45 Hz order-4 bandpass leaves 50 Hz mains at ~0.3 amplitude
(measured). MODMA has a strong line spike, SAM40 is already vendor-notched — so mains is a
corpus signature that survives into the domain probe. The plan specifies the bandpass only; if
the anchor probe stays well above 0.333, add `scipy.signal.iirnotch(50)` in `bandpass()` and
rebuild. Flagged with a `ponytail:` comment at `preprocess.py`.
