# RUN.md

## Explainable AI — saved screening models

```bash
.venv/bin/python explainable_ai/test_explain.py
.venv/bin/python -u explainable_ai/explain.py
```

Writes channel/family/band permutation plots, local feature effects, EEG traces and a
report to `explainable_ai/results/`, using the existing fitted models without training.
These are descriptive explanations on sampled training data. See
[`explainable_ai/RUN.md`](explainable_ai/RUN.md) for inputs, outputs and limitations.

## Paper diagrams

```bash
.venv/bin/python diagrams/test_generate_figures.py
.venv/bin/python diagrams/generate_figures.py
```

Writes five publication figures to `diagrams/` as editable SVG and 300-DPI PNG files.
The generator reads the committed result JSON files; the transfer panel uses the two-draw
summary in `arch_metd/PROGRESS.md` and retains the Mumtaz montage-confound warning.

## Deployment — classify a recording as healthy / stressed / depressed

```bash
.venv/bin/python -u scripts/train_deployable.py          # ~8 min, writes models/*.joblib
.venv/bin/python predict.py recording.npy                # uses models/A_binary_pair.joblib
.venv/bin/python predict.py recording.npy --model models/B_flat_3class.joblib --fs 256
```

Input: `.npy` of shape `(6, n_samples)`, channels **in order** `Fp1, Fp2, F7, F3, F4, F8`,
any sampling rate (`--fs`, resampled to 128 Hz). Cut into 4 s windows at 50% overlap, each
classified, votes aggregated. `--channels` reindexes a different order.

| artifact | CV accuracy | note |
|---|---|---|
| `B_flat_3class.joblib` | **0.6256** | best available; one model, one call |
| `C_hierarchical.joblib` | 0.6079 | context router (resting vs task) → specialist |
| `A_binary_pair.joblib` | 0.3198 | **do not deploy** — see below |

**These accuracies will not transfer to new hardware.** All models are validated
subject-disjoint but **never device-disjoint**; corpus identity is ~97% recoverable from the
features. Two independent measurements put real-world accuracy on an unseen amplifier near
**chance (~0.32)**: removing the device signature drops 3-class accuracy to 0.3255, and
approach A — whose specialists are necessarily applied to corpora they never trained on —
scores 0.3198. Treat any output as a research signal, never a diagnosis.


## all3 experiments

Run one at a time — `free -h` first, never two trainings concurrently. Use `-u`, or a
25-minute run is unmonitorable. Redirect to a file rather than piping through `tail`.

```bash
# 1. Confound diagnostic + corpus probe (~2 min, reuses the Riemannian cache)
.venv/bin/python -u scripts/all3_diagnostic.py

# 2. Classical baselines + fusion (~10 min, CPU)
.venv/bin/python -u scripts/all3_models.py

# 3. Deep baselines at nb_classes=3 (~19 min). Export run.sh's LD_LIBRARY_PATH first
#    or this silently falls back to CPU.
.venv/bin/python -u scripts/all3_deep.py

# 4. Domain-adversarial models, 2 backbones x 4 alphas (~30 min, GPU)
.venv/bin/python -u scripts/all3_dann.py

# 5. Collect everything into Results/all3/TRACKS.md (seconds)
.venv/bin/python scripts/all3_tracks.py
```

**Read `Results/all3/TRACKS.md`, Track A first.** Pooled 3-class accuracy is not
interpretable alone — a classifier knowing only which amplifier recorded the segment scores
0.6267. Track A (within-corpus) is what a model actually learned about mental state.

`Results/SUMMARY.md` contains **no** all3 row — `consolidate_results.py` reads only
`Results/phase1/` and `phase2/`.

## Raw-EEG deep learning, no feature extraction (`DL_Without_Features/`)

braindecode architectures consuming the `(6, 512)` windows directly, plus a per-corpus-teacher
distillation into one **corpus-blind** 3-class student. Pure torch — does not import `src/`.

```bash
# self-checks, seconds
.venv/bin/python -m DL_Without_Features.data           # data invariants
.venv/bin/python -m DL_Without_Features.core           # 9 asserts incl. the KD mask

# harness proof: 18 subjects, 2 folds, 2 epochs -> Results/DL_Without_Features/_smoke/
.venv/bin/python -u DL_Without_Features/benchmark.py --smoke     # ~1.5 min, CPU
.venv/bin/python -u DL_Without_Features/distill.py  --smoke      # ~4 min, CPU

# real runs — GPU box; local torch is 2.13.0+cpu, no CUDA
.venv/bin/python -u DL_Without_Features/benchmark.py  > bench.log 2>&1   # 8 archs x 2 norms
.venv/bin/python -u DL_Without_Features/benchmark.py --all              # +9 architectures
.venv/bin/python -u DL_Without_Features/distill.py    > distill.log 2>&1
```

Input `datasets/all3/{X,y,subjects,corpus}.npy`, `X` is `(15728, 6, 512)`. Output
`Results/DL_Without_Features/{benchmark,distill}.json` plus out-of-fold `preds/` and `emb/`.
`benchmark.py` saves after **every model**, so a killed session keeps its finished rows.
`distill.py --backbone auto` reads `benchmark.json`; run the benchmark first or pass
`--backbone <Arch>`.

**Read accuracy and corpus probe together.** Every row carries a 3-class accuracy against the
0.6267 corpus-ID shortcut *and* a corpus probe on the model's own penultimate embedding
(floor 0.4217). Beating 0.6267 with a probe near 0.97 is the device shortcut in a new costume,
not a result. Anchor: braindecode `EEGNet [segment]` should land near **0.4306**, matching the
Keras EEGNet on identical folds; above ~0.85 means the split broke.

On a GPU box install torch **and torchaudio from the same CUDA index** — braindecode
hard-imports torchaudio and a mismatched pair fails at import. See
`DL_Without_Features/README.md` and `dl_plan.md`.

## all3 feature extraction — v2 (corrected preprocessing, 273 features)

```bash
.venv/bin/python -u scripts/extract_features_v2.py --smoke   # 600 segments, ~1 min
.venv/bin/python -u scripts/extract_features_v2.py           # full, ~15 min, CPU only
```

Writes `features_v2/all3/`, one `.npy` per block plus `combined.npy`:

| block | dim | content |
|---|---|---|
| `bandpower_rel` | 30 | relative band power, **denominator fixed to 0.5–45 Hz** |
| `de` | 30 | differential entropy |
| `faa` | 3 | frontal alpha asymmetry |
| `amplitude` | 12 | STD, RMS |
| `spectral_shape` | 24 | spectral entropy, 95% edge freq, 1/f slope + offset |
| `hjorth` | 18 | activity, mobility, complexity |
| `perm_entropy` | 6 | permutation entropy (order 3) |
| `coherence` | 75 | 15 channel pairs × 5 bands |
| `plv` | 75 | phase-locking value, 15 pairs × 5 bands |
| **`combined`** | **273** | all of the above, in that order |
| `riemannian_percorpus` / `_global` | 105 each | as v1 but zero-phase filtered |

**Why v2 exists:** v1 divides band power by total power over 0–64 Hz while the bands span
only 0.5–45 Hz. The unused gap holds ~88–98% of MODMA's power (50 Hz mains), 19–75% of
CogLoad's, and ~34% of SAM40's — corpus-specific contamination in every relative-power
feature. v2 notches 50 Hz and bandpasses 0.5–45 Hz zero-phase first.

The script prints a **corpus probe** at the end — whether the recording device is still
identifiable — against v1's 0.9161. That comparison is the point of the exercise.

`src/proposed_model/features.py` is **not** modified; v1 stays reproducible. Self-test:
`.venv/bin/python -m src.proposed_model.features_v2` (synthetic, seconds).

## all3 feature extraction — v1 (81 features)

```bash
.venv/bin/python -u scripts/extract_all3_features.py
```

Reads `datasets/all3/` (run `scripts/build_all3.py` first if missing), writes `features/all3/`:

| file | shape | what |
|---|---|---|
| `handcrafted.npy` | (15728, 81) | 30 relative band power + 30 differential entropy + 3 FAA + 6 spectral entropy + 6 STD + 6 RMS |
| `riemannian_percorpus.npy` | (15728, 105) | per-band covariance → per-corpus re-centering → tangent space |
| `riemannian_global.npy` | (15728, 105) | same, one global re-centering group |
| `y.npy` `corpus.npy` `subjects.npy` | (15728,) | labels (0 healthy / 1 stress / 2 depression), corpus id, subject id |
| `features.json` | — | dims, feature names, baseline caveat |

~2 min if `datasets/all3/_cache_riemann_*.npy` exist (they do — reused as-is); ~15 min otherwise.
CPU only, no GPU env needed.

Loading:

```python
import numpy as np
F = np.load("features/all3/handcrafted.npy")
y = np.load("features/all3/y.npy")
groups = np.load("features/all3/subjects.npy", allow_pickle=True)  # split on this
```

**Always split by `subjects`, and score 3-class accuracy against 0.6267 (corpus-ID shortcut),
never against chance.** See `datasets/all3/README.md`.

## Dataset structure reports

```bash
.venv/bin/python -u scripts/analyze_dataset_structure.py            # ~65 s, all 10 reports
.venv/bin/python -u scripts/analyze_dataset_structure.py --smoke    # 2 files/corpus, ~20 s
.venv/bin/python -u scripts/analyze_dataset_structure.py --only SAM40,AMIGOS
```

Scans every corpus on disk and writes one report per dataset to `datasets/structure/`:
`MODMA_128ch_resting`, `MODMA_128ch_ERP`, `MODMA_3ch_resting`, `SAM40`, `CognitiveLoad`,
`AMIGOS`, `WESAD`, `OSF_park`, `combined_data`, `all3`, plus a `README.md` index. Each
covers file layout, filename→label parsing, montage, sampling rate, label schema, and
per-channel min/max/variance/NaN/zero counts.

`OSF_park` is the odd one out — CSV feature tables, no time series — and its report also
recomputes a 4-class baseline (majority / demographics-only / band power / coherence), so
that section takes a few seconds of SVM fitting rather than just reading files.

Stats are accumulated streaming (count/sum/sumsq/min/max per channel), so the 17 GB of
WESAD never lands in RAM at once. Peak RSS ~3 GB (the one WESAD pickle it loads). Heavy
corpora are file-sampled — 12 of 53 MODMA `.mat`, 6 of 53 ERP `.raw` — and every report
prints its own sample size. SAM40, CogLoad and AMIGOS are read in full.

Generated, not live: **re-run after adding data or changing a loader**, or the reports go
stale silently. Do not hand-edit them.

⚠️ `datasets/.gitignore` is `*`, so these reports are **not version-controlled**. Add
`!structure/` and `!structure/*.md` to that file if you want them tracked.

## OSF_Park 4-class — healthy / depression / anxiety / stress (`OSF_Park_model/`)

The one cohort in this project where all four target classes share a single feature schema,
so class is **not** confounded with recording corpus the way `datasets/all3/` is. Tabular
features (114 band powers + 1026 coherences), **not** raw EEG — nothing here touches the
`(6, 512)` window pipeline.

```bash
# self-checks, seconds each
.venv/bin/python -m OSF_Park_model.data     # 528 / 293 rows, band-major round-trip, symmetry
.venv/bin/python -m OSF_Park_model.models   # 8 models forward; coherence bias wired in

# harness proof: 1 seed, torch models at 8 epochs (~40 s)
.venv/bin/python -u -m OSF_Park_model.run --task 4class --smoke

# real runs: 3 seeds x 5-fold StratifiedKFold, 11 configs, CPU
.venv/bin/python -u -m OSF_Park_model.run --task 4class    # n=528, 4 classes
.venv/bin/python -u -m OSF_Park_model.run --task binary    # n=293, healthy vs depressed
```

Writes `Results/OSF_Park/<task>.json` after **every** model (a killed run keeps its finished
rows), out-of-fold probabilities to `Results/OSF_Park/preds/`, and `OSF_Park_model/RESULTS_<task>.md`.

⚠️ **The bar is demographics (~0.43), not chance (0.25) and not majority (0.377).** Age, sex,
IQ and education alone beat every EEG configuration measured so far. `run.py` prints all three
baselines plus a **recording-year probe** before any model, and will not write a table without
them. Never quote an accuracy from this track alone.

⚠️ **This file has no subject-identifier column** — `no.` indexes rows, not people, across
2011–2018. `StratifiedKFold` is grouping by necessity, not by proof. Expected anchors: SVM on
EEG ≈ 0.43, demographics ≈ 0.43. **Above ~0.75 means the split broke.**

Full detail, including the two silent data-integrity fixes, in `OSF_Park_model/README.md`.


---

## METD — protocol-gated classifier (`arch_metd/`)

Spec: `arch_metd/metd_plan.md`. Schedule and gates: `arch_metd/metd_phases.md`.
Rooted outside `src/` on purpose — `import src.*` pulls in the whole TensorFlow **and**
torch stack via `src/proposed_model/__init__.py`, which a corpus loader has no use for.

**Phase 0 — contract and registry.** `arch_metd/contract.py` is the only place a dataset
path, native sampling rate or channel index is written down; everything downstream reads
the registry rather than restating a path.

```bash
.venv/bin/python -m arch_metd.contract     # seconds, read-only
```

Asserts, in order: all five corpus directories plus `subjects_information_*.xlsx` and
`subject-info.csv` exist; the §4 array contract holds on a synthetic 5-domain batch
(dtypes, `(N, 6, 512)` float32, finite, row alignment across all nine fields, `y_domain ⊆
{0..4}`, `anchor == (y_dep==0)|(y_str==0)`, no segment with both labels, protocol matching
the corpus); and that the contract **rejects** five known bugs — both-labels, stale anchor,
reserved domain 5, a dropped row in one field, non-finite `X`.

Expected output:

```
paths OK — 5 corpora, subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx, subject-info.csv
contract OK — 10 synthetic segments, domains [0, 1, 2, 3, 4], 5 anchors
rejects OK — 5 known bugs caught
```

A `FAIL — contract accepted:` line means the contract stopped catching something it used
to catch; that is more serious than an assertion firing.

**Phases 1-2 — loaders for all five corpora.** Continuous `(6, n)` arrays at native
rate; no filtering and no windowing (that is phase 3).

```bash
.venv/bin/python -u -m arch_metd.data.loaders    # ~5 min, reads ~5 GB, read-only
```

```
sam40   OK — 480 trials, 40 subjects, 5280 pre-gate windows, 120 relax / 360 task
modma   OK — 53/53 subjects resolved, 24 MDD / 29 HC, 7959 pre-gate windows
cogload: 120 files -> 103 recordings (15 duplicate copies merged, 2 cross-subject duplicates dropped)
cogload OK — 103 recordings, 15 subjects, 4441 pre-gate windows, 15 natural / 88 load
mumtaz: 120 EC/EO files -> 115 recordings (5 duplicate copies merged, 0 ambiguous dropped)
mumtaz  OK — 115 recordings, 61 subjects (32 MDD / 29 HC), 17054 pre-gate windows
          incomplete sessions (7): mumtaz_H12=EO, mumtaz_H18=EO, mumtaz_H25=EC, mumtaz_MDD12=EO, mumtaz_MDD16=EO, mumtaz_MDD4=EO, mumtaz_MDD7=EC
eegmat  OK — 72 recordings, 36 paired subjects, 4266 pre-gate windows
phase 1-2 gate: pass
```

⚠️ **`5280` is the anchor.** 3200 samples at a true 128 Hz gives 11 windows per trial.
If SAM40 reports 5 windows / 2,400 total, the `data[:, ::2]` decimation bug is back and
every SAM40 segment is time-stretched 2x — stop and fix it before phase 3.

**CogLoad deduplication corrects `metd_plan.md` §2.3.** Content hashing found two
distinct problems, not one: `natural-N.txt` is byte-identical between the two task
folders for all 15 subjects (recorded once, copied into both), and
`Stroop_Data/highlevel-{14,15}.txt` are identical across two different subjects. The
rule in `cogload.py` is content-based — same bytes under one subject collapse to one
recording, same bytes under two subjects drop both — giving **103** recordings, not 120
and not the ~105 the spec estimates. Keeping the duplicated baselines would double the
negative class and put identical windows on both sides of a fold, which subject-grouping
cannot catch.

**Depression N is 114 subjects** — MODMA 53 (24 MDD / 29 HC) + Mumtaz 61 (32 MDD /
29 HC), against the spec's ~117 estimate. Seven Mumtaz subjects have only one of EC/EO;
they are kept and logged, not dropped. §7.1 stands: every depression number ships with
a bootstrap CI, and at this N the CI is roughly ±0.13 even at AUC 0.80.

**Duplicate handling is a three-case rule** (`loaders/__init__.py:dedupe`), and the
distinction is load-bearing rather than pedantic:

| case | example | action |
|---|---|---|
| same bytes, one subject | CogLoad `natural-N` in both task folders; Mumtaz's two `H S15 EO` copies | keep one |
| same bytes, subjects identical throughout | Mumtaz `H S27`/`H S30`, `MDD S33`/`MDD S34` — duplicated across **both** sessions | merge into one id, keep one copy |
| same bytes, subjects otherwise distinct | CogLoad `Stroop_Data/highlevel-{14,15}` | drop both — unrecoverable |

Merging matters because a subject retained under two ids defeats grouped CV **silently**:
the same person sits in train and test and no split assertion can see it. `_common()` in
the gate now hashes every retained recording and asserts identical bytes never appear
under two subject ids, across all five corpora.

**Phase 3 — preprocessing, K calibration, build.** Writes the arrays every later phase
reads. Re-runnable; the output is overwritten, never merged.

```bash
.venv/bin/python -u -m arch_metd.data.preprocess           # synthetic self-check, seconds
.venv/bin/python -u -m arch_metd.data.build --calibrate-k  # ~2 min, MODMA only, writes nothing
.venv/bin/python -u -m arch_metd.data.build                # ~12 min, writes 485 MB
.venv/bin/python -u -m arch_metd.data.verify               # ~2 min, §11 on the built arrays
```

Output is `datasets/metd/metd.npz` — one file, so row alignment across the nine fields
cannot drift, and no `allow_pickle` anywhere. `datasets/` is gitignored wholesale, so
485 MB cannot be staged by accident. Load it with `build.load()`, which validates the
§4 contract on the way in.

**`VAR_GATE_K = 10.0`, calibrated on MODMA** (zero dropout and zero NaN by measurement):
rejects 1.43% of its 7,959 windows, inside the 1-2% target. Ratio distribution is
p50 = 1.60, p90 = 3.86, p99 = 11.01, so K = 10 sits at 6.3x the median — clear of the
3-5x Fp1/F3 spread normal frontal EEG shows because blinks load on the prefrontal pair.
The old K = 5 rejects 6.89%. **K is a property of the filter chain and the data
together — re-run `--calibrate-k` after any change to either.**

**`BAND_ORDER = 8`, not the 4 that §3.1 specifies.** Measured: order 4 leaves MODMA a
2.3x mains peak at 1.24e-3 of total power, failing §11. A 45 Hz corner sits only 11%
below 50 Hz, so even order 8 passes ~18% of the power there — the notch does the
rejecting and the order decides how much transition-band noise rides along. This
independently reproduces the order-8 result in `CLAUDE.md`'s filter-design note.

| corpus | recordings | windows | kept | flatline | variance | kept % |
|---|---:|---:|---:|---:|---:|---:|
| modma | 53 | 7959 | 7845 | 0 | 114 | 98.6 |
| sam40 | 480 | 5280 | 5273 | 0 | 7 | 99.9 |
| cogload | 103 | 4441 | 4168 | 61 | 212 | 93.9 |
| mumtaz | 115 | 17054 | 16897 | 0 | 157 | 99.1 |
| eegmat | 72 | 4266 | 4266 | 0 | 0 | 100.0 |
| **all** | 823 | 39000 | **38449** | 61 | 490 | 98.6 |

⚠️ **CogLoad is the divergent row (93.9% vs EEGMAT's 100%) and it is a finding.** Its
raw signal is 99.9999% DC — a ~31,000-count pedestal with tiny AC on top — so once the
1 Hz high-pass removes the pedestal, the residue is wideband amplifier and EMG noise,
not EEG rhythms. It holds no mains peak (0.37x its stopband neighbours) but 8x EEGMAT's
above-45 Hz power after gating, and 29x before it. **That is a corpus signature the
phase 6 probes must be read against.** No filter reaches it: order 10 at a 40 Hz corner
still leaves 6.2e-3 pre-gate, and lowering the corner *widens* the corpus disparity to
80x because MODMA benefits more than CogLoad does. Per-corpus filtering would fix the
number and make preprocessing itself the signature, which §3.2 forbids.

**Baselines — recomputed from the rebuilt arrays, never hardcoded** (`results/baselines.json`):

| baseline | value | use |
|---|---:|---|
| **corpus dummy, 3-class** | **0.6109** | the bar every headline number is quoted against |
| majority class, 3-class | 0.4650 | |
| majority domain | 0.4395 | target for §8.1's full-set probe |
| depression majority | 0.5085 | n = 24,742 segments |
| stress majority | 0.6135 | n = 13,707 segments |

`0.6109` replaces the old pipeline's 0.6267/0.6755. A classifier knowing only which
corpus a segment came from, with zero EEG understanding, scores it.

`verify.py` asserts §11 on the built arrays: no 50 Hz peak **and** <1e-3 absolute in
every corpus, per-segment |mean| ≤ 2e-8 and |std−1| ≤ 1.2e-7, label masking per corpus,
subject class counts against the source spreadsheets (MODMA 24/29, Mumtaz 32/29), and
SAM40 edge/middle amplitude at 1.054 — the reflect padding leaves no filter transient.

**Phase 4 — splits.** Written before any model, because leakage is a splitting bug and
it is provable on fold indices alone.

```bash
.venv/bin/python -u -m arch_metd.train.folds     # ~30 s, assertions only, writes nothing
```

```
  outer 0: train  27656 / test  6527 segments,  32 test subjects, domains [0, 1, 2, 3]
  ...
  outer 4: train  27127 / test  7056 segments,  35 test subjects, domains [0, 1, 2, 3]
  cogload LOSO: 15 folds
  stress_sam40cog_to_eegmat    train domains [1, 2] -> test [4]  (  9441 /  4266 segments)
  stress_eegmat_to_sam40cog    train domains [4] -> test [1, 2]  (  4266 /  9441 segments)
  dep_modma_to_mumtaz          train domains [0] -> test [3]  (  7845 / 16897 segments)
  dep_mumtaz_to_modma          train domains [3] -> test [0]  ( 16897 /  7845 segments)
phase 4 gate: pass
```

**Folds are built per corpus and unioned, never over the pooled array** — for both the
outer and the inner level. Pooled `GroupKFold` balances group sizes, not corpora, so a
fold can come out with no MODMA in it at all; the depression head then trains or
early-stops on nothing while the run reports a plausible-looking number. MODMA and
Mumtaz additionally use `StratifiedGroupKFold` on the class label, since depression is a
per-subject trait and plain 5-fold on 53 subjects can hand you a fold with almost no MDD.
SAM40 and CogLoad cannot be stratified that way — every one of their subjects holds both
classes, so no subject-level label exists.

Asserted: no `subject_id` on both sides of any split at either level, inner validation
never touches an outer test subject, every outer fold spans all four training corpora,
EEGMAT in **zero** folds on either side, `y_domain` never 5, and MODMA/Mumtaz never on
the same side of a depression transfer run — without that last one, "transfer" is just
within-corpus CV wearing a different name.

**Phase 5 — model and training.** Runs on the GPU workstation; local torch is CPU-only.

```bash
.venv/bin/python -m arch_metd.models.mtdann        # architecture self-check, seconds, CPU
.venv/bin/python -u -m arch_metd.train.loop --smoke   # 1/20 of the data, 3 epochs, CPU
./arch_metd/run_gpu.sh                             # the real run — ON THE WORKSTATION
```

`run_gpu.sh` is the whole workstation workflow in one script: it md5-checks the
transferred `metd.npz`, verifies the GPU, builds the image if missing, runs the gates and
the smoke run, then the full nested CV — see `GPU_Run.md` §7. **Only `metd.npz` (485 MB)
needs to go up, never the 5 GB of raw corpora**, which is why the loader imports in
`build.py` and `verify.py` are lazy: `load()` needs numpy alone.

The architecture self-check asserts the four things that fail silently:

- **non-anchor segments put exactly zero gradient on the adversary** (§5.1). A marginal
  adversary over all segments is asked to erase MODMA-MDD vs SAM40-task — different
  classes in different domains — so erasing the domain erases the label.
- **the GRL flips gradient sign**, and λ = 0 passes none at all, so the λ = 0 control is
  genuinely method-off.
- **masked BCE divides by the valid-sample count, not the batch size** — otherwise each
  head's gradient scales with how much of the batch happened to carry its label, which
  moves under corpus-balanced sampling.
- **the anchor weighting** gives every present corpus equal total mass, since MODMA's HC
  segments outnumber SAM40's relax windows by an order of magnitude.

⚠️ **λ crosses 0.5 at p = 0.11, not near the middle** — it is 0.905 by 30% of training.
So §5.3's "lock early stopping until λ ≥ 0.5" covers only the first 11% of the budget. It
prevents shipping a completely un-adapted model; it is not evidence that adaptation had
time to work. `lambda_at_stop` is logged per fold for exactly this reason and is the first
thing to read in `cv.json`.

**Phase 5 results — read `aggregate.py`, never `cv.json`.**

```bash
.venv/bin/python -u -m arch_metd.eval.aggregate    # §7 units + bootstrap CIs
```

`cv.json` carries **segment-level** AUCs, which §7 and §12 item 7 say are never primary.
`aggregate.py` reduces out-of-fold predictions to the unit the label actually varies over
by **logit median** — depression to subject, SAM40 to trial, CogLoad to within-subject,
EEGMAT to §7.3's paired sign test — and attaches a 1,000-iteration bootstrap CI.

| result | subject-level AUC | 95% CI | |
|---|---:|---|---|
| depression / **MODMA** | 0.5647 | [0.416, 0.721] | **CI crosses chance** |
| depression / **Mumtaz** | 0.9386 | [0.866, 0.992] | above §13's expected 0.70-0.85 |
| depression / pooled *(secondary)* | 0.8051 | [0.722, 0.882] | **do not quote** |

⚠️ **The pooled 0.8051 is a pooling artifact**, exactly as §7.1 predicted. It sits between
a corpus at chance and one at 0.94, and describes neither. Report the two separately.

⚠️ **Mumtaz's 0.9386 has a batch confound.** Recording length alone predicts the class
there at AUC 0.709 (MDD 270 ± 54 segments vs HC 285 ± 48); MODMA's equivalent is 0.514.
Session lengths that differ by class mean the two groups were acquired under systematically
different conditions, which is a signature the model can use. §6.2 anticipated the pattern:
when the two depression corpora disagree this sharply, neither number stands alone.

**Three fixes landed after the first run, so its results are superseded — re-run to use them:**

- **Checkpoint selection is gated on λ ≥ 0.5** (`LAMBDA_FLOOR`), not just early stopping.
  Measured on the first run, the best validation epoch was 4-8 with λ = 0.20-0.38 in 3 of
  5 folds, so the shipped checkpoint was un-adapted while `lambda_at_stop` read a
  reassuring 0.74-0.82. The lock delayed stopping; it did not gate what got selected.
- **Predictions now carry `subject_id`, `trial_id` and `condition`.** Without them,
  subject-level aggregation had to reconstruct identity from the fold indices, and that
  only worked for two of four corpora — see below.
- **`build.py` writes a fingerprint** (`results/build_info.json`, and `--fingerprint`),
  stamped into `cv.json` and printed by `run_gpu.sh`. The first run's arrays turned out
  **not** to be the local ones: SAM40 rows were reordered in 5/5 folds and CogLoad in 2/5,
  which is invisible in every summary statistic and cost a long forensic detour.
  `run_gpu.sh` no longer md5s the file — two honest rebuilds legitimately differ at
  floating-point gate boundaries — it prints the fingerprint instead.

`aggregate.py` still recovers identity for predictions saved before that fix, but only for
corpora whose label sequence matches the split **exactly**, and it drops any corpus that
fails in even one fold. That is why the table above has no stress rows: SAM40 and CogLoad
were excluded, 9,441 of 34,183 segments.

**Phase 5 controls (§8.3) — main plus three, in one session.**

```bash
.venv/bin/python -u -m arch_metd.train.loop --smoke --all-modes   # ~1 min CPU, mechanics only
./arch_metd/run_gpu.sh                                            # runs --all-modes
```

| mode | what it is | reads as |
|---|---|---|
| `main` | the model | the result |
| `lambda0` | λ fixed at 0, adversary off | **the** comparison — without it "adaptation helped" is unfalsifiable |
| `permuted` | labels shuffled within corpus | must collapse to 0.50 |
| `randbb` | frozen random backbone, heads trained | separates learned representation from input artifact |

Outputs are tagged per mode (`cv_lambda0.json`, `preds_fold*_lambda0.npz`, …) so nothing
overwrites the main run — the same separation `DL_Without_Features/` needs for its smoke
output, and for the same reason.

**Trained weights are now saved** (`model_fold*<tag>.pt`). §8.1's frozen domain probe and
§8.2's silhouette both freeze a *trained* backbone; the first run kept weights in memory
only, which left phase 6 unable to start without retraining.

`permute_within_corpus` shuffles labels at the **aggregation unit** — subject for the
depression corpora, trial for the stress ones — not at the segment. Shuffling segment
labels inside a subject would leave that subject's mean label almost unchanged, so a
subject-level metric would barely move and the control would pass while testing nothing.
It asserts the unit-level label multiset survives; the segment-level balance legitimately
shifts, because units differ in length by 3-4x (Mumtaz 93-336 segments each).

Verified on the smoke run: `randbb`'s backbone is bit-identical to its initialisation
while its heads moved, `main`'s backbone moved, `lambda0` reports λ = 0.0 at every stop,
and predictions carry `subject_id` / `trial_id` / `condition`.

## Phase 6 — diagnostics (§8)

```bash
.venv/bin/python -u -m arch_metd.eval.probes      # ~12 min CPU, frozen probes + silhouette
.venv/bin/python -u -m arch_metd.eval.controls    # seconds, cross-head extrapolation
```

**§8.1 frozen domain probes.** The backbone is frozen and an **independent** logistic
regression is fitted on `z`, subject-disjoint. The adversary's own head accuracy is not a
probe (§12 item 9) — it is one side of a min-max game being actively suppressed.

| mode | anchor probe (chance 0.250) | full-set probe (majority 0.4943) | silhouette |
|---|---:|---:|---:|
| main (λ=1) | **0.3970** | **0.5626** | −0.0028 |
| lambda0 (λ=0) | 0.4761 | 0.6053 | +0.0025 |
| randbb (random) | 0.4649 | 0.5997 | +0.0033 |

Read together, three things follow:

- **The adversary does something real**: −0.079 on the anchor probe and −0.043 on the
  full set against its own λ=0 control. That is the method working in the direction it
  was designed to work.
- **It does not reach invariance.** 0.3970 against a 0.250 balanced chance means corpus
  identity stays substantially recoverable *from exactly the embeddings where invariance
  was enforced*. The full-set probe stays above the majority-domain baseline too.
- **Task training alone adds no domain separation**: λ=0 (0.4761) and a never-trained
  random backbone (0.4649) are within noise of each other. The domain signal is inherent
  in the input, not manufactured by fitting the task.

**§8.2 manifold gate: FAILED** against the criterion pre-registered in
`results/preregistration.md` before measurement. The caveat is recorded there rather than
used to revise the criterion: all three silhouettes sit within ±0.004 of zero, so the
metric carries no signal on this data in any condition. Corpus structure here is diffuse,
not compactly clustered — §8.2's absolute 0.3 would have passed trivially at the same
time. The probes, not the silhouette, are what measure the confound.

**§8.4 cross-head extrapolation**, compared against in-domain positives from the same
head — never against zero, since a head that outputs one value for everything would
otherwise look like a head that fails to extrapolate.

| head | in-domain pos | in-domain neg | separation | out-of-domain |
|---|---:|---:|---:|---|
| stress | 0.808 | 0.782 | **+0.026** | 0.812 — **above** its own positives |
| depression | 0.755 | 0.263 | +0.492 | 0.364 — near its negatives |

⚠️ **The stress head is near-constant and fires on MODMA at 0.812**, which contains zero
stress segments. This reproduces the cross-device failure already documented in
`CLAUDE.md` (the legacy model labelled 90% of MODMA "stressed" at confidence 0.839). §0's
protocol gate is what contains it — MODMA is `resting`, so the stress head is never
consulted there — but the head has no calibrated meaning off its training corpora, and a
cross-head argmax would be catastrophic. That is why §9 forbids one.

The depression head behaves correctly by contrast: out-of-domain segments land near its
negative distribution, not above its positives.

## Phase 6.5 — fixes that phase 6 forced

```bash
.venv/bin/python -u -m arch_metd.train.folds       # asserts row-order invariance
.venv/bin/python -u -m arch_metd.models.mtdann     # asserts class weighting works
./arch_metd/run_gpu.sh                             # re-run, then re-check probes
```

**1. The split is now row-order invariant.** `folds.py` computes each corpus's split on a
canonically ordered view (sorted by subject id, ties by position) and maps back. Before
this, plain `GroupKFold` broke size ties in encounter order and every SAM40 subject has
exactly 12 trials, so two builds of the same corpora differing only in row order put
**4-10 SAM40/CogLoad subjects in different folds**. Both splits were valid; they were not
the same split, so no stress number was reproducible. `folds.py` now asserts that
shuffling the input rows leaves all 169 subject-to-fold assignments identical, over three
permutations.

**2. The heads are class-weighted — and the stress head still does not work.** The gate
FAILED: separation reached +0.044 against a 0.10 bar. Calibration was fixed (out-of-domain
medians moved from 0.812, *above* the in-domain positives, to 0.520, below them) but
**calibration on a head with no discrimination is calibration to chance.** The head was
miscalibrated and weak; only the first was a bug. Measured on the pre-fix run:

| | stress | depression |
|---|---:|---:|
| training positive rate | **0.776** | 0.492 |
| final-layer bias | **+0.249** | +0.025 |
| logit, positives / negatives | +1.375 / **+1.174** | +0.892 / −1.247 |
| separation | **+0.201** | +2.139 |

Both stress logit distributions sat **above zero** — the head predicted "stressed" by
default and ranked weakly on top of that offset, which is why it emitted ~0.81 on MODMA, a
corpus with no stress in it. Corpus-balanced sampling equalises **corpora, not classes**,
so it never addressed this. `masked_bce` now takes `pos_weight`, computed from the
**training split only** (0.287 stress, 0.929 depression) — computing it over all data
would leak the test fold's class balance into the loss.

⚠️ **EEGMAT is 25.3% positive against training's 77.6%.** A head biased to the training
prior cannot transfer to it, so this fix is a precondition for P7, not a refinement.

**3. Mumtaz is confounded with its acquisition setup.** Full detail in
`results/mumtaz_audit.md`. The EDF channel count predicts diagnosis at **0.8889**
(baseline 0.5238): 23 signals is 24 H / 1 MDD, 21 signals is 6 H / 32 MDD. Montage alone
scores **0.8685** against the model's 0.9418. Within a single montage the model still
reaches 0.9107, so it is not reading setup alone — but neither subgroup is powered to
settle the residual (4 HC in one, 3 MDD in the other), and **no adequately sized subgroup
holds both classes on one montage**. Quote Mumtaz's AUC only alongside that table.

**4. §8.2 is replaced.** The silhouette gate could not adjudicate — all three modes within
±0.004 of zero. The frozen probe, which separated them cleanly, is now the gate:
`anchor_probe(main) < anchor_probe(lambda0) - 0.05`, recorded in
`results/preregistration.md` **before** the fixed pipeline was re-run. Absolute probe
values are reported alongside the verdict, so a passing margin on a still-corpus-separable
representation cannot be quoted as invariance.

## Phase 7 — cross-corpus transfer and the report

```bash
.venv/bin/python -u -m arch_metd.train.loop --smoke --transfer   # ~1.5 min CPU, mechanics
./arch_metd/run_gpu.sh                                           # CV + controls, then transfer
.venv/bin/python -m arch_metd.eval.report                        # assembles arch_metd/REPORT.md
```

Within-corpus CV measures **decodability only** — corpus identity is present in the test
fold too, so a model can ride the device signature and no accuracy column will say so.
The four §6.2 runs are the only device-disjoint measurements in the project.

⚠️ **Three of the four transfer directions train on a single corpus**, so the adversary
has no second domain to align against. This is not a detail: with one domain the
cross-entropy over five domain logits still has a gradient, the GRL reverses it, and the
backbone gets pushed to make a trivial one-class prediction *harder* — measured, the
single-corpus depression transfer trained to different weights at λ=1 than at λ=0 with
nothing to align. `domain_loss` now returns zero when the anchor set holds fewer than two
domains, so those runs are λ-independent by construction (verified: bit-identical weights,
max |Δp| = 0.000000). **Only `stress_sam40cog_to_eegmat` can test domain adaptation.**

`run_gpu.sh` runs transfer **after** the CV deliberately: the §6.5 gates are read off the
CV output, and a stress head still pinned near 0.8 would make every transfer number
unanswerable — better to see that in the same log.

`eval/report.py` assembles `arch_metd/REPORT.md` from `results/*.json`. Sections whose
inputs are missing are printed as *not run* rather than omitted, so the report states what
has not happened as plainly as what has.

## Running the whole pipeline locally

```bash
./arch_metd/run_local.sh --check      # preflight only, starts nothing
./arch_metd/run_local.sh --smoke      # smoke CV + smoke transfer, ~6 min
./arch_metd/run_local.sh              # everything: gates, CV, transfer, analysis
./arch_metd/run_local.sh --cv         # gates + CV only
./arch_metd/run_local.sh --transfer   # transfer only
./arch_metd/run_local.sh --analyse    # no training: aggregate, probes, controls, report
```

No Docker and no GPU — it runs in the project `.venv`. `run_gpu.sh` remains the
workstation path; this is the same stages against CPU torch, plus the analysis steps that
`run_gpu.sh` leaves for afterwards.

⚠️ **Local torch is `2.13.0+cpu`, so this is hours, not the ~12 minutes the GPU box
takes** — roughly 4-5 h for the four CV modes, 2-4 h for the 16 transfer runs, ~15 min for
analysis (the frozen probes dominate). `--cv` and `--transfer` exist so the work can be
split across sessions; run it under `tmux`.

**Preflight refuses to let you confound the robustness comparison.** It prints the array
fingerprint and the split hash, and compares the fingerprint against the kept draw in
`arch_metd/results_split_A/`. They currently differ — Run A was built on the workstation
(`3d017f08c581c5fd`), this machine has `59cc093f949968b0` — so a local run changes **both**
the arrays and the split, and a result that moves between the two draws could not be
attributed to either. Copy the workstation's `metd.npz` down first, or treat a local run
as an independent third draw and do not diff it against Run A.

It also prints `free -h` and warns below 2.5 GB available, and logs everything to
`arch_metd/results/run_local.log`.
