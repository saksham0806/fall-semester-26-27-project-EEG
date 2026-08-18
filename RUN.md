# RUN.md

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

