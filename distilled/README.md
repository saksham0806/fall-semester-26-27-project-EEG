# `distilled/` — channel-agnostic 3-class EEG with high→low density distillation

Keeps the project's novelty (one architecture, three classes) while fixing the upstream flaw:
`scripts/combine_datasets.py` crushes every corpus to a 6-channel lowest common denominator,
discarding 123 of MODMA's 129 channels and 26 of SAM40's 32 *before any model sees the data*.
Here each corpus is read at native resolution, and a 6-channel **student** is distilled from a
high-density **teacher** so the deployable model still matches low-channel hardware.

Pure torch. Does not import `src/` (that pulls in TensorFlow) — only the raw datasets.

## Status

| piece | state |
|---|---|
| `montages.py` | ✅ done, selfcheck passes for all 4 corpora |
| `scripts/00_validate_premise.py` | ✅ done, **gate PASSED** — see below |
| `data.py`, `models.py`, `distill.py`, `evaluate.py` | ⬜ not written |

### Gate result — PASSED, but the teacher is 32 channels, not 128

`Results/distilled/premise_gate.json`, MODMA depression, 5-fold subject-grouped, EEGNet,
identical windows/folds/seed with only `n_chans` varying and the frontal 6 nested inside every
larger set:

| chans | subject AUC | segment AUC | time |
|---|---|---|---|
| 6 (baseline) | 0.6667 | 0.6641 | 3.7 min |
| **32** | **0.7371** | 0.7195 | 14.2 min |
| 128 | 0.6279 | 0.6172 | 46.0 min |

Gain **+0.0704** vs the required +0.05 → **PASS**.

**The shape of this result matters more than the verdict.** Performance is an inverted U:
32 channels beats 6, but 128 is *worse than 6*. With 53 subjects, full density overfits. This
independently reproduces `Results/phase2/channel_density.json`, which peaked at 16 ch (0.6825)
and fell back by 64 ch (0.6580) under a completely different (Riemannian) model family. Two
model families, same curve — treat it as a property of the dataset, not of either method.

Consequences for the build:
- **The teacher is ~32 channels.** Not 128. Also 3× cheaper to train.
- The plan's premise ("128 channels carry structure the frontal 6 do not") is true only up to a
  point; past ~32 channels the extra channels cost more in parameters than they add in signal.
- **A caveat worth keeping in view:** the best deep result here (0.7371) is still below the
  classical SVM on 81 handcrafted 6-channel features (**0.7716**, `Results/phase1/depression_cv.json`).
  Higher density helps *within* the deep family; it has not yet beaten the classical baseline.
  A 32-channel classical/Riemannian teacher is worth measuring before assuming a deep teacher.

Rerun with:

```bash
.venv/bin/python -u distilled/scripts/00_validate_premise.py     # ~64 min, CPU, needs ~5 GB free
```

Peak memory is at the 128-channel folds (`X[tr][:, chans]` copies the training split on top of
the 2.09 GB base array).

## Findings this work has already produced

**1. SAM40 is sampled at 128 Hz, not 256 — `combine_datasets.py` is wrong.**
The [SAM 40 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8749216/) states *"EEG was sampled at
128 SPS"* with 25-second trials; 128 × 25 = 3200 samples, exactly the array length on disk.
`combine_datasets.py:171` decimates `data[:, ::2]` on the stated assumption of 256 Hz, which
halves SAM40 to a true 64 Hz while labelling it 128 Hz.

Measured on 40 Relax trials, same channels: spectral peak **3.00 Hz → 5.50 Hz** (1.83×, theory
says 2×); beta power share 0.329 → 0.204, theta 0.138 → 0.181.

So **every SAM40 segment in `datasets/combined_data/` and `datasets/all3/` is time-stretched 2×**
and all its spectral features are computed at the wrong frequencies. This affects a quarter of
the corpus and plausibly inflates the 97% corpus probe, since a shifted spectrum makes SAM40
trivially identifiable.

`combine_datasets.py` is deliberately **not patched** — doing so would silently invalidate all 36
existing all3 results. `distilled/` reads from raw at the correct rate instead. Fixing the main
pipeline is a separate decision with a full rebuild attached.

**2. MODMA filename parsing drops 16 of 53 subjects if done naively.**
Filenames are inconsistent — `02010002rest 20150416 1017..mat`, `02010008_rest ....mat`,
`02010021 20150805 1730.mat.mat`. Splitting on `rest` or `_` matches 37/53; the leading
zero-padded digit run matches 53/53.

**3. MODMA `.mat` key detection can match `Impedances_0`**, which is also `(129, 1)`.
`combine_datasets.py:114` survives only because dict insertion order happens to favour the data
key. Here the guard also requires `shape[1] > 1000`.

## Electrode positions — two corpora are guesses

`montages.py` carries `assumed: bool` per corpus rather than a comment, because any result on an
assumed montage is only as good as the guess.

| corpus | n | positions | source |
|---|---|---|---|
| MODMA | 128 | **real** | GSN-HydroCel-128; cross-checkable against `chan_info_egi_128.mat` |
| SAM40 | 32 | **real** | `datasets/SAM40/Coordinates.locs`, real 10-20 labels |
| AMIGOS | 14 | **ASSUMED** | Emotiv EPOC layout is documented, but the channel *order* is inferred from `combine_datasets.py:231` |
| CogLoad | 8 | **ASSUMED, high risk** | no placement documented anywhere on disk; reverse-engineered from `combine_datasets.py:193` |

If CogLoad's assumed positions are wrong, spatial interpolation for that corpus is meaningless.
Report its results separately and discount them.

## What this cannot fix

`depression` is a boolean identity with `corpus == MODMA` — every depression segment is MODMA,
every stress segment is not. No channel count changes that. Expect **within-corpus** accuracy to
improve and the pooled 3-class number to stay confound-limited. Every script therefore reports
the corpus probe alongside its result, so the confound is tracked rather than hidden.

Also absent: **MODMA's audio modality was never downloaded** (zero `.wav/.mp3/...` in the repo),
so the published 97.53% EEG+audio results are not reproducible here without a fetch. And
**6.2 GB of MODMA 128-channel ERP task data** sits unused — the largest untapped asset on disk.
