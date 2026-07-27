# RUN.md

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
