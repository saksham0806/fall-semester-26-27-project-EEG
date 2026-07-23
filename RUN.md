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

## all3 feature extraction

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
