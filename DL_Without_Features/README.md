# `DL_Without_Features/` — raw-EEG deep learning, corpus-blind 3-class

Deep models that consume the `(6, 512)` windows **directly**. No band powers, no differential
entropy, no covariances, no `extract_all_features` anywhere in the path.

Pure torch. Does **not** import `src/` — `src/proposed_model/__init__.py` pulls in the whole
TensorFlow stack. Results go to `Results/DL_Without_Features/`.

## What is actually new here

Raw-EEG DL already existed in this repo: `scripts/all3_deep.py` runs six architectures at
`nb_classes=3` (`Results/all3/models_deep.json`), plus `run_models.py` and `all3_dann.py`.
Best raw-only 3-class was **0.5631** (LSTM) against the **0.6267** corpus-ID shortcut. Two
things here that those did not do:

1. **braindecode 1.6.1 architectures.** 61 are installed and not one had been tried. 17 are
   verified to construct and forward at `n_chans=6, n_times=512, n_outputs=3, sfreq=128`.
2. **A corpus probe on the learned embedding.** Every existing all3 number is graded under
   subject-grouped CV where corpus is ~97% recoverable in the *test* fold too — so a model can
   ride device signature and the accuracy column will never say so. Here every row reports
   both.

## Read every row on this grid

```
high acc + high probe  ->  riding the device signature, not real
high acc + low  probe  ->  the goal
low  acc + low  probe  ->  invariance removed the signal (the known 0.3255 outcome)
```

| reference | value |
|---|---|
| chance | 0.3333 |
| majority class | 0.4217 |
| **corpus-ID shortcut — the bar** | **0.6267** |
| best flat 3-class in the repo (SVM on 81 features) | 0.6256 |
| best 3-class overall (hierarchical, uses corpus at inference) | 0.6820 |
| corpus-probe floor (majority corpus) | 0.4217 |
| corpus probe from 81 handcrafted features | 0.9161 |

**Never quote a 3-class accuracy without the 0.6267 bar next to it**, and never read the
accuracy without the probe.

## Honest expectation

A corpus-free 3-class model is what this repo has already measured as failing. Strip the
device signature → 0.3255 (chance). Apply a model to an unseen amplifier → 0.3198 (chance).
DANN, built for exactly this, failed in all 8 configs. The root cause is a property of the
data, not of the model: **no corpus contains more than two of the three classes**, so
`depression` is a boolean identity with `corpus == MODMA`.

This track does not claim to fix that. It builds the model and instruments it so you can tell
whether it worked.

## Files

| file | what it is |
|---|---|
| `data.py` | all3 loader · `normalize()` · subject-grouped `folds()` · `shortcut_baseline()` |
| `core.py` | fold trainer · embedding hook · masked KD loss · corpus probe · reporter |
| `benchmark.py` | N braindecode architectures × 2 normalizations |
| `distill.py` | per-corpus teachers → one corpus-blind 3-class student |

Each module has a self-check under `__main__`:

```bash
.venv/bin/python -m DL_Without_Features.data     # data invariants, ~10 s
.venv/bin/python -m DL_Without_Features.core     # 9 asserts incl. the KD mask, ~40 s
```

## Running

```bash
# harness proof — 18 subjects, 2 folds, 2 epochs; writes Results/DL_Without_Features/_smoke/
.venv/bin/python -u DL_Without_Features/benchmark.py --smoke     #  ~1.5 min, CPU
.venv/bin/python -u DL_Without_Features/distill.py  --smoke      #  ~4 min, CPU

# real runs (GPU box)
.venv/bin/python -u DL_Without_Features/benchmark.py             > bench.log 2>&1
.venv/bin/python -u DL_Without_Features/benchmark.py --all       # +9 more architectures
.venv/bin/python -u DL_Without_Features/distill.py               # --backbone auto reads benchmark.json
```

Use `-u` and redirect to a file — piping through `tail`/`head` buffers everything until exit
and makes a long run unmonitorable.

`benchmark.py` writes its JSON **after every model**, so a killed session keeps the rows it
finished. `distill.py` caches teacher soft targets to `_teacher_soft_<arch>_<norm>_<f>f<i>i.npy`
and reuses them across every `(T, λ)` config — delete that file if you change the teacher
definition, or the stale cache is silently reused.

## Two design decisions worth knowing before changing anything

**The KD mask.** Each teacher speaks about two classes only. The distillation term is masked
to those two and the student is renormalized over the same pair. An unmasked KL would force
`P(depression) = 0` on every SAM40/CogLoad segment, teaching the student
*"not-MODMA ⇒ not depressed"* — the corpus shortcut, re-derived from the loss function.
`core.py` asserts the property this exists for: a teacher that never covers class 2 puts
**exactly zero gradient** on the class-2 logit.

**Teacher soft targets are out-of-fold.** A teacher scoring its own training data emits
near-one-hot targets and distillation collapses to the hard labels. Teachers are fitted under
an inner subject-grouped CV inside each outer training split.

`λ = 0` (no distillation at all) runs as the control. Following the DANN lesson: a result
without its method-off control is uninterpretable.

## Anchor row

**braindecode `EEGNet` under `--norm segment` should land near 0.4306** — what the Keras
EEGNet scored on identical data and folds (`Results/all3/models_deep.json`). Far off means the
harness is wrong, not the model. **Above ~0.85 means the split broke**: a random rather than
subject-grouped split reports 0.9393 on this task purely from 50%-overlapping windows of the
same person. Every split here goes through `data.folds()`, which asserts subject-disjointness.

## Environment

Local torch is `2.13.0+cpu` — **no CUDA**; only `--smoke` is worth running here. On a GPU box
install torch **and torchaudio from the same CUDA index**, because braindecode hard-imports
torchaudio and a mismatched pair fails at import:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

`X.npy` is 193 MB, so the whole array stays resident comfortably.

## Caveat carried in, not fixed

`distilled/README.md` documents that **SAM40 is 128 Hz, not 256** — `combine_datasets.py:171`
decimates it anyway, so every SAM40 segment in `datasets/all3/X.npy` is time-stretched 2×.
This track reads `all3` as-is on purpose, to keep every number comparable to the existing 36
all3 results. Fixing it is a full rebuild and a separate decision.

Progress and findings are logged in `dl_plan.md` at the repo root.
