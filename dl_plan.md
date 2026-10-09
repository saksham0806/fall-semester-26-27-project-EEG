# `DL_Without_Features/` — build plan & progress log

Raw-EEG deep learning with **no feature extraction**, aimed at a **corpus-blind 3-class**
model: raw `(6, 512)` in → stress / depression / healthy out, with no corpus input at
inference and an explicit measurement of whether the model is leaning on device signature.

Full design: `/home/Saksham/.claude/plans/right-now-we-have-memoized-pizza.md`

---

## Why this exists (the premise correction)

Raw-EEG DL already ran in this repo — `scripts/all3_deep.py` (6 architectures at
`nb_classes=3`), `run_models.py`, `all3_dann.py`. None call `extract_all_features`. Best
raw-only 3-class was **0.5631** (LSTM) against the **0.6267** corpus-ID shortcut.

New here:
1. **braindecode 1.6.1 architectures** — 61 installed, never tried. 17 verified to construct
   and forward at `n_chans=6, n_times=512, n_outputs=3, sfreq=128`.
2. **The corpus probe on the learned embedding.** Every existing all3 number is graded under
   subject-grouped CV where corpus is 97% recoverable in the *test* fold too, so a model can
   ride device signature and the accuracy column never says so. This track measures it.

## Honest expectation

Corpus-free 3-class is what this repo has already measured as failing: strip the device
signature → 0.3255 (chance); unseen amplifier → 0.3198 (chance); DANN → 8/8 configs failed.
Root cause is a data property — no corpus holds more than two of the three classes, so
`depression` is a boolean identity with `corpus == MODMA`. This track does not claim to fix
that. It builds the model and **instruments it so you can tell whether it worked.**

## Read the result on this grid

```
high acc + high probe  -> riding the device signature, not real
high acc + low  probe  -> the goal
low  acc + low  probe  -> invariance removed the signal (the known 0.3255 outcome)
```
3-class accuracy bar = **0.6267** (corpus-ID shortcut). Corpus-probe floor = **0.4217**
(majority corpus).

---

## Milestones

- [x] **M0** — plan agreed, `dl_plan.md` created
- [x] **M1** — `data.py`: all3 loader, per-subject vs per-segment norm, subject-grouped folds
- [x] **M2** — `core.py`: torch train loop, embedding hook, masked-KD loss, corpus probe, reporter
- [x] **M3** — `core.py` self-check passes (`python -m DL_Without_Features.core`)
- [x] **M4** — `benchmark.py` written; `--smoke` passes on local CPU
- [x] **M5** — `distill.py` written (per-corpus teachers → corpus-blind student); `--smoke` passes
- [x] **M6** — `README.md` + `RUN.md` entries
- [ ] **M7** — full `benchmark.py` on GPU → `Results/DL_Without_Features/benchmark.json`
- [ ] **M8** — full `distill.py` on GPU → `Results/DL_Without_Features/distill.json`
- [ ] **M9** — findings written back into this file and `README.md`

## Progress log

<!-- append one line per milestone: date · milestone · what landed · what it proved -->

- **2026-08-03 · M0** — design settled: braindecode + all3 3-class + per-corpus-teacher
  distillation + subject-grouped CV with an embedding corpus probe + per-subject raw
  standardization. `distilled/` explicitly ignored; this track is standalone and pure torch.
- **2026-08-03 · M1** — `DL_Without_Features/data.py`. Loader for `datasets/all3/`, `normalize()`
  with `segment` (pass-through control) vs `subject` (per-channel standardization inside each
  subject), `folds()` with a hard subject-leakage assert, `shortcut_baseline()`, and `subsample()`
  for `--smoke`. Self-check passes: subject-norm gives per-subject per-channel mean ~0 / std ~1,
  folds are subject-disjoint, shortcut reproduces 0.6267 exactly.
- **2026-08-03 · M2/M3** — `DL_Without_Features/core.py`. Torch fold trainer with inner
  subject-held-out early stopping, `final_layer` embedding hook, the masked KD loss, out-of-fold
  corpus probe, and a reporter writing the same JSON schema as `all3_diagnostic.report()`.
  Self-check passes on 9 asserts: `lam=0` reduces exactly to weighted CE; masked KD matches a
  hand-computed value; a teacher that never covers class 2 produces **zero gradient** on that
  logit (the property that stops the student re-learning "not-MODMA ⇒ not depressed"); the hook
  fires and returns 2-D for 7 of 8 default architectures, ATCNet falls back to logits as expected.
- **2026-08-03 · M4** — `benchmark.py`. 8 default architectures × 2 norm variants, incremental
  JSON write after every model so a killed session keeps its results. `--smoke` (18 subjects /
  ~1.1k segments / 2 folds / 2 epochs) passes on local CPU in 84 s.
- **2026-08-03 · M5** — `distill.py`. Per-outer-fold teachers with inner 3-fold subject-grouped
  CV producing honest out-of-fold soft targets, cached to `_teacher_soft_<backbone>_<norm>.npy`
  and reused across every (T, λ) config. Student is corpus-blind. `λ=0` control included.
  `--smoke` passes on local CPU in 219 s; the λ=0 vs λ>0 rows differ, so the KD term is live.
- **2026-08-03 · M6** — `DL_Without_Features/README.md` + `RUN.md` section.
- **2026-08-03 · bug found and fixed** — `core._inner_split` held out 1/8 of training subjects
  for early stopping unconditionally, which empties the training set when a split has one or
  two subjects. It surfaced in the distillation teachers and would also bite CogLoad (15
  subjects total) at higher inner-fold counts. Guard added in the shared function, not at the
  call site: fewer than 3 training subjects ⇒ no hold-out.
- **Smoke artifacts live in `Results/DL_Without_Features/_smoke/`**, never beside the real
  ones — the prediction slugs are identical and a smoke run would otherwise silently overwrite
  a real 15,728-length array with a 1,128-length one.

## Known caveat carried in, not fixed

`distilled/README.md` documents that **SAM40 is 128 Hz, not 256** — `combine_datasets.py:171`
decimates it anyway, so every SAM40 segment in `datasets/all3/X.npy` is time-stretched 2×.
This track reads `all3` as-is on purpose, to keep every number comparable to the existing 36
results. Fixing it is a full rebuild and a separate decision.

## Out of scope

Ramped-alpha DANN · SSL pretraining on the AMIGOS unlabelled pool · hyperparameter tuning ·
the high-density teacher from `distilled/`. Add whichever the probe grid points at, once the
baseline table exists.
