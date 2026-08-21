# OSF_Park_model — plans

Four documents. **Read `00` first and implement it before touching `01`–`03`** — it is the
only one that changes what the other numbers *mean*.

| plan | question it answers | status |
|---|---|---|
| [00_protocol.md](00_protocol.md) | how do we stop selection bias inflating the next result? | **gate — do this first** |
| [01_corrected_features.md](01_corrected_features.md) | does the null survive correct preprocessing? | run second |
| [02_residualization.md](02_residualization.md) | does EEG carry anything demographics don't? | run third |
| [03_hierarchical_stacked.md](03_hierarchical_stacked.md) | where exactly does the 4-class problem break? | run last |

## Corrections to the originating brief

The brief that motivated these plans is accurate on the data and on the strategy. Three of
its specifics do not survive contact with the code, and one would be a silent bug.

**1. "No class weighting" — already implemented.** `class_weight='balanced'` is set on SVM,
RandomForest and LogisticRegression (`models.py:32-34`), and the torch models weight their
loss via `compute_class_weight('balanced')` (`models.py:178`). The demographics baseline in
`run.py:55` is balanced too.

This makes the finding **stronger**, not weaker: anxiety recall 0.093 and stress recall 0.078
occurred *despite* balanced reweighting. Reweighting is not the missing ingredient, so no plan
below prescribes it as a fix. What *is* wrong is presentation — `run.py`'s `markdown()` sorts
by raw accuracy. Fix that in the same commit as plan 00:

```python
for r in sorted(rows, key=lambda r: -r["balanced_accuracy"]):     # was -r["accuracy"]
```

and move `balanced_accuracy` and `macro_f1` to the leading columns. Both are already computed
and stored in every JSON, so no re-run is needed to restate existing results.

**2. "log₁₀ the AB block → convert to relative power" — order is wrong.** Ratios of logged
values are not relative power. Relative power must be computed on **linear** power, then
transformed. See plan 01 §2, which specifies a centered log-ratio instead — CLR removes the
per-channel gain factor exactly, which is the skull-thickness/impedance nuisance the brief
correctly identifies.

**3. `arctanh(COH/100)` returns `inf`.** 24 cells are exactly 100.0. Clip before transforming
(plan 01 §3). Unclipped, this poisons standardization for every subject in the fold.

Confirmed as stated in the brief: AB spans 0.0258–531.8 (**20,645×**), median per-column skew
**3.37** → **0.18** under log₁₀; row **944**'s AB block is byte-identical to row 226
(Behavioral addiction). `networkx 3.6.1` and `scipy 1.17.1` are installed — graph metrics need
no new dependency. `statsmodels` is absent; plan 02 uses sklearn instead.

## Shared decisions

These apply to all three architectures. Deviating from any of them makes results
non-comparable across plans.

- **Quarantine rows 206 *and* 944.** `data.DROP_NO = [206, 944]`. Cohort becomes **n=527**
  (healthy 93 / depression 199 / anxiety 107 / stress 128); binary becomes **n=292**.
  Every number in `RESULTS_*.md` predates this and must be regenerated, not compared against.
- **Primary metrics: balanced accuracy and macro-F1.** Raw accuracy is reported but never
  ranked on. On 93/199/107/128 it rewards ignoring half the problem.
- **The bar is demographics, not chance.** Unchanged from `run.py`. Every table keeps the
  chance / majority / demographics header rows.
- **All transforms fit inside the training fold.** CLR, Fisher-z and standardization are
  per-fold; the normative model in plan 02 is fit on training-fold healthy controls only.
- **Report the recording-year probe** alongside, unchanged. It came back clean (0.2179 vs
  0.2443 majority), and that is a load-bearing negative result — it is what lets this cohort
  claim the corpus confound that caps `datasets/all3/` does not apply here.

## Implementation deviations

Two things were measured during implementation that changed the plans. Both are recorded in
full where they belong; noted here so they are not missed.

- **Proportional thresholding fragments 94% of the graphs** at the specified `d = 0.20`, and
  never reaches zero at any usable density. Replaced with MST-union thresholding, which gives
  0 fragmented graphs at the same density. See plan 01 §4.
- **Correlation is the wrong diagnostic for residualization** — it is scale-invariant, so it
  stays near 1.0 after a successful residualization. Use variance ratio. See plan 02.

## Expected outcome

Stated in advance so the result cannot be rationalised afterwards:

| | expectation |
|---|---|
| 4-class balanced accuracy | 0.42 – 0.48 |
| binary accuracy | 0.80 – 0.85 |
| **EEG increment over demographics** | **0.00 – 0.06** |

If plan 01 delivers correct relative-log power and graph-reduced coherence and the increment
is still zero, **that is the result** — a properly controlled null on a dataset where the
published literature reports 90%+. Most of those papers use random splits on these same 1,140
features with no demographics baseline. A careful null is the more useful contribution, and
plan 02 exists specifically to make it a publishable one rather than a failed experiment.
