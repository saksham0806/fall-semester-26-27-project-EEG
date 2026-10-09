# Plan 03 — hierarchical decomposition, stacking, and the other 417 subjects

Three independent changes, each targeting something visible in the measured results. Run last:
each depends on plan 01's corrected features, and the third has a caveat that may cancel it.

New module: `OSF_Park_model/hierarchical.py`.

## A. Hierarchy — because two classes are never predicted at all

`SVM [EEG]` on 4-class produced this:

```
                pred:  healthy  depression  anxiety  stress
   healthy  (94)          42        52         0        0
depression (199)          16       183         0        0
   anxiety (107)          12        95         0        0
    stress (128)           9       119         0        0
```

Zero anxiety and zero stress predictions, *with* balanced class weighting already active. A
flat 4-way boundary is not merely inaccurate here — it is degenerate. Decompose:

```
stage 1   healthy (93)  vs  ill (434)
stage 2   among the ill:  mood (199) / anxiety (107) / trauma (128)
```

Each stage gets more data per decision and its own operating threshold.

**Reuse the guard from `scripts/all3_hierarchical.py`**: a stage is used only if it beats a
constant predictor in **inner** CV, otherwise it falls back to the majority class. That guard
was worth +0.055 on the all3 track and its critical property is that the decision is made on
inner folds — deciding it on outer-fold accuracy is test-set selection.

> ⚠️ **MEASURED DURING IMPLEMENTATION — the comparator must match the metric.**
> `all3_hierarchical.py` scores **raw accuracy** and compares against the **majority rate**,
> which is consistent. This guard scores **balanced accuracy**, whose value for any constant
> predictor is exactly **1/K** regardless of imbalance. Comparing balanced accuracy against the
> majority rate is an apples-to-oranges test that an imbalanced stage can never pass.
>
> Cost of getting it wrong, measured: stage 1 has majority 0.8242 and its classifier reaches
> balanced accuracy **0.7171** — good against a 0.5 chance level, but structurally incapable of
> exceeding 0.8242. The guard rejected it in **15/15** folds. Both stages then fell back to
> their majority class, and composing two majority fallbacks predicted `depression` for all 421
> subjects: composed balanced accuracy exactly **0.2500**, and a `specific.disorder` table
> reading recall 1.000 for depression and 0.000 for everything else.
>
> The failure is loud only because the unguarded score is reported next to the guarded one.
> Report both; a guard that silently swallows a working classifier looks identical to a stage
> that genuinely has no signal.

**Report stage-wise.** The expected outcome is stage 1 working (healthy vs ill is the one
contrast with demonstrated structure) and stage 2 landing near chance. That is not a
disappointing result, it is the precise localisation of the failure: it says the EEG separates
*ill from well* but not *which illness*, which is a far more defensible claim than a pooled
0.43 and is consistent with everything else measured here.

Stage-2 majority is 199/434 = **0.4585**. Report against that, never against 0.3333.

## B. Stacking instead of gating

`GatedHybrid` scored 0.3447 (no demo) and 0.3239 (+demo) — below several linear models. The
diagnosis is straightforward: a learned gate is extra parameters trained on 421 subjects, and
its own gate weights (0.70 coherence / 0.23 spectral / 0.07 demographics) were learned from
the same data that trains the branches. Same fusion idea, fewer parameters:

1. Three base learners on the plan-01 blocks: **spectral** (`CLR+FAA+TBR`), **connectivity**
   (`GRAPH`), **demographics**.
2. Collect **out-of-fold** probabilities for each.
3. Combine with a logistic meta-learner (`class_weight='balanced'`, L2).

**The out-of-fold requirement is the entire method.** Base-learner probabilities must come from
an inner CV *inside* the training fold. A base learner scoring its own training data emits
near-one-hot probabilities, the meta-learner learns to trust whichever base overfits hardest,
and the stack silently collapses to that single model. This repo has the identical lesson
recorded for knowledge distillation in `DL_Without_Features/`: teacher soft targets are
generated out-of-fold for exactly this reason.

The meta-learner's coefficients are also the honest version of the gate diagnostic — a
3-vector per class, fitted on held-out probabilities, rather than a 96→3 layer fitted jointly
with everything it weighs.

## C. The other 417 subjects — worth trying, with one caveat that may cancel it

The cohort restriction to 527 discards schizophrenia (117), addiction (186) and OCD (46). The
brief's proposal is to train a healthy-vs-any-disorder representation on all 945 and use it as
an auxiliary task or initialisation for the 4-class problem.

**The caveat: those 417 subjects contain no additional healthy controls.** There are 95 in the
entire file, 93 after quarantine, and all of them are already in the 527. So
healthy-vs-any-disorder on all 945 is **93 vs 850** — an even more imbalanced version of a
contrast that already has 93 positives.

If stage 1 is limited by how few healthy controls exist — which is the likely constraint — then
adding 417 more ill subjects does not relieve it. What the extra data can genuinely provide is
a better model of the *structure of ill resting-state EEG*, which may help stage 2 and may
sharpen the ill-side decision boundary in stage 1.

So it is worth running, but as a specific hypothesis rather than a free win:

- **Auxiliary task, not pretraining.** Train the stage-2 representation on all 850 ill subjects
  across 6 disorder classes, then transfer to the 3-way mood/anxiety/trauma problem. Stage 2 is
  where extra ill subjects plausibly help.
- **Control required**: the same architecture trained on the 434 ill subjects only. Without it
  there is no way to attribute a change to the extra data. This repo's DANN run is the cautionary
  case — 8 configurations, no interpretable conclusion, because the α=0 control was the only
  thing that revealed the method had not worked.
- Subjects appearing in both the auxiliary and target tasks must respect the plan-00 holdout.
  The 417 extra subjects need their own test-set exclusion, or the auxiliary task sees held-out
  people.

## D. Report the confusion at `specific.disorder` level

Independent of the above, and cheap — it is a regrouping of predictions already saved to
`Results/OSF_Park/preds/`, not a training run.

The `stress` class pools three conditions and `anxiety` pools two:

| pooled label | components | n |
|---|---|---:|
| anxiety | Panic disorder / Social anxiety disorder | 59 / 48 |
| stress | Acute stress / Adjustment / PTSD | 38 / 38 / 52 |

These are clinically distinct. Classify at `main.disorder` level as planned, but break the
confusion matrix down by `specific.disorder`. **If PTSD separates and adjustment disorder does
not, the pooled label is hiding a real effect** — and that is the single most likely place in
this dataset for a genuine finding, because a pooled label that mixes a separable condition
with a non-separable one will always look like a null.

Sub-group n is 38–59, so per-subgroup recall has a standard error around ±0.07. Report counts
alongside rates and resist reading a 2-subject difference as a finding.

## Verification

- Stage routing is exhaustive and disjoint: every subject reaches exactly one stage-2 decision
  or is stopped at stage 1.
- The composed 3-stage prediction reproduces a flat 4-class prediction vector of the right
  shape, and composed accuracy is recomputed from saved predictions rather than re-derived.
- Stacking: assert base-learner OOF probabilities were produced by an inner split — a direct
  test is that a base learner's OOF accuracy is *strictly lower* than its in-fold accuracy. If
  they match, the inner CV is not doing anything.
- The auxiliary-task control (434-only) is run in the same session as the 850 version, on the
  same folds.
- Stage-2 results reported against 0.4585, not 0.3333.

## Expected result

Stage 1 (healthy vs ill) **0.72–0.80** accuracy against a 0.824 majority — note that majority
is high enough that accuracy alone may look like failure; balanced accuracy and AUC are the
meaningful figures. Stage 2 **near its 0.4585 majority**. Composed 4-class balanced accuracy
**0.42–0.48**.

If stage 2 lands at its majority, the conclusion is that this cohort's EEG distinguishes ill
from well but not one disorder from another — which, combined with plan 02's bounded ΔAUC, is
a complete and defensible account of what is in this file.
