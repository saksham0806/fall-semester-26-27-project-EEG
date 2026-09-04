# Plan 02 — demographic residualization

Changes the question from *"does EEG beat demographics"* to **"does EEG carry anything
demographics don't"**. That is the question worth answering here, and it is the one that turns
a null into a publishable result rather than a failed experiment.

Depends on plan 01's `features_corrected.py`. Run it on the corrected view, not on raw AB/COH.

New module: `OSF_Park_model/normative.py`.

## Why the current framing cannot answer it

`SVM [EEG]` scores 0.4261 and `SVM [EEG+demographics]` scores 0.4261 — identical to four
decimals. That is not evidence that EEG contains demographic information; it is evidence that
1,140 columns drown out 4. Concatenation cannot separate "EEG is redundant with age" from
"EEG is informative but numerically swamped". Residualization can, because it removes the
demographic component from each feature *before* the classifier sees it.

## Method

Standard normative modelling, as used in psychiatric neuroimaging:

1. Fit, **on healthy controls only**, a linear model per EEG feature:
   `feature ~ age + sex + education` (see §IQ below for the fourth term).
2. Apply that model to **all** subjects and take residuals: `r = observed − predicted`.
3. Classify the residuals.

The healthy-only fit is what makes it normative: the model encodes what a feature *should*
look like for a person of that age and sex, and the residual is deviation from that norm.
Fitting on all 527 would absorb disorder-related variance into the "normal" model and
residualize away the signal.

Use `sklearn.linear_model.LinearRegression` — `statsmodels` is not installed, and nothing here
needs its inference machinery since confidence intervals come from the bootstrap in §3.

## 1. The leak, which is the whole difficulty

**The normative model must be fit inside the training fold, on training-fold healthy controls
only.** Fitting it once on all healthy controls and then cross-validating the classifier leaks
test-subject information into every feature of every fold. It is the same class of error as
fitting a scaler on the full dataset, and it will silently inflate the result.

This is tighter than it sounds. Dev is ~421 subjects after plan 00, of which ~74 are healthy.
A 5-fold split leaves roughly **59 healthy controls** to fit each normative model — 59 points
per feature, 3–4 predictors. That is workable for a linear fit but it is thin, and it will make
residuals noisier than a naive full-data fit would suggest.

Consequences to build in:

- Regularise the normative fit (`Ridge(alpha=1.0)`) rather than using OLS. With 59 points and
  correlated predictors, OLS coefficients are unstable enough to inject noise into every
  residual.
- Report the normative model's own fit quality (mean R² across features, on held-out healthy
  controls within the fold). If age explains ~nothing, residualization is a no-op and the
  ΔAUC in §3 will be near zero for an uninteresting reason. Distinguishing "EEG adds nothing"
  from "residualization did nothing" requires this number.
- Assert in code that the fold's normative fit never saw a test index.

## 2. IQ — handle separately, and default to excluding it

Healthy controls average **116.2** IQ against 89.6–107.8 for every patient group. That is a
14–27 point gap, larger than any EEG effect this cohort is likely to show.

IQ was measured after diagnosis. It is therefore partly a *consequence* of illness, medication
and current symptom burden — a mediator or collider, not an independent predictor.
Conditioning on it can induce as much bias as it removes.

- **Primary analysis excludes IQ**: `feature ~ age + sex + education`.
- **Sensitivity analysis includes it**: `feature ~ age + sex + education + IQ`.
- Report both. If the increment survives only when IQ is excluded, say so plainly — that is a
  finding about the confound structure, not a result to pick between.

The same argument applies with less force to education, which is at least partly pre-morbid.
Keep education in the primary model, and note the assumption.

## 3. The comparison — ΔAUC with a paired bootstrap

The output of this plan is one number with an interval, not a table of accuracies.

On **identical folds**, fit two models:

- **M_demo**: demographics alone (age, sex, education).
- **M_demo+res**: demographics + residualized EEG.

Nested, so the comparison is fair. Then:

- **Metric**: macro one-vs-rest AUC for 4-class; plain AUC for binary. AUC because it is
  threshold-free — the same property that caught all three threshold bugs documented in
  `CLAUDE.md`'s evaluation section.
- **ΔAUC** = `AUC(M_demo+res) − AUC(M_demo)`, computed on pooled out-of-fold predictions.
- **Paired bootstrap**, 2,000 resamples over *subjects*: draw one index set per resample and
  score **both** models on that same set. Pairing is essential — an unpaired bootstrap of two
  correlated estimates gives an interval far too wide and will report "no difference" for
  everything.
- Report the **percentile 95% CI** on ΔAUC.

**A CI containing 0 is the result, not a failure.** Write it as such: "residualized EEG adds
ΔAUC = 0.01 [−0.04, 0.06] over demographics alone". That is a bounded, interpretable null.
"Accuracy 0.43 vs 0.43" is not.

For binary, DeLong's test is the conventional alternative and is worth reporting alongside the
bootstrap. It has no sklearn implementation; the bootstrap is sufficient on its own if
implementing DeLong is not worth the time.

## 4. Interpreting a nonzero increment

If ΔAUC is meaningfully positive, one confound remains before it can be called an EEG effect.
Residualizing on age/sex/education does **not** remove medication status, time of day, symptom
severity or recording-year drift. The year probe already came back clean (0.2179 vs 0.2443
majority), which handles drift. The others are not in the file and cannot be checked — state
them as limitations rather than leaving them implied.

## Verification

- Residuals of healthy controls have near-zero mean **within the training fold** (that is what
  was fitted) and are *not* required to be zero-mean in the test fold — checking the latter
  would be checking for the leak, and it should fail.
- A synthetic self-check: inject a feature that is exactly `2·age + noise` and confirm its
  residual retains only the noise.

  ⚠️ **Check variance, not correlation.** Ridge shrinkage leaves a residual that is tiny but
  still exactly *proportional* to age, so `|corr(residual, age)|` stays at ~0.97 even when
  residualization worked perfectly — correlation is scale-invariant and cannot see that the
  variance is gone. The measured result on the real `age` vector is **0.0024 of unit variance
  remaining** (99.76% removed) while a pure-noise control column is preserved at 1.43. Asserting
  on correlation here produces a false failure; this was hit during implementation.
- Paired bootstrap sanity: bootstrapping M_demo against *itself* must give ΔAUC = 0 with a
  degenerate interval. If it does not, the pairing is broken.
- Assert no test index appears in any normative fit.

## Expected result

ΔAUC in **0.00–0.06**, CI plausibly containing 0 on the 4-class task. Binary is the more
likely place to see a positive increment, since healthy-vs-depressed is the one contrast where
the current pipeline already shows real per-class structure (depression recall 0.91).
