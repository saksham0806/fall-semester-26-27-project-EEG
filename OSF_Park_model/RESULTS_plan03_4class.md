# Plan 03 — hierarchy, stacking, auxiliary data (4class)

Dev only, n=421; majority 0.3777.

## A. Hierarchy

| stage | balanced acc | accuracy | majority | guard passed |
|---|---:|---:|---:|---:|
| 1 · healthy vs ill (guarded) | 0.7171 | 0.7878 | 0.8242 | 15/15 |
| 1 · healthy vs ill (unguarded classifier) | 0.7171 | 0.7878 | 0.8242 | — |
| 2 · mood/anxiety/trauma | 0.3489 | 0.3650 | 0.4582 | 10/15 |

Stage 2 is reported against its own majority (199/434), never against 0.3333.

## B. Stacking

Meta weights: {'spectral': 0.2395546054377896, 'graph': 0.2590451997701743, 'demo': 0.501400194792036}. In-sample minus OOF accuracy per base learner: {'spectral': 0.24149474824549008, 'graph': 0.4101396542791201, 'demo': 0.03107719843624889} (all > 0, confirming the inner CV is real).

## C. The other 417 subjects

| arm | stage-2 balanced acc |
|---|---:|
| control: 434 in-cohort ill only | 0.3437 |
| auxiliary: all 850 ill | 0.3521 |

Delta from the extra data: **+0.0083**. Those 417 add no healthy controls (all 93 are already in the cohort), so they cannot relieve a stage-1 bottleneck.

## D. Specific-disorder breakdown

| specific.disorder | pooled as | n | recall | ± |
|---|---|---:|---:|---:|
| Healthy control | healthy | 74 | 0.622 | 0.056 |
| Acute stress disorder | stress | 27 | 0.481 | 0.096 |
| Depressive disorder | depression | 159 | 0.396 | 0.039 |
| Panic disorder | anxiety | 45 | 0.178 | 0.057 |
| Social anxiety disorder | anxiety | 41 | 0.146 | 0.055 |
| Adjustment disorder | stress | 33 | 0.121 | 0.057 |
| Posttraumatic stress disorder | stress | 42 | 0.119 | 0.050 |

Sub-group n is 30–60, so recall SE is ±0.06–0.09. Read counts, not rates.

## Composed

| model | balanced acc | macro-F1 | accuracy | macro-AUC |
|---|---:|---:|---:|---:|
| stacked (spectral+graph+demo) | **0.5031** | 0.4390 | 0.4347 | 0.7118 |
| hierarchical (composed) | **0.3491** | 0.3315 | 0.3444 | 0.5592 |
