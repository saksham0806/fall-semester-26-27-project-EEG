# trial_cc_pre — rebuilt pipeline results

Subject-grouped throughout. Segment counts after artifact gating.

## 1. Segment rejection table

| corpus | recordings | windows | kept | flatline | variance |
|---|---|---|---|---|---|
| MODMA | 53 | 7959 | 7183 | 0 | 776 |
| SAM40 | 480 | 5280 | 5019 | 0 | 261 |
| CogLoad | 103 | 4441 | 3861 | 61 | 462 |

## 2. Within-corpus (5-fold nested CV)

| target | level | N | AUC | 95% CI | bal.acc | segment AUC |
|---|---|---|---|---|---|---|
| MODMA dep (subject-level) | subject | 53 | **0.3434** | [0.192, 0.486] | 0.3700 | 0.3874 |
| SAM40 stress (trial-level) | trial | 480 | **0.6837** | [0.627, 0.734] | 0.6458 | 0.6305 |
| CogLoad stress (recording-level) | recording | 99 | **0.9294** | [0.848, 0.982] | 0.8160 | 0.8227 |

Within-subject (mean over subjects, bootstrapped over subjects):

| target | mean AUC | 95% CI | subjects | frac > 0.5 |
|---|---|---|---|---|
| SAM40 stress (trial-level) | **0.7546** | [0.682, 0.825] | 40 | 0.85 |
| CogLoad stress (recording-level) | **1.0000** | [1.000, 1.000] | 14 | 1.00 |

**Read CogLoad from the within-subject row, not the pooled one.** Pooled recording-level AUC there rests on 14 negative recordings in total, so its interval cannot separate a working model from a broken one. Every subject has a natural baseline and load recordings, so within-subject is well posed — but at 1 baseline vs ~6 load per subject the per-subject AUC is quantised to 1/6, hence the wide bootstrap. MODMA is subject-level; SAM40 is trial-level.

## 3. CogLoad LOSO

| target | level | N | AUC | 95% CI | bal.acc | segment AUC |
|---|---|---|---|---|---|---|
| CogLoad stress (recording-level, LOSO) | recording | 99 | **0.7924** | [0.676, 0.896] | 0.6916 | 0.7415 |

Within-subject (mean over subjects, bootstrapped over subjects):

| target | mean AUC | 95% CI | subjects | frac > 0.5 |
|---|---|---|---|---|
| CogLoad stress (recording-level, LOSO) | **0.9000** | [0.764, 1.000] | 14 | 0.93 |

## 4. Cross-corpus stress transfer (headline)

| target | level | N | AUC | 95% CI | bal.acc | segment AUC |
|---|---|---|---|---|---|---|
| SAM40->CogLoad stress (recording-level, lambda=0.0) | recording | 99 | **0.6437** | [0.447, 0.813] | 0.5000 | 0.5686 |
| SAM40->CogLoad stress (recording-level, lambda=1.0) | recording | 99 | **0.4580** | [0.305, 0.609] | 0.4824 | 0.4531 |
| CogLoad->SAM40 stress (trial-level, lambda=0.0) | trial | 480 | **0.7174** | [0.665, 0.765] | 0.6694 | 0.6617 |
| CogLoad->SAM40 stress (trial-level, lambda=1.0) | trial | 480 | **0.7009** | [0.641, 0.751] | 0.6556 | 0.6486 |

Within-subject (mean over subjects, bootstrapped over subjects):

| target | mean AUC | 95% CI | subjects | frac > 0.5 |
|---|---|---|---|---|
| SAM40->CogLoad stress (recording-level, lambda=0.0) | **0.5500** | [0.360, 0.738] | 14 | 0.43 |
| SAM40->CogLoad stress (recording-level, lambda=1.0) | **0.3690** | [0.195, 0.574] | 14 | 0.36 |
| CogLoad->SAM40 stress (trial-level, lambda=0.0) | **0.7861** | [0.712, 0.853] | 40 | 0.88 |
| CogLoad->SAM40 stress (trial-level, lambda=1.0) | **0.7741** | [0.707, 0.841] | 40 | 0.82 |

## 5. Dual frozen domain probes

| probe | lambda=0 | lambda=1 | target |
|---|---|---|---|
| full-set accuracy | 0.7629 | 0.4470 | 0.447 (majority domain) |
| anchor accuracy | 0.7496 | 0.5761 | 0.333 (chance) |
| anchor silhouette S_corpus | 0.0133 | -0.0240 | < 0.30 |

**Manifold gate:** S_corpus = -0.0240 at lambda=1 -> alignment viable.

## 6. Cross-head extrapolation

| head fed out-of-domain | N | mean p | median p | frac p>0.5 |
|---|---|---|---|---|
| stress_head_on_MODMA_MDD | 3266 | 0.3716 | 0.3804 | 0.2789 |
| dep_head_on_CogLoad_task | 3192 | 0.4298 | 0.4148 | 0.3534 |
| stress_head_on_SAM40_task | 3742 | 0.4862 | 0.4905 | 0.4797 |
| dep_head_on_MODMA_MDD | 3266 | 0.3910 | 0.3743 | 0.3325 |

## 7. Historical diagnostic context (not for model comparison)

- Corpus shortcut baseline (per-corpus 3-class majority, zero EEG): **0.6755**
- Leaky random segment split: dep AUC 0.7203, stress AUC 0.7338
- Same model, subject-grouped: dep AUC 0.3874, stress AUC 0.7256

Depression cannot be transfer-tested: no second depression corpus is present.
