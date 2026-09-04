# OSF_Park BRMH — binary

n = 293 recordings. Out-of-fold over 15 folds (StratifiedKFold x 3 seeds). No subject IDs exist in this file, so grouping is impossible — see README.

## Baselines — read every row below against these

| baseline | accuracy |
|---|---:|
| chance | 0.5000 |
| majority class | 0.6792 |
| **demographics only** (age, sex, IQ, education) | **0.7576** |

Recording-year probe: **0.2423** vs 0.2355 majority over 8 year classes.

## Models

| model | accuracy | vs demo | balanced acc | macro-F1 | sec |
|---|---:|---:|---:|---:|---:|
| SVM [demographics only] | 0.7952 ±0.060 | +0.0376 | 0.7650 | 0.7650 | 1 |
| SVM [EEG+demographics] | 0.7645 ±0.056 | +0.0069 | 0.6863 | 0.6998 | 2 |
| SVM [EEG] | 0.7577 ±0.056 | +0.0001 | 0.6729 | 0.6860 | 2 |
| RandomForest [EEG] | 0.7440 ±0.084 | -0.0136 | 0.6684 | 0.6788 | 10 |
| GradientBoosting [EEG] | 0.7440 ±0.066 | -0.0136 | 0.6432 | 0.6531 | 3 |
| LogisticRegression [EEG] | 0.7270 ±0.056 | -0.0307 | 0.6755 | 0.6793 | 2 |
| MLP [EEG] | 0.7235 ±0.096 | -0.0341 | 0.6870 | 0.6854 | 4 |
| GatedHybrid [+demographics] | 0.7201 ±0.061 | -0.0375 | 0.6957 | 0.6888 | 12 |
| GatedHybrid [no demographics] | 0.6997 ±0.086 | -0.0580 | 0.6722 | 0.6660 | 9 |
| CohTransformer [AB+COH bias] | 0.5290 ±0.080 | -0.2286 | 0.5859 | 0.5278 | 9 |
| ChannelTransformer [AB] | 0.5154 ±0.078 | -0.2423 | 0.5674 | 0.5137 | 9 |

## Per-class — SVM [demographics only]

| class | precision | recall | F1 | n |
|---|---:|---:|---:|---:|
| healthy | 0.681 | 0.681 | 0.681 | 94 |
| depression | 0.849 | 0.849 | 0.849 | 199 |

## Explainability — SVM [demographics only]

**Permutation importance (balanced-acc drop, base 0.780)**

| feature | importance |
|---|---:|
| DEMO block | +0.2976 ±0.0366 |
| FP1 | +0.0000 ±0.0000 |
| FP2 | +0.0000 ±0.0000 |
| F7 | +0.0000 ±0.0000 |
| F3 | +0.0000 ±0.0000 |
| Fz | +0.0000 ±0.0000 |
| F4 | +0.0000 ±0.0000 |
| F8 | +0.0000 ±0.0000 |

## Explainability — SVM [EEG]

**Permutation importance (balanced-acc drop, base 0.648)**

| feature | importance |
|---|---:|
| COH block | +0.0943 ±0.0584 |
| delta | +0.0167 ±0.0223 |
| theta | +0.0129 ±0.0183 |
| P4 | +0.0125 ±0.0000 |
| P3 | +0.0083 ±0.0059 |
| Cz | +0.0042 ±0.0059 |
| T6 | +0.0042 ±0.0059 |
| O2 | +0.0042 ±0.0059 |

## Explainability — SVM [EEG+demographics]

**Permutation importance (balanced-acc drop, base 0.648)**

| feature | importance |
|---|---:|
| COH block | +0.0855 ±0.0477 |
| theta | +0.0254 ±0.0183 |
| delta | +0.0167 ±0.0223 |
| alpha | +0.0167 ±0.0118 |
| T3 | +0.0125 ±0.0102 |
| P3 | +0.0125 ±0.0000 |
| P4 | +0.0125 ±0.0000 |
| T6 | +0.0125 ±0.0000 |

## Explainability — LogisticRegression [EEG]

**Permutation importance (balanced-acc drop, base 0.739)**

| feature | importance |
|---|---:|
| COH block | +0.2454 ±0.0734 |
| beta | +0.0471 ±0.0322 |
| alpha | +0.0454 ±0.0364 |
| gamma | +0.0287 ±0.0218 |
| C4 | +0.0083 ±0.0059 |
| highbeta | +0.0046 ±0.0242 |
| F7 | +0.0042 ±0.0059 |
| F8 | +0.0042 ±0.0059 |

**Native importance by band**

| feature | importance |
|---|---:|
| delta | 3.6395 |
| beta | 3.2363 |
| gamma | 2.4573 |
| alpha | 2.0516 |
| theta | 1.8949 |
| highbeta | 1.5837 |

## Explainability — RandomForest [EEG]

**Permutation importance (balanced-acc drop, base 0.612)**

| feature | importance |
|---|---:|
| COH block | +0.0961 ±0.0327 |
| delta | +0.0318 ±0.0431 |
| beta | +0.0221 ±0.0156 |
| theta | +0.0092 ±0.0162 |
| gamma | +0.0004 ±0.0107 |
| FP2 | +0.0000 ±0.0000 |
| F7 | +0.0000 ±0.0000 |
| F3 | +0.0000 ±0.0000 |

**Native importance by band**

| feature | importance |
|---|---:|
| delta | 0.0583 |
| theta | 0.0309 |
| beta | 0.0260 |
| gamma | 0.0190 |
| highbeta | 0.0131 |
| alpha | 0.0120 |

## Explainability — GradientBoosting [EEG]

**Permutation importance (balanced-acc drop, base 0.595)**

| feature | importance |
|---|---:|
| COH block | +0.1215 ±0.0301 |
| O1 | +0.0614 ±0.0161 |
| T5 | +0.0263 ±0.0000 |
| theta | +0.0263 ±0.0000 |
| delta | +0.0184 ±0.0162 |
| T4 | +0.0175 ±0.0124 |
| Pz | +0.0134 ±0.0107 |
| FP1 | +0.0088 ±0.0124 |

**Native importance by band**

| feature | importance |
|---|---:|
| delta | 0.0800 |
| gamma | 0.0304 |
| theta | 0.0235 |
| beta | 0.0186 |
| highbeta | 0.0066 |
| alpha | 0.0001 |

## Explainability — MLP [EEG]

**Permutation importance (balanced-acc drop, base 0.574)**

| feature | importance |
|---|---:|
| COH block | +0.1048 ±0.0424 |
| C3 | +0.0004 ±0.0107 |
| DEMO block | +0.0000 ±0.0000 |
| FP2 | -0.0042 ±0.0223 |
| F3 | -0.0083 ±0.0059 |
| Cz | -0.0083 ±0.0059 |
| Pz | -0.0083 ±0.0059 |
| O1 | -0.0125 ±0.0000 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| delta | 0.1351 |
| beta | 0.1129 |
| theta | 0.0781 |
| alpha | 0.0679 |
| highbeta | 0.0578 |
| gamma | 0.0423 |

## Explainability — ChannelTransformer [AB]

**Permutation importance (balanced-acc drop, base 0.584)**

| feature | importance |
|---|---:|
| delta | +0.0623 ±0.0338 |
| beta | +0.0581 ±0.0686 |
| theta | +0.0526 ±0.0278 |
| alpha | +0.0480 ±0.0065 |
| gamma | +0.0384 ±0.0183 |
| F7 | +0.0292 ±0.0059 |
| C3 | +0.0254 ±0.0183 |
| P3 | +0.0250 ±0.0000 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| delta | 0.3513 |
| alpha | 0.2812 |
| beta | 0.2426 |
| highbeta | 0.1722 |
| theta | 0.1592 |
| gamma | 0.0915 |

**Attention received per channel**

| feature | importance |
|---|---:|
| T5 | 0.0557 |
| O1 | 0.0554 |
| FP2 | 0.0545 |
| O2 | 0.0542 |
| F8 | 0.0534 |
| T4 | 0.0529 |
| T6 | 0.0527 |
| T3 | 0.0525 |

## Explainability — CohTransformer [AB+COH bias]

**Permutation importance (balanced-acc drop, base 0.570)**

| feature | importance |
|---|---:|
| beta | +0.0581 ±0.0435 |
| delta | +0.0456 ±0.0245 |
| T6 | +0.0346 ±0.0245 |
| highbeta | +0.0235 ±0.0220 |
| gamma | +0.0226 ±0.0218 |
| O1 | +0.0175 ±0.0218 |
| F8 | +0.0171 ±0.0162 |
| theta | +0.0143 ±0.0322 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| delta | 0.2530 |
| beta | 0.2272 |
| theta | 0.1847 |
| highbeta | 0.1794 |
| alpha | 0.1508 |
| gamma | 0.1156 |

**Attention received per channel**

| feature | importance |
|---|---:|
| T5 | 0.0554 |
| O1 | 0.0552 |
| O2 | 0.0549 |
| T3 | 0.0543 |
| FP2 | 0.0542 |
| T6 | 0.0539 |
| T4 | 0.0533 |
| F7 | 0.0527 |

## Explainability — GatedHybrid [no demographics]

**Permutation importance (balanced-acc drop, base 0.574)**

| feature | importance |
|---|---:|
| COH block | +0.0697 ±0.0519 |
| T4 | +0.0221 ±0.0059 |
| P4 | +0.0180 ±0.0059 |
| highbeta | +0.0138 ±0.0102 |
| P3 | +0.0092 ±0.0162 |
| alpha | +0.0088 ±0.0124 |
| F3 | +0.0050 ±0.0124 |
| O2 | +0.0050 ±0.0161 |

**Attention received per channel**

| feature | importance |
|---|---:|
| O2 | 0.0543 |
| T5 | 0.0540 |
| O1 | 0.0538 |
| T3 | 0.0535 |
| T6 | 0.0535 |
| FP2 | 0.0531 |
| T4 | 0.0529 |
| F7 | 0.0527 |

**Gate weight per branch**

| feature | importance |
|---|---:|
| COH projection | 0.7691 |
| AB transformer | 0.2309 |

## Explainability — GatedHybrid [+demographics]

**Permutation importance (balanced-acc drop, base 0.613)**

| feature | importance |
|---|---:|
| DEMO block | +0.0292 ±0.0338 |
| alpha | +0.0167 ±0.0405 |
| beta | +0.0162 ±0.0364 |
| FP1 | +0.0000 ±0.0000 |
| Fz | +0.0000 ±0.0000 |
| F4 | +0.0000 ±0.0000 |
| F8 | +0.0000 ±0.0000 |
| T3 | +0.0000 ±0.0000 |

**Attention received per channel**

| feature | importance |
|---|---:|
| O2 | 0.0552 |
| O1 | 0.0549 |
| T5 | 0.0548 |
| T3 | 0.0537 |
| T6 | 0.0536 |
| T4 | 0.0533 |
| FP2 | 0.0532 |
| F7 | 0.0530 |

**Gate weight per branch**

| feature | importance |
|---|---:|
| COH projection | 0.4330 |
| demographics | 0.3121 |
| AB transformer | 0.2549 |
