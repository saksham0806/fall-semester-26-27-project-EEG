# OSF_Park BRMH — 4class

n = 528 recordings. Out-of-fold over 15 folds (StratifiedKFold x 3 seeds). No subject IDs exist in this file, so grouping is impossible — see README.

## Baselines — read every row below against these

| baseline | accuracy |
|---|---:|
| chance | 0.2500 |
| majority class | 0.3769 |
| **demographics only** (age, sex, IQ, education) | **0.3939** |

Recording-year probe: **0.2179** vs 0.2443 majority over 8 year classes.

## Models

| model | accuracy | vs demo | balanced acc | macro-F1 | sec |
|---|---:|---:|---:|---:|---:|
| SVM [demographics only] | 0.4299 ±0.038 | +0.0360 | 0.3914 | 0.3461 | 1 |
| SVM [EEG] | 0.4261 ±0.018 | +0.0322 | 0.3416 | 0.2626 | 8 |
| SVM [EEG+demographics] | 0.4261 ±0.016 | +0.0322 | 0.3430 | 0.2632 | 8 |
| GradientBoosting [EEG] | 0.3864 ±0.034 | -0.0076 | 0.3276 | 0.3172 | 19 |
| RandomForest [EEG] | 0.3466 ±0.033 | -0.0473 | 0.3265 | 0.3175 | 12 |
| LogisticRegression [EEG] | 0.3447 ±0.050 | -0.0492 | 0.3389 | 0.3430 | 26 |
| GatedHybrid [no demographics] | 0.3447 ±0.057 | -0.0492 | 0.3707 | 0.3442 | 14 |
| MLP [EEG] | 0.3390 ±0.054 | -0.0549 | 0.3710 | 0.3392 | 6 |
| GatedHybrid [+demographics] | 0.3239 ±0.061 | -0.0701 | 0.3560 | 0.3252 | 14 |
| CohTransformer [AB+COH bias] | 0.2917 ±0.049 | -0.1023 | 0.3246 | 0.2898 | 13 |
| ChannelTransformer [AB] | 0.2879 ±0.039 | -0.1060 | 0.3133 | 0.2794 | 12 |

## Per-class — SVM [demographics only]

| class | precision | recall | F1 | n |
|---|---:|---:|---:|---:|
| healthy | 0.516 | 0.670 | 0.583 | 94 |
| depression | 0.413 | 0.724 | 0.526 | 199 |
| anxiety | 0.357 | 0.093 | 0.148 | 107 |
| stress | 0.345 | 0.078 | 0.127 | 128 |

## Explainability — SVM [demographics only]

**Permutation importance (balanced-acc drop, base 0.382)**

| feature | importance |
|---|---:|
| DEMO block | +0.1208 ±0.0308 |
| FP1 | +0.0000 ±0.0000 |
| FP2 | +0.0000 ±0.0000 |
| F7 | +0.0000 ±0.0000 |
| F3 | +0.0000 ±0.0000 |
| Fz | +0.0000 ±0.0000 |
| F4 | +0.0000 ±0.0000 |
| F8 | +0.0000 ±0.0000 |

## Explainability — SVM [EEG]

**Permutation importance (balanced-acc drop, base 0.369)**

| feature | importance |
|---|---:|
| COH block | +0.1240 ±0.0158 |
| delta | +0.0171 ±0.0162 |
| T3 | +0.0132 ±0.0000 |
| theta | +0.0129 ±0.0054 |
| beta | +0.0109 ±0.0081 |
| T6 | +0.0065 ±0.0091 |
| FP1 | +0.0044 ±0.0062 |
| F7 | +0.0044 ±0.0062 |

## Explainability — SVM [EEG+demographics]

**Permutation importance (balanced-acc drop, base 0.363)**

| feature | importance |
|---|---:|
| COH block | +0.1113 ±0.0230 |
| delta | +0.0127 ±0.0054 |
| beta | +0.0046 ±0.0081 |
| DEMO block | +0.0044 ±0.0062 |
| Fz | +0.0000 ±0.0000 |
| T4 | +0.0000 ±0.0000 |
| T6 | +0.0000 ±0.0000 |
| O1 | +0.0000 ±0.0000 |

## Explainability — LogisticRegression [EEG]

**Permutation importance (balanced-acc drop, base 0.313)**

| feature | importance |
|---|---:|
| COH block | +0.0905 ±0.0502 |
| delta | +0.0543 ±0.0191 |
| theta | +0.0533 ±0.0087 |
| O2 | +0.0461 ±0.0159 |
| C3 | +0.0399 ±0.0126 |
| Pz | +0.0355 ±0.0107 |
| beta | +0.0336 ±0.0071 |
| F4 | +0.0331 ±0.0174 |

**Native importance by band**

| feature | importance |
|---|---:|
| delta | 2.9835 |
| beta | 2.9762 |
| gamma | 2.3842 |
| highbeta | 2.2242 |
| alpha | 2.1373 |
| theta | 2.1073 |

## Explainability — RandomForest [EEG]

**Permutation importance (balanced-acc drop, base 0.342)**

| feature | importance |
|---|---:|
| COH block | +0.1111 ±0.0335 |
| beta | +0.0437 ±0.0185 |
| F4 | +0.0168 ±0.0122 |
| T3 | +0.0147 ±0.0065 |
| C3 | +0.0147 ±0.0065 |
| Fz | +0.0125 ±0.0051 |
| theta | +0.0115 ±0.0140 |
| Cz | +0.0085 ±0.0065 |

**Native importance by band**

| feature | importance |
|---|---:|
| beta | 0.0279 |
| delta | 0.0260 |
| theta | 0.0208 |
| gamma | 0.0201 |
| alpha | 0.0184 |
| highbeta | 0.0180 |

## Explainability — GradientBoosting [EEG]

**Permutation importance (balanced-acc drop, base 0.334)**

| feature | importance |
|---|---:|
| COH block | +0.0842 ±0.0225 |
| FP2 | +0.0223 ±0.0055 |
| gamma | +0.0149 ±0.0186 |
| beta | +0.0149 ±0.0071 |
| T3 | +0.0119 ±0.0000 |
| P4 | +0.0102 ±0.0056 |
| highbeta | +0.0102 ±0.0056 |
| O2 | +0.0096 ±0.0000 |

**Native importance by band**

| feature | importance |
|---|---:|
| beta | 0.0364 |
| delta | 0.0333 |
| theta | 0.0230 |
| alpha | 0.0178 |
| gamma | 0.0171 |
| highbeta | 0.0135 |

## Explainability — MLP [EEG]

**Permutation importance (balanced-acc drop, base 0.330)**

| feature | importance |
|---|---:|
| COH block | +0.0637 ±0.0255 |
| beta | +0.0372 ±0.0079 |
| FP2 | +0.0209 ±0.0059 |
| F7 | +0.0108 ±0.0107 |
| C3 | +0.0105 ±0.0056 |
| highbeta | +0.0094 ±0.0068 |
| F3 | +0.0083 ±0.0029 |
| gamma | +0.0083 ±0.0029 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| beta | 0.1283 |
| delta | 0.0987 |
| theta | 0.0791 |
| highbeta | 0.0592 |
| gamma | 0.0484 |
| alpha | 0.0473 |

## Explainability — ChannelTransformer [AB]

**Permutation importance (balanced-acc drop, base 0.328)**

| feature | importance |
|---|---:|
| beta | +0.0686 ±0.0205 |
| theta | +0.0364 ±0.0226 |
| O1 | +0.0302 ±0.0211 |
| delta | +0.0288 ±0.0242 |
| FP1 | +0.0244 ±0.0213 |
| alpha | +0.0208 ±0.0130 |
| gamma | +0.0176 ±0.0079 |
| highbeta | +0.0126 ±0.0135 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| beta | 0.3621 |
| highbeta | 0.2530 |
| delta | 0.2474 |
| gamma | 0.2075 |
| alpha | 0.2043 |
| theta | 0.1953 |

**Attention received per channel**

| feature | importance |
|---|---:|
| T3 | 0.0599 |
| F7 | 0.0593 |
| FP1 | 0.0555 |
| T5 | 0.0551 |
| F8 | 0.0550 |
| FP2 | 0.0546 |
| O1 | 0.0544 |
| T4 | 0.0540 |

## Explainability — CohTransformer [AB+COH bias]

**Permutation importance (balanced-acc drop, base 0.322)**

| feature | importance |
|---|---:|
| beta | +0.0361 ±0.0190 |
| gamma | +0.0306 ±0.0023 |
| theta | +0.0285 ±0.0084 |
| alpha | +0.0187 ±0.0216 |
| F7 | +0.0118 ±0.0184 |
| FP2 | +0.0113 ±0.0127 |
| O2 | +0.0100 ±0.0091 |
| C3 | +0.0054 ±0.0082 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| beta | 0.1708 |
| delta | 0.1671 |
| alpha | 0.1202 |
| gamma | 0.0887 |
| theta | 0.0767 |
| highbeta | 0.0762 |

**Attention received per channel**

| feature | importance |
|---|---:|
| F7 | 0.0547 |
| T3 | 0.0541 |
| FP1 | 0.0538 |
| FP2 | 0.0536 |
| T5 | 0.0533 |
| F8 | 0.0532 |
| T4 | 0.0531 |
| C4 | 0.0529 |

## Explainability — GatedHybrid [no demographics]

**Permutation importance (balanced-acc drop, base 0.367)**

| feature | importance |
|---|---:|
| COH block | +0.0917 ±0.0437 |
| beta | +0.0448 ±0.0037 |
| theta | +0.0393 ±0.0124 |
| delta | +0.0283 ±0.0208 |
| C4 | +0.0064 ±0.0045 |
| alpha | +0.0053 ±0.0013 |
| T6 | +0.0044 ±0.0062 |
| FP2 | +0.0043 ±0.0040 |

**Attention received per channel**

| feature | importance |
|---|---:|
| F7 | 0.0536 |
| T3 | 0.0535 |
| FP1 | 0.0532 |
| FP2 | 0.0531 |
| P3 | 0.0530 |
| T5 | 0.0529 |
| F8 | 0.0529 |
| O1 | 0.0527 |

**Gate weight per branch**

| feature | importance |
|---|---:|
| COH projection | 0.7000 |
| AB transformer | 0.3000 |

## Explainability — GatedHybrid [+demographics]

**Permutation importance (balanced-acc drop, base 0.352)**

| feature | importance |
|---|---:|
| COH block | +0.0825 ±0.0261 |
| DEMO block | +0.0405 ±0.0129 |
| highbeta | +0.0324 ±0.0229 |
| theta | +0.0181 ±0.0065 |
| T4 | +0.0129 ±0.0091 |
| P4 | +0.0129 ±0.0091 |
| T5 | +0.0129 ±0.0071 |
| T6 | +0.0095 ±0.0045 |

**Attention received per channel**

| feature | importance |
|---|---:|
| F7 | 0.0564 |
| T3 | 0.0545 |
| O1 | 0.0542 |
| F8 | 0.0539 |
| P3 | 0.0538 |
| FP1 | 0.0536 |
| T5 | 0.0532 |
| T4 | 0.0528 |

**Gate weight per branch**

| feature | importance |
|---|---:|
| COH projection | 0.7004 |
| AB transformer | 0.2316 |
| demographics | 0.0680 |
