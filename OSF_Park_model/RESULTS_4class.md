# OSF_Park BRMH — 4class

n = 528 recordings. Out-of-fold over 5 folds (StratifiedKFold x 1 seeds). No subject IDs exist in this file, so grouping is impossible — see README.

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
| SVM [demographics only] | 0.4356 ±0.044 | +0.0417 | 0.4008 | 0.3720 | 1 |
| SVM [EEG+demographics] | 0.4356 ±0.011 | +0.0417 | 0.3535 | 0.2733 | 4 |
| SVM [EEG] | 0.4337 ±0.020 | +0.0398 | 0.3480 | 0.2693 | 4 |
| GradientBoosting [EEG] | 0.3655 ±0.054 | -0.0284 | 0.3184 | 0.3178 | 6 |
| RandomForest [EEG] | 0.3485 ±0.035 | -0.0454 | 0.3264 | 0.3172 | 7 |
| GatedHybrid [no demographics] | 0.3258 ±0.040 | -0.0682 | 0.3451 | 0.3236 | 2 |
| LogisticRegression [EEG] | 0.3239 ±0.043 | -0.0701 | 0.3159 | 0.3221 | 7 |
| GatedHybrid [+demographics] | 0.3144 ±0.075 | -0.0795 | 0.3401 | 0.3130 | 2 |
| MLP [EEG] | 0.3011 ±0.067 | -0.0928 | 0.3378 | 0.3009 | 2 |
| ChannelTransformer [AB] | 0.2917 ±0.014 | -0.1023 | 0.3175 | 0.2805 | 2 |
| CohTransformer [AB+COH bias] | 0.2652 ±0.031 | -0.1288 | 0.2963 | 0.2607 | 2 |

## Per-class — SVM [demographics only]

| class | precision | recall | F1 | n |
|---|---:|---:|---:|---:|
| healthy | 0.521 | 0.649 | 0.578 | 94 |
| depression | 0.414 | 0.698 | 0.520 | 199 |
| anxiety | 0.500 | 0.131 | 0.207 | 107 |
| stress | 0.340 | 0.125 | 0.183 | 128 |

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

**Permutation importance (balanced-acc drop, base 0.267)**

| feature | importance |
|---|---:|
| theta | +0.0251 ±0.0097 |
| C3 | +0.0211 ±0.0121 |
| Pz | +0.0211 ±0.0056 |
| Fz | +0.0207 ±0.0062 |
| O2 | +0.0175 ±0.0053 |
| FP1 | +0.0171 ±0.0159 |
| highbeta | +0.0166 ±0.0115 |
| Cz | +0.0163 ±0.0062 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| delta | 0.0309 |
| beta | 0.0289 |
| alpha | 0.0261 |
| highbeta | 0.0254 |
| theta | 0.0248 |
| gamma | 0.0241 |

## Explainability — ChannelTransformer [AB]

**Permutation importance (balanced-acc drop, base 0.335)**

| feature | importance |
|---|---:|
| theta | +0.0816 ±0.0117 |
| beta | +0.0763 ±0.0222 |
| alpha | +0.0308 ±0.0204 |
| gamma | +0.0302 ±0.0121 |
| P4 | +0.0283 ±0.0107 |
| highbeta | +0.0234 ±0.0185 |
| F7 | +0.0202 ±0.0161 |
| C4 | +0.0195 ±0.0128 |

**Gradient x input by band**

| feature | importance |
|---|---:|
| beta | 0.2160 |
| delta | 0.1699 |
| alpha | 0.1277 |
| theta | 0.0772 |
| highbeta | 0.0717 |
| gamma | 0.0592 |

**Attention received per channel**

| feature | importance |
|---|---:|
| T3 | 0.0546 |
| F7 | 0.0545 |
| FP1 | 0.0541 |
| FP2 | 0.0537 |
| T5 | 0.0536 |
| F8 | 0.0531 |
| T4 | 0.0531 |
| O1 | 0.0530 |

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

**Permutation importance (balanced-acc drop, base 0.363)**

| feature | importance |
|---|---:|
| COH block | +0.0953 ±0.0234 |
| DEMO block | +0.0547 ±0.0084 |
| beta | +0.0304 ±0.0166 |
| highbeta | +0.0262 ±0.0088 |
| delta | +0.0196 ±0.0091 |
| FP1 | +0.0129 ±0.0054 |
| F7 | +0.0129 ±0.0141 |
| theta | +0.0128 ±0.0121 |

**Attention received per channel**

| feature | importance |
|---|---:|
| F7 | 0.0553 |
| T3 | 0.0540 |
| O1 | 0.0536 |
| FP1 | 0.0535 |
| F8 | 0.0534 |
| P3 | 0.0533 |
| T5 | 0.0530 |
| FP2 | 0.0530 |

**Gate weight per branch**

| feature | importance |
|---|---:|
| COH projection | 0.6649 |
| AB transformer | 0.2549 |
| demographics | 0.0802 |
