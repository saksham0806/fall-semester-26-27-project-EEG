# Plan 02 — demographic residualization (4class)

Dev only, n=421. Normative model fit on training-fold healthy controls only.

## The result

| arm | ΔAUC | 95% CI | verdict | normative R² |
|---|---:|---|---|---:|
| no IQ | +0.0242 | [-0.0144, +0.0604] | **null** | -0.2812 |
| with IQ | -0.0094 | [-0.0455, +0.0262] | **null** | -0.2896 |

Primary analysis is **no IQ**: IQ was measured after diagnosis (healthy 116.2 vs 89.6–107.8 for patient groups), so it is partly a consequence of illness and medication — a mediator, not an independent predictor.

| model | balanced acc | macro-F1 | accuracy | macro-AUC |
|---|---:|---:|---:|---:|
| demographics only [with IQ] | **0.4938** | 0.4189 | 0.4228 | 0.7052 |
| demographics + residualized EEG [with IQ] | **0.4517** | 0.4448 | 0.4323 | 0.6958 |
| demographics + residualized EEG [no IQ] | **0.4305** | 0.4179 | 0.4086 | 0.6814 |
| demographics only [no IQ] | **0.3993** | 0.3386 | 0.3444 | 0.6572 |

Residualizing on age/sex/education does not remove medication status, symptom severity or time of day. None are in this file. The recording-year probe came back clean (0.2179 vs 0.2443 majority), which handles drift; the rest stand as limitations.
