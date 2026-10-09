# Plan 02 — demographic residualization (binary)

Dev only, n=233. Normative model fit on training-fold healthy controls only.

## The result

| arm | ΔAUC | 95% CI | verdict | normative R² |
|---|---:|---|---|---:|
| no IQ | +0.0937 | [+0.0222, +0.1650] | increment | -0.2992 |
| with IQ | +0.0233 | [-0.0347, +0.0789] | **null** | -0.3065 |

Primary analysis is **no IQ**: IQ was measured after diagnosis (healthy 116.2 vs 89.6–107.8 for patient groups), so it is partly a consequence of illness and medication — a mediator, not an independent predictor.

| model | balanced acc | macro-F1 | accuracy | macro-AUC |
|---|---:|---:|---:|---:|
| demographics + residualized EEG [with IQ] | **0.7773** | 0.7661 | 0.7897 | 0.8593 |
| demographics only [with IQ] | **0.7535** | 0.7261 | 0.7425 | 0.8360 |
| demographics + residualized EEG [no IQ] | **0.7350** | 0.7217 | 0.7468 | 0.8275 |
| demographics only [no IQ] | **0.5987** | 0.5819 | 0.6052 | 0.7337 |

Residualizing on age/sex/education does not remove medication status, symptom severity or time of day. None are in this file. The recording-year probe came back clean (0.2179 vs 0.2443 majority), which handles drift; the rest stand as limitations.
