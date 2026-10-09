# Model Plan — `datasets/all3` (3-class: healthy / stress / depression)

Roster of baseline and multi-representation fusion models for the 3-class dataset.
Dataset: 15,728 segments — healthy 6,632 · stress 5,497 · depression 3,599, 108 subjects.
Labels verified against raw source metadata (53/53 MODMA subjects, 0 mismatches).

> ## ✅ EXECUTED — 18 models run, none beat the shortcut
>
> Scripts: `scripts/all3_models.py` (Tier 1 + F1/F2), `scripts/all3_deep.py` (Tier 2 + F3).
> Results: `Results/all3/{diagnostic,models_tier1_fusion,models_deep}.json`.
>
> | Model | acc | vs 0.6267 |
> |---|---|---|
> | **Corpus ID only — no EEG** | **0.6267** | — **unbeaten** |
> | SVM / 81 features | 0.6256 | −0.001 |
> | F2 late fusion (best weight) | 0.5989 | −0.028 |
> | F3 DSFFN gated fusion | 0.5954 | −0.031 |
> | Logistic regression / 81 features | 0.5931 | −0.034 |
> | F1 early fusion (Riemann ⊕ 81feat) | 0.5898 | −0.037 |
> | Random Forest | 0.5781 | −0.049 |
> | XGBoost | 0.5706 | −0.056 |
> | EEGTransformer · CNN-LSTM · LSTM | 0.525 · 0.514 · 0.506 | −0.10 to −0.12 |
> | TSception · 1D CNN · EEGNet | 0.490 · 0.443 · 0.430 | −0.14 to −0.20 |
> | Riemannian, per-corpus re-centering | 0.3255 | −0.301 |
>
> **Nothing clears 0.6267. The predicted outcome held.**
>
> - **All three fusion strategies lost to a plain SVM.** F1 −0.036, F2 −0.027, F3 −0.030
>   relative to the best single-view model. Combining representations did not help.
> - **All six deep architectures lost badly** — 0.10 to 0.20 below the shortcut, consistent
>   with the three prior lines of evidence.
> - **F4 (attention fusion) and F5 (ST²-ADN) were not built.** The plan gated them on
>   "F1–F3 show fusion helps at all". Three fusion variants showed it hurts, so building a
>   fourth would have been ignoring the plan's own stopping rule.
>
> **Two corrections to this document, found during execution:**
> 1. F3 was described as ready via `nb_classes=3`. False — `build_dsffn` hardcoded two
>    sigmoid heads (`depression`, `stress`), as does `build_st2adn`. An optional
>    `nb_classes` parameter was added; omitting it preserves the original dual-head model.
> 2. `dsffn` must be imported as `src.dsffn.model`, not as top-level `dsffn` with `src/` on
>    the path — a consequence of the Phase 3 relative-import fix.
>
> Tier 2's claim did hold: all six architectures ran at `nb_classes=3` unmodified, via the
> softmax branch at `eeg_models.py:4-9` that had never previously been called.

---

## Read this before the model list

Class is confounded with recording corpus. MODMA holds only {healthy, depression};
SAM40/CogLoad only {healthy, stress}; each used a different amplifier.

| Baseline | 3-class accuracy |
|---|---|
| Chance | 0.3333 |
| Majority class | 0.4217 |
| **Corpus ID only — zero EEG understanding** | **0.6267** |

**0.6267 is the bar.** Corpus identity is 92% recoverable from the 81 features alone
(measured, `Results/all3/diagnostic.json`). And the task collapses without it:

| Riemannian variant | accuracy |
|---|---|
| Global re-centering (device signature **kept**) | 0.5078 |
| Per-corpus re-centering (device signature **removed**) | **0.3255** ≈ chance |

Nothing measured so far beats 0.6267.

## Evaluation protocol — applies to every model below

- 5-fold **subject-grouped** CV (`GroupKFold` on `subjects.npy`), assert zero leakage
- Report five numbers **together**, never accuracy alone:
  1. 3-class accuracy 2. balanced accuracy 3. **margin over 0.6267**
  4. within-corpus accuracy vs that corpus's majority 5. confusion matrix
- Reuse `report()` and `shortcut_baseline()` from `scripts/all3_diagnostic.py`
- A model clearing 0.6267 by **< 0.02** has not demonstrated EEG understanding

---

## Tier 1 — Baselines (classical + geometric)

| ID | Model | Input | Status |
|---|---|---|---|
| B1 | Majority class | — | measured **0.4217** |
| B2 | Corpus-ID lookup | corpus only | measured **0.6267** — the bar |
| B3 | SVM (RBF) | 81 features | measured **0.5931** (−0.034) |
| B4 | Logistic regression | 81 features | to run |
| B5 | Random Forest | 81 features | to run |
| B6 | XGBoost | 81 features | to run |
| B7 | Riemannian, global re-centering | 105 tangent dims | measured **0.5078** |
| B8 | Riemannian, per-corpus re-centering | 105 tangent dims | measured **0.3255** |
| B9 | **Riemannian ⊕ 81 features** | 186 dims | **to run — cheapest untried idea** |

Reuse: `get_svm_pipeline` / `get_random_forest_pipeline` / `get_xgboost_pipeline`
(`src/proposed_model/classical_models.py`), `extract_all_features`
(`src/proposed_model/features.py:197`), `riemannian_features()`
(`scripts/all3_diagnostic.py`).

## Tier 2 — Deep baselines (low priority)

All six existing architectures at `nb_classes=3`: **1D CNN, LSTM, CNN-LSTM, EEGNet,
TSception, EEGTransformer**.

**Nearly free to run.** `_get_classification_layer` (`src/proposed_model/eeg_models.py:4-9`)
already branches to a softmax head when `nb_classes > 1`, and the loss auto-switches to
`sparse_categorical_crossentropy`. That path exists and has **never been called** — every
call site in this project uses `nb_classes=1`. This is a parameter change, not new code.

Low priority because three independent lines of evidence say this family loses here: all six
at chance on stress (~0.52); a foundation model with working pretraining landing 0.18 below a
105-dim linear model; SSL scoring below its own random-init control. Run for completeness of
the comparison, not in expectation of a win.

## Tier 3 — Multi-representation fusion

Four views of the same EEG. "Multi-modal" means this, not sensor fusion: no non-EEG signal is
shared across the three corpora (accelerometer CogLoad-only, clinical scores MODMA-only,
ECG/GSR AMIGOS-only), so adding one would hand the model an even cleaner corpus shortcut than
the amplifier signature.

| View | Representation | Reuse from |
|---|---|---|
| V1 | raw waveform (6, 512) | any backbone in `src/proposed_model/eeg_models.py` |
| V2 | log-STFT spectrogram | `SpectrogramLayer`, `src/st2adn/model.py:81` |
| V3 | Riemannian tangent space (105) | `riemannian_features()`, `scripts/all3_diagnostic.py` |
| V4 | 81 handcrafted features | `extract_all_features`, `src/proposed_model/features.py:197` |

| ID | Fusion model | Notes |
|---|---|---|
| F1 | **Early** — concat V3+V4 → linear/SVM | Same as B9. Do first: if plain concatenation gains nothing, elaborate fusion probably will not either. |
| F2 | **Late** — per-view classifier, soft-vote or stack | No new architecture; stacking meta-learner exists at `src/codae/model.py:57` |
| F3 | **Gated** — learned gate between deep and handcrafted streams | **Already written**: `build_dsffn` (`src/dsffn/model.py:9`) gates a TSception stream against the 81 features. Set `nb_classes=3` and it is ready. |
| F4 | **Attention** — learned attention over all four view embeddings | The only genuinely new architecture in this roster |
| F5 | **Spectro-temporal transformer** — STFT → ViT | `build_st2adn` (`src/st2adn/model.py:156`) — ⚠️ see below |

> ⚠️ **F5 carries a trap.** ST²-ADN's Adaptive Instance Norm is *conditioned on `dataset_id`*
> — it takes corpus identity as an explicit input. On `all3` that hands the model the 0.6267
> shortcut directly instead of making it earn the result. Either disable the AdaIN
> conditioning or treat any F5 number as uninterpretable.

---

## Priority order

1. **B9 / F1** — Riemannian ⊕ features. One line, most likely cheap win.
2. **B4–B6** — complete the classical row.
3. **F2, F3** — late and gated fusion; both reuse existing code.
4. **Tier 2** — deep baselines, for completeness.
5. **F4** — attention fusion, only if F1–F3 show fusion helps at all.
6. **F5** — only with AdaIN conditioning disabled.

## Honest expectation

Given nothing yet beats 0.6267, and that removing the device signature drops the task to
chance, the realistic outcome is that **no model in this roster clears the shortcut by a
meaningful margin**. That is a publishable finding — it demonstrates the 3-class formulation
is confounded, not that the models are inadequate.

The within-corpus rows are where genuine improvement would show. Current SVM within-corpus,
against the best constant prediction inside each corpus:

| Corpus | 3-class SVM | majority | margin |
|---|---|---|---|
| MODMA | 0.5638 | 0.5478 | +0.016 |
| SAM40 | 0.7279 | 0.7500 | −0.022 |
| CogLoad | 0.5765 | 0.6886 | **−0.112** |

Two of three are below a constant predictor — forcing one model to do both tasks makes it
worse at each. The same data as two separate binary tasks gives **0.7716** subject-level AUC
for depression and **0.7560** within-subject AUC for stress, which remain the stronger
vehicle for any scientific claim.
