# Plan 01 — corrected features, simple model

Isolates one question: **does the null survive correct preprocessing?**

Everything measured so far fed an RBF SVM raw absolute power that had merely been z-scored.
Band power is log-normal, absolute power is dominated by skull thickness and electrode
impedance rather than brain state, and 1,026 of 1,140 columns were raw coherence against 527
subjects. Any of those three alone is enough to explain a null. This plan removes all three
and changes nothing else, so a null afterwards is attributable to the data.

New module: `OSF_Park_model/features_corrected.py`. **Do not modify `data.py`'s existing
blocks** — `AB`/`COH`/`DEMO` stay exactly as they are, and this file adds a parallel view.
Same discipline as `features.py` vs `features_v2.py` in `src/proposed_model/`: the version
that produced published numbers is never edited in place.

## 1. Quarantine

`data.DROP_NO = [206, 944]` → n=527 (healthy 93 / depression 199 / anxiety 107 / stress 128).
944's AB block is byte-identical to row 226 (Behavioral addiction disorder), so its band
powers are probably not its own. Its coherence block is distinct, but a row whose spectral
half is known-corrupt does not belong in a spectral experiment.

Cost is real and falls on the smallest class: healthy 94 → 93. State it rather than absorbing
it silently.

## 2. Spectral transform — CLR, not log-then-ratio

The originating brief specified "log₁₀ the AB block → convert to relative power per channel".
Applied in that order it computes ratios of logarithms, which is not relative power. Correct
order is relative on **linear** power, then transform.

Relative power per channel is a **compositional** vector — six values that sum to 1 — and the
correct transform for compositional data is the centered log-ratio:

```
P    = AB reshaped to (n, 19, 6)                    # band-major; use data.Cohort.channel_tensor
CLR  = log(P) - log(P).mean(axis=-1, keepdims=True)  # per channel, across its 6 bands
```

CLR is worth the extra sentence because of *what it removes*. Multiply one channel's six band
powers by any constant `k` — which is precisely what a thicker skull, a worse electrode
contact or a different impedance does — and `log(kP) - mean(log(kP))` is unchanged. CLR
removes the per-channel gain factor **exactly**, not approximately. That is the nuisance the
brief correctly identifies, and this is the transform that deletes it.

Two consequences to record:

- CLR rows sum to zero, so the 6 bands per channel carry 5 degrees of freedom. Keep all 114
  columns anyway — dropping one arbitrarily breaks the symmetry that makes the features
  interpretable, and the elastic net handles the collinearity.
- CLR discards absolute level entirely. If absolute power turns out to matter, it is only
  recoverable from the separate `log10(AB)` view below.

Verified justification for the log: median per-column skew **3.37** raw, **0.18** after log₁₀.

**Feature blocks:**

| block | cols | definition |
|---|---:|---|
| `CLR` | 114 | as above — the primary spectral view |
| `LOGABS` | 114 | `log10(AB)`, kept as a secondary view for the ablation in §6 |
| `FAA` | 3 | frontal alpha asymmetry, `log(alpha_right) − log(alpha_left)` for FP2/FP1, F4/F3, F8/F7 |
| `TBR` | 19 | theta/beta ratio per channel, computed on linear power then logged |

FAA is computed from log absolute alpha, not from CLR — asymmetry is a *between-channel*
contrast and CLR is a within-channel normalisation, so taking FAA on CLR values would cancel
the very thing being measured. This is the same reason `combine_datasets.py` normalises
globally rather than per channel.

## 3. Coherence transform — Fisher-z, and the `inf` trap

```
Z = arctanh(clip(COH / 100.0, 0.0, 1.0 - 1e-6))
```

**The clip is mandatory.** 24 cells in this file are exactly 100.0, and `arctanh(1.0)` is
`inf`. Unclipped, a single `inf` propagates through the fold's mean and standard deviation and
destroys standardization for every subject in that fold — silently, since sklearn will happily
fit on it and return plausible-looking numbers. Assert `np.isfinite(Z).all()` immediately
after the transform.

`Z` is not used as 1,026 model inputs. It is the input to §4.

## 4. Coherence → graph metrics (1,026 → 144)

The highest-leverage change in this plan. Every permutation run says the COH block carries the
signal (+0.124 for the SVM, versus +0.017 for the strongest single band) and `GatedHybrid`
independently put 0.70 of its gate on the coherence branch — but 1,026 features against 421
dev subjects is a regime where an untuned RBF kernel cannot work. The information is there;
the parameterisation is what fails.

This is also the native idiom of this OSF deposit: Study A in the same folder ships
`Data3_EEG_smallworldness_DMN.region.csv`, precomputed graph metrics on the same montage.

Per subject, per band, build the 19×19 symmetric matrix (`data.Cohort.coh_matrix()` already
does this) and extract:

| metric | count | note |
|---|---:|---|
| nodal strength | 19 | row sums — the one metric that survives any thresholding choice |
| average clustering coefficient | 1 | `nx.average_clustering(G, weight="weight")` |
| characteristic path length | 1 | on `distance = 1/w`, **not** on `w` |
| global efficiency | 1 | `mean(1/d_ij)` over the Dijkstra distance matrix |
| modularity | 1 | `nx.community.greedy_modularity_communities(G, weight="weight")` |
| small-world σ | 1 | see the cost warning below |
| | **24 × 6 bands = 144** | |

**Thresholding is a real decision, not a detail.** These graphs are fully connected — all 171
edges present and strictly positive — and on a complete graph path length and clustering are
near-degenerate. Standard practice is proportional thresholding: keep the top *d* fraction of
edges by weight, discard the rest.

> ⚠️ **MEASURED DURING IMPLEMENTATION — plain proportional thresholding does not work here.**
> The connectivity check below was run over all 527 subjects × 6 bands and it fails badly:
>
> | density | edges kept | fragmented graphs |
> |---:|---:|---:|
> | 0.20 | 34 | **2973 / 3162 (94%)** |
> | 0.30 | 51 | 2157 (68%) |
> | 0.40 | 68 | 1276 (40%) |
> | 0.50 | 86 | 690 (22%) |
>
> Coherence has hub structure, so the strongest edges cluster on a few nodes and the rest of
> the graph detaches. Fragmentation never reaches zero at any usable density, and a fragmented
> graph has **infinite** characteristic path length — the metric is undefined, not merely noisy.
>
> **Resolution: MST-union thresholding**, which is the standard fix in network neuroscience for
> exactly this failure. Take the minimum spanning tree of the *distance* graph (`1/w`, so the
> MST keeps the strongest connecting edges — 18 edges on 19 nodes), union it with the top-*d*
> edges by weight, and connectivity is guaranteed by construction. Deterministic, no seed.
> Measured result at `d = 0.20`: **0 fragmented graphs out of 3,162.**
>
> Implemented in `features_corrected._threshold`. `d = 0.20` is retained unchanged.

**Fix `d = 0.20` and do not sweep it.** Sweeping density in inner CV adds a degree of freedom,
and this repo has already measured what that costs on a cohort this size: letting the
hierarchical model choose its feature view lost −0.025 because 3-fold inner CV over ~40
subjects could not tell two views apart. With 421 dev subjects the same trap applies. Record
`d = 0.20` as a fixed preprocessing constant; if it is ever varied, it is varied on dev only
and reported as a sensitivity analysis, never as a selected hyperparameter.

**Cost warning on small-world σ.** `nx.sigma` builds many randomised null graphs per call; at
527 subjects × 6 bands = 3,162 graphs it is prohibitively slow. Two acceptable routes:

1. Omit σ. `C` and `L` are already in the block and σ is a ratio of them against a null — it
   adds little independent information. Block becomes 23 × 6 = 138.
2. Normalise against **one shared null ensemble per band**, computed once over the dev set (10
   degree-preserving rewires, fixed seed), not per subject. Cheap, deterministic, defensible.

Take route 1 first. Add σ only if plan 01 shows a nonzero increment and σ is worth the cost.

**Cache the graph block** to `features/osf_park/graph_d20.npy`, keyed by the density and the
quarantine list. It is deterministic given raw COH and takes minutes to compute. Same stale-
cache hazard as `extracted_features*.npy` and `_cache_riemann_*.npy` elsewhere in this repo:
**if the density, the metric set or `DROP_NO` changes, delete the file** — nothing in the
filename will tell you it went stale.

## 5. Model — regularised linear, not RBF

```python
LogisticRegression(penalty="elasticnet", solver="saga", class_weight="balanced",
                   max_iter=5000, l1_ratio=<grid>, C=<grid>)
```

- `l1_ratio ∈ {0.1, 0.5, 0.9}`, `C ∈ {0.01, 0.1, 1, 10}` — 12 configurations.
- **Nested CV**: the grid is searched in an inner 5-fold on the training fold only; the outer
  fold is scored once with the selected config. Selecting on outer-fold accuracy is test-set
  selection, and this repo has the lesson on record from the all3 hierarchical guard.
- Scoring for selection: `balanced_accuracy`, matching the primary metric.
- Compare against L2-only (`penalty="l2"`) as the simpler control. If elastic net does not beat
  it, report L2 — fewer moving parts.

Keep an RBF SVM in the table as a **control**, not a candidate. It is what produced the current
numbers; showing it on corrected features is what separates "the preprocessing was wrong" from
"the model family was wrong".

## 6. Ablation grid

The point of the plan is attribution, so run the blocks separately, not just pooled:

| # | blocks | p | isolates |
|---|---|---:|---|
| 1 | `DEMO` | 4 | the bar |
| 2 | `CLR + FAA + TBR` | 136 | spectral only, gain-corrected |
| 3 | `GRAPH` | 138 | connectivity only, reduced |
| 4 | `CLR + FAA + TBR + GRAPH` | 274 | full EEG |
| 5 | #4 `+ DEMO` | 278 | does EEG add to demographics? |
| 6 | `LOGABS + COH` (raw, z-scored) | 1140 | **the current pipeline, as a baseline row** |

Row 6 is what makes the plan interpretable. Without it there is no way to attribute a change
to preprocessing rather than to the model swap.

## Verification

`features_corrected.py` gets a `__main__` self-check asserting:

- CLR rows sum to ~0 per channel (`atol=1e-5`), and CLR is invariant to per-channel rescaling:
  multiply one channel's 6 bands by 7.3 and the CLR output is unchanged. **Do this in float64**
  — `data.Cohort.AB` is float32, where the invariance holds only to ~7e-7 and a tight `atol`
  produces a false failure. All transforms in `features_corrected.py` cast to float64, and no
  epsilon is added inside the log (AB's minimum is 0.0258, strictly positive).
- `np.isfinite` on every block, especially `Z` after the clip.
- Graph block shape `(527, 138)`, all finite, nodal strengths strictly positive.
- Thresholded graphs stay connected at `d = 0.20`. **This check fires on plain proportional
  thresholding** (94% fragment) and is the reason for MST-union above; the self-check asserts
  the cached metadata records `fragmented == 0`, so a regression cannot pass silently.
- Feature count matches the table above exactly.

## Expected result, and what would falsify the plan

4-class balanced accuracy **0.42–0.47**; EEG increment over demographics **0.00–0.06**.

- **Increment ≥ 0.06 with anxiety and stress recall both above ~0.25** — preprocessing was the
  blocker. Proceed to 02 and 03 with the corrected features as the default view.
- **Increment ≈ 0, but rows 2–4 now predict all four classes** — the features are fixed and
  the signal is genuinely absent. This is the publishable null; plan 02 is what makes it one.
- **Row 4 still predicts zero anxiety and zero stress** — preprocessing was not the problem at
  all, and plan 03's hierarchy is the diagnostic that says where it breaks.
- **Anything above ~0.75 4-class** — the split broke. Suspect that before believing it.
