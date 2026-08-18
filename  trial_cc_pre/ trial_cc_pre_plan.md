**1. Unified Preprocessing & Artifact Harmonization Pipeline**

All filtering is performed prior to windowing to eliminate edge discontinuities and boundary ringing. Every recording is filtered using a zero-phase forward-backward filter (`scipy.signal.filtfilt`), resampled via polyphase filtering (`scipy.signal.resample_poly`), windowed into 4.0-second segments (512 samples at 128 Hz, 50% overlap / 256 stride), and mapped to canonical frontal channels: `Fp1, Fp2, F7, F3, F4, F8`.

| Corpus | Native $f_s$ | Ingestion, Filtering & Edge-Handling Protocol | Channel Mapping | Rebuilt Segment Yield |
| --- | --- | --- | --- | --- |
| **MODMA**<br> | 250 Hz

 | 1. 0.5–45 Hz zero-phase bandpass (`filtfilt`, high-pass first to eliminate the $\sim -10^4$ DC baseline).

<br>

<br>2. Polyphase resample ($250 \to 128\text{ Hz}$, `up=64, down=125`).

 | E22, E9, E33, E24, E124, E122

 | ~7,959 segments (53 subjects: 24 MDD, 29 HC)

 |
| **SAM40**<br> | 128 Hz

 | 1. **Do not decimate.** Ingest native `raw_data/` (`Data`) directly.

<br>

<br>2. Apply 3.0-second explicit reflection padding (`padlen=384`) on each 25-second trial to suppress filter transients, apply 0.5–45 Hz bandpass (`filtfilt`), and strip padding.

 | Indices: 2, 31, 3, 4, 29, 30

 | **~5,280 segments** (40 subjects: 120 relax, 360 task trials)

 |
| **Cognitive Load**<br> | 250 Hz

 | 1. 0.5–45 Hz bandpass (`filtfilt`) to remove the $\sim 10^4$ DC pedestal.

<br>

<br>2. Polyphase resample ($250 \to 128\text{ Hz}$, `up=64, down=125`).

 | Columns: 1, 2, 3, 4, 6, 7

 | ~4,800–5,300 segments (15 subjects: 8 trials/sub)

 |
| **AMIGOS & WESAD**<br> | N/A | **Purged entirely.** AMIGOS contains chance-level arousal labels (0.4875 AUC) and spatial approximations; WESAD contains zero EEG.

 | Excluded

 | 0

 |

* **Uniform Artifact Gating:** To ensure rejection thresholds do not introduce corpus-specific biases, apply the exact same two rejection rules across all corpora post-filtering:
* *Flatline Masking:* Mark zero-runs on raw continuous signals; reject any window overlapping a flatline span $\ge 50\text{ ms}$.


* *Channel Variance Gate:* Reject any window where $\max_{c}(\text{Var}(c)) > 5 \times \text{median}_{c}(\text{Var}(c))$ computed across the 6 channels within that 512-sample segment (eliminates single-electrode spikes like the CogLoad F4 blowout).


* *Reporting:* Log and publish the explicit segment rejection table across MODMA, SAM40, and CogLoad.




* **Robust Segment Normalization:** Apply global scaling per segment:

$$\tilde{X} = \text{clip}\left(\frac{X - \text{median}(X)}{\text{MAD}(X)}, -5, 5\right)$$



Computing median and MAD across the entire $(6, 512)$ matrix preserves inter-channel amplitude ratios and frontal asymmetry.



---

**2. Target Space & Data Representations**

Represent each segment as a multi-task tuple with masked supervised loss branches:

```
Data Tuple: (X[i], y_dep, y_stress, y_domain, anchor_mask, subject_id, trial_id)
  ├── y_dep       ∈ {0 (MODMA HC), 1 (MODMA MDD), -1 (Mask)}
  ├── y_stress    ∈ {0 (SAM40 Relax / CogLoad Natural), 1 (Task/Load Trials), -1 (Mask)}
  ├── y_domain    ∈ {0 (MODMA), 1 (SAM40), 2 (CogLoad)}
  └── anchor_mask ∈ True if y_dep == 0 or y_stress == 0, else False

```

* `y_dep` is supervised exclusively on MODMA segments (`-1` on SAM40 and CogLoad).


* `y_stress` is supervised exclusively on SAM40 and CogLoad segments (`-1` on MODMA).


* `anchor_mask` restricts domain alignment strictly to non-pathological baseline recordings.


* **Dynamic Domain Baselines:** On the rebuilt array ($N_{\text{total}} \approx 18,200$ segments), the majority domain baseline is computed dynamically:

$$\text{Baseline}_{\text{domain}} = \frac{\max(N_{\text{MODMA}}, N_{\text{SAM40}}, N_{\text{CogLoad}})}{N_{\text{total}}} \approx \frac{7,959}{18,200} \approx \mathbf{0.437}$$



---

**3. Model Architecture & Anchor-Only Adaptation**

```
                      ┌────────────────────────────────────────┐
                      │    Input EEG Segment: X ∈ R^(6×512)    │
                      └────────────────────┬───────────────────┘
                                           │
                        ┌──────────────────▼───────────────────┐
                        │        Shared Feature Backbone       │
                        │     Temporal Conv2D (1×64, 8 filt)   │
                        │     DepthwiseConv2D (6×1, D=2)       │
                        │     SeparableConv2D (1×16, 16 filt)  │
                        └──────────────────┬───────────────────┘
                                           │ Latent Vector: z ∈ R^d
                 ┌─────────────────────────┼─────────────────────────┐
                 │ (Balanced Anchor GRL)   │                         │
       ┌─────────▼──────────┐    ┌─────────▼──────────┐    ┌─────────▼──────────┐
       │ Anchor Adversary   │    │  Depression Head   │    │    Stress Head     │
       │ Dense(64) -> (3)   │    │  Dense(32) -> (1)  │    │  Dense(32) -> (1)  │
       │ (Inverse-Weighted) │    │  (MODMA Segments)  │    │(SAM40/CogLoad Segs)│
       └────────────────────┘    └────────────────────┘    └────────────────────┘

```

* **Pre-Alignment Manifold Gate:** Measure the corpus silhouette score $S_{\text{corpus}}$ on anchor subsets ($y=0$) at $\lambda=0$ and $\lambda=1$.



$$\text{If } S_{\text{corpus}}(\text{anchors}) > 0.30 \text{ at } \lambda=1 \implies \text{Healthy manifold cannot align; abort GRL and use independent backbones.}$$


* **Balanced Anchor-GRL:** Domain gradients are reversed ($-\lambda_p$) exclusively on anchor segments (`anchor_mask == True`) with inverse class-weighting ($w_c = \frac{N_{\text{anchor}}}{3 \cdot N_{\text{anchor}, c}}$) to prevent MODMA HC from overwhelming SAM40 Relax and CogLoad Natural:



$$\mathcal{L}_{\text{adversary}} = -\sum_{c \in \{0,1,2\}} w_c \cdot \mathbb{I}(y_{\text{domain}} = c) \log(\hat{y}_{\text{domain}})$$


* **Total Loss:**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{dep\_masked}}(y_{\text{dep}}, \hat{y}_{\text{dep}}) + \mathcal{L}_{\text{stress\_masked}}(y_{\text{stress}}, \hat{y}_{\text{stress}}) + \lambda_p \mathcal{L}_{\text{adversary}}$$


* **Theoretical Limitation to Acknowledge:** Anchor-only GRL aligns the healthy manifold across recording hardware but does not align the pathological space. On an unseen recording device, the out-of-distribution MDD representation remains unconstrained.



---

**4. Statistical Validation Hierarchy & Aggregation**

All evaluation uses nested cross-validation grouped by subject to prevent overlapping-window data leakage.

```
Nested Cross-Validation Schema:
Outer Loop (5-Fold StratifiedGroupKFold on Subjects) ──> Unbiased Test Evaluation
  └── Inner Loop (GroupKFold on Train Subjects) ───────> Early Stopping & Threshold Tuning (θ)

```

* **Inner Validation Splits:**
* *MODMA & SAM40:* Inner 4-fold `GroupKFold` on training subjects for early stopping and Youden threshold calibration ($J = \text{TPR} - \text{FPR}$).


* *Cognitive Load ($N=15$):* Outer Leave-One-Subject-Out (LOSO); train for a fixed epoch budget (derived from SAM40 inner runs) to eliminate high-variance early-stopping jitter on 3-subject inner splits.




* **Robust Aggregation Function:** For any test unit containing $K$ segments, aggregate logit medians rather than mean probabilities to resist outlier segments:

$$\text{Logit}_{\text{group}} = \text{median}\left(\log\left(\frac{p_k}{1 - p_k}\right)\right)_{k=1}^K$$


* **Evaluation Metrics:**
* **MODMA (Depression):** Primary metric is **Subject-Level ROC-AUC ($N=53$)** with 1,000-iteration bootstrap 95% confidence intervals.


* **SAM40 (Stress):** Primary metric is **Trial-Level ROC-AUC ($N=480$)** with 1,000-iteration bootstrap 95% CIs. Report mean within-subject AUC as secondary (noting the discrete 0.037 granularity over 3 relax vs 9 task pairs).


* **Cognitive Load (Stress):** Primary metric is **Recording-Level ROC-AUC ($N=120$)** with 1,000-iteration bootstrap 95% CIs.





---

**5. Benchmark Suite & Out-of-Distribution Validation**

#### Core Model Evaluation & Transfer Suite

| Experiment / Evaluation Target | Validation Scheme | Primary Metrics | Objective / Hypothesis |
| --- | --- | --- | --- |
| **Within-Corpus Performance** | Outer Nested Subject-Grouped CV

 | MODMA Subject AUC, SAM40 Trial AUC, CogLoad Recording AUC

 | Measures task decodability within consistent recording environments.

 |
| **Cross-Corpus Stress Generalization** (Headline Benchmark) | Train on SAM40 $\to$ Test on CogLoad; Train on CogLoad $\to$ Test on SAM40

 | Zero-Shot Cross-Corpus Trial/Recording ROC-AUC

 | **Direct test of hardware invariance.** Verifies that acute stress features survive an amplifier and laboratory shift.

 |
| **Dual Frozen Domain Probes** | Freeze backbone $f_\theta$; fit logistic regression on $z$ | 1. Anchor Probe Accuracy (Target: $\approx 0.333$)<br>

<br>2. Full Set Probe Accuracy (Target: $\approx 0.437$)

 | Quantifies residual corpus decodability on aligned baselines vs. full latent representations.

 |
| **Cross-Head Extrapolation Diagnostic** | Feed MODMA MDD to Stress Head; feed CogLoad Task to Depression Head

 | Output Probability Density Histograms (Target: Near 0) | Confirms heads do not produce uncalibrated false positives outside their supervised domain.

 |

#### Historical Diagnostic Context (Not for Direct Model Comparison)

Document the legacy metrics separately to contextualize the architectural changes:

* *Leaky Random Segment Split:* $\approx \mathbf{0.939}$ 3-class accuracy (demonstrates window-overlap data leakage).


* *Recomputed Corpus Shortcut Baseline:* $\approx \mathbf{0.645}$ majority accuracy on rebuilt arrays (a dummy classifier knowing only amplifier identity with zero EEG features).



#### External Transfer Limitation

Cross-corpus transfer is currently possible for stress (arithmetic/Stroop tasks shared across SAM40 and CogLoad). It cannot be performed on MODMA because no second depression dataset is present in this pipeline. Future work should evaluate the depression head on external cohorts such as PRED+CT or Mumtaz et al.