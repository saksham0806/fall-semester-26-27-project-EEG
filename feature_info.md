# Feature Reference

Every feature extracted in this project, in exact column order.

**Common input:** 6 frontal EEG channels at 128 Hz, 4-second windows = `(N, 6, 512)`.
**Channels (fixed order):** `Fp1, Fp2, F7, F3, F4, F8` (`CHANNEL_MAP`, `src/proposed_model/features.py`).
**Bands:** delta 0.5–4 · theta 4–8 · alpha 8–12 · beta 12–30 · gamma 30–45 Hz (`BANDS`).

Two feature families exist:

| View | Dims | Source | Used by |
|---|---|---|---|
| Handcrafted | **81** | `extract_all_features()` in `features.py` | every classical model, DSFFN second input |
| Riemannian tangent | **105** | `riemannian_features()` / `bandwise_tangent()` | M1 stress model, all3 Riemannian rows |

---

## 1. Handcrafted — 81 features

Concatenation order (the `np.hstack` in `extract_all_features`, `features.py:197`):

`[ relative PSD (30) | differential entropy (30) | FAA (3) | spectral entropy (6) | STD (6) | RMS (6) ]`

All spectral features use Welch PSD, `nperseg = min(256, n_samples)` → 0.5 Hz resolution.

### 1a. Relative band power — 30 (cols 0–29)
`extract_psd_features(relative=True)`. Band power = ∫PSD over the band, divided by total power
(0–64 Hz). **Band-major layout**: all 6 channels of delta, then theta, alpha, beta, gamma.

| cols | band | channels (in order) |
|---|---|---|
| 0–5 | delta | Fp1 Fp2 F7 F3 F4 F8 |
| 6–11 | theta | Fp1 Fp2 F7 F3 F4 F8 |
| 12–17 | alpha | Fp1 Fp2 F7 F3 F4 F8 |
| 18–23 | beta | Fp1 Fp2 F7 F3 F4 F8 |
| 24–29 | gamma | Fp1 Fp2 F7 F3 F4 F8 |

### 1b. Differential entropy — 30 (cols 30–59)
`extract_differential_entropy()`. Per band: bandpass filter (Butterworth order 3), then
`DE = 0.5·ln(2πe·σ²)` on the filtered signal's variance. For a Gaussian, DE is a monotone
function of log band power. Same band-major, channel-minor layout as 1a.

| cols | band | channels |
|---|---|---|
| 30–35 | delta | Fp1 … F8 |
| 36–41 | theta | Fp1 … F8 |
| 42–47 | alpha | Fp1 … F8 |
| 48–53 | beta | Fp1 … F8 |
| 54–59 | gamma | Fp1 … F8 |

### 1c. Frontal alpha asymmetry — 3 (cols 60–62)
`extract_faa_features()`. `FAA = ln(alpha_power_right) − ln(alpha_power_left)`. A
well-established **valence** marker (not arousal — see the note in `CLAUDE.md`).

| col | pair (right − left) | region |
|---|---|---|
| 60 | Fp2 − Fp1 | prefrontal |
| 61 | F4 − F3 | mid-frontal |
| 62 | F8 − F7 | lateral frontal |

### 1d. Spectral entropy — 6 (cols 63–68)
`extract_spectral_entropy()`. Shannon entropy (log base 2) of the per-channel PSD normalized
to a probability distribution. One value per channel.

| cols | channels |
|---|---|
| 63–68 | Fp1 Fp2 F7 F3 F4 F8 |

### 1e. Standard deviation — 6 (cols 69–74)
`extract_amplitude_features()`, first half. `std` over time, per channel.

| cols | channels |
|---|---|
| 69–74 | Fp1 Fp2 F7 F3 F4 F8 |

### 1f. RMS — 6 (cols 75–80)
`extract_amplitude_features()`, second half. `sqrt(mean(x²))` over time, per channel.

| cols | channels |
|---|---|
| 75–80 | Fp1 Fp2 F7 F3 F4 F8 |

> Column names are generated programmatically by `handcrafted_names()` in
> `scripts/extract_all3_features.py`, in exactly this order — that function is the
> machine-readable source of truth for these 81 labels.

---

## 2. Riemannian tangent space — 105 features

`riemannian_features(X, group)` (`scripts/all3_diagnostic.py:59`) /
`bandwise_tangent()` (`scripts/phase2_riemannian.py:63`). Not per-channel — these are
geometric features of the inter-channel covariance structure.

Per band: bandpass filter → **6×6 covariance matrix** (`Covariances(estimator="oas")`) →
re-centre by the group's Riemannian mean → project to the tangent space at the identity.
A 6×6 symmetric matrix has `6·7/2 = 21` unique entries → **21 tangent dims per band**.

`5 bands × 21 = 105`, hstacked in band order:

| cols | band | content |
|---|---|---|
| 0–20 | delta | 21 tangent-space coords of the delta-band 6×6 covariance |
| 21–41 | theta | " |
| 42–62 | alpha | " |
| 63–83 | beta | " |
| 84–104 | gamma | " |

### Two variants — the re-centering group is load-bearing

The only difference is what `mean_riemann` is computed over before whitening:

| variant | group | file | effect |
|---|---|---|---|
| `riemannian_percorpus` | corpus | `all3_diagnostic.py` | whitens each device's mean → **removes** the amplifier signature. Honest for all3 (drops to chance). |
| `riemannian_global` | all segments | `phase2_channel_density.py`, all3 fusion | one reference → **keeps** the signature. |
| (stress) subject | subject | `phase2_riemannian.py` | removes per-subject offset; correct for within-subject stress contrast. |
| (depression) global | all segments | `phase2_riemannian.py` | per-subject centering would delete the between-subject label → AUC 0.5. |

Getting the group wrong fails **silently** — see the "Re-centering group choice is
load-bearing" section in `CLAUDE.md`.

---

## 3. Where the matrices live

Once `scripts/extract_all3_features.py` is run, `features/all3/` holds:

| file | shape | this doc |
|---|---|---|
| `handcrafted.npy` | (15728, 81) | §1 |
| `riemannian_percorpus.npy` | (15728, 105) | §2, percorpus |
| `riemannian_global.npy` | (15728, 105) | §2, global |
| `y.npy` `corpus.npy` `subjects.npy` | (15728,) | labels / corpus id / subject id |
| `features.json` | — | dims + `handcrafted_names()` output |

> As of this writing the extractor has **not** been run, so `features/all3/` does not exist
> yet. The definitions above come straight from the code and are correct regardless.
