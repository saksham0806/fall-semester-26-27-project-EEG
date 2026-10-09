# Feature Reference

Every feature extracted in this project, in exact column order.

> **Two versions exist.** §1–2 document **v1** (81 handcrafted + 105 Riemannian,
> `src/proposed_model/features.py`), which produced every published result and is unmodified.
> **§4 documents v2** (273 handcrafted, `src/proposed_model/features_v2.py`), which corrects a
> measured spectral defect in v1 and adds four feature families v1 lacks. Read §4 before using
> v2 numbers — the two are **not** interchangeable.

**Common input:** 6 frontal EEG channels at 128 Hz, 4-second windows = `(N, 6, 512)`.
**Channels (fixed order):** `Fp1, Fp2, F7, F3, F4, F8` (`CHANNEL_MAP`, `src/proposed_model/features.py`).
**Bands:** delta 0.5–4 · theta 4–8 · alpha 8–12 · beta 12–30 · gamma 30–45 Hz (`BANDS`).

| View | Dims | Source | On disk? |
|---|---|---|---|
| v1 handcrafted | **81** | `extract_all_features()` in `features.py` | computed inline by each script; `features/all3/` **never generated** |
| v1 Riemannian tangent | **105** | `riemannian_features()` / `bandwise_tangent()` | cached at `datasets/all3/_cache_riemann_*.npy` |
| **v2 handcrafted** | **273** | `extract_all_features_v2()` in `features_v2.py` | ✅ `features_v2/all3/` — 15,728 × 273, generated in 168 s |
| **v2 Riemannian tangent** | **105 × 2** | `riemannian()` in `scripts/extract_features_v2.py` | ✅ `features_v2/all3/riemannian_{percorpus,global}.npy` |

Full v2 column-by-column index: **§5**.

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

> **`features/all3/` has never been generated** — `scripts/extract_all3_features.py` has not
> been run. Every script that needs the v1 81 recomputes them inline via
> `extract_all_features()`. The definitions above come straight from the code and are correct
> regardless. (v2 *has* been generated — see §4e.)

---

## 4. v2 — `src/proposed_model/features_v2.py`

### 4a. Why v2 exists — a measured defect in v1

v1's relative band power divides by total power integrated over the **full 0–64 Hz**
(`features.py:90`), but the five bands span only **0.5–45 Hz**. The 45–64 Hz gap is therefore
in every denominator while contributing to no numerator. Measured on `datasets/all3/X.npy`:

| corpus | power above 45 Hz | shape |
|---|---|---|
| MODMA | ~88–98% of total | sharp 50 Hz mains spike |
| CogLoad | ~19–75% | weak 50 Hz spike |
| SAM40 | ~34% | **no line peak** — broadband (vendor already notched) |

The contaminant's *shape* differs per corpus, not just its size — so it is a per-device
signature sitting inside 30 of the 81 v1 features, plus spectral entropy, plus (via
`combine_datasets.py`'s global z-scoring) the amplitude features.

### 4a-bis. The motivating hypothesis was TESTED and REFUTED

v2 was built to test whether this contamination drives the 91.6% corpus recoverability behind
the all3 confound. **It does not.** Measured on all 15,728 segments, 5-fold subject-grouped,
RBF SVM predicting corpus:

| features | preprocessing | corpus probe |
|---|---|---|
| v1's 81 | none, 0–64 Hz denominator | 0.9161 |
| **the same 81** | **notch + 0.5–45 bandpass, matched denominator** | **0.9057** |
| all 273 | notch + bandpass | 0.9703 |

Removing 34–98% of total spectral power changed device recoverability by **−0.010**. The
amplifier signature is therefore *inside the 0.5–45 Hz band*, not in the discarded gap — which
makes it much harder to remove than a filter.

Two further consequences, both worth knowing before building on v2:

- **More features make the shortcut easier.** Going 81 → 273 raised the probe by **+0.065**.
  Every family added here gives a corpus classifier more to work with. For the all3 3-class
  framing that is actively counterproductive.
- **A 600-segment pilot pointed the opposite way** (−0.124) purely because 273 features are
  starved at that sample size. Do not trust a probe measured on a subset; dimensionality and
  sample size interact.

**This does not make v2 useless.** Corpus recoverability and class discrimination are separate
questions — v2's value for within-corpus depression and stress accuracy is untested and
remains plausible, since connectivity and aperiodic features are genuinely new information.
It does mean v2 is *not* a fix for the confound, and should not be presented as one.

### 4b. v2 preprocessing

Applied once, before any feature is computed:

1. **Notch at 50 Hz** (`iirnotch`, Q=30)
2. **Bandpass 0.5–45 Hz, Butterworth order 8, SOS, zero-phase** (`sosfiltfilt`)

Both stages are required — measured, not assumed:

| design | MODMA | SAM40 | CogLoad |
|---|---|---|---|
| none (= v1) | 0.883 | 0.341 | 0.191 |
| bandpass order 4 @45 | 0.092 | 0.008 | 0.007 |
| bandpass order 8 @45 | 0.0007 | 0.0043 | 0.0013 |
| **notch50 + order 8 @45** | **0.0001** | **0.0039** | **0.0012** |

Order 4 is insufficient: 45 and 50 Hz are only 11% apart, and Butterworth roll-off cannot
separate them. A `@42 Hz` corner scores better still but was **rejected** — it attenuates the
top of the gamma band (30–45 Hz) and would distort a real feature.

Windows are **not** re-normalised after filtering, so amplitude features stay on v1's scale.
Expect MODMA's RMS to drop sharply: the mains hum was most of its apparent amplitude.

Per-band filters (DE, PLV, Riemannian) stay at **order 4** — they run after preprocessing and
never see the gap.

### 4c. v2 blocks — 273 handcrafted columns

Column order of `combined.npy` is the order below. `features.json` in the output directory
carries per-block names and index ranges; `block_names()` is the source of truth.

| block | dim | cols | content |
|---|---|---|---|
| `bandpower_rel` | 30 | 0–29 | relative band power, **denominator now 0.5–45 Hz** — band-major, as v1 |
| `de` | 30 | 30–59 | differential entropy, zero-phase filtered |
| `faa` | 3 | 60–62 | frontal alpha asymmetry (unchanged definition) |
| `amplitude` | 12 | 63–74 | STD (6) then RMS (6) |
| `spectral_shape` | 24 | 75–98 | spectral entropy (6), 95% edge freq (6), 1/f slope (6), 1/f offset (6) |
| `hjorth` | 18 | 99–116 | activity (6), mobility (6), complexity (6) |
| `perm_entropy` | 6 | 117–122 | permutation entropy, order 3, delay 1, normalised to [0,1] |
| `coherence` | 75 | 123–197 | magnitude-squared coherence, band-major: 5 bands × 15 pairs |
| `plv` | 75 | 198–272 | phase-locking value, same layout |

Channel pairs are `itertools.combinations(range(6), 2)` in order:
`Fp1_Fp2, Fp1_F7, Fp1_F3, Fp1_F4, Fp1_F8, Fp2_F7, Fp2_F3, Fp2_F4, Fp2_F8, F7_F3, F7_F4,
F7_F8, F3_F4, F3_F8, F4_F8`.

**What is genuinely new** (none of these exist anywhere in v1 or the wider repo):

- **1/f aperiodic slope + offset** — the non-oscillatory component band power ignores entirely.
  Fitted as least squares of `log10(PSD)` on `log10(f)` over **2–45 Hz**; the fit starts at
  2 Hz because below that the bandpass roll-off would dominate and the slope would describe
  the filter rather than the brain.
- **Hjorth parameters** — activity `var(x)`, mobility `sqrt(var(x')/var(x))`,
  complexity `mobility(x')/mobility(x)`.
- **Permutation entropy** — ordinal-pattern complexity; nonlinear temporal structure that no
  spectral feature captures.
- **Coherence and PLV** — the repo previously had *no* connectivity measure beyond covariance.
  **PLV is the load-bearing one**: it discards amplitude entirely and measures phase coupling,
  which the Riemannian covariance view is structurally blind to. Coherence is partly redundant
  with per-band covariance; PLV is not.

### 4d. Two implementation choices that would fail silently if wrong

- **Coherence uses `nperseg=128, noverlap=64`** → 7 averaged segments across a 512-sample
  window. At `nperseg=256` only 2 segments are averaged and coherence is biased hard toward
  1.0 — producing a plausible-looking but meaningless block.
- **PLV is chunked band-by-band.** One band of Hilbert output at full size is ~770 MB as
  complex128; a naive 5-band pass would swap on a 15 GB machine.

### 4e. Output — `features_v2/all3/` (generated)

**Status: generated on the full dataset.** 15,728 segments, `smoke_mode: false`, 168 s total
(filter 3.3 s, nine blocks 44.9 s, Riemannian 100.3 s, corpus probe 18 s). CPU only.

One `.npy` per block (named as above), plus `combined.npy` (15728 × 273),
`riemannian_percorpus.npy` and `riemannian_global.npy` (15728 × 105 each), `y.npy` /
`corpus.npy` / `subjects.npy`, and `features.json` recording block index ranges, all 273
column names, the exact preprocessing parameters, measured before/after gap power per corpus,
and the paired corpus-probe result.

Measured filter effect on the full set (fraction of power above 45 Hz):

| corpus | before | after |
|---|---|---|
| MODMA | 0.8389 | 0.00012 |
| SAM40 | 0.3478 | 0.00401 |
| CogLoad | 0.2293 | 0.00119 |

```python
import numpy as np, json
F = np.load("features_v2/all3/combined.npy")            # (15728, 273)
y = np.load("features_v2/all3/y.npy")
groups = np.load("features_v2/all3/subjects.npy", allow_pickle=True)   # split on this
idx = json.load(open("features_v2/all3/features.json"))["blocks"]
plv = F[:, slice(*idx["plv"]["cols"])]                  # one block by name
```

The block layout makes **leave-one-block-out ablation a slicing operation**, which is what
answers "which features actually matter" — never measured for v1.


---

## 5. Complete v2 feature list — all 273 columns

Every column of `features_v2/all3/combined.npy`, with its exact index. Generated from
`features.json`, so this table and the data cannot disagree.

### bandpower_rel — cols 0–29 (Relative band power)

| band | Fp1 | Fp2 | F7 | F3 | F4 | F8 |
|---|---|---|---|---|---|---|
| delta | 0 | 1 | 2 | 3 | 4 | 5 |
| theta | 6 | 7 | 8 | 9 | 10 | 11 |
| alpha | 12 | 13 | 14 | 15 | 16 | 17 |
| beta | 18 | 19 | 20 | 21 | 22 | 23 |
| gamma | 24 | 25 | 26 | 27 | 28 | 29 |

### de — cols 30–59 (Differential entropy)

| band | Fp1 | Fp2 | F7 | F3 | F4 | F8 |
|---|---|---|---|---|---|---|
| delta | 30 | 31 | 32 | 33 | 34 | 35 |
| theta | 36 | 37 | 38 | 39 | 40 | 41 |
| alpha | 42 | 43 | 44 | 45 | 46 | 47 |
| beta | 48 | 49 | 50 | 51 | 52 | 53 |
| gamma | 54 | 55 | 56 | 57 | 58 | 59 |

### faa — cols 60–62

| col | name | pair |
|---|---|---|
| 60 | `faa_Fp2_Fp1` | Fp2 − Fp1 |
| 61 | `faa_F4_F3` | F4 − F3 |
| 62 | `faa_F8_F7` | F8 − F7 |

### amplitude — cols 63–74

| feature | Fp1 | Fp2 | F7 | F3 | F4 | F8 |
|---|---|---|---|---|---|---|
| STD (`std_*`) | 63 | 64 | 65 | 66 | 67 | 68 |
| RMS (`rms_*`) | 69 | 70 | 71 | 72 | 73 | 74 |

### spectral_shape — cols 75–98

| feature | Fp1 | Fp2 | F7 | F3 | F4 | F8 |
|---|---|---|---|---|---|---|
| Spectral entropy (`specent_*`) | 75 | 76 | 77 | 78 | 79 | 80 |
| 95% spectral edge freq (`edge95_*`) | 81 | 82 | 83 | 84 | 85 | 86 |
| 1/f slope (`aperiodic_slope_*`) | 87 | 88 | 89 | 90 | 91 | 92 |
| 1/f offset (`aperiodic_offset_*`) | 93 | 94 | 95 | 96 | 97 | 98 |

### hjorth — cols 99–116

| feature | Fp1 | Fp2 | F7 | F3 | F4 | F8 |
|---|---|---|---|---|---|---|
| Hjorth activity (`hjorth_activity_*`) | 99 | 100 | 101 | 102 | 103 | 104 |
| Hjorth mobility (`hjorth_mobility_*`) | 105 | 106 | 107 | 108 | 109 | 110 |
| Hjorth complexity (`hjorth_complexity_*`) | 111 | 112 | 113 | 114 | 115 | 116 |

### perm_entropy — cols 117–122

| feature | Fp1 | Fp2 | F7 | F3 | F4 | F8 |
|---|---|---|---|---|---|---|
| Permutation entropy (`perment_*`) | 117 | 118 | 119 | 120 | 121 | 122 |

### coherence — cols 123–197 (Magnitude-squared coherence, 15 pairs × 5 bands)

| channel pair | delta | theta | alpha | beta | gamma |
|---|---|---|---|---|---|
| Fp1_Fp2 | 123 | 138 | 153 | 168 | 183 |
| Fp1_F7 | 124 | 139 | 154 | 169 | 184 |
| Fp1_F3 | 125 | 140 | 155 | 170 | 185 |
| Fp1_F4 | 126 | 141 | 156 | 171 | 186 |
| Fp1_F8 | 127 | 142 | 157 | 172 | 187 |
| Fp2_F7 | 128 | 143 | 158 | 173 | 188 |
| Fp2_F3 | 129 | 144 | 159 | 174 | 189 |
| Fp2_F4 | 130 | 145 | 160 | 175 | 190 |
| Fp2_F8 | 131 | 146 | 161 | 176 | 191 |
| F7_F3 | 132 | 147 | 162 | 177 | 192 |
| F7_F4 | 133 | 148 | 163 | 178 | 193 |
| F7_F8 | 134 | 149 | 164 | 179 | 194 |
| F3_F4 | 135 | 150 | 165 | 180 | 195 |
| F3_F8 | 136 | 151 | 166 | 181 | 196 |
| F4_F8 | 137 | 152 | 167 | 182 | 197 |

### plv — cols 198–272 (Phase-locking value, 15 pairs × 5 bands)

| channel pair | delta | theta | alpha | beta | gamma |
|---|---|---|---|---|---|
| Fp1_Fp2 | 198 | 213 | 228 | 243 | 258 |
| Fp1_F7 | 199 | 214 | 229 | 244 | 259 |
| Fp1_F3 | 200 | 215 | 230 | 245 | 260 |
| Fp1_F4 | 201 | 216 | 231 | 246 | 261 |
| Fp1_F8 | 202 | 217 | 232 | 247 | 262 |
| Fp2_F7 | 203 | 218 | 233 | 248 | 263 |
| Fp2_F3 | 204 | 219 | 234 | 249 | 264 |
| Fp2_F4 | 205 | 220 | 235 | 250 | 265 |
| Fp2_F8 | 206 | 221 | 236 | 251 | 266 |
| F7_F3 | 207 | 222 | 237 | 252 | 267 |
| F7_F4 | 208 | 223 | 238 | 253 | 268 |
| F7_F8 | 209 | 224 | 239 | 254 | 269 |
| F3_F4 | 210 | 225 | 240 | 255 | 270 |
| F3_F8 | 211 | 226 | 241 | 256 | 271 |
| F4_F8 | 212 | 227 | 242 | 257 | 272 |

All 273 indices above verified against `features.json` at generation time.
