# METD — progress log

Running record of what has been built, what was measured, and what it means.
Last updated **2026-09-06**.

- `metd_plan.md` — the specification (what is true, what the thresholds are, why)
- `metd_phases.md` — the schedule (what order, what must pass before moving on)
- `REPORT.md` — generated results, regenerate with `python -m arch_metd.eval.report`
- **this file** — the narrative: decisions, bugs, findings, what is still open

---

## Phase status

| phase | state | gate |
|---|---|---|
| P0 contract & registry | ✅ complete | paths, §4 contract, 5 seeded bugs rejected |
| P1 loaders MODMA/SAM40/CogLoad | ✅ complete | SAM40 **5,280** windows, MODMA 53/53, CogLoad 103 recordings |
| P2 loaders Mumtaz/EEGMAT | ✅ complete | 115 recordings / 61 subjects, 72 / 36 paired |
| P3 preprocess + K calibration ⛔ | ✅ complete | §11 asserted on built arrays; K = 10 (1.43%) |
| P4 splits | ✅ complete | subject-disjoint, EEGMAT held out, row-order invariant |
| P5 model + training ⛔ | ✅ complete | λ floor held; 5 modes × 5 folds |
| P6 diagnostics §8 | ✅ complete | probes, gate, controls, cross-head |
| P6.5 fixes §6.5 | ✅ complete | 4 items; item 2 **failed** its gate (see below) |
| P7 transfer + report | ✅ complete | 16 runs × 2 splits, aggregated, reported |
| P8 optional | ⛔ **not recommended** | see "why P8 is the wrong move" |

Two independent split draws exist: **Run A** (`results_split_A/`, hash `22810e2548e07f90`)
and **Run B** (`results/`, hash `317d4e6107d8e065`). A result that holds on both is real; a
result that moves between them was never a result.

---

## The finding

**The mechanism works and the outcome it was built to protect has no signal to protect.**

The pre-registered frozen-probe gate — does adversarial training remove domain information
relative to its own method-off control — **passes on both draws**. On the held-out
amplifier that mechanism exists to serve, a model trained on **shuffled labels transfers as
well as either real arm**. The stress transfer is inert.

A probe can confirm a mechanism does exactly what it claims while saying nothing about
whether the task it serves is learnable at all.

### Retracted

An earlier version of this project claimed the adversary **destroyed** stress transfer,
on McNemar p = 0.0225 (split A). It did not survive split B (p = 0.5811), and the
transfer-level permutation control — run for the first time with split B — shows why:
there was no signal to destroy. **Two arbitrary draws plus a label-permutation null are
what separated a finding from a fluctuation.** Neither existed before they were asked for.

---

## Results

### Cross-corpus transfer (§6.2, §7.5) — the only device-disjoint measurements

| direction | Run A | Run B | permuted null (B) | verdict |
|---|---:|---:|---:|---|
| SAM40+CogLoad → EEGMAT (λ=1) | 0.4167 | 0.5000 | **0.5556** | **inert** |
| SAM40+CogLoad → EEGMAT (λ=0 locked) | — | 0.5833 | 0.5556 | inert |
| **Mumtaz → MODMA** | **0.6638** | **0.6667** | 0.5431 | **real, survives both** |
| **MODMA → Mumtaz** | **0.8675** | **0.8438** | 0.3998 | **real**, montage confound attached |
| EEGMAT → SAM40 | 0.3548 | 0.3539 | 0.5925 | inverted, below its own null |
| EEGMAT → CogLoad | 0.7406 | 0.8276 | 0.4803 | real |

On EEGMAT the permutation null sits **between** the two real arms. McNemar on the clean
locked contrast: **p = 0.5811**.

### Within-corpus CV, both draws

| metric | A main | B main | B λ=0 locked | B random backbone | B permuted |
|---|---:|---:|---:|---:|---:|
| depression / MODMA [subject] | 0.4784 | 0.5862 | 0.5575 | 0.5445 | 0.5934 |
| depression / Mumtaz [subject] | 0.9375 | 0.9364 | 0.9138 | 0.8912 | 0.5787 |
| stress / SAM40 [trial] | 0.5884 | 0.5717 | 0.6002 | **0.6390** | 0.5053 |
| stress / CogLoad [within-subject] | 0.8690 | 0.8405 | 0.9048 | **0.9381** | 0.5992 |
| stress / CogLoad [recording] | 0.7906 | 0.7332 | 0.7348 | **0.8415** | 0.5131 |

**MODMA moved 0.108 between two valid draws** — larger than most effects under discussion.
It is *underpowered*, not "at chance"; no claim rests on it in either direction.

**A never-trained random backbone matches or beats the trained one on every stress metric,
on both draws.** 33,287 of the model's 34,487 parameters are in the heads; the
convolutional trunk is 1,200.

### Frozen domain probes (§8.1) and the gate

| | main (λ=1) | λ=0 locked | λ=0 | random |
|---|---:|---:|---:|---:|
| anchor probe (chance 0.250) | **0.4019** | 0.4580 | 0.4613 | 0.4669 |
| full-set probe (majority 0.4943) | 0.5575 | 0.5928 | 0.5928 | 0.5998 |

**Gate: PASS on both draws** — A by 0.0010 (2/5 folds), B by 0.0094 (3/5 folds), per-fold
margin 0.0594 ± 0.0356. Also passes against the clean locked control (by 0.0061).

**It passes and the representation is not invariant**: +0.1519 above balanced chance,
+0.0632 above the majority domain. Those coexist because the gate tests *relative* removal
against the method-off control, not invariance.

### Cross-head extrapolation (§8.4)

| head | in-domain pos | neg | separation | out-of-domain |
|---|---:|---:|---:|---|
| stress | 0.520 | 0.476 | **+0.044** | 0.523 — **above** its own positives |
| depression | 0.802 | 0.245 | +0.556 | 0.325 — near its negatives |

**§6.5 item 2 gate: FAILED on Run B**, both criteria. Class weighting fixed the
calibration on split A and did not hold on B. **Calibration on a head with no
discrimination is calibration to chance** — the stress head does not work.

### Baselines (recomputed from the arrays, never hardcoded)

corpus dummy 0.6109 · majority class 0.4650 · majority domain 0.4395 ·
depression majority 0.5085 (n=24,742) · stress majority 0.6135 (n=13,707)

---

## Data findings

**Mumtaz is confounded with its acquisition setup.** EDF channel count predicts diagnosis
at **0.8889** (baseline 0.5238): 23 signals is 24 H / 1 MDD, 21 signals is 6 H / 32 MDD.
Montage alone scores 0.8685 against the model's 0.9418. Within one montage the model still
reaches 0.9107, so it is not reading setup alone — but no adequately sized subgroup holds
both classes on one montage, so the residual cannot be measured here. Full detail in
`results/mumtaz_audit.md`. **Never quote a Mumtaz figure without it.**

**CogLoad is 99.9999% DC** — a ~31,000-count pedestal with tiny AC on top. After the 1 Hz
high-pass the residue is wideband amplifier and EMG noise, leaving it with **8× EEGMAT's
above-45 Hz power after gating** (29× before). No filter reaches it; per-corpus filtering
would fix the number and make preprocessing itself the signature, which §3.2 forbids.

**Duplicate recordings, three distinct cases** — same bytes under one subject (keep one),
same bytes across subjects whose recording sets are identical throughout (merge the ids),
same bytes across subjects who otherwise hold distinct recordings (drop both). CogLoad and
Mumtaz each contain two of the three. A subject kept under two ids defeats grouped CV
silently.

**Depression N = 114 subjects** (MODMA 53 + Mumtaz 61), against the spec's ~117 estimate.

---

## Bugs found and fixed

| what | how it would have failed |
|---|---|
| `GroupKFold` tie-breaking is **sklearn-version dependent** | every SAM40 subject has 12 trials, so ties are everywhere; two machines produced splits differing on 17 of 169 subjects running identical code. Replaced with a self-contained greedy assignment — verified identical across sklearn 1.8.0 and 1.9.0 |
| λ=0 confounded the adversary with the early-stopping lock | λ=0 could select epoch 0 where λ=1 could not select before 11. Added `lambda0_locked`; verified `main` ≡ `lambda0_locked` bit-identically where the adversary is off |
| adversary ran against a **single domain** | cross-entropy over 5 logits with one class still has a gradient, and the GRL reverses it — the backbone was pushed to make a trivial prediction harder. Now returns zero below two domains |
| checkpoint selected before the adversary was on | λ at selection was 0.20–0.38 in 3/5 folds while `lambda_at_stop` read 0.74–0.82. Selection is now gated on the lock |
| predictions carried no identity | subject-level aggregation had to reconstruct it from fold indices, which worked for two corpora of four |
| `load_oof` globbed `preds_fold*.npz` | also matched `_lambda0`, `_randbb`, `_smoke` — would have pooled every mode and reported the average as the model |
| `auc_ci(n_boot=1)` | crashed when the single resample came back single-class |
| bandpass order 4 could not meet §11 | a 45 Hz corner is 11% below 50 Hz; MODMA kept a 2.3× mains peak. Order 8 |
| `normalize` on an empty array | a recording fully rejected by the gates killed the build |

---

## Decisions taken

| decision | by | rationale |
|---|---|---|
| root at `arch_metd/`, pure torch | assistant | `import src.*` drags in the whole TF stack; GRL is 8 lines in torch |
| `BAND_ORDER = 8`, not §3.1's 4 | user | order 4 cannot meet §11's mains test |
| Mumtaz whole-subject duplicates → merge, not drop | user | class label unambiguous; only the subject number was |
| §8.2 replaced by a frozen-probe gate | user | silhouette landed within ±0.004 of zero in every mode — it could not adjudicate |
| keep Run A, add Run B as a second draw | user | two valid draws are a robustness check, not a supersession |
| greedy split, not a version pin | user | pinning would version-lock an arbitrary tie-break and call it reproducible |

---

## Open items

1. **`run_local.sh`** — requested, not delivered.
2. **`cv_permuted` is not split-matched** — permuting `y_dep` changes the stratified split
   (hash `b94c96e0d8848d03`). Valid null, imperfect control.
3. **`results_split_A/` predates the locked arm** — the draws are comparable on
   main/λ0/randbb only.
4. **`arch_metd/results/` is 43 MB of `.pt`/`.npz`**, now gitignored, so Run A's binaries
   live on disk only.

## Why P8 is the wrong move

P8 is aux heads, the `filtered_data` ablation, and TDBRAIN. None addresses what the results
say. Its one item with a claim here — the CogLoad ordinal head, which forces a
within-subject arousal axis that cannot be device identity — is a probe of *why* the stress
head failed, and the permutation control says the cross-device stress target is inert, so a
better stress representation has nothing to be measured against.

**The work worth doing is the writeup.** §13 advises headlining stress; that advice is now
inverted by evidence. The defensible structure is: depression transfer as the positive
result, stress transfer as a *demonstrated* null, and the probe/transfer dissociation as
the methodological contribution.
