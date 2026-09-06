# METD Build — Phase Plan

Execution plan for `metd_plan.md`. That file is the **specification** (what is true,
what the thresholds are, why); this file is the **schedule** (what order, what must
pass before moving on). Section references like §3.2 point into `metd_plan.md` — do
not duplicate its content here, read it there.

**All five corpora are on disk.** The spec's §1.2 "to download" is void; §14 step 3
loses its download dependency. Verified paths and counts are in Phase 1/2 below.

## Rules that apply to every phase

- **Every phase ends at a gate that is a runnable command, not a judgement.** If the
  gate does not fail loudly when the phase is wrong, the gate is not written yet.
- **Do not start phase N+1 to "unblock" a red gate at phase N.** The two stop-and-read
  gates (P3, P5) exist because the previous pipeline shipped a result that was 92%
  amplifier identity and nothing downstream noticed.
- Each phase leaves one `assert`-based self-check runnable as `python -m arch_metd.<mod>`.
  No test framework — the repo has none and does not need one for this.
- **Every phase ends with a "Your turn" block** listing what you have to do before it
  starts and what you have to read once it finishes. Where it says nothing, do nothing.
- **I ask before running anything**, per your `CLAUDE.md` §5 — loaders included, not just
  training. `RUN.md` gets the new commands as each phase lands, not at the end.

## Two decisions made up front

**Root the build at `arch_metd/`, not `src/`.** The spec's §10 layout is kept verbatim
*inside* it (`arch_metd/data/loaders/`, `arch_metd/models/`, …). Reason: `import src.*`
triggers `src/proposed_model/__init__.py`, which imports the entire TF **and** torch
stack (documented in CLAUDE.md's Gotchas). A clean corpus loader should not cost 40 s
and 2 GB of import. Nothing in this build imports the legacy `src/` package.

**Torch, not Keras.** The GRL (§5.1) is ~8 lines as a `torch.autograd.Function` and
awkward under Keras 3. `DL_Without_Features/` already establishes the pure-torch
pattern in this repo. Local torch is CPU-only (2.13.0+cpu), so P5 onward runs on the
GPU box per `GPU_Run.md`; P1–P4 are CPU work and run locally.

---

## Phase 0 — Contract and skeleton

**Goal.** The array contract exists and is checkable before any corpus is read.

**Build.**
- `arch_metd/contract.py` — the §4 dtypes/fields, plus a `CORPORA` registry: one row
  per domain holding `domain`, `path`, `fs`, `protocol`, `task`, `channel_rows`.
  This registry is the only place a path or a channel index is written down.
- `arch_metd/{data,models,train,eval,infer}/` package dirs, `configs/`, `results/`.

**Gate.** `python -m arch_metd.contract` — asserts on a synthetic 10-segment batch:
dtypes match §4, `y_domain ⊆ {0..4}`, `anchor == (y_dep==0)|(y_str==0)`, no segment
carries both labels, `protocol` is `resting` for {0,3} and `task` for {1,2,4}, and
every `CORPORA` path exists on disk.

**Effort.** Small. Do not expand it — the registry is a dict, not a config framework.

**Your turn.**
- *Before:* confirm or overrule the two decisions above (`arch_metd/` root, torch). Nothing else — this phase touches no data.
- *After:* nothing.

---

## Phase 1 — Loaders A: MODMA, SAM40, CogLoad

The three corpora with known, previously-shipped bugs. Spec §2.1–2.3.

| corpus | domain | path (verified) |
|---|---:|---|
| MODMA | 0 | `datasets/MODMA/EEG_128channels_resting_lanzhou_2015/` |
| SAM40 | 1 | `datasets/SAM40/raw_data/` |
| CogLoad | 2 | `datasets/Cognitive Load Assessment Through EEG A Dataset from Arithmetic and Stroop Tasks/raw_data/` |

**Build.** `arch_metd/data/loaders/{modma,sam40,cogload}.py`, each returning
`[(array (6, n_samples) at native fs, metadata)]`. No filtering, no windowing — that
is Phase 3. Loaders only read, select channels, and attach labels.

**Gate** (§11 Ingestion, the subset covering these three):
- SAM40 **is not decimated**; a 3200-sample trial yields **11** windows at 512/256,
  giving **5,280** pre-gate segments. 5 windows / 2,400 segments means the `[:, ::2]`
  bug is back. This is spec §12 item 1 and the most damaging bug in the old pipeline.
- MODMA subject IDs resolve **53/53** via `re.match(r"0*(\d+)", filename)` against the
  spreadsheet; splitting on `rest` or `_` gives 37 and is wrong.
- MODMA `.mat` key search requires `shape[1] > 1000` (rejects `Impedances_0`), and row
  128 (`E129`) is all-zero in every file and never selected.
- CogLoad glob matches the **trailing-space** folder name, and subject IDs collapse to
  **15**, not 120. Log the final recording count and the dropped cross-subject duplicate.

**Your turn.**
- *Before:* nothing. Loaders only read; no corpus file is written to.
- *After:* eyeball three numbers in the gate output — SAM40 **5,280** pre-gate segments, MODMA **53/53** subject IDs, CogLoad **15** subject IDs plus the final recording count. If SAM40 reports 2,400, stop me: the decimation bug is back and everything downstream is time-stretched.

---

## Phase 2 — Loaders B: Mumtaz, EEGMAT (EDF)

Different shape of work — `mne` 1.12.1 is installed and declared, so this is a read
problem, not an environment problem. Spec §2.4–2.5.

| corpus | domain | path (verified) | files |
|---|---:|---|---:|
| Mumtaz | 3 | `datasets/Mumtaz/` | 181 `.edf` (58 EC, 62 EO, rest TASK) |
| EEGMAT | 4 | `datasets/eeg-during-mental-arithmetic-tasks-1.0.0/` | 72 `.edf` + `subject-info.csv` |

**Three Mumtaz filename facts measured on this copy, all traps:**
- **`H S1` and `MDD S1` are different people.** Subject numbering restarts per group,
  so `subject_id` must carry the group (`mumtaz_H1` / `mumtaz_MDD1`). Keying on the
  digit run alone silently merges 34 pairs of subjects across the class boundary.
- **Spacing is inconsistent** — `MDD S1  EO.edf` has a double space, `MDD S30 TASK.edf`
  a single. Parse with `re.match(r".*?(H|MDD)\s+S(\d+)\s+(EC|EO|TASK)", fn, re.I)`.
- **Two duplicate files carry numeric prefixes**: `6921143_H S15 EO.edf` and
  `6921959_H S15 EO.edf`. Same subject, same session, two copies. Pick one
  deterministically or drop both — but log it, per the CogLoad duplicate precedent.

**Build.** `mumtaz.py`, `eegmat.py`. Mumtaz: use **EC and EO, drop TASK** (evoked
potential, not resting). EEGMAT: `_1.edf` → `y_str = 0`, `_2.edf` → `y_str = 1`.

**Gate.**
- All six frontal channel names resolve in **every** EDF, matched case-insensitively on
  the bare label after stripping `EEG ` prefixes and `-LE`/`-REF` suffixes. Assert, do
  not warn.
- Session inventory printed: which Mumtaz subjects lack EC, lack EO, or have TASK only.
  Expect **fewer than 64 usable subjects** — the number is a result, log it.
- EEGMAT yields exactly 36 subjects × 2 recordings, paired.
- Subject-ID collision check: `len(set(subject_ids)) == 34 + n_mdd`, i.e. no H/MDD merge.

**Your turn.**
- *Before:* decide the duplicate policy for `6921143_H S15 EO.edf` / `6921959_H S15 EO.edf` — keep one deterministically, or drop both. My default is drop both, following the CogLoad duplicate precedent; say so if you want one kept.
- *After:* read the Mumtaz session inventory. The usable-subject count that comes out of it **is** the depression N you report in the paper, and it will be under 64. Confirm the number before P3 bakes it into the arrays.

---

## Phase 3 — Preprocess, build, and **K calibration** ⛔ STOP AND READ

The spec's own verdict (§14): *"Steps 1–2 will tell you more about whether this works
than any amount of further architecture work."* This is that step.

**Build.**
- `arch_metd/data/preprocess.py` — the §3 chain, in order, no reordering: mark flatlines
  (mark, never excise) → reflect-pad 3.0 s → 50 Hz notch → 1–45 Hz Butterworth SOS
  `sosfiltfilt` → strip pad → `resample_poly` to 128 Hz → window 512/256 → reject
  flatline-overlapping → reject variance-gate failures → per-segment normalize (§3.3,
  **global over the whole (6,512), not per channel**).
- `arch_metd/data/build.py` — loaders → arrays → disk, **shuffled on write**, emitting
  the rejection table. Re-runnable, and it **deletes any cached array on rebuild**
  (spec §10; filename-keyed caches have shipped broken artifacts in this repo twice).

**The one thing to actually calibrate.** `K` in the channel-variance gate (§3.2) is
**swept on MODMA**, not assumed. Pick the K that rejects 1–2%. `K = 5` rejected ~9.7%
and is too aggressive. The gate runs on **filtered-but-unnormalized** data.

**Gate** (§11 Preprocessing) — and then *stop and read the output before P4*:
- 49–51 Hz band power `< 0.001` of total, **every corpus** (MODMA was 92.3% pre-notch).
- No NaN/Inf in `X`. Per-segment `|mean| < 1e-6`, `|std - 1| < 1e-6`.
- SAM40 mean absolute amplitude is flat across trial time — first/last second above the
  middle means the filter transient survived the reflect padding.
- Rejection table published: rows = corpus, columns = recordings, windows, kept,
  flatline-rejected, variance-rejected, %. **Divergent rates across corpora are a
  finding.** A gate that rejects unevenly makes the rejection rate itself a corpus
  signature (§3.2).
- **Corpus-dummy baseline recomputed from the rebuilt arrays** and written to
  `results/baselines.json`. The old figure was 0.6755 — do not carry it over, and do
  not hardcode any baseline anywhere (§12 item 10).
- Label asserts (§11 Labels): `y_dep == -1` across domains {1,2,4}, `y_str == -1` across
  {0,3}, no segment with both, class counts match the source spreadsheets exactly.

**Your turn.**
- *Before:* run `free -h` and close the browser/Discord/IDE if it is tight — the box is ~15 GB and a full build pushes it into swap. Then say go: this phase **writes new arrays to disk**, and I will not start it without that. Nothing existing is overwritten; the output is a new directory.
- *After:* **this is the real checkpoint — read two things before I go anywhere near P4.**
  1. **The rejection table.** Divergent rejection rates across corpora are a finding about the corpora, not a number I can adjudicate for you. If one corpus loses 15% of its windows and another loses 1%, that asymmetry becomes a corpus signature and you decide whether to retune `K`, accept it and document it, or stop.
  2. **The recomputed corpus-dummy baseline.** It replaces the old 0.6755, and every headline number in P7 gets quoted against it. If it lands far from 0.6755, that is worth understanding now rather than in the writeup.

---

## Phase 4 — Splits

Written **before** any model, because leakage is a splitting bug and it is cheap to
prove correct in isolation. Spec §6.1–6.2.

**Build.** `arch_metd/train/folds.py` — outer `StratifiedGroupKFold(5)` on `subject_id`
per corpus unioned into joint folds, inner `GroupKFold(4)`. MODMA and Mumtaz stratified
on the class label (plain 5-fold on 53 subjects can hand you a fold with almost no MDD).
**CogLoad uses outer LOSO with a fixed epoch budget** derived from the SAM40 inner runs
— 14–15 subjects cannot support a 3-subject inner split.

**Gate** (§11 Splitting) — pure assertions over generated fold indices, no training:
- No `subject_id` in both sides of any split, inner or outer.
- **EEGMAT appears in zero training folds.**
- In the depression transfer runs, MODMA and Mumtaz never sit on the same side.
- `y_domain ⊆ {0..4}`; index 5 (TDBRAIN) never emitted.
- Row alignment: any stage that drops rows applies the same mask to `y`, `subjects`,
  `domain`. Assert `len` equality across all five arrays after every masking step.

**Your turn.**
- *Before:* nothing.
- *After:* nothing — the gate is pure assertions over fold indices and fails loudly on its own. No judgement call here.

---

## Phase 5 — Model and training loop ⛔ SIGN CHECK FIRST

Spec §5–6.3. Runs on the GPU box (`GPU_Run.md`); local torch is CPU-only.

**Build.**
- `models/eegnet.py` (EEGNet-8,2 backbone → `z ∈ R^256`), `models/grl.py` (reversal +
  the §5.3 λ schedule), `models/heads.py` (masked BCE heads + inverse-anchor-frequency
  weighted adversary), `models/mtdann.py` (assembly).
- `train/loop.py` — AdamW 1e-3 / wd 1e-2, cosine warm restarts, batch 64 with
  **corpus-balanced sampling**, early stopping patience 15 on inner-fold sum of valid
  AUCs, **locked until λ ≥ 0.5**, and λ-at-stop logged for every fold.

**Three things that are load-bearing and easy to undo:**
- The adversary sees **anchor segments only** (§5.1). A marginal adversary over all
  segments is asked to erase MODMA-MDD vs SAM40-task — different classes in different
  domains — so erasing domain erases the label.
- **Prefer GroupNorm over BatchNorm**, or verify BN running stats survive single-corpus
  test batches. A held-out-corpus test batch is by definition single-domain.
- **Every run has a `λ = 0` twin** as the method-off control. The DANN post-mortem in
  CLAUDE.md is exactly what happens without it.

**Gate — run this before accepting any number:**
- **§8.6 sign check.** A prior run returned MODMA subject-level AUC **0.3434** with one
  fold at exactly 0.0000. That is inversion, not chance — a model that found signal and
  assigned it backwards. Assert the label join: `int()` of the MODMA digit run lands on
  the correct spreadsheet row. **Do not accept any depression AUC below 0.5 without
  clearing this.**
- λ at the stopping epoch logged for every fold, and `≥ 0.5` in all of them.

**Your turn.**
- *Before:* three things, and I do not start training until all three land.
  1. GPU box access (`GPU_Run.md`) — local torch is CPU-only, so this phase cannot run here.
  2. If P0–P4 added any dependency, `Dockerfile` gets updated **in the same edit** as `requirements.txt`. A stale image fails after the upload and the queue wait, not before.
  3. rsync the **built arrays**, not the raw corpora. Say go explicitly — per your CLAUDE.md I never start a training run on my own.
- *After:* read the λ-at-stop value for every fold (all should be ≥ 0.5) and the §8.6 sign-check result. **Any depression AUC below 0.5 is inversion until proven otherwise** — a label-join bug, not bad luck. Do not let me explain one away.

---

## Phase 6 — Diagnostics and controls, *before* any headline number

Spec §8. None optional. The previous iteration shipped a 92%-amplifier-identity result
and only these surfaced it.

**Build.** `eval/probes.py`, `eval/controls.py`, `eval/aggregate.py` (logit medians —
not mean probabilities; sigmoid means saturate), `eval/metrics.py` (AUC + 1,000-iter
bootstrap CI, paired sign test).

**Gate.**
- **Frozen domain probes**, independent logistic regression on `z`, at both λ = 0 and
  λ = 1, run **twice**: anchor probe on balanced subsets (target ≈ 0.200) and full-set
  probe (target ≈ the majority-domain baseline **from `results/baselines.json`**). The
  adversary's own accuracy is not a probe (§12 item 9).
- **Manifold gate (§8.2)**, criterion pre-registered numerically before it is measured:
  corpus silhouette on anchors > 0.3 after λ reaches 1 → fall back to independent
  single-task backbones. Report the delta, not a verdict.
- **Negative controls (§8.3)**: permuted-within-corpus labels must collapse AUC to 0.50;
  random frozen backbone with heads trained.
- **§8.4 cross-head extrapolation**, plotted against **in-domain positives from the same
  head**, not against zero — otherwise "heads don't extrapolate" is indistinguishable
  from "heads output 0.4 for everything", which is what the last run actually showed.

**Your turn.**
- *Before:* **pre-register the manifold-gate threshold, in writing, before I measure the silhouette.** The spec proposes 0.3. Picking the number after seeing the value is not pre-registration, and this is the one gate where that distinction decides whether the whole domain-adaptation approach gets abandoned.
- *After:* read both frozen probes. Anchors aligned at ≈0.200 while the full-set probe stays corpus-separable is an **honest result**, not a failure — the protocol-gated inference still stands. How you frame that in the writeup is your call, not mine.

---

## Phase 6.5 — Fix what phase 6 exposed ⛔ BEFORE ANY TRANSFER NUMBER

Not in `metd_plan.md`. Added after phase 6, because its diagnostics invalidated three of
the preconditions P7 assumes. P7 is the only measurement that tests what this architecture
actually claims, and it is worth doing once against a sound pipeline rather than three
times against a moving one. Four items, in order — each depends on the one before.

### 1. Make the split reproducible

**The defect.** `folds.py` uses plain `GroupKFold` for SAM40 and CogLoad. Every SAM40
subject has exactly 12 trials, so group sizes tie and `GroupKFold` breaks ties by
encounter order — which changes with array row order. Measured: the workstation split
differs from the local one by **4-10 subjects per fold**, in SAM40 and CogLoad only.
MODMA and Mumtaz are unaffected (unequal sizes, seeded `StratifiedGroupKFold`).

**Consequence.** Both splits are valid — subject-disjoint, no leakage — but they are
different, so no stress number is reproducible or comparable across runs. Nothing
downstream means anything until this is deterministic.

**Fix.** Canonicalise subject order before splitting, so fold assignment depends only on
*which* subjects exist, not on how rows happen to be laid out.

**Gate.** Shuffling the rows of the input arrays leaves every fold membership identical.
Assert it, in `folds.py`'s self-check, over several permutations.

### 2. Diagnose the stress head

**The defect.** §8.4 measured in-domain separation of **+0.026** with out-of-domain MODMA
at **0.812 — above its own in-domain positives**. The head is near-constant and says
"stressed" to a corpus containing zero stress segments.

**Where to look, in order.** Raw logits before the sigmoid, then the class balance the
masked BCE actually sees. Stress is ~77% positive pooled and 25/75 within SAM40, and
corpus-balanced sampling rebalances *corpora*, not *classes* — a head that predicts the
positive class everywhere gets most of the loss right, which is exactly what saturation
near 0.8 looks like. Then re-run.

**Gate.** In-domain separation > 0.10, and the out-of-domain median below the in-domain
positive median. A collapsed head makes the transfer test unanswerable: EEGMAT would score
whatever a constant predictor scores, and that number would say nothing about transfer.

**Outcome: FAILED.** Class weighting fixed the calibration — medians moved 0.808/0.782/
0.812 to 0.525/0.480/0.520, so the out-of-domain no longer sits above the in-domain
positives — but separation reached only **+0.044** against the 0.10 bar. This is not
partial success. **Calibration on a head with no discrimination is calibration to
chance**: the head was miscalibrated *and* weak, and only the first was a bug. It is
consistent with 15/36 on the held-out corpus and with SAM40 trial AUC 0.5884. Report the
stress head as not working, not as improved.

### 3. Audit Mumtaz's batch structure

**The finding to explain.** Mumtaz sits at subject-level AUC **0.9418**, above §13's
expected 0.70-0.85 — and **recording length alone predicts its class at AUC 0.709**
(MODMA's equivalent: 0.514). `randbb` reaches 0.9235 there with a never-trained backbone,
and the permutation control kills it to 0.4601. That combination reads as a real but
possibly non-neural signal.

**What to check.** EDF headers carry recording date, start time, and equipment fields.
Test whether acquisition metadata alone separates H from MDD. If it does, the two groups
were recorded in different sessions or batches and the model may be reading the batch.

**Gate.** A written statement of whether H and MDD are separable by acquisition metadata
alone, with the number. One afternoon of work, and it determines how every depression
figure in this project gets read — including whether 0.9418 belongs in a paper at all.

### 4. Pre-register a replacement for §8.2

**Why.** The silhouette gate carried no signal: all three modes landed within ±0.004 of
zero, so the pre-registered relative test could not have passed and §8.2's absolute 0.3
would have passed trivially. Neither form measured anything.

**What replaces it.** The frozen probe already *is* the informative measure — it separated
the three modes cleanly (main 0.3970, λ=0 0.4761, random 0.4649, against 0.250 chance).
Make it the gate. Register the criterion, in writing, before re-measuring on the fixed
pipeline; the existing numbers do not count as the pre-registration, because they are
already known.

**Gate.** A dated criterion in `results/preregistration.md`, recorded before the re-run.

**Your turn.**
- *Before:* nothing for items 1-3. Item 4 needs your criterion, and it must be written down before the fixed pipeline is measured — the same rule that governed §8.2, for the same reason.
- *After:* read the Mumtaz audit. If acquisition metadata separates the classes there, that is a finding about the corpus rather than about the model, and how it is reported is a judgement call I should not make alone. It plausibly changes the paper from "depression transfer is weak" to "one of our two depression corpora is confounded".

---

## Phase 7 — Transfer, aggregation, report

Spec §7, §7.5, §13. Only now does a headline number get written down.

**Runs.**
```
SAM40 + CogLoad -> EEGMAT           primary stress generalization
EEGMAT          -> SAM40, CogLoad   diagnostic only (EEGMAT is small)
MODMA           -> Mumtaz (EC)      primary depression transfer
Mumtaz          -> MODMA (EC)       reciprocal
```
Each with its λ = 0 control. EO transfer is secondary — MODMA is eyes-closed only, so
a MODMA→Mumtaz EO test confounds eye state with the amplifier change.

**Aggregation units are per corpus and not negotiable:** depression → subject; SAM40 →
trial (480); EEGMAT → **paired sign test** over 36 subjects, exact binomial vs 0.5
(one baseline + one task recording per subject makes within-subject AUC degenerate);
CogLoad → within-subject primary, pooled secondary with its honest interval. A previous
CogLoad pooled AUC of exactly 1.0000 on 14 subjects was corpus design, not decoding.

**Build.** `eval/report.py` → `REPORT.md` from `results/*.json`, and `infer/gate.py`
(§9, protocol-gated, **no cross-head argmax** — the heads train on disjoint data and
their outputs share no scale).

**Gate** (§11 Reporting): every headline AUC ships with a bootstrap CI **in the same
sentence**; every headline number ships with the corpus-dummy baseline alongside; if
accuracy appears anywhere, the majority baseline sits next to it.

**Your turn.**
- *Before:* nothing.
- *After:* the framing call. Spec §13's own recommendation is to headline the stress arm (three corpora, a genuine held-out amplifier) and report depression as an underpowered secondary finding with its CI. Confirm or overrule — and note that a depression AUC of 0.73 here is a better result than 0.94 on the old pipeline, so a lower number is not a reason to keep tuning.

---

## Phase 8 — Optional, only after P7 lands

- §8.5 auxiliary heads: PHQ-9/GAD-7 regression on MODMA, ordinal load level on CogLoad.
  The CogLoad ordinal is the only ordinal label in the project and forces a
  **within-subject** arousal axis, which by construction cannot be device identity.
- SAM40 `filtered_data/` as a **separate ablation** — never as the primary read, since
  the authors' uncontrolled per-corpus filter chain would make preprocessing itself a
  corpus signature.
- TDBRAIN at `domain = 5` if access clears; §6.2 reverts to a held-out depression corpus
  with no other change. The index is reserved — do not reuse it.

**Your turn.**
- *Before:* your call whether any of this happens at all. None of it is on the critical path to a reportable result.
- *After:* n/a.

---

## Dependencies

```
P0 ──> P1 ──┐
       P2 ──┴──> P3 ⛔ ──> P4 ──> P5 ⛔ ──> P6 ──> P6.5 ⛔ ──> P7 ──> P8
```

P1 and P2 are independent and can be built in either order. Everything downstream of
P3 depends on the arrays P3 writes, so a change to the filter chain, the gate, or `K`
invalidates every result after it — rebuild, do not patch.

P6.5 is not in the specification. It exists because phase 6 measured three things that
make a P7 result uninterpretable: a split that changes between runs, a stress head that
outputs one value, and a depression corpus whose class may be confounded with its
acquisition batch. Its four items are ordered by dependency — a re-run for item 2 is
wasted unless item 1 has made the split stable.

## What this plan drops from `metd_plan.md`

- §1.2 downloads, and §14 step 3's download dependency. All five corpora are on disk;
  paths and file counts verified above.
- Nothing else. Every constraint, threshold and acceptance test in the spec survives.
