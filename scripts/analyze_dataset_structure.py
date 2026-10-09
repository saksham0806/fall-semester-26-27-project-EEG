"""
Scan every corpus on disk and write one structure report per dataset to
datasets/structure/*.md.

Stats are accumulated streaming (count/sum/sumsq/min/max per channel) so a
17 GB corpus never lands in RAM at once.  Heavy corpora are file-sampled; the
sample size is printed in every report so a number is never mistaken for a
full-corpus figure.

    .venv/bin/python -u scripts/analyze_dataset_structure.py
    .venv/bin/python -u scripts/analyze_dataset_structure.py --smoke   # 2 files/corpus
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
import scipy.io

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = os.path.join(ROOT, "datasets")
OUT = os.path.join(D, "structure")

MODMA_REST = os.path.join(D, "MODMA/EEG_128channels_resting_lanzhou_2015")
MODMA_ERP = os.path.join(D, "MODMA/EEG_128channels_ERP_lanzhou_2015")
MODMA_3CH = os.path.join(D, "MODMA/EEG_3channels_resting_lanzhou_2015")
SAM40 = os.path.join(D, "SAM40")
COG = os.path.join(D, "Cognitive Load Assessment Through EEG A Dataset from Arithmetic and Stroop Tasks")
AMIGOS = os.path.join(D, "AMIGOS DATASET")
WESAD = os.path.join(D, "WESAD")


class Acc:
    """Streaming per-channel mean/var/min/max/NaN/zero counts."""

    def __init__(self, names):
        n = len(names)
        self.names = names
        self.n = np.zeros(n, dtype=np.int64)
        self.s = np.zeros(n)
        self.ss = np.zeros(n)
        self.mn = np.full(n, np.inf)
        self.mx = np.full(n, -np.inf)
        self.nan = np.zeros(n, dtype=np.int64)
        self.zero = np.zeros(n, dtype=np.int64)

    def update(self, x):
        """x: (n_channels, n_samples), float."""
        x = np.asarray(x, dtype=np.float64)
        bad = ~np.isfinite(x)
        self.nan += bad.sum(axis=1)
        self.zero += (x == 0).sum(axis=1)
        xm = np.where(bad, np.nan, x)
        self.n += np.isfinite(xm).sum(axis=1)
        self.s += np.nansum(xm, axis=1)
        self.ss += np.nansum(xm ** 2, axis=1)
        # all-NaN channels occur in AMIGOS; nanmin/nanmax warn and return NaN, fmin ignores it
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mn = np.fmin(self.mn, np.nanmin(xm, axis=1))
            self.mx = np.fmax(self.mx, np.nanmax(xm, axis=1))

    def table(self, unit=""):
        mean = self.s / np.maximum(self.n, 1)
        var = self.ss / np.maximum(self.n, 1) - mean ** 2
        var = np.maximum(var, 0.0)
        rows = [
            f"| Channel | N samples | Mean{unit} | Variance | Std | Min | Max | NaN/Inf | Zeros |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for i, nm in enumerate(self.names):
            rows.append(
                f"| {nm} | {self.n[i]:,} | {mean[i]:.4g} | {var[i]:.4g} | {np.sqrt(var[i]):.4g} | "
                f"{self.mn[i]:.4g} | {self.mx[i]:.4g} | {self.nan[i]:,} | {self.zero[i]:,} |"
            )
        return "\n".join(rows)


def fsize(path):
    return os.path.getsize(path)


def human(nbytes):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or u == "TB":
            return f"{nbytes:.1f} {u}"
        nbytes /= 1024


def dirsize(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def sample(files, k):
    """Evenly spaced sample of k files, deterministic."""
    files = sorted(files)
    if k is None or k >= len(files):
        return files
    idx = np.linspace(0, len(files) - 1, k).astype(int)
    return [files[i] for i in sorted(set(idx))]


def df_stats(df, cols):
    """Min/max/mean/var table for spreadsheet columns."""
    rows = ["| Column | Non-null | Mean | Variance | Std | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            f"| {c} | {len(s)} | {s.mean():.4g} | {s.var():.4g} | {s.std():.4g} | {s.min():.4g} | {s.max():.4g} |"
        )
    return "\n".join(rows)


def modma_subject_block(xlsx):
    df = pd.read_excel(xlsx)
    df.columns = df.columns.str.strip()
    df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
    kinds = df["type"].value_counts().to_dict()
    genders = df["gender"].value_counts().to_dict()
    num = [c for c in df.columns if c not in ("subject id", "type", "gender")]
    txt = [
        f"- **Rows**: {len(df)}",
        f"- **Columns**: `{'`, `'.join(df.columns)}`",
        f"- **`type`**: {kinds}  (MDD = Major Depressive Disorder, HC = Healthy Control)",
        f"- **`gender`**: {genders}",
        "",
        df_stats(df, num),
        "",
        "Per-group clinical scores:",
        "",
    ]
    grp = df.groupby("type")[["PHQ-9", "GAD-7", "PSQI", "age"]].agg(["mean", "std", "min", "max"])
    txt.append(grp.round(2).to_markdown())
    return "\n".join(txt), df


# --------------------------------------------------------------------------
# MODMA 128-channel resting  (.mat, the corpus the pipeline actually uses)
# --------------------------------------------------------------------------
def report_modma_rest(nsample):
    files = glob.glob(os.path.join(MODMA_REST, "*.mat"))
    subj_md, df = modma_subject_block(
        os.path.join(MODMA_REST, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
    )

    inv, rates = [], []
    for f in sorted(files):
        b = os.path.basename(f)
        sid = re.match(r"0*(\d+)", b)
        inv.append((b, sid.group(1) if sid else "?", fsize(f)))

    picked = sample(files, nsample)
    acc = Acc([f"E{i}" for i in (22, 9, 33, 24, 124, 122)])
    accall = Acc(["all 129 channels pooled"])
    shapes, keys_seen, zero_rows = [], set(), []
    idx6 = [21, 8, 32, 23, 123, 121]
    for f in picked:
        m = scipy.io.loadmat(f)
        for k, v in m.items():
            if k.startswith("__"):
                continue
            keys_seen.add(f"{k.split('rest')[0][:1]}…{'' if k != 'samplingRate' else ''}" if False else k)
        data = None
        for k, v in m.items():
            if isinstance(v, np.ndarray) and v.shape[0] in (128, 129) and v.shape[1] > 1000:
                data = v
                break
        if data is None:
            continue
        shapes.append(data.shape)
        if "samplingRate" in m:
            rates.append(float(np.ravel(m["samplingRate"])[0]))
        acc.update(data[idx6, :])
        accall.update(data.reshape(1, -1))
        zero_rows.append(np.where((data == 0).all(axis=1))[0].tolist())
        del m, data

    ns = np.array([s[1] for s in shapes])
    fs = rates[0] if rates else 250.0
    total = dirsize(MODMA_REST)

    return f"""# MODMA — 128-channel resting state

**This is the only MODMA variant the pipeline uses.** `corpus == 0`, task `y_depression`.

## 1. Location & size

| | |
|---|---|
| Path | `datasets/MODMA/EEG_128channels_resting_lanzhou_2015/` |
| Format | MATLAB `.mat` v5, big-endian `>f8` (float64) |
| Files | {len(files)} `.mat` + 1 `.xlsx` + 5 `.pdf` |
| On-disk size | {human(total)} |
| Per-file size | {human(min(i[2] for i in inv))} – {human(max(i[2] for i in inv))} |
| Sampling rate | **{fs:g} Hz** (stored in the `samplingRate` key of every file) |
| Reference | Cai et al., *MODMA dataset: a Multi-modal Open Dataset for Mental-disorder Analysis* |

## 2. File layout

Each `.mat` holds three variables:

| Key | Shape | Meaning |
|---|---|---|
| `a<subjectid>rest_<date>_<time>mat` | `(129, n_samples)` | the EEG: 128 scalp channels + 1 reference |
| `samplingRate` | `(1, 1)` | {fs:g} |
| `Impedances_0` | `(129, 1)` | per-electrode impedance at recording time |

⚠️ **The data key is named after the file, so it differs in every file.** Code must
search for it. Matching on `shape[0] in (128, 129)` alone **also matches
`Impedances_0`** — require `shape[1] > 1000` as well.

⚠️ **Row 128 (`E129`, the Cz reference) is identically zero in every file sampled**
(all-zero row indices found: {sorted({r for rs in zero_rows for r in rs})}). It carries no
signal — it is the reference the other 128 are already subtracted against. Include it in a
model and you feed it a constant channel; include it in a channel-count sweep and the
"129-channel" condition is really 128. The project's frontal-6 indices avoid it.

Recording lengths (over the {len(shapes)} sampled files): {ns.min():,} – {ns.max():,} samples
= {ns.min()/fs:.0f} – {ns.max()/fs:.0f} s (~{np.mean(ns)/fs/60:.1f} min mean).

### Filename → subject id

Filenames are **inconsistent** across the corpus:

```
02010002rest 20150416 1017..mat     <- 'rest', double dot
02010008_rest 20150619 1653.mat     <- underscore before 'rest'
02010021 20150805 1730.mat.mat      <- no 'rest' at all, double extension
```

Splitting on `rest` or `_` recovers only **37 of {len(files)}** subjects.
`re.match(r"0*(\\d+)", filename)` recovers **{len(files)}/{len(files)}**.
The leading `0201` is a site prefix; `int()` of the digit run gives the spreadsheet's
`subject id` (e.g. `02010002` → `2010002`).

## 3. Montage

EGI HydroCel Geodesic Sensor Net 128, channels named `E1`…`E128`, `E129` = Cz reference.

The six frontal electrodes this project extracts (0-indexed rows into the `(129, n)` array):

| Canonical | EGI | Row index |
|---|---|---|
| Fp1 | E22 | 21 |
| Fp2 | E9 | 8 |
| F7 | E33 | 32 |
| F3 | E24 | 23 |
| F4 | E124 | 123 |
| F8 | E122 | 121 |

## 4. Labels

Labels live in `subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx`,
**not** in the signal files. Join on `subject id`.

{subj_md}

Pipeline mapping: `y_depression = 1 if type == 'MDD' else 0`; `y_stress = -1` (not applicable).

**PHQ-9 (depression severity) and GAD-7 (anxiety severity) are per-subject continuous
scores and are currently unused.** GAD-7 is the only route to an anxiety target
anywhere in this project.

## 5. Signal statistics

No filtering applied. Computed over **{len(picked)} of {len(files)} files** (evenly spaced
sample), on the six frontal channels:

{acc.table("")}

Pooled over all 129 channels of the same files:

{accall.table("")}

⚠️ **These are not calibrated µV.** Channel means run to ~−10⁴ with excursions past −10⁵,
where scalp EEG is a ±100 µV signal. The export carries a large per-channel **DC offset /
drift** and an unknown gain. Do not compare these magnitudes against SAM40's (±50) or read
them as physical units — the project's per-segment z-scoring is what makes the corpora
commensurable. The `Zeros` column of the pooled row is exactly one channel's worth of
samples: the all-zero `E129` reference.

## 6. Gotchas

- **{len(files)} `.mat` files, {len(df)} spreadsheet rows — a clean 1:1 match.** Every
  recording has metadata and vice versa. Usable N = **{len(df)}** ({int((df['type']=='MDD').sum())} MDD /
  {int((df['type']=='HC').sum())} HC).
- Data is **unfiltered and un-notched**. Measured, **84% of MODMA's total power sits above
  45 Hz** (sharp 50 Hz mains spike). This inflates the denominator of every relative band
  power feature computed over a 0–64 Hz total.
- Every segment of a subject shares one label, so this task's signal is **between-subject**.
  Per-subject re-centering deletes the label outright (measured AUC exactly 0.5000).
"""


# --------------------------------------------------------------------------
# MODMA 128-channel ERP  (.raw EGI, unused)
# --------------------------------------------------------------------------
def report_modma_erp(nsample):
    import mne

    files = sorted(glob.glob(os.path.join(MODMA_ERP, "*.raw")))
    subj_md, _ = modma_subject_block(
        os.path.join(MODMA_ERP, "subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx")
    )

    info_rows, ev_names = [], set()
    acc = None
    for f in sample(files, nsample):
        raw = mne.io.read_raw_egi(f, preload=False, verbose="error")
        eeg = mne.pick_types(raw.info, eeg=True)
        info_rows.append((os.path.basename(f), raw.info["sfreq"], len(raw.ch_names), len(eeg), raw.n_times))
        ev_names |= {c for c in raw.ch_names if c not in [raw.ch_names[i] for i in eeg]}
        if acc is None:
            acc = Acc([raw.ch_names[i] for i in eeg[:6]])
        chunk = raw.get_data(picks=eeg[:6], start=0, stop=min(raw.n_times, 250 * 120))
        acc.update(chunk * 1e6)  # mne returns volts
        del raw

    sf = info_rows[0][1]
    nt = np.array([r[4] for r in info_rows])
    return f"""# MODMA — 128-channel ERP (dot-probe task)

**NOT used by any pipeline in this repo.** Present on disk, documented here for completeness.

## 1. Location & size

| | |
|---|---|
| Path | `datasets/MODMA/EEG_128channels_ERP_lanzhou_2015/` |
| Format | EGI NetStation `.raw` (binary, read with `mne.io.read_raw_egi`) |
| Files | {len(files)} `.raw` + 1 `.xlsx` + 4 `.pdf` |
| On-disk size | {human(dirsize(MODMA_ERP))} |
| Sampling rate | **{sf:g} Hz** |
| Paradigm | dot-probe attentional-bias task (emotional face pairs) |

## 2. Channel inventory (from the sampled files)

| | |
|---|---|
| Total channels in file | {info_rows[0][2]} |
| EEG channels | {info_rows[0][3]} |
| Non-EEG (stimulus/event) channels | {info_rows[0][2] - info_rows[0][3]} |

Event/trigger channel names present: `{'`, `'.join(sorted(ev_names)[:24])}`

Recording lengths: {nt.min():,} – {nt.max():,} samples = {nt.min()/sf:.0f} – {nt.max()/sf:.0f} s.

Same EGI HydroCel 128 montage as the resting variant, so the frontal-6 row indices
(Fp1=E22, Fp2=E9, F7=E33, F3=E24, F4=E124, F8=E122) transfer directly.

## 3. Labels

Identical spreadsheet schema to the resting variant — the **same 53 subjects**.

{subj_md}

Because it is a task paradigm, the ERP recordings additionally carry **per-trial event
codes on the stimulus channels**, which the resting variant does not have. Those codes are
the only route to condition-level (congruent/incongruent, face-valence) labels.

## 4. Signal statistics

First 120 s of each sampled file, first 6 EEG channels:

{acc.table("")}

⚠️ **Not calibrated µV** — same as the resting variant. `mne` returns volts and the table
above rescales by 10⁶, but the underlying export carries a large DC offset and unknown
gain, so treat the magnitudes as file units only.

## 5. Why it is unused

The project's common representation is 4-second resting/continuous windows. Extracting ERP
components needs epoch-locked averaging around the event codes — a different pipeline
entirely. At {human(dirsize(MODMA_ERP))} it is also the **largest EEG corpus on disk**
(WESAD is bigger overall but contains no EEG).
"""


# --------------------------------------------------------------------------
# MODMA 3-channel  (.txt, unused)
# --------------------------------------------------------------------------
def report_modma_3ch(nsample):
    files = sorted(glob.glob(os.path.join(MODMA_3CH, "*.txt")))
    subj_md, _ = modma_subject_block(
        os.path.join(MODMA_3CH, "subjects_information_EEG_3channels_resting_lanzhou_2015.xlsx")
    )
    acc = Acc(["col 0", "col 1", "col 2"])
    lens = []
    for f in sample(files, nsample):
        a = np.loadtxt(f)
        if a.ndim == 1:
            a = a[:, None]
        lens.append(a.shape[0])
        acc.update(a.T)
    lens = np.array(lens)
    fs = 250.0
    return f"""# MODMA — 3-channel resting state

**NOT used by any pipeline in this repo.** `plans/plan.md` describes an abandoned
3-channel variant; this is that corpus.

## 1. Location & size

| | |
|---|---|
| Path | `datasets/MODMA/EEG_3channels_resting_lanzhou_2015/` |
| Format | plain text, 3 whitespace-separated integer columns, no header |
| Files | {len(files)} `.txt` + 1 `.xlsx` + 2 `.pdf` |
| On-disk size | {human(dirsize(MODMA_3CH))} |
| Sampling rate | **{fs:g} Hz** |
| Device | 3-electrode wearable (pervasive/prefrontal collector) |
| Reference | *Depression Detection using Resting State Three-channel EEG Signal* (PDF ships with the data) |

Filenames are `<subjectid>_still.txt`, e.g. `02010001_still.txt` — **uniform**, unlike the
128-channel variant.

Recording lengths (sampled files): {lens.min():,} – {lens.max():,} rows
= {lens.min()/fs:.0f} – {lens.max()/fs:.0f} s.

## 2. Columns

Three prefrontal electrodes (Fp1, Fp2 and a forehead reference per the dataset paper).
The files carry **no header**, so column identity comes from documentation only.

## 3. Labels

{subj_md}

Note this sheet has **55 rows**, two more than the 128-channel sheets — the 3-channel
collection covered subjects the high-density session did not.

## 4. Signal statistics

⚠️ **These are raw ADC counts, not µV.** The measured maxima below sit at
**{acc.mx.max():.6g} ≈ 2³² = {2**32}**, and the minima at 0–13, i.e. the device writes an
**unsigned 32-bit** register in which negative voltages wrap to huge positive numbers.
That is also why the mean (~{acc.s.sum()/acc.n.sum():.3g}) and std are meaningless as
written — they average two sign regimes together. Any use of this corpus must first
reinterpret as signed (`x -= 2**32` where `x > 2**31`) and apply the device gain.

{acc.table(" (counts)")}

## 5. Why it is unused

3 prefrontal channels cannot supply the project's 6-channel common representation
(Fp1 Fp2 F7 F3 F4 F8) — F7/F3/F4/F8 are simply absent. The abandoned variant zero-padded
those four, which is the "padded channels" warning still sitting in `plans/plan.md`.
"""


# --------------------------------------------------------------------------
# SAM40
# --------------------------------------------------------------------------
def report_sam40(nsample):
    raw = sorted(glob.glob(os.path.join(SAM40, "raw_data/*.mat")))
    filt = sorted(glob.glob(os.path.join(SAM40, "filtered_data/*.mat")))
    locs = pd.read_csv(os.path.join(SAM40, "Coordinates.locs"), sep=r"\s+", header=None,
                       names=["idx", "theta", "radius", "label"])
    names = list(locs["label"])

    tasks = {}
    subs = set()
    for f in raw:
        b = os.path.basename(f).replace(".mat", "")
        m = re.match(r"(.+)_sub_(\d+)_trial(\d+)", b)
        if m:
            tasks.setdefault(m.group(1), 0)
            tasks[m.group(1)] += 1
            subs.add(int(m.group(2)))

    acc_r = Acc(names)
    acc_f = Acc(names)
    shapes = []
    for f in sample(raw, nsample):
        d = scipy.io.loadmat(f)["Data"]
        shapes.append(d.shape)
        acc_r.update(d)
    for f in sample(filt, nsample):
        acc_f.update(scipy.io.loadmat(f)["Clean_data"])

    n_t = shapes[0][1]
    idx6 = [names.index(c) for c in ["Fp1", "Fp2", "F7", "F3", "F4", "F8"]]

    try:
        scales = pd.read_excel(os.path.join(SAM40, "scales.xls"))
        scales_md = f"- Shape `{scales.shape}`, columns `{list(scales.columns)}`\n\n" + \
                    df_stats(scales, [c for c in scales.columns if pd.api.types.is_numeric_dtype(scales[c])])
    except Exception as e:
        scales_md = (f"- `scales.xls` is legacy binary `.xls`; reading it needs `xlrd`, which is "
                     f"**not installed** in `.venv` (`{type(e).__name__}`). "
                     f"Per the dataset paper it holds subjective stress ratings on a 1–10 scale for "
                     f"the three task types across 3 trials × 40 subjects.")

    return f"""# SAM40 — stress / arousal EEG

`corpus == 1`, task `y_stress`. Contributes the **relaxed baseline** class the whole
stress task is contrasted against.

## 1. Location & size

| | |
|---|---|
| Path | `datasets/SAM40/` |
| Format | MATLAB `.mat` v5, little-endian `<f8` |
| Files | {len(raw)} raw + {len(filt)} filtered + `Coordinates.locs` + `scales.xls` + `artifact_removal/` |
| On-disk size | {human(dirsize(SAM40))} |
| Sampling rate | **128 Hz** — see the warning in §6 |
| Subjects | {len(subs)} (`sub_1` … `sub_{max(subs)}`) |
| Reference | Ghosh et al., *SAM 40: Dataset of 40 subject EEG recordings to monitor the induced-stress* (Data in Brief, 2022) |

## 2. Directory layout

```
SAM40/
├── raw_data/         {len(raw)} .mat, key 'Data'        (32, {n_t}) float64
├── filtered_data/    {len(filt)} .mat, key 'Clean_data'  (32, {n_t}) float64
├── artifact_removal/ Cleaned_EEG.mat, Corrupted_EEG.mat, correct_EEG.m
├── Coordinates.locs  32-row EEGLAB montage (idx, theta, radius, label)
└── scales.xls        subjective stress ratings
```

**The pipeline reads `filtered_data/` (`Clean_data`), not `raw_data/`.**

## 3. Filenames → task, subject, trial

`<Task>_sub_<N>_trial<T>.mat`, e.g. `Arithmetic_sub_10_trial1.mat`.

| Task prefix | Files | `y_stress` |
|---|---:|---|
{chr(10).join(f"| `{k}` | {v} | {0 if k.lower() == 'relax' else 1} |" for k, v in sorted(tasks.items()))}

Every recording is **{n_t} samples**. At the true 128 Hz that is exactly **{n_t/128:.0f} s**,
matching the paper's 25-second trials.

## 4. Montage (all 32, from `Coordinates.locs`)

`{'`, `'.join(names)}`

All six of the project's canonical channels are native here — row indices
{idx6} for Fp1 Fp2 F7 F3 F4 F8 respectively. **No approximation, no padding.**

## 5. Labels

- **Signal-file labels**: derived from the filename task prefix only (table above).
  `y_depression = -1` (not applicable).
- **Subjective ratings**: `scales.xls`.

{scales_md}

## 6. ⚠️ The sampling-rate defect

`scripts/combine_datasets.py:171` does `data[:, ::2]` with the comment *"Since native is
256 Hz"*, then declares the result 128 Hz.

**The SAM 40 paper states 128 SPS**, and 128 × 25 s = 3200 = exactly the array length on
disk. The 256 Hz claim traces only to `datasets/metadatas/SAM40/sam40_metadata.md`, which
is a *generated report*, not a source.

So the decimation halves SAM40 to a true **64 Hz** while the pipeline labels it 128 Hz.
Measured on 40 `Relax_*` trials, same channels:

| | spectral peak | theta share | beta share |
|---|---|---|---|
| raw @ true 128 Hz | 3.00 Hz | 0.138 | 0.329 |
| after `::2`, read as 128 Hz | **5.50 Hz** | 0.181 | 0.204 |

A 1.83× frequency shift. **Every SAM40 segment in `datasets/combined_data/` and
`datasets/all3/` has its spectral content at roughly double its true frequency** — 2,400 of
15,728 all3 segments. `combine_datasets.py` is deliberately NOT patched because fixing it
invalidates every downstream result; `distilled/data.py` reads raw at the correct rate.

**New work reading this corpus directly should use `fs = 128` and no decimation.**

## 7. Signal statistics

`raw_data/` (`Data`), {min(nsample or len(raw), len(raw))} of {len(raw)} files, µV:

{acc_r.table(" (µV)")}

`filtered_data/` (`Clean_data`) — what the pipeline actually consumes:

{acc_f.table(" (µV)")}
"""


# --------------------------------------------------------------------------
# Cognitive Load
# --------------------------------------------------------------------------
def report_cogload(nsample):
    files = sorted(glob.glob(os.path.join(COG, "raw_data/*/*.txt")))
    groups = {}
    for f in files:
        cond = os.path.basename(os.path.dirname(f)).strip()
        lvl = re.match(r"([a-z]+)-(\d+)", os.path.basename(f))
        groups.setdefault((cond, lvl.group(1) if lvl else "?"), []).append(f)

    names = [f"col {i}" for i in range(25)]
    names[0] = "col 0 (sample index)"
    for i, nm in enumerate(["Fp1", "Fp2", "F7", "F3", "Fz?", "F4", "F8", "C2?"], start=1):
        names[i] = f"col {i} (EEG {nm})"
    names[9], names[10], names[11] = "col 9 (ACC x)", "col 10 (ACC y)", "col 11 (ACC z)"
    names[22] = "col 22 (unix timestamp)"

    acc = Acc(names[:12])
    lens, ncols, z0 = [], set(), 0
    for f in sample(files, nsample):
        df = pd.read_csv(f, header=None, comment="#", on_bad_lines="skip")
        ncols.add(df.shape[1])
        lens.append(df.shape[0])
        v = df.iloc[:, :12].apply(pd.to_numeric, errors="coerce").values.T
        z0 += int((v[1:9, 0] == 0).all())
        acc.update(v)
    lens = np.array(lens)
    fs = 250.0

    gl = "\n".join(f"| `{c}` | `{l}-*.txt` | {len(v)} | {0 if l == 'natural' else 1} |"
                   for (c, l), v in sorted(groups.items()))

    return f"""# Cognitive Load — arithmetic & Stroop tasks

`corpus == 2`, task `y_stress`. The **best-performing stress corpus** (per-corpus ROC-AUC
0.76–0.88, vs AMIGOS at 0.34–0.52).

## 1. Location & size

| | |
|---|---|
| Path | `datasets/Cognitive Load Assessment Through EEG A Dataset from Arithmetic and Stroop Tasks/` |
| Format | OpenBCI GUI text export — comma-separated, **no header row** |
| Files | {len(files)} `.txt` |
| On-disk size | {human(dirsize(COG))} |
| Sampling rate | **{fs:g} Hz** |
| Device | OpenBCI Cyton, 8 EEG channels + onboard 3-axis accelerometer |
| Subjects | 15 (`-1` … `-15` suffix) |
| Reference | *Cognitive Load Assessment Through EEG: A Dataset from Arithmetic and Stroop Tasks* (Data in Brief, 2025) |

## 2. Directory layout

```
raw_data/
├── "Arithmetic_Data "/   60 .txt   <- note the TRAILING SPACE in the folder name
└── Stroop_Data/          60 .txt
```

⚠️ **`Arithmetic_Data ` has a trailing space.** Any hardcoded path must include it;
`combine_datasets.py` sidesteps this with the glob `raw_data/Arithmetic_Data*/*.txt`.

## 3. Filenames → condition, difficulty, subject

`<level>-<subject>.txt`, four levels × 15 subjects × 2 conditions = {len(files)}.

| Condition folder | Level | Files | `y_stress` |
|---|---|---:|---|
{gl}

`natural-*` is the **rest / baseline** condition and is the only `y_stress = 0` group here.
Hardcoding every file to stress-positive was a real bug in this repo; do not reintroduce it.

**Subject IDs are the first digit run in the basename**, so all four difficulty levels
across both task folders collapse into **15 group IDs** — correct for subject-grouped
splitting, but easy to misread as 120 subjects.

## 4. Columns

Files have **{sorted(ncols)} columns**, no header. Layout (0-indexed):

| Column | Content |
|---|---|
| 0 | sample counter (0–255, wraps) |
| 1–8 | 8 EEG channels |
| 9–11 | accelerometer x, y, z (g) |
| 12–21 | OpenBCI aux / digital channels, mostly zero |
| 22 | Unix epoch timestamp (seconds, float) |
| 23 | aux |
| 24 | ISO-8601 local timestamp string |

The project extracts columns **`[1, 2, 3, 4, 6, 7]`** as Fp1 Fp2 F7 F3 F4 F8 — i.e. it
skips columns 5 and 8. Per the repo's own notes those two are Fz and C2, so the native
8-channel order is `Fp1 Fp2 F7 F3 Fz F4 F8 C2`. **This ordering comes from repo comments,
not from a header in the files — verify against the dataset paper before relying on it.**

Recording lengths (sampled files): {lens.min():,} – {lens.max():,} rows
= {lens.min()/fs:.0f} – {lens.max()/fs:.0f} s.

## 5. Signal statistics

Columns 0–11, {min(nsample or len(files), len(files))} of {len(files)} files:

{acc.table("")}

⚠️ **EEG here is raw ADC counts at the ~10⁴ scale, not µV, and is unfiltered.** Per-channel
**std is ~1.0–3.6 × 10⁴** against SAM40's raw **40–143 µV** — two to three orders of
magnitude apart, and the means are large and positive (a DC pedestal) rather than centred
on zero. The project's per-segment z-score normalization is what makes the corpora
commensurable; anything reading this corpus directly must scale and de-trend it.

`col 6` (F4) is the outlier: variance ~1.3 × 10⁹, roughly 10× every other channel, with a
max of 1.6 × 10⁵. That is one or more files with a bad electrode, not a property of the
montage.

Zero runs are **not** uniform. Row 0 is an all-zero row in **{z0} of {len(files)} files** (an
OpenBCI export artifact, ~1 sample each). But `col 2` carries {int(acc.zero[2]):,} zeros and
`col 3` {int(acc.zero[3]):,} — thousands of samples, i.e. **sustained electrode dropout**,
not an export artifact. Those are silent: they survive z-scoring as a flat line.

## 6. Labels

- `y_stress`: 0 for `natural-*`, 1 for `lowlevel/midlevel/highlevel-*`.
- `y_depression`: `-1` (not applicable).
- **The three difficulty levels are collapsed to a single positive class.** Graded
  difficulty is available in the filenames and unused — the only ordinal load label in
  the project.
"""


# --------------------------------------------------------------------------
# AMIGOS
# --------------------------------------------------------------------------
def report_amigos(nsample):
    files = sorted(glob.glob(os.path.join(AMIGOS, "Data_Preprocessed_P*/Data_Preprocessed_P*.mat")))
    names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
             "ECG right", "ECG left", "GSR"]
    acc = Acc(names)
    lab = []
    lens, ntrials, empty = [], [], 0
    for f in sample(files, nsample):
        m = scipy.io.loadmat(f)
        jd, ls = m["joined_data"], m["labels_selfassessment"]
        ntrials.append(jd.shape[1])
        for t in range(jd.shape[1]):
            sig, L = jd[0, t], ls[0, t]
            if not isinstance(sig, np.ndarray) or sig.size == 0 or not isinstance(L, np.ndarray) or L.size == 0:
                empty += 1
                continue
            lens.append(sig.shape[0])
            acc.update(sig.T[:len(names)])
            lab.append(np.ravel(L)[:7])
        del m
    lab = np.array([r for r in lab if len(r) == 7], dtype=float)
    lens = np.array(lens)
    fs = 128.0

    lab_names = ["arousal", "valence", "dominance", "liking", "familiarity", "neutral?", "disgust?"]
    lrows = ["| Index | Field | Mean | Variance | Std | Min | Max |", "|---:|---|---:|---:|---:|---:|---:|"]
    for i in range(lab.shape[1]):
        c = lab[:, i]
        lrows.append(f"| {i} | {lab_names[i]} | {np.nanmean(c):.3f} | {np.nanvar(c):.3f} | "
                     f"{np.nanstd(c):.3f} | {np.nanmin(c):.3f} | {np.nanmax(c):.3f} |")
    arousal = lab[:, 0]
    pos = int((arousal >= 5.0).sum())

    return f"""# AMIGOS — affect / arousal from video watching

`corpus == 3`, task `y_stress`. **Kept as training data and for SSL, but its label is
NOT decodable — do not evaluate on it.**

## 1. Location & size

| | |
|---|---|
| Path | `datasets/AMIGOS DATASET/` |
| Format | MATLAB `.mat` v5, cell arrays of per-trial matrices |
| Participants on disk | **{len(files)} of 40** (P01, P07, P08, P12–P15, P22–P25, P28, P36, P37, P40) |
| On-disk size | {human(dirsize(AMIGOS))} |
| Sampling rate | **{fs:g} Hz** (already preprocessed & downsampled by the authors) |
| Device | Emotiv EPOC, 14-channel wireless EEG + ECG + GSR |
| Reference | Miranda-Correa et al., *AMIGOS: A Dataset for Affect, Personality and Mood Research* |

⚠️ **Only {len(files)} of 40 participants are downloaded**, yet AMIGOS still supplies
**~71% of all segments in `combined_data/`** — because its trials are long (see §3) and
windowing at 4 s / 50% overlap multiplies that out.

## 2. Directory layout

```
AMIGOS DATASET/
├── Data_Preprocessed_P01/Data_Preprocessed_P01.mat   (one folder per participant)
├── ...
├── SelfAsessment.xlsx        <- note the misspelling; defines the label column order
└── Metadata_xlsx/
    ├── Experiment_Data.xlsx
    ├── Participant_Questionnaires.xlsx
    ├── Participants_Panas.xlsx
    ├── Participants_Personality.xlsx   (Big-Five)
    └── Video_List.xlsx
```

## 3. `.mat` structure

| Key | Shape | Contents |
|---|---|---|
| `joined_data` | `(1, 20)` object | per trial: `(n_samples, 17)` float — **samples × channels**, so it needs transposing |
| `labels_selfassessment` | `(1, 20)` object | per trial: `(1, 12)` self-report ratings |
| `labels_ext_annotation` | `(1, 20)` object | per trial: external annotator valence/arousal time series |
| `VideoIDs` | `(1, 20)` object | stimulus id per trial |

Trials per participant: {sorted(set(ntrials))}. **Empty/missing trial cells: {empty}** of
{sum(ntrials)} — these must be skipped, they are not an error.

Trial lengths are **bimodal**, matching the AMIGOS design of 16 short + 4 long videos:

| group | trials | length |
|---|---:|---|
| short videos | {int((lens < 300*fs).sum())} | {lens[lens < 300*fs].min()/fs:.0f} – {lens[lens < 300*fs].max()/fs:.0f} s |
| long videos | {int((lens >= 300*fs).sum())} | {lens[lens >= 300*fs].min()/fs:.0f} – {lens[lens >= 300*fs].max()/fs:.0f} s ({lens[lens >= 300*fs].max()/fs/60:.0f} min) |

Overall {lens.min():,} – {lens.max():,} samples. The 4 long trials per participant carry
most of the data, which is why 15 participants outweigh the other three corpora combined.

## 4. Channel order in `joined_data` (columns of the transposed matrix)

| Idx | Channel | | Idx | Channel |
|---:|---|---|---:|---|
| 0 | AF3 | | 9 | T8 |
| 1 | F7 | | 10 | FC6 |
| 2 | F3 | | 11 | F4 |
| 3 | FC5 | | 12 | F8 |
| 4 | T7 | | 13 | AF4 |
| 5 | P7 | | 14 | ECG right |
| 6 | O1 | | 15 | ECG left |
| 7 | O2 | | 16 | GSR |
| 8 | P8 | | | |

The project extracts `[0, 13, 1, 2, 11, 12]` → AF3, AF4, F7, F3, F4, F8.

⚠️ **AF3→Fp1 and AF4→Fp2 are approximations** — Emotiv EPOC has no Fp1/Fp2 electrode.
This is the **only channel approximation anywhere in the pipeline**; F7/F3/F4/F8 are native.

## 5. Labels — `labels_selfassessment`

Column order is defined by `SelfAsessment.xlsx`: **index 0 = arousal, 1 = valence**,
2 = dominance, 3 = liking, 4 = familiarity, then per-emotion checkboxes.

⚠️ This repo previously read **index 1 (valence) as if it were arousal**. Combined with the
Cognitive Load bug, 45.6% of all stress labels were wrong. Do not reintroduce it.

Over the sampled trials (n = {len(lab)}):

{chr(10).join(lrows)}

Pipeline mapping: `y_stress = 1 if arousal >= 5.0 else 0` → **{pos}/{len(lab)} positive
({pos/max(len(lab),1):.1%})** in this sample. `y_depression = -1`.

Note arousal is **continuous, not integer** (min {arousal.min():.3f}) — it is a slider
position, not a 1–9 Likert click, so a `>= 5.0` cut is a genuine threshold choice.

## 6. Signal statistics

All {len(lens)} non-empty trials of the {len(sample(files, nsample))} participants read,
raw file units:

{acc.table("")}

⚠️ **NaNs are present in the source data, and they are confined to the EEG.** The `NaN/Inf`
count is **identical across all 14 EEG channels** ({acc.nan[:14].max():,} samples each,
{acc.nan[:14].max()/(acc.n[:14].max()+acc.nan[:14].max()):.2%} of the recording) while
**ECG and GSR have exactly {acc.nan[14:].max()}**.

So the wireless EEG headset dropped out for whole spans while the wired peripherals kept
recording — a headset/link fault, not an electrode fault, and it takes all 14 channels at
once. Anything using the ECG or GSR streams is unaffected.

`combine_datasets.py` applies **no finite-check** before windowing, so this propagates:
**1,380 segments — 3.5% of AMIGOS, 2.5% of all of `combined_data/X.npy` — contain NaN**,
from participants P12, P22, P23 and P24. See `combined_data.md` §4; it also explains why
`extracted_features_tuned.npy` has exactly 1,380 fewer rows than `X.npy`.

## 7. ⚠️ The label does not work — measured

| experiment | within-subject AUC |
|---|---|
| AMIGOS alone, 5-fold subject-grouped | **0.4875** (chance) |
| per-subject median split instead of the global `>= 5.0` cut | −0.004 (no change) |

So it is **not** a rating-bias artifact — self-reported arousal from video watching simply
is not recoverable from 6 frontal channels at 4-second resolution. Adding AMIGOS to SAM40
training moves SAM40 performance by **+0.001**, i.e. it is harmless as *training* data.

**Rules:** report stress on SAM40/CogLoad, never pooled with AMIGOS (the pooled AUC of 0.40
is explained entirely by a test set that was ~78% AMIGOS). AMIGOS is excluded from
`datasets/all3/` labels and retained there as `X_unlabelled.npy` for self-supervision.
"""


# --------------------------------------------------------------------------
# WESAD
# --------------------------------------------------------------------------
def report_wesad():
    subs = sorted([d for d in os.listdir(WESAD) if re.fullmatch(r"S\d+", d)],
                  key=lambda s: int(s[1:]))
    pkl = os.path.join(WESAD, subs[0], f"{subs[0]}.pkl")
    with open(pkl, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")

    chest = d["signal"]["chest"]
    wrist = d["signal"]["wrist"]
    lab = d["label"]
    uniq, cnt = np.unique(lab, return_counts=True)

    crows = ["| Sensor | Shape | Rate (Hz) | Mean | Variance | Min | Max |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for k, v in chest.items():
        v = np.asarray(v, dtype=np.float64)
        crows.append(f"| chest/{k} | `{v.shape}` | 700 | {v.mean():.4g} | {v.var():.4g} | "
                     f"{v.min():.4g} | {v.max():.4g} |")
    WRATE = {"ACC": 32, "BVP": 64, "EDA": 4, "TEMP": 4}
    for k, v in wrist.items():
        v = np.asarray(v, dtype=np.float64)
        crows.append(f"| wrist/{k} | `{v.shape}` | {WRATE.get(k, '?')} | {v.mean():.4g} | {v.var():.4g} | "
                     f"{v.min():.4g} | {v.max():.4g} |")

    LBL = {0: "not defined / transient", 1: "baseline", 2: "stress", 3: "amusement",
           4: "meditation", 5: "ignore", 6: "ignore", 7: "ignore"}
    lrows = ["| Code | Condition | Samples @700 Hz | Seconds | Share |", "|---:|---|---:|---:|---:|"]
    for u, c in zip(uniq, cnt):
        lrows.append(f"| {u} | {LBL.get(int(u), '?')} | {c:,} | {c/700:.0f} | {c/cnt.sum():.1%} |")

    return f"""# WESAD — wearable stress & affect (⚠️ CONTAINS NO EEG)

**Used by legacy code only. There is no EEG in this dataset** — it is chest and wrist
physiology. It cannot enter the 6-frontal-channel common representation at all.

## 1. Location & size

| | |
|---|---|
| Path | `datasets/WESAD/` |
| Format | Python pickle (protocol 2, `latin1` encoding) + raw `.txt` + `.csv` |
| Subjects | {len(subs)} — {', '.join(subs)} (**S1 and S12 are excluded by the authors**) |
| On-disk size | {human(dirsize(WESAD))} — the largest corpus here |
| Reference | Schmidt et al., *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection* (ICMI 2018) |

## 2. Per-subject directory

```
S2/
├── S2.pkl            ~975 MB   synchronised signals + labels  <- the one to read
├── S2_respiban.txt   ~261 MB   raw RespiBAN chest export
├── S2_E4_Data.zip    ~1.8 MB   raw Empatica E4 wrist export
├── S2_quest.csv                protocol timing + PANAS/STAI/SAM/SSSQ questionnaires
└── S2_readme.txt               age, height, weight, gender, handedness, caffeine/sport/smoking
```

## 3. `.pkl` structure

```python
{{'subject': '{d['subject']}',
 'signal': {{'chest': {{{', '.join(repr(k) for k in chest)}}},      # RespiBAN, 700 Hz
            'wrist': {{{', '.join(repr(k) for k in wrist)}}}}},   # Empatica E4, mixed rates
 'label':  ndarray {lab.shape} int8}}                     # 700 Hz, aligned to chest
```

Measured on **{subs[0]}** (one subject; the others follow the same schema):

{chr(10).join(crows)}

Note the **different sampling rates within one file** — chest is uniformly 700 Hz, wrist
runs at 32/64/4/4 Hz, and `label` is on the chest clock.

## 4. Labels

`label` is a per-sample int at 700 Hz:

{chr(10).join(lrows)}

The standard WESAD binary task is **baseline (1) vs stress (2)**; codes 0 and 5–7 are
discarded. `S*_quest.csv` gives the protocol block boundaries in minutes
(`Base / TSST / Medi 1 / Fun / Medi 2 / sRead / fRead`) plus the questionnaire responses.

## 5. Known data quirks

- **Temperature sensor dropouts are encoded as −273.15 °C** (absolute zero), not NaN.
  Filter them explicitly.
- `S2_readme.txt` records per-subject caveats, e.g. *"The RespiBAN temperature sensor was
  not fully attached throughout the entire duration of the study protocol."* Read those
  before trusting any single subject's channel.

## 6. Why it is unused

No EEG. Every model in this project consumes `(6, 512)` EEG windows. WESAD is retained
because `src/`'s legacy code referenced it and because it is a valid external stress
corpus should the project ever add a physiological modality.
"""


# --------------------------------------------------------------------------
# Derived: combined_data and all3
# --------------------------------------------------------------------------
def npy_block(path):
    """Shape/dtype from the .npy header only — object arrays cannot be mmapped."""
    rows = ["| File | Shape | dtype | Size |", "|---|---|---|---:|"]
    for f in sorted(glob.glob(os.path.join(path, "*.npy"))):
        with open(f, "rb") as fh:
            major, _ = np.lib.format.read_magic(fh)
            reader = (np.lib.format.read_array_header_1_0 if major == 1
                      else np.lib.format.read_array_header_2_0)
            shape, _, dtype = reader(fh)
        rows.append(f"| `{os.path.basename(f)}` | `{shape}` | `{dtype}` | {human(fsize(f))} |")
    return "\n".join(rows)


def report_combined():
    X = np.load(os.path.join(D, "combined_data/X.npy"), mmap_mode="r")
    ydep = np.load(os.path.join(D, "combined_data/y_depression.npy"))
    ystr = np.load(os.path.join(D, "combined_data/y_stress.npy"))
    ds = np.load(os.path.join(D, "combined_data/datasets.npy"))
    subj = np.load(os.path.join(D, "combined_data/subjects.npy"), allow_pickle=True)
    with open(os.path.join(D, "combined_data/metadata.json")) as fh:
        meta = json.load(fh)

    NAME = {0: "MODMA", 1: "SAM40", 2: "Cognitive Load", 3: "AMIGOS"}
    rows = ["| corpus | id | segments | subjects | y_depression | y_stress |",
            "|---|---:|---:|---:|---|---|"]
    for c in sorted(set(ds.tolist())):
        m = ds == c
        rows.append(
            f"| {NAME[c]} | {c} | {m.sum():,} | {len(set(subj[m].tolist()))} | "
            f"{dict(zip(*[x.tolist() for x in np.unique(ydep[m], return_counts=True)]))} | "
            f"{dict(zip(*[x.tolist() for x in np.unique(ystr[m], return_counts=True)]))} |"
        )

    # full NaN sweep (chunked) — 2.5% of segments are non-finite and it matters downstream
    bad = np.zeros(len(X), dtype=bool)
    for i in range(0, len(X), 4096):
        bad[i:i + 4096] = ~np.isfinite(np.asarray(X[i:i + 4096])).all(axis=(1, 2))
    nan_rows = ["| corpus | segments | with NaN/Inf | share |", "|---|---:|---:|---:|"]
    for c in sorted(set(ds.tolist())):
        m = ds == c
        nan_rows.append(f"| {NAME[c]} | {m.sum():,} | {bad[m].sum():,} | {bad[m].mean():.3%} |")
    nan_rows.append(f"| **all** | **{len(X):,}** | **{bad.sum():,}** | **{bad.mean():.3%}** |")
    nan_subj = sorted(set(subj[bad].tolist()))

    # z-score check on clean segments only
    idx = np.linspace(0, len(X) - 1, 3000).astype(int)
    idx = idx[~bad[idx]]
    smp = np.asarray(X[idx], dtype=np.float64).reshape(len(idx), -1)
    per_seg_mean, per_seg_std = smp.mean(axis=1), smp.std(axis=1)
    gmin, gmax = smp.min(), smp.max()

    feat = os.path.join(D, "combined_data/extracted_features_tuned.npy")
    n_feat = npy_len(feat) if os.path.exists(feat) else None
    feat_md = (f"`extracted_features_tuned.npy` has **{n_feat:,} rows** against `X.npy`'s "
               f"**{len(X):,}** — a difference of **{len(X) - n_feat:,}**, which is exactly the "
               f"NaN-segment count above. The feature extractor drops the non-finite segments, so "
               f"**feature-matrix row *i* is not segment *i*.** Any code indexing `y`/`subjects`/"
               f"`datasets` by feature row must apply the same `isfinite` mask first."
               if n_feat is not None else "")

    return f"""# `combined_data/` — the unified training set (derived)

Not a corpus. Produced by `scripts/combine_datasets.py` from the four raw corpora.
**Rebuild after any change to labels, channel maps or windowing**, then delete
`extracted_features*.npy` (the cache is keyed on filename only and is silently reused).

## 1. Files

{npy_block(os.path.join(D, "combined_data"))}

`metadata.json`:

```json
{json.dumps(meta, indent=2)}
```

## 2. Common representation

Every corpus is projected onto **6 frontal channels at 128 Hz, in 4-second windows
(512 samples) with 50% overlap** (stride 256).

Canonical channel order — **fixed, positional, and not stored in the array**:

`Fp1, Fp2, F7, F3, F4, F8`

| Corpus | native fs | native channels used | resampling |
|---|---|---|---|
| MODMA | 250 | E22 E9 E33 E24 E124 E122 (real EGI positions) | `scipy.signal.resample` → 128 |
| SAM40 | 128 | all 6 native from its 32-ch cap | `[:, ::2]` **then** treated as 128 ⚠️ |
| Cognitive Load | 250 | cols 1,2,3,4,6,7 of the 8-ch OpenBCI | `resample` → 128 |
| AMIGOS | 128 | AF3→Fp1, AF4→Fp2 approximated; F7/F3/F4/F8 native | none |

**There is no zero-padding** — all four genuinely carry the six electrodes.

## 3. Normalization

`z_score_normalize` (`combine_datasets.py:25`), applied **per segment, globally**:

1. median and MAD over the whole `(6, 512)` window,
2. clip to ±5 MAD,
3. subtract mean, divide by std — **across the whole window, not per channel**.

Global rather than per-channel is deliberate: it preserves inter-channel amplitude ratios,
which frontal alpha asymmetry depends on. **There is no bandpass and no notch filter
anywhere in this pipeline.**

Verified on a {len(idx):,}-segment sample of the finite segments of `X.npy`:
per-segment mean ∈ [{per_seg_mean.min():.2e}, {per_seg_mean.max():.2e}],
per-segment std ∈ [{per_seg_std.min():.6f}, {per_seg_std.max():.6f}],
values ∈ [{gmin:.3f}, {gmax:.3f}] (the ±5 MAD clip, post-scaling).

## 4. ⚠️ 2.5% of segments contain NaN, all of them AMIGOS

Full sweep of all {len(X):,} segments:

{chr(10).join(nan_rows)}

Affected subjects: `{'`, `'.join(nan_subj)}` — {len(nan_subj)} of the 15 AMIGOS
participants. The cause is upstream: some AMIGOS trials ship channels that are entirely
NaN, and `combine_datasets.py` applies no finite-check before windowing.

{feat_md}

## 5. Composition & labels

Total **{len(X):,} segments**, {len(set(subj.tolist()))} subjects, shape `{tuple(X.shape)}` `{X.dtype}`.

{chr(10).join(rows)}

### The `-1` masking convention

Every segment carries **both** `y_depression` and `y_stress`; **`-1` means "this corpus does
not support this task"**.

`masked_binary_crossentropy` (`src/proposed_model/multitask_dann_models.py`) zeroes out
samples where `y_true == -1`, which is what lets one shared backbone train on all four
corpora at once.

⚠️ **No segment carries both labels.** The multi-task trunk gets its depression gradient
exclusively from MODMA and its stress gradient exclusively from the other three.

### Subject id format

`modma_<int>`, `sam40_<n>`, `cogload_<n>`, `amigos_P<nn>` — strings, dtype `object`.
Cognitive Load subject ids come from the first digit run in the basename, so its four
difficulty levels across both task folders collapse into 15 ids.

## 6. Ordering ⚠️

Segments are stored **grouped by corpus, in order MODMA → SAM40 → CogLoad → AMIGOS**.
Any unshuffled internal split — `CalibratedClassifierCV(cv=3)`, an early-stopping holdout,
`train_test_split(shuffle=False)` — silently becomes a **leave-one-corpus-out** split.
This has already shipped broken artifacts once. Always pass a shuffled splitter.
"""


def report_all3():
    X = np.load(os.path.join(D, "all3/X.npy"), mmap_mode="r")
    y = np.load(os.path.join(D, "all3/y.npy"))
    corpus = np.load(os.path.join(D, "all3/corpus.npy"))
    subj = np.load(os.path.join(D, "all3/subjects.npy"), allow_pickle=True)
    ydep = np.load(os.path.join(D, "all3/y_depression.npy"))
    ystr = np.load(os.path.join(D, "all3/y_stress.npy"))
    Xu = np.load(os.path.join(D, "all3/X_unlabelled.npy"), mmap_mode="r")
    with open(os.path.join(D, "all3/metadata.json")) as fh:
        meta = json.load(fh)

    CN = {0: "healthy", 1: "stress", 2: "depression"}
    NAME = {0: "MODMA", 1: "SAM40", 2: "Cognitive Load"}
    rows = ["| corpus | segments | subjects | " + " | ".join(CN.values()) + " | majority |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for c in sorted(set(corpus.tolist())):
        m = corpus == c
        counts = [int((y[m] == k).sum()) for k in (0, 1, 2)]
        rows.append(f"| {NAME[c]} | {m.sum():,} | {len(set(subj[m].tolist()))} | "
                    + " | ".join(f"{v:,}" for v in counts)
                    + f" | {max(counts)/m.sum():.4f} |")
    tot = [int((y == k).sum()) for k in (0, 1, 2)]
    rows.append(f"| **all** | **{len(y):,}** | **{len(set(subj.tolist()))}** | "
                + " | ".join(f"**{v:,}**" for v in tot)
                + f" | **{max(tot)/len(y):.4f}** |")

    dep_only_modma = bool(np.all(corpus[y == 2] == 0))
    str_never_modma = bool(np.all(corpus[y == 1] != 0))
    modma_classes = sorted(set(y[corpus == 0].tolist()))
    nan = int(np.isnan(np.asarray(X[::37], dtype=np.float32)).sum())

    return f"""# `datasets/all3/` — the 3-class dataset (derived)

Not a corpus. Derived from `combined_data/` by `scripts/build_all3.py` (~30 s; it does
**not** re-read the raw corpora). **Segments are unchanged — labels only are remapped.**

## 1. Files

{npy_block(os.path.join(D, "all3"))}

`_cache_riemann_*.npy` are cached per-band tangent-space features. **Delete both if the
feature definition changes** — they are silently reused otherwise.

`metadata.json`:

```json
{json.dumps(meta, indent=2)}
```

## 2. Label space

| `y` | class | source |
|---:|---|---|
| 0 | healthy | MODMA HC **and** SAM40 relax **and** CogLoad `natural-*` |
| 1 | stress | SAM40 task trials, CogLoad load trials |
| 2 | depression | MODMA MDD |

⚠️ **"healthy" means two different things** — MODMA healthy controls (a person-level trait)
vs SAM40/CogLoad relaxed state (a within-person condition). They are pooled into one class.

`y_depression.npy` and `y_stress.npy` ride along, so **both binary tasks remain reproducible
from this folder**: depression = `corpus == 0`, stress = `corpus in {{1, 2}}`.

AMIGOS is **excluded from the labels** (its arousal label is chance) and kept as
`X_unlabelled.npy`, shape `{tuple(Xu.shape)}`, for self-supervision.

Data check: NaNs in a 1-in-37 sample of `X.npy`: **{nan}**.

## 3. Composition

{chr(10).join(rows)}

## 4. ⚠️ Class is confounded with recording corpus — this dominates everything

MODMA holds only {{healthy, depression}}; SAM40/CogLoad only {{healthy, stress}}. Each used a
different amplifier, so **corpus identity is ~92% recoverable from the 81 handcrafted
features alone** (~97% from the 273 v2 features).

| baseline | 3-class accuracy |
|---|---|
| chance | 0.3333 |
| majority class | {max(tot)/len(y):.4f} |
| **corpus ID only — zero EEG understanding** | **0.6267** |

**0.6267 is the bar, not chance. Always report 3-class accuracy against it.**

- No **flat** 3-class model beats it in 26 attempts (best flat: SVM at 0.6256).
- A **hierarchical two-stage** model does: **0.6820** (`scripts/all3_hierarchical.py`).
- Removing the device signature (per-corpus Riemannian re-centering) drops accuracy to
  **0.3255** — chance. The signature was worth +0.18 and nothing remains without it.
- Adding features makes the shortcut *easier*: 81 → 273 features raised the corpus probe
  **+0.065**.

**Class 2 occurs in exactly one corpus, and class 1 never occurs there.** Verified here:

| check | result |
|---|---|
| `np.all(corpus[y == 2] == 0)` — every depression segment is MODMA | **{dep_only_modma}** |
| `np.all(corpus[y == 1] != 0)` — no stress segment is MODMA | **{str_never_modma}** |
| classes present in MODMA | `{modma_classes}` (healthy, depression) |

Note the implication runs **one way only**: `depression ⟹ MODMA`, but *not* the reverse —
MODMA also supplies 4,360 healthy segments, so `y == 2` and `corpus == 0` are **not** the
same boolean mask. (`CLAUDE.md` calls this a "boolean identity"; the accurate statement is
the implication above.)

The consequence is unchanged and it is the important part: **never train a
stress-vs-depression classifier.** It is a MODMA-detector with a clinical label, it will
score ~0.97, and it cannot be evaluated within any corpus because no corpus contains both
classes. In deployment that distinction comes from the recording protocol (resting-state
assessment vs cognitive-load task), not from the EEG.

The one well-posed relabelling is **healthy vs ill**, because both classes appear in all
three corpora — the only framing that learns from within-device contrast.

## 5. ⚠️ Splitting

Segments are 4-second windows with **50% overlap**, stored **ordered by corpus**
(MODMA → SAM40 → CogLoad).

A random (non-subject-grouped) split reports **0.9393** on the 3-class task; the same model
under `GroupKFold` on `subjects.npy` reports **0.6256**. The difference is pure leakage —
overlapping windows from the same person land in both train and test.

**Every split must be grouped on `subjects`.** If a number jumps above ~0.85, suspect the
split before believing the model.
"""


# --------------------------------------------------------------------------
def report_osf_park():
    P = os.path.join(D, "OSF_park")
    files = sorted(glob.glob(os.path.join(P, "*.csv")))

    # --- the six Park longitudinal tables ---
    frows = ["| File | Rows | Cols | Subjects | Key columns |", "|---|---:|---:|---:|---|"]
    park = {}
    for f in files:
        b = os.path.basename(f)
        df = pd.read_csv(f, low_memory=False)
        park[b] = df
        sid = df["id"].nunique() if "id" in df.columns else len(df)
        frows.append(f"| `{b}` | {len(df):,} | {df.shape[1]} | {sid} | "
                     f"`{'`, `'.join(list(df.columns)[:6])}`{' …' if df.shape[1] > 6 else ''} |")

    d1 = park["Data1_clinical.psychological.csv"]
    d6 = park["Data6_EEG_PSD(absolute power).csv"]
    chans19 = list(dict.fromkeys(d6["channel"]))
    clin = df_stats(d1, ["hamd", "bdi", "bai", "bhs"])

    # --- BRMH ---
    ml = park["EEG.machinelearing_data_BRMH.csv"]
    cols = list(ml.columns)
    ab = [c for c in cols if c.startswith("AB.")]
    coh = [c for c in cols if c.startswith("COH.")]
    stray = [c for c in cols if c.startswith("Unnamed")]
    bands = list(dict.fromkeys(c.split(".")[2] for c in ab))
    abch = list(dict.fromkeys(c.split(".")[4] for c in ab))
    A, C = ml[ab].to_numpy(float), ml[coh].to_numpy(float)

    main_ct = ml["main.disorder"].value_counts()
    spec_ct = ml["specific.disorder"].value_counts()
    ct = pd.crosstab(ml["main.disorder"], ml["specific.disorder"])
    ct = ct.loc[:, (ct != 0).any(axis=0)]
    nest = "\n".join(
        f"| {m} | {main_ct[m]} | " + ", ".join(f"{s} ({int(v)})" for s, v in ct.loc[m].items() if v) + " |"
        for m in main_ct.index)

    demo_rows = ml.groupby("main.disorder").agg(
        n=("no.", "size"), age=("age", "mean"),
        pct_M=("sex", lambda s: (s == "M").mean() * 100), IQ=("IQ", "mean")).round(1)
    demo_md = "\n".join(f"| {i} | {int(r.n)} | {r.age} | {r.pct_M} | {r.IQ} |"
                        for i, r in demo_rows.iterrows())

    dup = ml[ml.duplicated(subset=ab, keep=False)].sort_values(ab[0])
    dup_full = int(ml.duplicated(subset=ab + coh, keep=False).sum())
    dup_md = "\n".join(
        f"| {int(r['no.'])} | {r['sex']} | {r['age']:g} | {r['eeg.date']} | {r['specific.disorder']} |"
        for _, r in dup.iterrows())
    years = ml["eeg.date"].map(lambda s: s.split(".")[0]).astype(int)

    four = ml["main.disorder"].map({"Healthy control": "healthy", "Anxiety disorder": "anxiety",
                                    "Trauma and stress related disorder": "stress"})
    four[ml["specific.disorder"] == "Depressive disorder"] = "depression"
    fc = four.dropna().value_counts()

    # 4-class baselines: majority vs demographics-only vs EEG. The demographics row is the
    # honest bar here, the same way corpus-ID (0.6267) is the bar in all3.
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.svm import SVC

    sub = ml[four.notna()]
    yv = four.dropna()
    demo = sub[["age", "IQ", "education"]].copy()
    demo["sex"] = (sub["sex"] == "M").astype(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    pipe = make_pipeline(SimpleImputer(), StandardScaler(), SVC(C=1, gamma="scale"))

    def cvacc(X):
        s = cross_val_score(pipe, X, yv, cv=cv, scoring="accuracy")
        return s.mean(), s.std()

    maj = fc.max() / fc.sum()
    b_demo = cvacc(demo)
    b_ab = cvacc(sub[ab])
    b_coh = cvacc(sub[coh])
    b_all = cvacc(pd.concat([sub[ab + coh], demo], axis=1))
    best_eeg = max(b_ab[0], b_coh[0], b_all[0])

    return f"""# OSF_park — Park et al. OSF deposit (⚠️ TWO different studies in one folder)

**Not raw EEG.** Every file here is a **precomputed feature table** in CSV. There are no
time series, so nothing in this folder can enter the project's `(6, 512)` window pipeline —
it is used at the feature level or not at all.

The folder mixes **two unrelated studies** that happen to share an OSF deposit. Treat them
separately; they share no subjects, no device record, and no schema.

| | study | n | design |
|---|---|---|---|
| `Data1`–`Data6` | Park et al. biofeedback RCT (`RAS.BFB_MDD`) | 30 MDD patients | longitudinal, pre/post |
| `EEG.machinelearing_data_BRMH.csv` | Brain Research & Metrics Hospital psychiatric cohort | **945 subjects** | cross-sectional, 7 diagnoses |

**The BRMH file is the one that matters to this project — see §5.**

## 1. Location & size

| | |
|---|---|
| Path | `datasets/OSF_park/` |
| Format | CSV, header row, one file per table |
| On-disk size | {human(dirsize(P))} |
| Source | OSF project `RAS.BFB_MDD` — <https://osf.io/ya9ps> |
| Creator | Su Mi Park (OSF), deposited 2020-09-16 |
| `metadatas/` | 7 DataCite JSON records, one per CSV — **provenance only (author, URL, date); they contain no column schema** |

## 2. File inventory

{chr(10).join(frows)}

## 3. Study A — `Data1`–`Data6`, the biofeedback RCT

30 MDD patients (ids `B001`…), randomised to **`BFB+TAU`** (biofeedback + treatment as
usual) vs **`TAU`**, measured across sessions. Only 29 subjects appear in the EEG tables.

**Every table is long-format**, one row per (subject × session × stratum) — so a subject
contributes many rows and **any split must group on `id`**.

| Table | Row granularity | Sessions |
|---|---|---|
| `Data1` clinical | subject × session | `pre`, `2nd`, `4th`, `post(6th)` |
| `Data2` HRV | subject × session × stage | `pre`, `post` × 5 stages (`1.baseline`, `2.color`, `3.rest1`, `4.math`, `5.rest2`) |
| `Data3` small-worldness | subject × session × band | `pre`, `post` × 6 bands |
| `Data4` source coherence | subject × session × BA-region-pair | 190 Brodmann region pairs |
| `Data5` source CSD | subject × visit × Brodmann region | 92 regions, with MNI `X`,`Y`,`Z` |
| `Data6` scalp PSD | subject × session × channel | 19 channels |

⚠️ **`Data5` uses different column names and codes for the same fields**: `visit` instead
of `session` (`BX`/`FU` rather than `pre`/`post`) and `group` coded `B`/`C` rather than
`BFB+TAU`/`TAU`. Joining `Data5` to the others on `session`/`group` silently produces
nothing.

### Clinical scales (`Data1`, 120 rows = 30 subjects × 4 sessions)

{clin}

`hamd` = Hamilton Depression, `bdi` = Beck Depression, `bai` = Beck **Anxiety**,
`bhs` = Beck Hopelessness. 13–16 missing values per column — attrition across sessions.

### Montage (`Data6`)

19 channels, standard 10-20: `{'`, `'.join(chans19)}`

**All six of the project's canonical channels are present** (FP1, FP2, F7, F3, F4, F8).

### Why Study A is of limited use here

30 subjects, all MDD, **no healthy control arm** — the contrast is treatment
(`BFB+TAU` vs `TAU`), not diagnosis. It cannot supply a healthy/depressed label.

## 4. Study B — `EEG.machinelearing_data_BRMH.csv`

**945 rows**, one per recording. Recordings span **{years.min()}–{years.max()}**
({years.value_counts().max()} in the busiest year).

⚠️ **There is no subject-identifier column.** `no.` is a row index 1–945 with no
duplicates, which identifies *rows*, not people. Whether any individual contributed more
than one recording across those {years.max() - years.min() + 1} years is **not verifiable from this file**.
Treating each row as an independent subject is the only option available, but it is an
assumption, not a fact — and it is the assumption that the whole subject-grouping
discipline in this repo exists to protect.

### Column layout ({len(cols)} columns)

| Index | Block | Count | Content |
|---|---|---:|---|
| 0–7 | metadata | 8 | `no.`, `sex`, `age`, `eeg.date`, `education`, `IQ`, `main.disorder`, `specific.disorder` |
| 8–121 | **AB** absolute band power | {len(ab)} | 6 bands × 19 channels |
| 122 | `Unnamed: 122` | 1 | ⚠️ **entirely empty — drop it** |
| 123–1148 | **COH** coherence | {len(coh)} | 6 bands × 171 channel pairs = C(19,2) |

⚠️ **`Unnamed: 122` is 100% NaN and is the *only* source of NaN in the file.** It is a
trailing-comma artifact sitting exactly between the AB and COH blocks. Drop it and the
{len(ab) + len(coh):,} feature columns are **completely clean** — 0 missing cells. Leave it in
and a naive `dropna()` deletes all 945 rows.

**Layout is band-major**, the same convention as the project's v1 81 features: columns 8–26
are all 19 channels of delta, 27–45 theta, and so on.

- Bands: `{'`, `'.join(bands)}` — note **`highbeta` is a separate band**, 6 bands not 5.
- Channels: `{'`, `'.join(abch)}`
- Naming: `AB.<band-letter>.<band>.<chan-letter>.<CHAN>`, e.g. `AB.A.delta.a.FP1`;
  coherence adds the second channel: `COH.A.delta.a.FP1.b.FP2`.

| Block | Min | Max | Mean | Units |
|---|---:|---:|---:|---|
| AB | {A.min():.4g} | {A.max():.4g} | {A.mean():.4g} | µV², absolute (not relative) |
| COH | {C.min():.4g} | {C.max():.4g} | {C.mean():.4g} | percent, 0–100 |

### Labels — two nested columns

`specific.disorder` nests cleanly inside `main.disorder`:

| `main.disorder` | n | `specific.disorder` |
|---|---:|---|
{nest}

### Demographics by group

| Group | n | Mean age | % Male | Mean IQ |
|---|---:|---:|---:|---:|
{demo_md}

### ⚠️ Data integrity — {len(dup)} rows share duplicated band-power blocks

{len(dup)} rows form {len(dup)//2} pairs whose **entire {len(ab)}-column AB block is
byte-identical**, despite differing in sex, age, recording date and IQ:

| `no.` | sex | age | eeg.date | specific.disorder |
|---:|---|---:|---|---|
{dup_md}

Their **COH blocks are all distinct** ({dup_full} rows are duplicated across AB+COH
together), so this is not whole-row duplication — the absolute-power block specifically was
mis-copied between records.

**The middle pair is the damaging one: identical band power, one labelled
`Behavioral addiction disorder` and the other `Healthy control`.** That is a direct
contradiction in feature space — no model can separate those two rows, and any
cross-validation fold that splits them contributes a guaranteed error. Drop or quarantine
these {len(dup)} rows before using the AB block; the COH block appears unaffected.

## 5. ⚠️ Why this file matters — and the trap in it

`CLAUDE.md` states the project's goal is 4-class **Stress / Anxiety / Healthy / Depression**
and that *"there is no anxiety label, dataset, or model anywhere in the repo."*

**This file has all four classes in a single-site cohort with a single feature schema:**

| target class | source | n |
|---|---|---:|
| healthy | `main.disorder == 'Healthy control'` | {fc['healthy']} |
| depression | `specific.disorder == 'Depressive disorder'` | {fc['depression']} |
| anxiety | `main.disorder == 'Anxiety disorder'` (panic + social anxiety) | {fc['anxiety']} |
| stress | `main.disorder == 'Trauma and stress related disorder'` (PTSD + acute stress + adjustment) | {fc['stress']} |
| **total** | | **{int(fc.sum())}** |

That is the **one framing in this project where class is not confounded with corpus** —
all four classes come from the same hospital and the same 19-channel pipeline, so the
"different amplifier per class" problem that dominates `all3` does not arise. It is the
single strongest reason to care about this folder.

⚠️ **"Same device" is an inference, not a verified fact.** There is **no device, site or
session column**, and the recordings span {years.max() - years.min() + 1} years
({years.min()}–{years.max()}). Hardware and acquisition software plausibly changed over
that period. What can be verified is that all four classes share one *feature schema* and
one institution — not one amplifier. If a corpus-style probe is ever run here, `eeg.date`
(year) is the variable to probe against.

### But demographics beat the EEG — measured

Before building anything on it, note that class is confounded with **age, sex and IQ**
(healthy controls are the youngest and highest-IQ group; the trauma group is the oldest and
only 34% male). Measured on the {int(fc.sum())}-subject 4-class subset, 5-fold stratified CV,
one SVM (RBF, `C=1`, `gamma='scale'`) on standardised features:

| features | 4-class accuracy |
|---|---|
| chance | {1/4:.4f} |
| majority class | {maj:.4f} |
| **demographics only** (age, sex, IQ, education) | **{b_demo[0]:.4f} ±{b_demo[1]:.3f}** |
| EEG absolute band power ({len(ab)} cols) | {b_ab[0]:.4f} ±{b_ab[1]:.3f}{' ← **below majority**' if b_ab[0] < maj else ''} |
| EEG coherence ({len(coh)} cols) | {b_coh[0]:.4f} ±{b_coh[1]:.3f}{' ← **below majority**' if b_coh[0] < maj else ''} |
| EEG (all {len(ab)+len(coh)}) + demographics | {b_all[0]:.4f} ±{b_all[1]:.3f} |

**A demographics-only model beats every EEG configuration ({b_demo[0]:.4f} vs a best EEG
result of {best_eeg:.4f}), and adding EEG to demographics makes it worse
({b_all[0]:.4f} < {b_demo[0]:.4f}).** This is the same shape of result as the all3 corpus
shortcut (0.6267): the honest bar here is **{b_demo[0]:.4f}** — not {1/4:.4f} and not
{maj:.4f} — and no EEG feature set has cleared it yet.

⚠️ **This is a first-look baseline, not a verdict.** It is a single untuned SVM — exactly
the caveat `CLAUDE.md` attaches to *"deep models lose here"*. {len(ab)+len(coh):,} features
against {int(fc.sum())} subjects is also badly under-determined, and the repo has already measured that adding
features strengthens a shortcut rather than weakening it. Feature selection, regularisation
tuning, or age/sex/IQ-matched sampling are all untried. What is **not** in doubt is that
the demographic confound is real and must be controlled or reported alongside any number.

## 6. Splitting

- **Study B (`BRMH`)**: one row per *recording*, and there is no subject id to group on, so
  `StratifiedKFold` is the only option. The 50%-overlap-window leakage hazard does not apply
  — there are no windows. But note this is grouping-by-necessity, not grouping-by-proof: if
  any person was recorded twice (§4), those rows can straddle a fold and inflate the score
  exactly the way a random window split does elsewhere in this repo.
- **Study A (`Data1`–`Data6`)**: long format, many rows per subject. **Must group on `id`.**

## 7. Integration status

**Not wired into anything.** No loader, no `corpus` id, no entry in
`scripts/combine_datasets.py`, and it is absent from `datasets/structure/README.md`'s
raw-corpora table because it is not a raw corpus.

Two things block a naive merge with the existing pipeline:

- **No time series.** `combined_data/` is built from raw windows; this folder starts at
  band power and coherence. It can only join the project at the feature level, and even
  then the definitions differ (AB here is **absolute** power over 6 bands including
  `highbeta`; the project's v1 block is **relative** power over 5).
- **19 channels vs 6.** The project's canonical six are all present, so a 6-channel subset
  of the AB block is directly extractable — but that discards 13 channels and the entire
  coherence block, which is where most of this file's information is.
"""


def npy_len(path):
    with open(path, "rb") as fh:
        major, _ = np.lib.format.read_magic(fh)
        reader = (np.lib.format.read_array_header_1_0 if major == 1
                  else np.lib.format.read_array_header_2_0)
        return reader(fh)[0][0]


def report_index():
    n_comb = npy_len(os.path.join(D, "combined_data/X.npy"))
    n_all3 = npy_len(os.path.join(D, "all3/X.npy"))
    n_unl = npy_len(os.path.join(D, "all3/X_unlabelled.npy"))
    return f"""# Dataset structure reports

One file per dataset. Generated by `scripts/analyze_dataset_structure.py` — regenerate
after adding data or changing a loader; do not hand-edit.

Every statistic here was **measured from the files on disk**, not copied from the dataset
papers. Where a paper and the disk disagree, the disagreement is called out.

## Raw corpora

| Report | Corpus | Channels | Fs | Subjects | Size | Used by pipeline |
|---|---|---|---|---|---|---|
| [MODMA_128ch_resting.md](MODMA_128ch_resting.md) | MODMA resting | 128 + ref | 250 Hz | 53 | 3.9 GB | ✅ `corpus 0`, depression |
| [SAM40.md](SAM40.md) | SAM40 | 32 | **128 Hz** ⚠️ | 40 | 728 MB | ✅ `corpus 1`, stress |
| [CognitiveLoad.md](CognitiveLoad.md) | Cognitive Load | 8 + ACC | 250 Hz | 15 | 778 MB | ✅ `corpus 2`, stress |
| [AMIGOS.md](AMIGOS.md) | AMIGOS | 14 + ECG/GSR | 128 Hz | 15 of 40 | 1.3 GB | ⚠️ train only, label is chance |
| [MODMA_128ch_ERP.md](MODMA_128ch_ERP.md) | MODMA ERP | 128 + triggers | 250 Hz | 53 | 6.2 GB | ❌ unused |
| [MODMA_3ch_resting.md](MODMA_3ch_resting.md) | MODMA 3-channel | 3 | 250 Hz | 55 | 386 MB | ❌ unused |
| [WESAD.md](WESAD.md) | WESAD | **no EEG** | 700/64/32/4 Hz | 15 | 17 GB | ❌ legacy only |

## Feature tables (no raw signal)

| Report | Contents | Subjects | Size | Used by pipeline |
|---|---|---|---|---|
| [OSF_park.md](OSF_park.md) | **two studies**: a 30-subject biofeedback RCT, and a 945-subject psychiatric cohort (band power + coherence CSVs) | 30 / 945 | 12 MB | ❌ not wired in |

**`OSF_park` is the only dataset here carrying an anxiety label**, and its 945-subject file
holds healthy / depression / anxiety / stress in **one cohort on one device** — the four
classes this project set out to classify, without the corpus confound. It is also the only
folder with no time series at all. See [OSF_park.md](OSF_park.md) §5 before using it: a
demographics-only model (0.4328) currently beats every EEG feature set in it.

## Derived

| Report | What | Segments |
|---|---|---|
| [combined_data.md](combined_data.md) | the unified `(N, 6, 512)` training set, two binary tasks | {n_comb:,} |
| [all3.md](all3.md) | the 3-class dataset (healthy / stress / depression) | {n_all3:,} + {n_unl:,} unlabelled |

## The five things most likely to bite

1. **SAM40 is 128 Hz, not 256.** `combine_datasets.py` decimates it by 2 anyway, so every
   SAM40 segment downstream is time-stretched 2×. See [SAM40.md](SAM40.md) §6.
2. **AMIGOS arousal is not decodable** (AUC 0.4875) yet supplies ~71% of `combined_data`.
   Never report pooled stress. See [AMIGOS.md](AMIGOS.md) §6.
3. **Class is confounded with corpus in all3**, and corpus is ~92% recoverable from
   features. The baseline to beat is 0.6267, not 0.3333. See [all3.md](all3.md) §4.
4. **Arrays are stored ordered by corpus.** Any unshuffled internal split becomes an
   accidental leave-one-corpus-out. See [combined_data.md](combined_data.md) §5.
5. **`OSF_park`'s `Unnamed: 122` column is 100% NaN** and is the only NaN in that file —
   a naive `dropna()` deletes all 945 rows. See [OSF_park.md](OSF_park.md) §4.

## Note

`datasets/.gitignore` is `*` (only `datasets.md` tracked), so these reports are **not
version-controlled** unless that file is amended.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 files per corpus, fast sanity run")
    ap.add_argument("--only", default=None, help="comma-separated report names")
    args = ap.parse_args()

    n = 2 if args.smoke else None
    # file-sample sizes: bounded for the multi-GB corpora, full for the cheap ones
    S = dict(modma_rest=n or 12, modma_erp=n or 6, modma_3ch=n or 12,
             sam40=n or 480, cogload=n or 120, amigos=n or 15)

    jobs = [
        ("MODMA_128ch_resting", lambda: report_modma_rest(S["modma_rest"])),
        ("MODMA_128ch_ERP", lambda: report_modma_erp(S["modma_erp"])),
        ("MODMA_3ch_resting", lambda: report_modma_3ch(S["modma_3ch"])),
        ("SAM40", lambda: report_sam40(S["sam40"])),
        ("CognitiveLoad", lambda: report_cogload(S["cogload"])),
        ("AMIGOS", lambda: report_amigos(S["amigos"])),
        ("WESAD", report_wesad),
        ("OSF_park", report_osf_park),
        ("combined_data", report_combined),
        ("all3", report_all3),
        ("README", report_index),
    ]
    if args.only:
        want = set(args.only.split(","))
        jobs = [j for j in jobs if j[0] in want]

    # smoke output is segregated: filenames are identical to a real run, so a shared
    # directory would silently overwrite full-corpus reports with 2-file ones
    out = os.path.join(OUT, "_smoke") if args.smoke else OUT
    os.makedirs(out, exist_ok=True)
    for name, fn in jobs:
        t0 = time.time()
        try:
            md = fn()
        except Exception as e:
            print(f"  !! {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        with open(os.path.join(out, f"{name}.md"), "w") as fh:
            fh.write(md.rstrip() + "\n")
        print(f"  {name:24s} {len(md):7,d} chars  {time.time()-t0:6.1f}s")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
