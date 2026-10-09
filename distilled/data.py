"""
Native-resolution loading and channel-generic features.

The repo's `scripts/combine_datasets.py` projects every corpus onto 6 frontal channels before
anything sees it. Everything here reads from raw instead, at whatever channel count is asked
for, so channel count becomes an experimental variable rather than a fixed constraint.

Preprocessing is kept byte-for-byte identical to `combine_datasets.py` (resample to 128 Hz,
512-sample windows at 50% overlap, MAD-clipped global z-score) so numbers stay comparable with
the rest of the project. The ONE deliberate difference is SAM40's sampling rate — see
`SAM40_FS` below.
"""
import os
import re

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, resample, sosfiltfilt, welch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE = os.path.join(PROJECT_ROOT, "distilled", "_cache")

FS, WINDOW, STRIDE = 128, 512, 256
MODMA_DIR = os.path.join(PROJECT_ROOT, "datasets", "MODMA",
                         "EEG_128channels_resting_lanzhou_2015")
MODMA_FS = 250

# SAM40 is 128 Hz, NOT 256. The dataset paper states "EEG was sampled at 128 SPS" with 25-second
# trials, and 128 x 25 = 3200 = the array length on disk.
# scripts/combine_datasets.py:171 decimates data[:, ::2] assuming 256 Hz, which halves SAM40 to a
# true 64 Hz while labelling it 128 -- time-stretching every SAM40 segment in combined_data/ and
# all3/ by 2x. Measured on 40 Relax trials: spectral peak moves 3.00 Hz -> 5.50 Hz.
# Nothing here decimates. See distilled/README.md.
SAM40_FS = 128

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "beta": (12.0, 30.0), "gamma": (30.0, 45.0)}


def z_score_normalize(seg):
    """Verbatim from scripts/combine_datasets.py:25. Global, not per-channel, on purpose:
    it preserves inter-channel amplitude ratios, which asymmetry measures depend on."""
    med = np.median(seg)
    mad = np.median(np.abs(seg - med)) + 1e-8
    clipped = np.clip(seg, med - 5 * mad, med + 5 * mad)
    return (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)


def window(arr):
    """(n_chans, n_samples) -> list of normalised (n_chans, WINDOW) windows."""
    return [z_score_normalize(arr[:, s:s + WINDOW]).astype(np.float32)
            for s in range(0, arr.shape[1] - WINDOW + 1, STRIDE)]


def channel_subset(n, frontal):
    """
    `n` channel indices that always CONTAIN the frontal set, so smaller sets nest inside larger.

    Nesting is what makes an n-vs-m comparison mean "does more help" rather than "do these other
    channels help" — phase2_channel_density.py:119 used np.linspace sets that exclude the frontal
    six, so its 6-vs-N rows changed which channels were used as well as how many.
    """
    frontal = list(frontal)
    if n <= len(frontal):
        return sorted(frontal)
    pool = [i for i in np.linspace(0, 127, n).astype(int).tolist() if i not in frontal]
    return sorted(frontal + pool[: n - len(frontal)])


def load_modma(cache=True):
    """
    All 128 MODMA channels, windowed. (X, y, subjects) with X (n_windows, 128, 512) float32.

    Slice channels from the result rather than reloading — the raw read is ~2 minutes.
    """
    os.makedirs(CACHE, exist_ok=True)
    paths = {k: os.path.join(CACHE, f"modma_{k}.npy") for k in ("X", "y", "subjects")}
    if cache and all(os.path.exists(p) for p in paths.values()):
        return (np.load(paths["X"], mmap_mode="r"),
                np.load(paths["y"]), np.load(paths["subjects"], allow_pickle=True))

    import pandas as pd
    from scipy.io import loadmat

    xlsx = os.path.join(MODMA_DIR,
                        "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
    info = pd.read_excel(xlsx)
    idcol = next(c for c in info.columns if "subject" in str(c).lower() and "id" in str(c).lower())
    typecol = next(c for c in info.columns if "type" in str(c).lower())
    label = {str(r[idcol]).strip(): int("mdd" in str(r[typecol]).lower())
             for _, r in info.iterrows()}

    X, y, subj = [], [], []
    files = sorted(f for f in os.listdir(MODMA_DIR) if f.endswith(".mat"))
    for i, fn in enumerate(files):
        # Filenames are inconsistent ('...rest ...', '..._rest ...', '... .mat.mat').
        # Splitting on 'rest' or '_' matches only 37/53; the zero-padded digit run matches 53/53.
        m = re.match(r"0*(\d+)", fn)
        sid = m.group(1) if m else None
        if sid not in label:
            print(f"\n  WARNING: no label for {fn!r} (id {sid!r}) — skipped")
            continue
        mat = loadmat(os.path.join(MODMA_DIR, fn))
        # Impedances_0 is also (129, 1); require a real time axis so key order cannot matter.
        arr = next(v for _, v in mat.items()
                   if isinstance(v, np.ndarray) and v.ndim == 2
                   and v.shape[0] in (128, 129) and v.shape[1] > 1000)
        arr = resample(arr[:128].astype(np.float64),
                       int(arr.shape[1] * FS / MODMA_FS), axis=1)
        w = window(arr)
        X.extend(w); y.extend([label[sid]] * len(w)); subj.extend([sid] * len(w))
        print(f"\r  loading MODMA {i + 1}/{len(files)}", end="", flush=True)
    print()

    X = np.stack(X); y = np.array(y); subj = np.array(subj)
    if cache:
        np.save(paths["X"], X); np.save(paths["y"], y); np.save(paths["subjects"], subj)
    return X, y, subj


# ------------------------------------------------------------------ channel-generic features

def handcrafted(X, fs=FS):
    """
    Per-channel spectral + amplitude features for ANY channel count.

    `src/proposed_model/features.py` is structurally 6-channel — its CHANNEL_MAP and
    ASYMMETRY_PAIRS name specific electrodes. This is the same feature families minus frontal
    alpha asymmetry (which needs named left/right pairs), generalised over channels:

        5 relative band powers + 5 differential entropies + spectral entropy + std + rms
        = 13 features per channel

    Relative power divides by total power over 0.5-45 Hz, matching features_v2 rather than v1 --
    v1 divides by 0-64 Hz, which puts mains hum in the denominator (see feature_info.md 4a).
    """
    n_ch = X.shape[1]
    freqs, psd = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    in_band = (freqs >= 0.5) & (freqs <= 45.0)
    total = trapezoid(psd[..., in_band], freqs[in_band], axis=-1) + 1e-12

    rel = np.stack([trapezoid(psd[..., (freqs >= lo) & (freqs <= hi)],
                              freqs[(freqs >= lo) & (freqs <= hi)], axis=-1) / total
                    for lo, hi in BANDS.values()], axis=1)          # (N, 5, ch)

    de = []
    nyq = 0.5 * fs
    for lo, hi in BANDS.values():
        sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
        de.append(0.5 * np.log(np.var(sosfiltfilt(sos, X, axis=-1), axis=-1) + 1e-8))
    de = np.stack(de, axis=1)                                        # (N, 5, ch)

    p = psd[..., in_band] / (psd[..., in_band].sum(-1, keepdims=True) + 1e-12)
    spec_ent = -np.sum(p * np.log2(p + 1e-12), axis=-1)               # (N, ch)

    return np.hstack([rel.reshape(len(X), -1), de.reshape(len(X), -1), spec_ent,
                      np.std(X, axis=-1),
                      np.sqrt(np.mean(X ** 2, axis=-1))]).astype(np.float32)


def handcrafted_dim(n_ch):
    return 13 * n_ch


def subject_auc(y, prob, subjects):
    """One score per subject then AUC over subjects — depression is a subject-level trait."""
    from sklearn.metrics import roc_auc_score
    su = np.unique(subjects)
    sy = np.array([y[subjects == s][0] for s in su])
    sp = np.array([prob[subjects == s].mean() for s in su])
    return float(roc_auc_score(sy, sp))
