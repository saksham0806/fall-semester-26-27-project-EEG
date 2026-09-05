"""Preprocessing chain — metd_plan.md §3. Order matters; do not reorder.

    mark flatline spans (mark only, never excise)
      -> reflect-pad 3.0 s -> 50 Hz notch -> 1-45 Hz bandpass -> strip padding
      -> resample to 128 Hz -> window 512/256
      -> reject flatline-overlapping -> reject variance-gate failures
      -> per-segment normalize

Applied identically to all five corpora. Excising a flatline span instead of marking it
would create a step discontinuity that filtfilt rings on; filtering before marking would
smear the zeros and make them undetectable.

Self-check: ``python -m arch_metd.data.preprocess``
"""
import math

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from ..contract import (BAND, BAND_ORDER, CLIP_MAD, FLATLINE_MS, FS, NOTCH_HZ,
                        NOTCH_Q, PAD_SEC, STRIDE, WINDOW, n_windows)


def flatline_mask(x, fs, min_ms=FLATLINE_MS):
    """Per-sample mask of zero-runs at least ``min_ms`` long, on ANY channel.

    Computed on the raw signal, before filtering — a filter smears exact zeros into
    small non-zero values and the dropout becomes invisible.
    """
    min_len = max(1, int(round(min_ms * fs / 1000)))
    bad = np.zeros(x.shape[1], dtype=bool)
    for channel in x == 0:
        edges = np.diff(np.concatenate(([0], channel.view(np.int8), [0])))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if end - start >= min_len:
                bad[start:end] = True
    return bad


def filter_recording(x, fs):
    """Notch then bandpass, zero-phase, with reflect padding stripped afterwards.

    Notch first: a sharp notch applied to a large DC pedestal rings badly, and MODMA
    runs to ~-1e4 while CogLoad runs to ~+1e4.
    """
    pad = int(round(PAD_SEC * fs))
    assert x.shape[1] > pad, f"recording shorter than {PAD_SEC}s of padding"
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    b, a = iirnotch(NOTCH_HZ, NOTCH_Q, fs)
    xp = filtfilt(b, a, xp, axis=-1)
    sos = butter(BAND_ORDER, BAND, btype="band", fs=fs, output="sos")
    xp = sosfiltfilt(sos, xp, axis=-1)
    return xp[:, pad:xp.shape[1] - pad]


def resample(x, fs):
    if fs == FS:
        return x
    g = math.gcd(int(fs), FS)
    return resample_poly(x, FS // g, int(fs) // g, axis=-1)


def variance_ratio(w):
    """max_c Var(c) / median_c Var(c), within-segment across the 6 channels.

    Computed on filtered-but-UNNORMALIZED data. Post-normalization is the wrong input:
    global MAD scaling forces the whole (6, 512) matrix to unit variance while leaving
    channels free to differ, and normal frontal EEG has Fp1 at 3-5x F3 because blinks
    load on the prefrontal pair.
    """
    v = w.var(axis=-1)
    med = np.median(v, axis=-1)
    return np.where(med > 0, v.max(axis=-1) / np.where(med > 0, med, 1), np.inf)


def normalize(w):
    """§3.3, global over the whole (6, 512) — not per channel.

    Global preserves inter-channel amplitude ratios, which frontal asymmetry depends on.
    """
    med = np.median(w, axis=(-2, -1), keepdims=True)
    mad = np.median(np.abs(w - med), axis=(-2, -1), keepdims=True)
    w = np.clip(w, med - CLIP_MAD * mad, med + CLIP_MAD * mad)
    std = w.std(axis=(-2, -1), keepdims=True)
    assert (std > 0).all(), "zero-variance window survived the gates"
    return (w - w.mean(axis=(-2, -1), keepdims=True)) / std


def prepare(x, fs):
    """Continuous (6, n) -> (unnormalized windows, flatline-keep mask, variance ratios).

    Everything up to the gates. Shared by ``segment`` and by build.py's K sweep so the
    calibration measures the exact pipeline the build will run.
    """
    flat = flatline_mask(x, fs)
    y = resample(filter_recording(x, fs), fs)
    n = n_windows(y.shape[1])
    if n == 0:
        return (np.empty((0, x.shape[0], WINDOW)), np.ones(0, bool), np.empty(0))
    starts = np.arange(n) * STRIDE
    w = np.stack([y[:, s:s + WINDOW] for s in starts])
    scale = fs / FS  # the mask lives at the native rate; map window bounds back into it
    keep = ~np.array([flat[int(s * scale):int((s + WINDOW) * scale)].any() for s in starts])
    return w, keep, variance_ratio(w)


def segment(x, fs, k):
    """Continuous (6, n) at native fs -> (normalized windows (m, 6, 512), counts).

    ``counts`` reports the fate of every window so build.py can publish the §3.2
    rejection table: divergent rates across corpora are a finding, not a detail.
    """
    w, keep, ratio = prepare(x, fs)
    counts = dict(windows=len(w), flatline=int((~keep).sum()), variance=0, kept=0)
    if not len(w):
        return np.empty((0, x.shape[0], WINDOW), np.float32), counts

    bad_var = (ratio > k) & keep
    counts["variance"] = int(bad_var.sum())
    keep = keep & ~bad_var
    counts["kept"] = int(keep.sum())
    if not counts["kept"]:
        # a recording can lose every window to the gates; np.median dies on empty input
        return np.empty((0, x.shape[0], WINDOW), np.float32), counts
    return normalize(w[keep]).astype(np.float32), counts


def _selfcheck():
    rng = np.random.default_rng(0)
    fs, n = 250, 30 * 250
    t = np.arange(n) / fs
    # pink-ish noise + a 10 Hz alpha + 50 Hz mains + a large DC pedestal
    x = (rng.standard_normal((6, n)) + 3 * np.sin(2 * np.pi * 10 * t)
         + 5 * np.sin(2 * np.pi * 50 * t) - 1e4)

    y = filter_recording(x, fs)
    f = np.fft.rfftfreq(y.shape[1], 1 / fs)
    p = np.abs(np.fft.rfft(y, axis=-1)) ** 2
    line = p[:, (f >= 49) & (f <= 51)].sum() / p.sum()
    assert line < 1e-3, f"mains survived the notch: {line:.4f}"
    assert abs(y.mean()) < 1e-6 * abs(x.mean()), "DC pedestal survived the high-pass"

    # a 3 s flatline is marked, and only where it is
    x2 = x.copy()
    x2[2, 1000:1000 + 3 * fs] = 0
    m = flatline_mask(x2, fs)
    assert m[1000:1000 + 3 * fs].all() and m.sum() == 3 * fs, m.sum()
    assert not flatline_mask(x, fs).any(), "clean signal marked as flatline"

    w, counts = segment(x2, fs, k=100.0)
    assert counts["windows"] == n_windows(n, fs) and counts["flatline"] > 0
    assert counts["kept"] == len(w) == (counts["windows"] - counts["flatline"]
                                        - counts["variance"])
    # The two gates are complements, and this is why both exist. Windows ADJACENT to a
    # dropout do not overlap it, so the flatline rule leaves them — but filtfilt rings on
    # the step at the dropout edge and their variance ratio blows up (measured: 388 and
    # 1.1e5 here, against 1.05-1.08 for clean windows). The variance gate is what catches
    # them. Dropping either gate leaves ringing artifacts in the training set.
    assert counts["variance"] > 0, "no ringing caught beside the dropout"
    _, keep, ratio = prepare(x2, fs)
    assert ratio[keep].max() > 100 > np.sort(ratio[keep])[-3], "ringing is not localized"
    assert w.dtype == np.float32 and w.shape[1:] == (6, WINDOW)
    assert np.isfinite(w).all()
    assert abs(w.mean(axis=(1, 2))).max() < 1e-6, "normalized mean is not 0"
    assert abs(w.std(axis=(1, 2)) - 1).max() < 1e-6, "normalized std is not 1"

    # the variance gate fires on a single blown-up channel and nothing else
    x3 = x.copy()
    x3[0] *= 50
    assert segment(x3, fs, k=5.0)[1]["variance"] > 0, "variance gate never fires"
    assert segment(x, fs, k=5.0)[1]["variance"] == 0, "variance gate fires on clean data"
    print(f"preprocess OK — mains {line:.2e} of total, {counts['kept']} windows kept, "
          f"mean/std within 1e-6")


if __name__ == "__main__":
    _selfcheck()
