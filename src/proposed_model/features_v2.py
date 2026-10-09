"""
Feature extraction v2 — corrected preprocessing plus the families v1 lacks.

WHY THIS EXISTS. v1 (`features.py`) divides band power by total power integrated over the
full 0-64 Hz, while the five bands only span 0.5-45 Hz. Measured on datasets/all3, the
unused 45-64 Hz gap holds the great majority of MODMA's total power (~88-98% across samples,
a sharp 50 Hz mains spike), 19-75% of CogLoad's, and ~34% of SAM40's. SAM40's is broadband
with no line peak at all — its vendor already notch-filtered — so the contaminant's *shape*
is corpus-specific, not just its size. It sits in the denominator of every relative-power
feature, which makes it a plausible driver of the 91.6% corpus recoverability behind the
all3 confound.

v2 therefore notch-filters the mains and bandpasses to 0.5-45 Hz, zero-phase, BEFORE
computing anything, and computes the relative-power denominator over 0.5-45 Hz only.

`features.py` is deliberately NOT modified — nine call sites and every published result
depend on it. v1 stays bit-for-bit reproducible.

Only numpy/scipy are used here, but note that importing this module as
`proposed_model.features_v2` still executes `proposed_model/__init__.py`, which imports
TensorFlow. That is a pre-existing property of the package, not of this file.

Blocks (273 handcrafted columns total):
    bandpower_rel   30   relative band power, denominator fixed to 0.5-45 Hz
    de              30   differential entropy (log absolute band power)
    faa              3   frontal alpha asymmetry
    amplitude       12   STD, RMS
    spectral_shape  24   spectral entropy, 95% edge freq, 1/f slope + offset
    hjorth          18   activity, mobility, complexity
    perm_entropy     6   permutation entropy (order 3, delay 1)
    coherence       75   magnitude-squared coherence, 15 pairs x 5 bands
    plv             75   phase-locking value, 15 pairs x 5 bands
"""
import itertools
import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import butter, coherence, hilbert, iirnotch, sosfiltfilt, tf2sos, welch

from .features import ASYMMETRY_PAIRS, BANDS, CHANNEL_MAP

FS = 128
# The band the signal is restricted to. Must match the span of BANDS, or the relative-power
# denominator reintroduces the v1 defect.
TOTAL_BAND = (0.5, 45.0)
# Mains. All three corpora show a 50 Hz peak or none at all; no 60 Hz peak was measured in
# any of them, and adding a 60 Hz notch measurably improved nothing (0.00009 vs 0.00010).
LINE_FREQ = 50.0
NOTCH_Q = 30.0
# Order 8, not 4. Measured on real data: order 4 cornered at 45 Hz leaves 9.2% of MODMA's
# power above 45 Hz, because 45 and 50 Hz are only 11% apart and Butterworth roll-off cannot
# separate them. Order 8 plus the notch takes MODMA to 0.01%. Per-band filters stay at
# order 4 — they run after preprocessing, so they never see the gap.
BANDPASS_ORDER = 8
# 1/f fit range. Starts at 2 Hz: below that the bandpass roll-off dominates and would drag
# the slope toward a value describing the filter rather than the neural aperiodic component.
APERIODIC_BAND = (2.0, 45.0)
CHANNELS = list(CHANNEL_MAP)
PAIRS = list(itertools.combinations(range(len(CHANNELS)), 2))  # 15 pairs
EPS = 1e-12


# ---------------------------------------------------------------------------- filtering

def _sos(low, high, fs=FS, order=4):
    """SOS form, not (b, a) — the ba form is numerically unstable at a 0.5 Hz corner."""
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def bandpass(X, low, high, fs=FS, order=4):
    """Zero-phase bandpass. Unlike v1's causal lfilter this introduces no phase distortion."""
    return sosfiltfilt(_sos(low, high, fs, order), X, axis=-1).astype(np.float32)


def _notch_sos(f0, q=NOTCH_Q, fs=FS):
    b, a = iirnotch(f0, q, fs=fs)
    return tf2sos(b, a)


def preprocess(X, fs=FS, line_freq=LINE_FREQ):
    """
    The v2 fix: notch the mains, then restrict to 0.5-45 Hz, so nothing from the 45-64 Hz
    gap reaches any feature.

    Both stages are needed and neither alone suffices, measured on this dataset:
      - the notch alone leaves SAM40's broadband HF content (SAM40 has no line peak at all,
        its vendor already notch-filtered, but 34% of its power sits above 45 Hz);
      - the bandpass alone cannot reject 50 Hz sharply enough next to a 45 Hz corner.
    """
    x = X.astype(np.float64)          # sosfiltfilt promotes anyway; be explicit about the cost
    if line_freq:
        x = sosfiltfilt(_notch_sos(line_freq, fs=fs), x, axis=-1)
    x = sosfiltfilt(_sos(TOTAL_BAND[0], TOTAL_BAND[1], fs, BANDPASS_ORDER), x, axis=-1)
    return x.astype(np.float32)


def gap_power_fraction(X, fs=FS, cutoff=45.0):
    """Fraction of spectral power above `cutoff`. The check that preprocess() worked."""
    freqs, psd = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    total = trapezoid(psd, freqs, axis=-1) + EPS
    m = freqs >= cutoff
    return float(np.mean(trapezoid(psd[..., m], freqs[m], axis=-1) / total))


# ---------------------------------------------------------------------- spectral blocks

def extract_bandpower_rel(X, fs=FS):
    """
    Relative band power, (N, 30). Band-major layout, matching v1.

    THE v2 FIX: the denominator integrates over TOTAL_BAND (0.5-45 Hz), not the full
    0-64 Hz Nyquist range as v1 does.
    """
    n_ch = X.shape[1]
    freqs, psd = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    m_tot = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    total = trapezoid(psd[..., m_tot], freqs[m_tot], axis=-1) + EPS

    out = np.zeros((len(X), n_ch * len(BANDS)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS.values()):
        m = (freqs >= lo) & (freqs <= hi)
        out[:, i * n_ch:(i + 1) * n_ch] = trapezoid(psd[..., m], freqs[m], axis=-1) / total
    return out


def extract_de(X, fs=FS):
    """Differential entropy, (N, 30). Same definition as v1 but zero-phase filtered."""
    n_ch = X.shape[1]
    const = 0.5 * np.log(2 * np.pi * np.e)
    out = np.zeros((len(X), n_ch * len(BANDS)), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS.values()):
        var = np.var(bandpass(X, lo, hi, fs), axis=-1) + 1e-8
        out[:, i * n_ch:(i + 1) * n_ch] = 0.5 * np.log(var) + const
    return out


def extract_faa(X, fs=FS):
    """Frontal alpha asymmetry, (N, 3). ln(alpha_right) - ln(alpha_left)."""
    freqs, psd = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)
    lo, hi = BANDS["alpha"]
    m = (freqs >= lo) & (freqs <= hi)
    log_alpha = np.log(np.maximum(trapezoid(psd[..., m], freqs[m], axis=-1), 1e-8))

    out = np.zeros((len(X), len(ASYMMETRY_PAIRS)), dtype=np.float32)
    for i, (right, left) in enumerate(ASYMMETRY_PAIRS):
        out[:, i] = log_alpha[:, CHANNEL_MAP[right]] - log_alpha[:, CHANNEL_MAP[left]]
    return out


def extract_amplitude(X):
    """STD then RMS, (N, 12). On the filtered signal, so mains hum no longer inflates it."""
    return np.hstack([np.std(X, axis=-1),
                      np.sqrt(np.mean(X ** 2, axis=-1))]).astype(np.float32)


def extract_spectral_shape(X, fs=FS):
    """
    (N, 24) = spectral entropy (6) | 95% edge frequency (6) | 1/f slope (6) | offset (6).

    The aperiodic slope is the non-oscillatory 1/f component that band power ignores
    entirely — the genuinely new information in this block.
    """
    freqs, psd = welch(X, fs=fs, nperseg=min(256, X.shape[-1]), axis=-1)

    m = (freqs >= TOTAL_BAND[0]) & (freqs <= TOTAL_BAND[1])
    f_b, p_b = freqs[m], psd[..., m]

    p_norm = p_b / (p_b.sum(axis=-1, keepdims=True) + EPS)
    entropy = -np.sum(p_norm * np.log2(p_norm + EPS), axis=-1)

    cum = np.cumsum(p_b, axis=-1)
    cum = cum / (cum[..., -1:] + EPS)
    edge = f_b[np.argmax(cum >= 0.95, axis=-1)]

    # Least-squares slope of log10(PSD) on log10(f), done closed-form so it vectorises.
    ma = (freqs >= APERIODIC_BAND[0]) & (freqs <= APERIODIC_BAND[1])
    lf = np.log10(freqs[ma])
    lp = np.log10(psd[..., ma] + 1e-20)
    lf_c = lf - lf.mean()
    slope = (lp * lf_c).sum(axis=-1) / (lf_c ** 2).sum()
    offset = lp.mean(axis=-1) - slope * lf.mean()

    return np.hstack([entropy, edge, slope, offset]).astype(np.float32)


# ------------------------------------------------------------------- time-domain blocks

def extract_hjorth(X):
    """(N, 18) = activity (6) | mobility (6) | complexity (6)."""
    d1 = np.diff(X, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = np.var(X, axis=-1) + EPS
    v1 = np.var(d1, axis=-1) + EPS
    v2 = np.var(d2, axis=-1) + EPS

    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / mobility
    return np.hstack([v0, mobility, complexity]).astype(np.float32)


def extract_perm_entropy(X, order=3, delay=1, chunk=2000):
    """
    Permutation entropy, (N, 6), normalised to [0, 1].

    Vectorised over segments — a Python loop across 94k series would dominate runtime.
    Chunked because the (N, C, T, order) embedding plus its argsort is the largest
    intermediate in this module.
    """
    N, C, T = X.shape
    n = T - (order - 1) * delay
    n_bins = order ** order          # sparse (only order! codes occur) but keeps encoding trivial
    norm = math.log2(math.factorial(order))
    out = np.zeros((N, C), dtype=np.float32)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        emb = np.stack([X[s:e, :, i * delay: i * delay + n] for i in range(order)], axis=-1)
        ranks = np.argsort(emb, axis=-1)

        codes = np.zeros(ranks.shape[:-1], dtype=np.int64)
        for i in range(order):
            codes = codes * order + ranks[..., i]

        flat = codes.reshape(-1, n)
        offs = (np.arange(flat.shape[0], dtype=np.int64) * n_bins)[:, None]
        counts = np.bincount((flat + offs).ravel(),
                             minlength=flat.shape[0] * n_bins).reshape(-1, n_bins)
        p = counts / n
        H = -np.sum(np.where(p > 0, p * np.log2(np.maximum(p, EPS)), 0.0), axis=-1)
        out[s:e] = (H / norm).reshape(e - s, C)
    return out


# ------------------------------------------------------------------- connectivity blocks

def extract_coherence(X, fs=FS):
    """
    Magnitude-squared coherence, (N, 75). Band-major: 5 bands x 15 channel pairs.

    nperseg=128 with 50% overlap gives 7 averaged segments across a 512-sample window.
    nperseg=256 would average only 2 and bias coherence hard toward 1.0 — a silent failure
    that still produces plausible-looking numbers.
    """
    out = np.zeros((len(X), len(BANDS) * len(PAIRS)), dtype=np.float32)
    for pi, (a, b) in enumerate(PAIRS):
        f, cxy = coherence(X[:, a, :], X[:, b, :], fs=fs,
                           nperseg=128, noverlap=64, axis=-1)
        for bi, (lo, hi) in enumerate(BANDS.values()):
            m = (f >= lo) & (f <= hi)
            out[:, bi * len(PAIRS) + pi] = cxy[:, m].mean(axis=-1)
    return out


def extract_plv(X, fs=FS, chunk=2000):
    """
    Phase-locking value, (N, 75). Band-major, same layout as coherence.

    PLV discards amplitude entirely, so it measures phase coupling — which the Riemannian
    covariance view is structurally blind to. That is what makes this block non-redundant.

    Chunked: one band of Hilbert output at full size is ~770 MB as complex128, so a naive
    5-band pass would swap on a 15 GB box.
    """
    N = len(X)
    out = np.zeros((N, len(BANDS) * len(PAIRS)), dtype=np.float32)

    for bi, (lo, hi) in enumerate(BANDS.values()):
        sos = _sos(lo, hi, fs)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            xb = sosfiltfilt(sos, X[s:e], axis=-1)
            phase = np.angle(hilbert(xb, axis=-1)).astype(np.float32)
            for pi, (a, b) in enumerate(PAIRS):
                d = phase[:, a, :] - phase[:, b, :]
                # |mean(e^{i d})| without materialising a complex array
                out[s:e, bi * len(PAIRS) + pi] = np.hypot(np.cos(d).mean(axis=-1),
                                                          np.sin(d).mean(axis=-1))
    return out


# ------------------------------------------------------------------------------ registry

# Order defines the column order of the combined matrix. Keep in sync with block_names().
BLOCKS = {
    "bandpower_rel": extract_bandpower_rel,
    "de": extract_de,
    "faa": extract_faa,
    "amplitude": extract_amplitude,
    "spectral_shape": extract_spectral_shape,
    "hjorth": extract_hjorth,
    "perm_entropy": extract_perm_entropy,
    "coherence": extract_coherence,
    "plv": extract_plv,
}


def block_names(block):
    """Column labels for one block, in the exact order that block emits them."""
    ch = CHANNELS
    pair = [f"{ch[a]}_{ch[b]}" for a, b in PAIRS]
    if block == "bandpower_rel":
        return [f"relpsd_{b}_{c}" for b in BANDS for c in ch]
    if block == "de":
        return [f"de_{b}_{c}" for b in BANDS for c in ch]
    if block == "faa":
        return [f"faa_{r}_{l}" for r, l in ASYMMETRY_PAIRS]
    if block == "amplitude":
        return [f"std_{c}" for c in ch] + [f"rms_{c}" for c in ch]
    if block == "spectral_shape":
        return ([f"specent_{c}" for c in ch] + [f"edge95_{c}" for c in ch]
                + [f"aperiodic_slope_{c}" for c in ch] + [f"aperiodic_offset_{c}" for c in ch])
    if block == "hjorth":
        return ([f"hjorth_activity_{c}" for c in ch] + [f"hjorth_mobility_{c}" for c in ch]
                + [f"hjorth_complexity_{c}" for c in ch])
    if block == "perm_entropy":
        return [f"perment_{c}" for c in ch]
    if block == "coherence":
        return [f"coh_{b}_{p}" for b in BANDS for p in pair]
    if block == "plv":
        return [f"plv_{b}_{p}" for b in BANDS for p in pair]
    raise KeyError(block)


def extract_all_features_v2(X, fs=FS, preprocessed=False, verbose=False):
    """
    Every block, as a dict {name: (N, k) float32}.

    `preprocessed=False` applies the 0.5-45 Hz bandpass first. Pass True only if X is
    already filtered, otherwise the v2 fix does not happen.
    """
    Xf = X if preprocessed else preprocess(X, fs)
    out = {}
    for name, fn in BLOCKS.items():
        if verbose:
            print(f"    {name} ...", flush=True)
        out[name] = fn(Xf) if name in ("amplitude", "hjorth", "perm_entropy") else fn(Xf, fs)
    return out


def _selfcheck():
    """Shape/finiteness/behaviour checks on synthetic data. Runs in a few seconds."""
    rng = np.random.default_rng(0)
    n, fs = 40, FS
    t = np.arange(512) / fs

    # 10 Hz alpha + a strong 50 Hz "mains" component, the situation v1 mishandles.
    X = (rng.standard_normal((n, 6, 512)).astype(np.float32) * 0.1
         + np.sin(2 * np.pi * 10 * t).astype(np.float32)
         + 5.0 * np.sin(2 * np.pi * 50 * t).astype(np.float32))

    before = gap_power_fraction(X)
    Xf = preprocess(X)
    after = gap_power_fraction(Xf)
    assert before > 0.5, f"synthetic mains should dominate, got {before:.3f}"
    assert after < 0.01, f"preprocess left {after:.4f} of power above 45 Hz"

    feats = extract_all_features_v2(X)
    expected = {"bandpower_rel": 30, "de": 30, "faa": 3, "amplitude": 12,
                "spectral_shape": 24, "hjorth": 18, "perm_entropy": 6,
                "coherence": 75, "plv": 75}
    for name, k in expected.items():
        F = feats[name]
        assert F.shape == (n, k), f"{name}: expected {(n, k)}, got {F.shape}"
        assert np.isfinite(F).all(), f"{name}: non-finite values"
        assert len(block_names(name)) == k, f"{name}: name count != column count"

    assert sum(expected.values()) == 273

    # Relative band power must sum to ~1 across bands now that the denominator matches.
    bp = feats["bandpower_rel"].reshape(n, len(BANDS), 6).sum(axis=1)
    assert np.allclose(bp, 1.0, atol=0.02), f"relative power sums to {bp.mean():.3f}, not 1"

    # Alpha should be the dominant band for a 10 Hz carrier.
    per_band = feats["bandpower_rel"].reshape(n, len(BANDS), 6).mean(axis=(0, 2))
    dominant = list(BANDS)[int(np.argmax(per_band))]
    assert dominant == "alpha", f"10 Hz signal should peak in alpha, got {dominant}"

    # PLV and coherence are bounded.
    assert (feats["plv"] >= 0).all() and (feats["plv"] <= 1.0001).all()
    assert (feats["coherence"] >= 0).all() and (feats["coherence"] <= 1.0001).all()
    # Permutation entropy is normalised.
    assert (feats["perm_entropy"] >= 0).all() and (feats["perm_entropy"] <= 1.0001).all()

    print(f"features_v2 selfcheck OK — 273 columns, "
          f"gap power {before:.3f} -> {after:.5f} after bandpass")


if __name__ == "__main__":
    _selfcheck()
