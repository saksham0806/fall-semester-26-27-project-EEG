"""
Classify EEG as healthy / stressed / depressed.

    .venv/bin/python predict.py recording.npy
    .venv/bin/python predict.py recording.npy --model models/B_flat_3class.joblib --fs 256

INPUT. A numpy array of shape (n_channels, n_samples) — one continuous recording — in
CHANNEL ORDER Fp1, Fp2, F7, F3, F4, F8. Any sampling rate; it is resampled to 128 Hz. The
recording is cut into 4-second windows with 50% overlap, each window is classified, and the
per-window votes are aggregated into one verdict.

Channel order is not checkable from the data — pass them in the right order or every number
below is meaningless. Use --channels to declare a different order and they will be reindexed.

READ THIS BEFORE TRUSTING AN OUTPUT. The models were trained on three research corpora
recorded on three amplifiers, and were validated subject-disjoint but NOT device-disjoint.
Corpus identity is ~97% recoverable from these features, and removing that device signature
collapses 3-class accuracy to chance. Cross-validated accuracy is therefore an UPPER BOUND
for new hardware. Treat any output as a research signal, never as a diagnosis.
"""
import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import joblib
from scipy.signal import resample

from proposed_model.features import extract_all_features

FS, WINDOW = 128, 512
STRIDE = WINDOW // 2                      # 50% overlap, as in scripts/combine_datasets.py
CANONICAL = ["Fp1", "Fp2", "F7", "F3", "F4", "F8"]
CLASSES = {0: "healthy", 1: "stressed", 2: "depressed"}


def z_score_normalize(segment):
    """Verbatim from scripts/combine_datasets.py:25 — must match training exactly.

    Global (not per-channel) on purpose: it preserves inter-channel amplitude ratios, which
    frontal alpha asymmetry depends on.
    """
    median = np.median(segment)
    mad = np.median(np.abs(segment - median)) + 1e-8
    clipped = np.clip(segment, median - 5 * mad, median + 5 * mad)
    return (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)


def validate(raw, fs, channels):
    """Fail loudly. A mis-shaped input otherwise yields a confident, meaningless answer."""
    if raw.ndim != 2:
        raise SystemExit(f"expected 2-D (n_channels, n_samples), got shape {raw.shape}")
    if raw.shape[0] > raw.shape[1]:
        raise SystemExit(f"shape {raw.shape} looks transposed — expected (channels, samples)")
    if len(channels) != raw.shape[0]:
        raise SystemExit(f"{raw.shape[0]} channels in data but {len(channels)} names given")
    if sorted(channels) != sorted(CANONICAL):
        raise SystemExit(f"need exactly these 6 channels {CANONICAL}, got {channels}")
    if fs <= 0:
        raise SystemExit(f"--fs must be positive, got {fs}")
    min_samples = int(np.ceil(WINDOW * fs / FS))
    if raw.shape[1] < min_samples:
        raise SystemExit(f"recording too short: need >= {min_samples} samples at {fs} Hz "
                         f"(4 s), got {raw.shape[1]}")
    if not np.isfinite(raw).all():
        raise SystemExit("input contains NaN or inf")


def preprocess(raw, fs, channels):
    """Reindex channels -> resample to 128 Hz -> 4 s windows @ 50% overlap -> normalise."""
    order = [channels.index(c) for c in CANONICAL]
    x = np.asarray(raw, dtype=np.float64)[order]

    if fs != FS:
        x = resample(x, int(round(x.shape[1] * FS / fs)), axis=1)

    segs = [z_score_normalize(x[:, s:s + WINDOW])
            for s in range(0, x.shape[1] - WINDOW + 1, STRIDE)]
    if not segs:
        raise SystemExit("no complete 4-second window after resampling")
    return np.stack(segs).astype(np.float32)


def predict_proba(art, F):
    """Per-window class probabilities, (n_windows, 3), for whichever approach was trained."""
    approach, m = art["approach"], art["models"]

    if approach == "flat_3class":
        return m.predict_proba(F)

    if approach == "binary_pair":
        p_dep = m["depression"].predict_proba(F)[:, 1]
        p_str = m["stress"].predict_proba(F)[:, 1]
        s = np.column_stack([(1 - p_dep) * (1 - p_str), p_str, p_dep])
        return s / s.sum(axis=1, keepdims=True)

    if approach == "hierarchical":
        route = m["router"].predict(F)
        out = np.zeros((len(F), 3))
        for ctx, (clf, pos) in m["specialists"].items():
            sel = route == ctx
            if not sel.any():
                continue
            p = clf.predict_proba(F[sel])[:, 1]
            out[sel, 0] = 1 - p
            out[sel, pos] = p
        return out

    raise SystemExit(f"unknown approach '{approach}' in artifact")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eeg", help=".npy file of shape (n_channels, n_samples)")
    ap.add_argument("--model", default="models/A_binary_pair.joblib")
    ap.add_argument("--fs", type=float, default=128.0, help="sampling rate of the input")
    ap.add_argument("--channels", default=",".join(CANONICAL),
                    help="channel order in the input file")
    ap.add_argument("--per-window", action="store_true", help="print every window's verdict")
    args = ap.parse_args()

    model_path = args.model if os.path.isabs(args.model) \
        else os.path.join(PROJECT_ROOT, args.model)
    if not os.path.exists(model_path):
        raise SystemExit(f"no model at {model_path} — run scripts/train_deployable.py first")

    raw = np.load(args.eeg)
    channels = [c.strip() for c in args.channels.split(",")]
    validate(raw, args.fs, channels)

    art = joblib.load(model_path)
    X = preprocess(raw, args.fs, channels)
    F = extract_all_features(X).astype(np.float32)
    proba = predict_proba(art, F)

    per_window = proba.argmax(axis=1)
    mean_proba = proba.mean(axis=0)
    verdict = int(mean_proba.argmax())

    print(f"\nmodel      {os.path.relpath(model_path, PROJECT_ROOT)} "
          f"({art['approach']}, CV accuracy {art['cv']['accuracy']:.4f})")
    print(f"input      {raw.shape[0]} ch x {raw.shape[1]} samples @ {args.fs:g} Hz "
          f"-> {len(X)} windows of 4 s")
    print(f"\nPREDICTION  {CLASSES[verdict].upper()}\n")
    print(f"  {'class':12s}{'mean prob':>11s}{'windows':>9s}")
    for k in sorted(CLASSES):
        n = int((per_window == k).sum())
        print(f"  {CLASSES[k]:12s}{mean_proba[k]:11.3f}{n:6d} ({100*n/len(X):.0f}%)")

    if args.per_window:
        print("\n  window  verdict")
        for i, k in enumerate(per_window):
            print(f"  {i:6d}  {CLASSES[k]}")

    agreement = float((per_window == verdict).mean())
    print(f"\n  window agreement {agreement:.0%}"
          f"{'  — LOW, treat as inconclusive' if agreement < 0.6 else ''}")
    print("\nResearch signal only, not a diagnosis. The models were validated subject-disjoint\n"
          "but NOT device-disjoint; accuracy on unseen hardware is unmeasured and likely lower.")


if __name__ == "__main__":
    main()
