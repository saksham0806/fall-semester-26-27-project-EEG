"""Section 1: rebuild MODMA / SAM40 / CogLoad into one 6-channel 128 Hz segment array.

Filter -> resample -> window -> gate -> normalize.  Reads datasets/ only.
Writes data/*.npy and data/rejection_table.json.
"""
import hashlib
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "datasets"
OUT = ROOT / "data"

FS = 128
WIN = 512
STRIDE = 256
BAND = (0.5, 45.0)
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "F4", "F8"]

# 0-based rows into each corpus's native layout, in CHANNELS order.
MODMA_IDX = [21, 8, 32, 23, 123, 121]           # EGI E22 E9 E33 E24 E124 E122
SAM40_IDX = [2, 31, 3, 4, 29, 30]               # verified against SAM40/Coordinates.locs
COG_COLS = [1, 2, 3, 4, 6, 7]                   # OpenBCI 8-ch, skipping Fz and C2

MODMA_DIR = DATA / "MODMA" / "EEG_128channels_resting_lanzhou_2015"
SAM40_DIR = DATA / "SAM40" / "raw_data"
COG_DIR = (DATA / "Cognitive Load Assessment Through EEG A Dataset from "
                  "Arithmetic and Stroop Tasks" / "raw_data")


# --------------------------------------------------------------------------- filtering

MAINS = 50.0


def bandpass(x, fs, pad=None):
    """Zero-phase 50 Hz notch + 0.5-45 Hz bandpass.

    The notch is not optional: an order-4 corner at 45 Hz leaves 50 Hz at ~0.3 amplitude, and
    MODMA has a line spike while SAM40 is already vendor-notched -- i.e. residual mains is a
    corpus signature sitting in the input to the domain probe.  Applied uniformly to all three
    corpora so the gating stays identical across them.
    SOS form: a direct b/a order-8 at 0.5/250 is numerically unstable.
    """
    sos = np.vstack([tf2sos(*iirnotch(MAINS, 30.0, fs)),
                     butter(4, [BAND[0] / (fs / 2), BAND[1] / (fs / 2)], btype="band", output="sos")])
    x = x - x.mean(axis=1, keepdims=True)       # kill the ~1e4 DC pedestal first
    if pad:                                      # short trials: explicit reflection padding
        x = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
        return sosfiltfilt(sos, x, axis=1, padlen=0)[:, pad:-pad]
    return sosfiltfilt(sos, x, axis=1)


def flat_mask(x, fs, ms=50):
    """True where a >=`ms` constant (incl. zero) run touches the sample, any channel."""
    k = max(2, int(round(fs * ms / 1000)))
    same = (np.diff(x, axis=1) == 0).any(axis=0).astype(float)
    run = np.convolve(same, np.ones(k - 1), mode="same") >= k - 1
    bad = np.convolve(run.astype(float), np.ones(k), mode="same") > 0
    return np.append(bad, bad[-1])


def to_128(x, fs):
    if fs == FS:
        return x
    from math import gcd
    g = gcd(int(fs), FS)
    return resample_poly(x, FS // g, int(fs) // g, axis=1)


# --------------------------------------------------------------------------- windowing

BLOWN = 100.0   # a channel this far above its siblings for a whole recording is a dead electrode


def segment(x, bad, counts):
    """(6, n) at 128 Hz -> list of gated, normalized (6, 512) segments.

    The variance gate is referenced to each channel's own median across the recording, not to
    the other channels within one window. The plan's intent is transient single-electrode
    spikes; the within-window form cannot tell a spike from an electrode that simply ran hot
    all session, so it rejects 100% of such a subject's windows and deletes them silently.
    """
    starts = list(range(0, x.shape[1] - WIN + 1, STRIDE))
    counts["total"] += len(starts)
    if not starts:
        return []
    V = np.array([x[:, s:s + WIN].var(axis=1) for s in starts])       # (n_win, 6)
    ref = np.median(V, axis=0)                                        # per-channel reference
    if ref.max() > BLOWN * np.median(ref):
        counts["blown_channel"] += len(starts)
        return []

    segs = []
    for i, s in enumerate(starts):
        if bad[s:s + WIN].any():
            counts["flatline"] += 1
            continue
        if (V[i] > 5 * ref).any():
            counts["variance"] += 1
            continue
        seg = x[:, s:s + WIN]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        segs.append(np.clip((seg - med) / (mad + 1e-12), -5, 5).astype(np.float32))
        counts["kept"] += 1
    return segs


# --------------------------------------------------------------------------- loaders

def modma():
    info = pd.read_excel(MODMA_DIR / "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
    lab = {int(r["subject id"]): int(r["type"] == "MDD") for _, r in info.iterrows()}
    for p in sorted(glob(str(MODMA_DIR / "*.mat"))):
        d = sio.loadmat(p)
        keys = [k for k, v in d.items()
                if not k.startswith("__") and getattr(v, "ndim", 0) == 2
                and v.shape[0] in (128, 129) and v.shape[1] > 1000]
        sid = int(re.match(r"0*(\d+)", os.path.basename(p)).group(1))
        x = d[keys[0]][MODMA_IDX].astype(np.float64)
        fs = float(d["samplingRate"][0, 0]) if "samplingRate" in d else 250.0
        yield dict(x=x, fs=fs, pad=None, subject=f"M{sid}", unit=f"M{sid}",
                   y_dep=lab[sid], y_stress=-1, y_domain=0)


def sam40():
    for p in sorted(glob(str(SAM40_DIR / "*.mat"))):
        stem = Path(p).stem
        sub = re.search(r"sub_(\d+)", stem).group(1)
        x = sio.loadmat(p)["Data"][SAM40_IDX].astype(np.float64)
        yield dict(x=x, fs=128.0, pad=384, subject=f"S{sub}", unit=f"S:{stem}",
                   y_dep=-1, y_stress=int(not stem.startswith("Relax")), y_domain=1)


def cogload():
    """120 files -> 104 unique recordings.

    Each subject's single `natural` baseline is filed in both task folders (15 byte-identical
    pairs), so the real design is 1 baseline + 6 task recordings per subject.  One further
    collision, Stroop/highlevel-14 == highlevel-15, spans two *different* subject ids: its
    provenance is unrecoverable, so both copies are dropped rather than filed under a subject
    that may not have produced it -- that would corrupt the grouping variable.
    """
    groups = {}
    for p in sorted(glob(str(COG_DIR / "*" / "*.txt"))):
        groups.setdefault(hashlib.md5(open(p, "rb").read()).hexdigest(), []).append(p)

    for paths in groups.values():
        subs = {re.search(r"(\d+)$", Path(q).stem).group(1) for q in paths}
        if len(subs) > 1:
            print(f"  ambiguous provenance, dropping: {[Path(q).name for q in paths]} -> {subs}")
            continue
        p = paths[0]
        stem = Path(p).stem
        sub = re.search(r"(\d+)$", stem).group(1)
        x = pd.read_csv(p, header=None, usecols=COG_COLS).to_numpy().T.astype(np.float64)
        nz = np.flatnonzero(~np.all(x == 0, axis=0))
        x = x[:, nz[0]:nz[-1] + 1]                # strip the all-zero header row
        yield dict(x=x, fs=250.0, pad=None, subject=f"C{sub}",
                   unit=f"C:{Path(p).parent.name.strip()}/{stem}",
                   y_dep=-1, y_stress=int(not stem.startswith("natural")), y_domain=2)


# --------------------------------------------------------------------------- build

def build(smoke=False):
    X, meta, rej = [], [], {}
    for name, gen in [("MODMA", modma()), ("SAM40", sam40()), ("CogLoad", cogload())]:
        c = dict(total=0, kept=0, flatline=0, variance=0, blown_channel=0)
        n_rec = 0
        for i, r in enumerate(gen):
            if smoke and i >= 3:
                break
            x = bandpass(r["x"], r["fs"], r["pad"])
            bad = flat_mask(r["x"], r["fs"])
            x = to_128(x, r["fs"])
            idx = np.minimum((np.arange(x.shape[1]) * r["fs"] / FS).astype(int), len(bad) - 1)
            segs = segment(x, bad[idx], c)
            n_rec += 1
            for s in segs:
                X.append(s)
                meta.append((r["subject"], r["unit"], r["y_dep"], r["y_stress"], r["y_domain"]))
            print(f"\r{name} {i + 1} {r['unit']:<32} kept={c['kept']}", end="", flush=True)
        c["recordings"] = n_rec
        rej[name] = c
        print()

    X = np.stack(X)
    sub, unit, dep, st, dom = zip(*meta)
    OUT.mkdir(exist_ok=True)
    np.save(OUT / "X.npy", X)
    np.save(OUT / "subject.npy", np.array(sub))
    np.save(OUT / "unit.npy", np.array(unit))
    np.save(OUT / "y_dep.npy", np.array(dep, np.int8))
    np.save(OUT / "y_stress.npy", np.array(st, np.int8))
    np.save(OUT / "y_domain.npy", np.array(dom, np.int8))
    # anchor: the non-pathological baseline of whichever task the corpus supports
    anchor = (np.array(dep) == 0) | (np.array(st) == 0)
    np.save(OUT / "anchor.npy", anchor)
    json.dump(rej, open(OUT / "rejection_table.json", "w"), indent=2)

    print(f"\nX {X.shape}  nan={np.isnan(X).sum()}  anchors={anchor.sum()}")
    for k, v in rej.items():
        print(f"  {k:<8} {v['recordings']:>3} rec  {v['kept']:>6} kept / {v['total']:>6} "
              f"({v['flatline']} flatline, {v['variance']} variance, "
              f"{v['blown_channel']} blown-channel)")


def selfcheck():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 3000))
    x[:, 1000:1040] = 0                                    # 40 samples flat @128 Hz = 312 ms
    m = flat_mask(x, 128)
    assert m[1000:1039].all() and not m[500], "flatline mask missed the run"
    assert not flat_mask(rng.normal(size=(6, 3000)), 128).any(), "false flatline"

    def tone(f, dc=1e4):
        # Q=30 rings for ~1 s, so measure well away from the filter edges.
        y = bandpass(np.tile(np.sin(2 * np.pi * f * np.arange(6000) / 250), (6, 1)) + dc, 250.0)
        return np.abs(y[:, 1000:-1000]).max(), abs(y.mean())
    assert tone(10)[0] > 0.9 and tone(40)[0] > 0.5, "passband should survive"
    assert tone(50)[0] < 0.02, "50 Hz mains survived the notch"
    assert tone(60)[0] < 0.25 and tone(80)[0] < 0.25, "stopband too leaky"
    assert tone(10)[1] < 1e-5 * 1e4, "DC pedestal survived"   # >120 dB down
    assert to_128(np.zeros((6, 2500)), 250.0).shape[1] == 1280, "resample ratio wrong"

    cnt = lambda: dict(total=0, kept=0, flatline=0, variance=0, blown_channel=0)
    rec = rng.normal(size=(6, 512 * 12))
    ok = np.zeros(rec.shape[1], bool)

    c = cnt()                                               # transient spike -> that window only
    spiked = rec.copy()
    spiked[3, 2560:3072] *= 30
    segment(spiked, ok, c)
    assert 1 <= c["variance"] <= 4 and c["kept"] >= 19, c   # 50% overlap: one spike hits ~3

    c = cnt()                                               # hot electrode all session -> keep it
    hot = rec.copy()
    hot[3] *= 10
    segment(hot, ok, c)
    assert c["variance"] == 0 and c["kept"] == c["total"], c

    c = cnt()                                               # dead electrode -> drop the recording
    dead = rec.copy()
    dead[3] *= 500
    segment(dead, ok, c)
    assert c["kept"] == 0 and c["blown_channel"] == c["total"], c

    s = segment(rec[:, :512], np.zeros(512, bool), cnt())[0]
    assert np.abs(s).max() <= 5 and abs(np.median(s)) < 1e-6, "normalization off"
    print("preprocess selfcheck ok")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        selfcheck()
    else:
        build(smoke="--smoke" in sys.argv)
