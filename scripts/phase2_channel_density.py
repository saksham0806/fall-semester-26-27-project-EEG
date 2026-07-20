"""
Phase 2 M5 — what does the 6-channel frontal bottleneck cost?

Every corpus is projected onto 6 frontal electrodes so they can be pooled, but MODMA
records 128. This sweeps channel count on MODMA depression (6 -> 16 -> 32 -> 64) with
the M1 Riemannian pipeline held fixed, so the only variable is spatial coverage.

Channels are subsampled evenly across the montage rather than picked by name: the
question is "does more coverage help", not "which electrodes matter". Covariances are
computed per subject and the raw signal is discarded, so peak memory stays ~1 GB.

Run:  .venv/bin/python -u scripts/phase2_channel_density.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from proposed_model.features import BANDS, butter_bandpass_filter

MODMA_DIR = os.path.join(PROJECT_ROOT, "datasets", "MODMA",
                         "EEG_128channels_resting_lanzhou_2015")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase2")
NATIVE_FS, FS, WINDOW, STRIDE = 250, 128, 512, 256
CHANNEL_COUNTS = [6, 16, 32, 64]
# The 6 frontal indices the main pipeline uses (EGI E22/E9/E33/E24/E124/E122).
FRONTAL6 = [21, 8, 32, 23, 123, 121]
N_SPLITS = 5


def zscore(seg):
    """Same outlier-clipped global normalization as scripts/combine_datasets.py."""
    med = np.median(seg)
    mad = np.median(np.abs(seg - med)) + 1e-8
    seg = np.clip(seg, med - 5 * mad, med + 5 * mad)
    return (seg - seg.mean()) / (seg.std() + 1e-8)


def load_modma_segments(channels):
    """Windowed MODMA segments at the given channel indices. Returns X, y, subjects."""
    meta = pd.read_excel(os.path.join(
        MODMA_DIR, "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx"))
    meta.columns = [str(c).strip() for c in meta.columns]
    id_col, type_col = meta.columns[0], [c for c in meta.columns if "type" in c.lower()][0]
    label = {str(r[id_col]).strip(): (1 if str(r[type_col]).strip() == "MDD" else 0)
             for _, r in meta.iterrows()}

    X, y, subj = [], [], []
    for f in sorted(glob.glob(os.path.join(MODMA_DIR, "*.mat"))):
        sid = os.path.basename(f).split("_")[0].replace("mat", "")
        key = next((k for k in label if sid.startswith(k) or k in sid), None)
        if key is None:
            continue
        mat = scipy.io.loadmat(f)
        arr = next((v for v in mat.values()
                    if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] in (128, 129)
                    and v.shape[1] > 1000), None)
        if arr is None:
            continue
        data = arr[channels, :]
        data = resample(data, int(data.shape[1] * FS / NATIVE_FS), axis=1)
        for s in range(0, data.shape[1] - WINDOW + 1, STRIDE):
            X.append(zscore(data[:, s:s + WINDOW]).astype(np.float32))
            y.append(label[key])
            subj.append(f"modma_{key}")
    return np.array(X), np.array(y), np.array(subj, dtype=object)


def tangent_features(X):
    """Filter-bank covariance -> global re-centering -> tangent space."""
    out = []
    for low, high in BANDS.values():
        covs = Covariances(estimator="oas").fit_transform(butter_bandpass_filter(X, low, high, FS))
        W = invsqrtm(mean_riemann(covs))
        out.append(TangentSpace(metric="riemann").fit_transform(W @ covs @ W))
    return np.hstack(out)


def subject_auc(y, prob, subjects):
    subs = np.unique(subjects)
    t = np.array([y[subjects == s][0] for s in subs])
    p = np.array([prob[subjects == s].mean() for s in subs])
    return float(roc_auc_score(t, p))


def evaluate(F, y, subj):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subj):
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(F[te]))[:, 1]
    return float(roc_auc_score(y, oof)), subject_auc(y, oof, subj)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    print(f"{'channels':>9s} {'tangent dims':>13s} {'segment AUC':>12s} {'subject AUC':>12s}")
    print("-" * 50)
    for n in CHANNEL_COUNTS:
        chans = FRONTAL6 if n == 6 else np.linspace(0, 127, n).astype(int).tolist()
        X, y, subj = load_modma_segments(chans)
        if len(X) == 0:
            print(f"  {n}: no segments loaded — check MODMA path/labels")
            continue
        F = tangent_features(X)
        seg, sub = evaluate(F, y, subj)
        results[n] = {"tangent_dims": int(F.shape[1]), "auc_segment": seg, "auc_subject": sub,
                      "n_segments": int(len(y)), "n_subjects": int(len(np.unique(subj)))}
        print(f"{n:9d} {F.shape[1]:13d} {seg:12.4f} {sub:12.4f}")
        del X, F

    if results:
        base = results.get(6, {}).get("auc_subject")
        best = max(results.items(), key=lambda kv: kv[1]["auc_subject"])
        print("-" * 50)
        print(f"6-ch subject AUC {base:.4f}  ->  best {best[0]}-ch {best[1]['auc_subject']:.4f} "
              f"(delta {best[1]['auc_subject'] - base:+.4f})")
        print("Phase 1 SVM on 81 handcrafted 6-ch features: 0.7716")

    with open(os.path.join(OUT_DIR, "channel_density.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'channel_density.json')}")


if __name__ == "__main__":
    main()
