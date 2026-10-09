"""
Phase 2 M2 — CBraMod foundation model, frozen backbone + linear probe.

Phase 1 showed every deep architecture trained from scratch sits at chance on stress
(~0.52) while an SVM on handcrafted features reaches 0.72, and M1 (Riemannian) reaches
0.7560. The stated cause is data volume: ~1,900 training segments per fold is far too
little to learn a representation. A foundation model pretrained on the TUH EEG Corpus
tests that explanation directly — if the constraint really is data, a frozen pretrained
representation should beat a from-scratch one by a wide margin.

Frozen probe rather than fine-tuning, deliberately: it is cheap, it needs no GPU, and it
measures the quality of the *pretrained representation* rather than this machine's ability
to fine-tune. If the probe is not competitive, fine-tuning is unlikely to rescue it.

CBraMod expects 200 Hz with 1-second (200-sample) patches; our windows are 512 samples at
128 Hz, so they are resampled to 800 samples at 200 Hz.

Run:  .venv/bin/python -u scripts/phase2_cbramod_probe.py
"""
import json
import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from braindecode.models import CBraMod
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase2")
CACHE = os.path.join(PROJECT_ROOT, ".cache_cbramod.pth")
WEIGHTS_URL = "https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth"
SAM40, N_SPLITS, BATCH = 1, 5, 64
TARGET_LEN = 800  # 4 s at 200 Hz = 4 patches of 200
M1_BASELINE = 0.7560


def load_sam40():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_stress.npy"))
    subj = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    ds = np.load(os.path.join(DATA_DIR, "datasets.npy"))
    v = ~np.isnan(X).any(axis=(1, 2))
    X, y, subj, ds = X[v], y[v], subj[v], ds[v]
    m = (ds == SAM40) & (y != -1)
    return X[m], y[m].astype(int), subj[m]


def build_backbone(pretrained=True):
    """CBraMod with the classification head removed, so forward() returns embeddings."""
    model = CBraMod(n_chans=6, n_times=TARGET_LEN, n_outputs=2)
    loaded = False
    if pretrained:
        if not os.path.exists(CACHE):
            print("  downloading CBraMod pretrained weights (~20 MB)...")
            urllib.request.urlretrieve(WEIGHTS_URL, CACHE)
        sd = torch.load(CACHE, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        sd = {k.replace("backbone.", "").replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        matched = len(sd) - len(unexpected)
        loaded = matched > 0
        print(f"  loaded {matched}/{len(sd)} pretrained tensors "
              f"({len(missing)} missing, {len(unexpected)} unexpected)")
        if not loaded:
            print("  WARNING: no pretrained tensors matched — this is a random init, "
                  "and the probe below measures nothing about pretraining.")
    model.final_layer = nn.Identity()
    model.eval()
    return model, loaded


@torch.no_grad()
def embed(model, X):
    """Resample 512@128Hz -> 800@200Hz, then extract frozen embeddings."""
    out = []
    for i in range(0, len(X), BATCH):
        b = resample(X[i:i + BATCH], TARGET_LEN, axis=2)
        e = model(torch.from_numpy(np.ascontiguousarray(b)).float())
        out.append(e.reshape(e.shape[0], -1).cpu().numpy())  # (B, ch, patch, d) -> (B, D)
    return np.vstack(out)


def within_subject_auc(y, prob, subjects):
    s = [roc_auc_score(y[subjects == k], prob[subjects == k]) for k in np.unique(subjects)
         if len(np.unique(y[subjects == k])) > 1]
    return float(np.mean(s)) if s else float("nan")


def probe(F, y, subj):
    scores = []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subj):
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        prob = clf.predict_proba(sc.transform(F[te]))[:, 1]
        scores.append(within_subject_auc(y[te], prob, subj[te]))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, subj = load_sam40()
    print(f"SAM40: {len(y)} segments, {len(np.unique(subj))} subjects\n")

    results = {}
    for tag, pre in [("pretrained", True), ("random init (control)", False)]:
        print(f"{tag}:")
        model, loaded = build_backbone(pretrained=pre)
        F = embed(model, X)
        mean, std = probe(F, y, subj)
        results[tag] = {"auc_within": [mean, std], "dims": int(F.shape[1]),
                        "weights_loaded": bool(loaded)}
        print(f"  embedding dim={F.shape[1]}  within-subj AUC = {mean:.4f} +/- {std:.4f}"
              f"   (M1 Riemannian {M1_BASELINE:.4f}, delta {mean - M1_BASELINE:+.4f})\n")
        del model, F

    # The control is the point: pretrained minus random isolates what pretraining bought,
    # separately from whatever the architecture alone contributes.
    gain = results["pretrained"]["auc_within"][0] - results["random init (control)"]["auc_within"][0]
    print(f"pretraining gain over random init: {gain:+.4f}")

    with open(os.path.join(OUT_DIR, "cbramod_probe.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {os.path.join(OUT_DIR, 'cbramod_probe.json')}")


if __name__ == "__main__":
    main()
