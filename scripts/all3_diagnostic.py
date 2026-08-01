"""
Confound diagnostic for datasets/all3/ — is a 3-class result real, or device detection?

Class is heavily confounded with recording corpus: MODMA holds only {healthy, depression},
SAM40/CogLoad only {healthy, stress}, and each used a different amplifier. Corpus identity
is therefore trivially recoverable and collapses the 3-class problem to 2 for free. A model
knowing ONLY the corpus scores 0.6267 with zero EEG understanding.

Three checks, each answering something the headline accuracy cannot:

  1. corpus probe      — how recoverable is corpus identity? If ~1.0, the shortcut is live.
  2. 3-class CV        — accuracy measured against the 0.6267 shortcut, not against chance.
  3. within-corpus     — accuracy inside MODMA and inside the task corpora separately.
                         This strips the free corpus split out entirely and is the only
                         honest measure of what was learned about mental state.

Riemannian is run with PER-CORPUS re-centering, which whitens each device's mean covariance
to the identity — i.e. it removes the very signature the shortcut exploits. Per-subject
re-centering is deliberately NOT used: MODMA subjects are entirely healthy or entirely
depressed, so it would delete that contrast outright (measured elsewhere as AUC 0.5000).

Run:  .venv/bin/python -u scripts/all3_diagnostic.py
"""
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from proposed_model.features import BANDS, butter_bandpass_filter, extract_all_features
from proposed_model import get_svm_pipeline

DATA = os.path.join(PROJECT_ROOT, "datasets", "all3")
OUT = os.path.join(PROJECT_ROOT, "Results", "all3")
CLASS_NAMES = {0: "healthy", 1: "stress", 2: "depression"}
CORPUS_NAMES = {0: "MODMA", 1: "SAM40", 2: "CogLoad"}
N_SPLITS, FS = 5, 128


def load():
    return (np.load(os.path.join(DATA, "X.npy")),
            np.load(os.path.join(DATA, "y.npy")).astype(int),
            np.load(os.path.join(DATA, "subjects.npy"), allow_pickle=True),
            np.load(os.path.join(DATA, "corpus.npy")))


def riemannian_features(X, corpus):
    """Per-band covariance -> per-corpus re-centering -> tangent space."""
    out = []
    for lo, hi in BANDS.values():
        covs = Covariances(estimator="oas").fit_transform(butter_bandpass_filter(X, lo, hi, FS))
        rc = np.empty_like(covs)
        for c in np.unique(corpus):          # per-corpus: whitens the device signature
            m = corpus == c
            W = invsqrtm(mean_riemann(covs[m]))
            rc[m] = W @ covs[m] @ W
        out.append(TangentSpace(metric="riemann").fit_transform(rc))
    return np.hstack(out)


def shortcut_baseline(y, corpus):
    """Best constant prediction within each corpus — the accuracy of knowing only the device."""
    return float(sum(np.bincount(y[corpus == c]).max()
                     for c in np.unique(corpus)) / len(y))


def oof_predict(F, y, subjects):
    """Out-of-fold 3-class predictions under subject-grouped CV."""
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(F[te]))
    return pred


def slug(name):
    """Filesystem-safe key for the saved prediction array."""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def report(name, y, pred, corpus, shortcut, results, elapsed_sec=None):
    acc = float((pred == y).mean())
    bal = float(balanced_accuracy_score(y, pred))
    macro_f1 = float(f1_score(y, pred, average="macro"))
    margin = acc - shortcut
    print(f"\n  {name}")
    print(f"    3-class accuracy      {acc:.4f}   balanced {bal:.4f}   macro-F1 {macro_f1:.4f}")
    print(f"    vs corpus-ID shortcut {shortcut:.4f}   margin {margin:+.4f}"
          f"   {'<-- REAL SIGNAL' if margin > 0.02 else '<-- NO BETTER THAN DEVICE DETECTION'}")

    # Per-class precision/recall/F1 — a 3-class accuracy can hide a class never predicted.
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=sorted(CLASS_NAMES),
                                                 zero_division=0)
    per_class = {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])}
                 for i in sorted(CLASS_NAMES)}
    print("      " + "".join(f"{CLASS_NAMES[i]:>12s}" for i in sorted(CLASS_NAMES)))
    for label, vals in (("precision", p), ("recall", r), ("f1", f)):
        print(f"    {label:9s}" + "".join(f"{v:12.4f}" for v in vals))

    within = {}
    for c in np.unique(corpus):
        m = corpus == c
        a = float((pred[m] == y[m]).mean())
        base = float(np.bincount(y[m]).max() / m.sum())
        within[CORPUS_NAMES[c]] = {"accuracy": a, "majority_within": base,
                                   "margin": a - base,
                                   "macro_f1": float(f1_score(y[m], pred[m], average="macro"))}
        print(f"    within {CORPUS_NAMES[c]:8s} {a:.4f}  (majority {base:.4f}, margin {a - base:+.4f})")

    results[name] = {"accuracy": acc, "balanced_accuracy": bal, "macro_f1": macro_f1,
                     "shortcut_baseline": shortcut, "margin": margin,
                     "per_class": per_class,
                     "confusion_matrix": confusion_matrix(y, pred, labels=sorted(CLASS_NAMES)).tolist(),
                     "within_corpus": within}
    if elapsed_sec is not None:
        results[name]["elapsed_sec"] = round(float(elapsed_sec), 1)
        print(f"    elapsed {elapsed_sec:.1f}s")

    # Persist out-of-fold predictions so a new metric never costs another training run.
    preds_dir = os.path.join(OUT, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    np.save(os.path.join(preds_dir, f"{slug(name)}.npy"), pred)
    return results


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, subjects, corpus = load()
    shortcut = shortcut_baseline(y, corpus)
    print(f"datasets/all3: {len(y)} segments, {len(set(subjects))} subjects")
    print(f"classes: " + ", ".join(f"{CLASS_NAMES[k]} {int((y == k).sum())}" for k in sorted(CLASS_NAMES)))
    print(f"\nbaselines: chance {1/3:.4f} | majority {np.bincount(y).max()/len(y):.4f} "
          f"| CORPUS-ID-ONLY {shortcut:.4f}")

    print("\nextracting 81 handcrafted features...")
    F = extract_all_features(X)

    # --- 1. how recoverable is corpus identity? -----------------------------------
    probe = cross_val_score(get_svm_pipeline(), F, corpus,
                            cv=GroupKFold(n_splits=N_SPLITS), groups=subjects,
                            scoring="accuracy", n_jobs=-1).mean()
    print(f"\n1. CORPUS PROBE — predicting which device recorded each segment: {probe:.4f}")
    print("   " + ("shortcut is fully available; treat pooled accuracy with suspicion"
                   if probe > 0.95 else "corpus only partly recoverable"))

    # --- 2 & 3. 3-class performance and the within-corpus breakdown ----------------
    print("\n2/3. 3-CLASS PERFORMANCE (5-fold subject-grouped CV)")
    results = {"corpus_probe_accuracy": float(probe), "shortcut_baseline": shortcut}
    t0 = time.perf_counter()
    # Named for the estimator oof_predict actually uses (LogisticRegression, line 87).
    # This row was previously labelled "SVM on 81 features", which it never was — it
    # reproduces B4 LogReg from all3_models.py (0.593146), not B3b SVM (0.625636).
    results = report("LogReg on 81 features", y, oof_predict(F, y, subjects), corpus, shortcut,
                     results, elapsed_sec=time.perf_counter() - t0)

    # Same array all3_models.py caches; recomputing it costs ~13 min for nothing.
    cache = os.path.join(DATA, "_cache_riemann_percorpus.npy")
    if os.path.exists(cache):
        print(f"\n   Riemannian features: reusing {os.path.basename(cache)}")
        FR = np.load(cache)
    else:
        print("\n   building Riemannian features (per-corpus re-centering)...")
        FR = riemannian_features(X, corpus)
        np.save(cache, FR)
    t0 = time.perf_counter()
    pred_r = oof_predict(FR, y, subjects)
    results = report("Riemannian tangent space", y, pred_r, corpus, shortcut, results,
                     elapsed_sec=time.perf_counter() - t0)

    print("\n   confusion matrix, Riemannian (rows = true, cols = predicted)")
    cm = confusion_matrix(y, pred_r)
    print("        " + "".join(f"{CLASS_NAMES[i]:>12s}" for i in sorted(CLASS_NAMES)))
    for i, row in enumerate(cm):
        print(f"   {CLASS_NAMES[i]:10s}" + "".join(f"{v:12d}" for v in row))
    results["confusion_matrix_riemannian"] = cm.tolist()

    with open(os.path.join(OUT, "diagnostic.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {os.path.join(OUT, 'diagnostic.json')}")
    print("\nRead the within-corpus rows, not the pooled accuracy. Pooled includes the free\n"
          "corpus split; within-corpus is what the model learned about mental state.")


if __name__ == "__main__":
    main()
