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

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
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


def report(name, y, pred, corpus, shortcut, results):
    acc = float((pred == y).mean())
    bal = float(balanced_accuracy_score(y, pred))
    margin = acc - shortcut
    print(f"\n  {name}")
    print(f"    3-class accuracy      {acc:.4f}   balanced {bal:.4f}")
    print(f"    vs corpus-ID shortcut {shortcut:.4f}   margin {margin:+.4f}"
          f"   {'<-- REAL SIGNAL' if margin > 0.02 else '<-- NO BETTER THAN DEVICE DETECTION'}")

    within = {}
    for c in np.unique(corpus):
        m = corpus == c
        a = float((pred[m] == y[m]).mean())
        base = float(np.bincount(y[m]).max() / m.sum())
        within[CORPUS_NAMES[c]] = {"accuracy": a, "majority_within": base}
        print(f"    within {CORPUS_NAMES[c]:8s} {a:.4f}  (majority {base:.4f}, margin {a - base:+.4f})")

    results[name] = {"accuracy": acc, "balanced_accuracy": bal,
                     "shortcut_baseline": shortcut, "margin": margin, "within_corpus": within}
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
    results = report("SVM on 81 features", y, oof_predict(F, y, subjects), corpus, shortcut, results)

    print("\n   building Riemannian features (per-corpus re-centering)...")
    FR = riemannian_features(X, corpus)
    pred_r = oof_predict(FR, y, subjects)
    results = report("Riemannian tangent space", y, pred_r, corpus, shortcut, results)

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
