"""
all3 Tier 1 baselines + Tier 3 feature-level fusion (F1 early, F2 late).

Implements the feature-based half of plans/all3_plan.md. Everything here is CPU and fast; the
deep roster lives in scripts/all3_deep.py.

Every model is reported against the 0.6267 corpus-ID shortcut, never against chance, and
with the within-corpus breakdown. A model clearing the shortcut by <0.02 has demonstrated
no EEG understanding.

Riemannian features are cached (5 bands x 15,728 segments is the slow part) so re-runs and
the deep script can reuse them.

Run:  .venv/bin/python -u scripts/all3_models.py
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from proposed_model.features import extract_all_features
# Reuse the diagnostic's protocol verbatim so numbers stay comparable.
from all3_diagnostic import (CLASS_NAMES, load, report, riemannian_features,
                             shortcut_baseline)

OUT = os.path.join(PROJECT_ROOT, "Results", "all3")
CACHE_R = os.path.join(PROJECT_ROOT, "datasets", "all3", "_cache_riemann_percorpus.npy")
CACHE_G = os.path.join(PROJECT_ROOT, "datasets", "all3", "_cache_riemann_global.npy")
N_SPLITS = 5

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False


def oof_predict(F, y, subjects, clf_factory):
    """Out-of-fold predictions under subject-grouped CV. Scaler fitted per fold."""
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        sc = StandardScaler().fit(F[tr])
        clf = clf_factory()
        clf.fit(sc.transform(F[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(F[te]))
    return pred


def oof_proba(F, y, subjects, clf_factory):
    """Out-of-fold class probabilities — needed for late fusion."""
    proba = np.zeros((len(y), len(CLASS_NAMES)))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        sc = StandardScaler().fit(F[tr])
        clf = clf_factory()
        clf.fit(sc.transform(F[tr]), y[tr])
        proba[te] = clf.predict_proba(sc.transform(F[te]))
    return proba


def cached_riemann(path, X, groups):
    if os.path.exists(path):
        return np.load(path)
    F = riemannian_features(X, groups)
    np.save(path, F)
    return F


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, subjects, corpus = load()
    shortcut = shortcut_baseline(y, corpus)
    print(f"all3: {len(y)} segments, {len(set(subjects))} subjects")
    print(f"shortcut to beat (corpus-ID only): {shortcut:.4f}\n")

    print("features: 81 handcrafted...")
    F81 = extract_all_features(X)
    print("features: Riemannian, per-corpus re-centering...")
    FRc = cached_riemann(CACHE_R, X, corpus)
    print("features: Riemannian, global re-centering...")
    FRg = cached_riemann(CACHE_G, X, np.zeros(len(y), int))
    print(f"  dims — 81feat {F81.shape[1]}, riemann {FRc.shape[1]}\n")

    results = {"shortcut_baseline": shortcut}
    print("=" * 78)
    print("TIER 1 — baselines")
    print("=" * 78)

    tier1 = [
        ("B4 LogisticRegression / 81feat", F81,
         lambda: LogisticRegression(max_iter=3000, class_weight="balanced")),
        ("B5 RandomForest / 81feat", F81,
         lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        n_jobs=-1, random_state=42)),
        ("B6 XGBoost / 81feat", F81,
         (lambda: XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                objective="multi:softmax", num_class=3, n_jobs=-1,
                                random_state=42, verbosity=0))
         if HAS_XGB else
         (lambda: GradientBoostingClassifier(n_estimators=100, random_state=42))),
        ("B3b SVM / 81feat (rerun)", F81,
         lambda: SVC(C=1.0, kernel="rbf", class_weight="balanced")),
    ]
    for name, F, factory in tier1:
        results = report(name, y, oof_predict(F, y, subjects, factory), corpus, shortcut, results)

    print("\n" + "=" * 78)
    print("TIER 3 — F1 early fusion (concatenation)")
    print("=" * 78)
    # The plan's cheapest untried idea: does covariance structure add anything the
    # spectral features lack? Both re-centering variants, since they answer different
    # questions (global keeps the device signature, per-corpus removes it).
    for tag, FR in [("global re-centering", FRg), ("per-corpus re-centering", FRc)]:
        F = np.hstack([FR, F81])
        results = report(f"B9/F1 Riemann+81feat ({tag})", y,
                         oof_predict(F, y, subjects,
                                     lambda: LogisticRegression(max_iter=3000,
                                                                class_weight="balanced")),
                         corpus, shortcut, results)

    print("\n" + "=" * 78)
    print("TIER 3 — F2 late fusion (soft-vote over per-view classifiers)")
    print("=" * 78)
    p81 = oof_proba(F81, y, subjects,
                    lambda: LogisticRegression(max_iter=3000, class_weight="balanced"))
    prg = oof_proba(FRg, y, subjects,
                    lambda: LogisticRegression(max_iter=3000, class_weight="balanced"))
    for w in (0.5, 0.3, 0.7):
        pred = (w * p81 + (1 - w) * prg).argmax(axis=1)
        results = report(f"F2 late fusion (w_81feat={w:.1f})", y, pred, corpus, shortcut, results)

    with open(os.path.join(OUT, "models_tier1_fusion.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    ranked = sorted(((v["accuracy"], k) for k, v in results.items() if isinstance(v, dict)),
                    reverse=True)
    print(f"{'RANKED':44s} {'acc':>8s} {'vs 0.6267':>11s}")
    for acc, k in ranked:
        print(f"  {k:42s} {acc:8.4f} {acc - shortcut:+11.4f}")
    beat = [k for a, k in ranked if a > shortcut + 0.02]
    print(f"\nmodels clearing the shortcut by >0.02: {beat if beat else 'NONE'}")
    print(f"written: {os.path.join(OUT, 'models_tier1_fusion.json')}")


if __name__ == "__main__":
    main()
