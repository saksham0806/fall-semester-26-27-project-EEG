"""
Phase 2 M1 — filter-bank Riemannian tangent space with per-subject re-centering.

Phase 1: SVM on 81 handcrafted features beat all six deep models on both tasks
(stress 0.7231 within-subject AUC on SAM40+CogLoad; SAM40-only 0.6663). Those
features carry no spatial covariance structure. This adds exactly that.

Per band: covariance -> whiten by that subject's Riemannian mean -> tangent space.
Re-centering is the principled form of what the DANN attempts adversarially: it
removes each subject's mean covariance analytically instead of via a GRL.

Run:  .venv/bin/python -u scripts/phase2_riemannian.py
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from proposed_model.features import BANDS, butter_bandpass_filter

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase2")
SAM40, COGLOAD = 1, 2
FS, N_SPLITS = 128, 5
# Phase 1 numbers this must beat (SVM, within-subject AUC).
BASELINE = {"SAM40 only": 0.6663, "SAM40 + CogLoad": 0.7231}


def recenter(covs, groups):
    """
    Whiten covariances by their group's Riemannian mean -> that mean becomes I.

    Choice of group is task-dependent and load-bearing:
      stress     -> group by SUBJECT. The contrast is within-subject (every SAM40
                    subject has both classes), so removing each subject's offset
                    removes nuisance, not signal.
      depression -> group GLOBALLY. Every segment of a subject shares one label, so
                    the signal *is* between-subject. Per-subject centering whitens
                    each subject's mean to identity and deletes the label entirely
                    (measured: AUC exactly 0.5000, Brier exactly 0.2500).
    """
    out = np.empty_like(covs)
    for g in np.unique(groups):
        m = groups == g
        W = invsqrtm(mean_riemann(covs[m]))
        out[m] = W @ covs[m] @ W
    return out


def bandwise_tangent(X, groups):
    """
    (N, 6, 512) -> (N, 5 bands * 21 tangent dims).

    TangentSpace is fitted on all segments rather than per fold. Verified to make no
    difference: fitting on train-only gives 0.7560 +/- 0.0503, identical to 4 dp. After
    re-centering, every group's mean covariance is the identity, so the tangent-space
    reference point is invariant to which subset it is fitted on. Both steps are
    unsupervised — no test label is ever touched.
    """
    feats = []
    for low, high in BANDS.values():
        Xb = butter_bandpass_filter(X, low, high, FS)
        covs = Covariances(estimator="oas").fit_transform(Xb)
        feats.append(TangentSpace(metric="riemann").fit_transform(recenter(covs, groups)))
    return np.hstack(feats)


def within_subject_auc(y, prob, subjects):
    s = [roc_auc_score(y[subjects == k], prob[subjects == k]) for k in np.unique(subjects)
         if len(np.unique(y[subjects == k])) > 1]
    return float(np.mean(s)) if s else float("nan")


def load(task="stress"):
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, f"y_{task}.npy"))
    subj = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    ds = np.load(os.path.join(DATA_DIR, "datasets.npy"))
    v = ~np.isnan(X).any(axis=(1, 2))
    X, y, subj, ds = X[v], y[v], subj[v], ds[v]
    m = y != -1
    return X[m], y[m].astype(int), subj[m], ds[m]


def subject_auc(y, prob, subjects):
    """Depression: one label per subject, so aggregate posteriors and rank subjects."""
    subs = np.unique(subjects)
    t = np.array([y[subjects == s][0] for s in subs])
    p = np.array([prob[subjects == s].mean() for s in subs])
    return float(roc_auc_score(t, p))


def run_depression():
    """M1 on depression + M6 calibration. Beat Phase 1 SVM subject-level AUC 0.7716."""
    X, y, subj, _ = load("depression")
    F = bandwise_tangent(X, np.zeros(len(y), int))  # global, not per-subject
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subj):
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(F[te]))[:, 1]

    seg, sub = float(roc_auc_score(y, oof)), subject_auc(y, oof, subj)
    # M6: temperature scaling. Monotone in the score, so AUC is unchanged by
    # construction - it calibrates probabilities, it does not reorder them.
    brier_raw = float(np.mean((oof - y) ** 2))
    T = min(np.linspace(0.5, 5.0, 46),
            key=lambda t: np.mean((1 / (1 + np.exp(-(np.log(oof / (1 - oof + 1e-9) + 1e-9)) / t)) - y) ** 2))
    cal = 1 / (1 + np.exp(-(np.log(oof / (1 - oof + 1e-9) + 1e-9)) / T))
    print(f"  depression  segment AUC={seg:.4f}  subject AUC={sub:.4f}  "
          f"(Phase 1 SVM 0.7716, delta {sub - 0.7716:+.4f})")
    print(f"  M6 calibration  T={T:.2f}  Brier {brier_raw:.4f} -> {np.mean((cal - y) ** 2):.4f}")
    return {"auc_segment": seg, "auc_subject": sub,
            "brier_raw": brier_raw, "brier_calibrated": float(np.mean((cal - y) ** 2)),
            "temperature": float(T)}


def selfcheck():
    """Re-centering must map each subject's Riemannian mean to the identity."""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(40, 6, 20))
    covs = Covariances(estimator="oas").fit_transform(A)
    subs = np.repeat(["a", "b"], 20)
    rc = recenter(covs, subs)
    for s in ("a", "b"):
        assert np.allclose(mean_riemann(rc[subs == s]), np.eye(6), atol=1e-6), "recenter broken"
    print("selfcheck: re-centering maps subject means to identity  OK")


def main():
    selfcheck()
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y, subj, ds = load()

    print("building filter-bank tangent-space features (5 bands x 21 dims)...")
    F = bandwise_tangent(X, subj)
    print(f"  {F.shape[1]} features for {F.shape[0]} segments\n")

    sam = np.where(ds == SAM40)[0]
    folds = list(GroupKFold(n_splits=N_SPLITS).split(sam, y[sam], groups=subj[sam]))
    results = {}

    for label, extra in [("SAM40 only", []), ("SAM40 + CogLoad", [COGLOAD])]:
        scores = []
        for tr_rel, te_rel in folds:
            tr, te = sam[tr_rel], sam[te_rel]
            add = np.where(np.isin(ds, extra))[0] if extra else np.array([], int)
            tr = np.concatenate([tr, add]) if len(add) else tr

            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            Xtr = StandardScaler().fit(F[tr])
            clf.fit(Xtr.transform(F[tr]), y[tr])
            prob = clf.predict_proba(Xtr.transform(F[te]))[:, 1]
            scores.append(within_subject_auc(y[te], prob, subj[te]))

        mean, std = float(np.mean(scores)), float(np.std(scores))
        results[label] = {"auc_within": [mean, std]}
        base = BASELINE[label]
        print(f"  {label:18s} within-subj AUC = {mean:.4f} +/- {std:.4f}   "
              f"(Phase 1 SVM {base:.4f}, delta {mean - base:+.4f})")

    print("\ndepression (M1 + M6):")
    results["depression"] = run_depression()

    with open(os.path.join(OUT_DIR, "riemannian.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'riemannian.json')}")


if __name__ == "__main__":
    main()
