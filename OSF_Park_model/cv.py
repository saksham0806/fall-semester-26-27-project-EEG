"""
Shared nested-CV harness for plans 01-03.  Dev-only by construction.

Every plan uses this so results are directly comparable, and so the holdout guarantee is
enforced in one place by code rather than by discipline in three.
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from .make_holdout import load


def dev_folds(task, y, seeds=(0, 1, 2), n_splits=5):
    """Folds over DEV indices only.  Yields (train, test) as indices into the full cohort."""
    dev, test = load(task)
    tset = set(test.tolist())
    for s in seeds:
        for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=s).split(dev, y[dev]):
            a, b = dev[tr], dev[te]
            assert not (set(a.tolist()) | set(b.tolist())) & tset, \
                "test-set indices leaked into a CV fold"
            yield a, b


def nested(X, y, est, grid, task, seeds=(0, 1, 2), scoring="balanced_accuracy", n_jobs=-1):
    """Out-of-fold probabilities with the hyperparameter grid searched on INNER folds only.

    Selecting on the outer fold is test-set selection.  Averaged over seeds because every
    dev row is predicted once per seed.
    """
    n, K = len(y), len(np.unique(y))
    oof, seen, picks = np.zeros((n, K)), np.zeros(n, int), []
    for tr, te in dev_folds(task, y, seeds):
        if grid:
            gs = GridSearchCV(clone(est), grid, scoring=scoring, n_jobs=n_jobs,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0))
            gs.fit(X[tr], y[tr])
            m, best = gs.best_estimator_, gs.best_params_
        else:
            m, best = clone(est).fit(X[tr], y[tr]), {}
        oof[te] += m.predict_proba(X[te])
        seen[te] += 1
        picks.append(best)
    dev_idx = load(task)[0]
    assert (seen[dev_idx] > 0).all(), "a dev row was never predicted"
    oof[dev_idx] /= seen[dev_idx][:, None]
    return oof, picks


def metrics(y, oof, idx, classes, label, picks=None, elapsed=None):
    yt, p = y[idx], oof[idx]
    pred = p.argmax(1)
    K = len(classes)
    try:
        auc = (roc_auc_score(yt, p[:, 1]) if K == 2 else
               roc_auc_score(yt, p / p.sum(1, keepdims=True), multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")
    r = {"model": label, "n": int(len(idx)),
         "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
         "macro_f1": float(f1_score(yt, pred, average="macro")),
         "accuracy": float((pred == yt).mean()),
         "macro_auc": float(auc),
         "recall_per_class": {c: float((pred[yt == i] == i).mean()) if (yt == i).any() else float("nan")
                              for i, c in enumerate(classes)},
         "confusion": confusion_matrix(yt, pred, labels=range(K)).tolist()}
    if picks:
        r["selected_params"] = _modal(picks)
    if elapsed is not None:
        r["elapsed_sec"] = round(elapsed, 1)
    return r


def _modal(picks):
    if not picks or not picks[0]:
        return {}
    out = {}
    for k in picks[0]:
        vals = [str(p[k]) for p in picks]
        out[k] = max(set(vals), key=vals.count) + f" ({vals.count(max(set(vals), key=vals.count))}/{len(vals)})"
    return out


def paired_bootstrap_delta(y, oofA, oofB, idx, classes, n_boot=2000, seed=0):
    """95% CI on AUC(B) - AUC(A), resampling SUBJECTS with the SAME indices for both models.

    Pairing is essential: an unpaired bootstrap of two highly correlated estimates gives an
    interval far too wide and reports 'no difference' for everything.
    """
    rng = np.random.RandomState(seed)
    K = len(classes)

    def auc(yy, pp):
        if len(np.unique(yy)) < K:
            return np.nan
        return (roc_auc_score(yy, pp[:, 1]) if K == 2 else
                roc_auc_score(yy, pp / pp.sum(1, keepdims=True), multi_class="ovr", average="macro"))

    yt, A, B = y[idx], oofA[idx], oofB[idx]
    obs = auc(yt, B) - auc(yt, A)
    d = []
    for _ in range(n_boot):
        b = rng.randint(0, len(yt), len(yt))
        v = auc(yt[b], B[b]) - auc(yt[b], A[b])
        if np.isfinite(v):
            d.append(v)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_auc": float(obs), "ci95": [float(lo), float(hi)],
            "n_boot": len(d), "excludes_zero": bool(lo > 0 or hi < 0)}


def table(rows, bar=None):
    L = ["| model | balanced acc | macro-F1 | accuracy | macro-AUC |" + (" vs demo |" if bar else ""),
         "|---|---:|---:|---:|---:|" + ("---:|" if bar else "")]
    for r in sorted(rows, key=lambda r: -r["balanced_accuracy"]):
        row = (f"| {r['model']} | **{r['balanced_accuracy']:.4f}** | {r['macro_f1']:.4f} | "
               f"{r['accuracy']:.4f} | {r['macro_auc']:.4f} |")
        if bar:
            row += f" {r['balanced_accuracy'] - bar:+.4f} |"
        L.append(row)
    return "\n".join(L)
