"""
Plan 02 -- demographic residualization.

Changes the question from "does EEG beat demographics" to "does EEG carry anything
demographics don't".  Output is one number with an interval, not a table of accuracies.

    .venv/bin/python -u -m OSF_Park_model.normative --task 4class
    .venv/bin/python -m OSF_Park_model.normative --selfcheck
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features_corrected as fc
from .cv import dev_folds, metrics, paired_bootstrap_delta, table
from .data import ROOT, cohort
from .make_holdout import load

OUT = os.path.join(ROOT, "Results/OSF_Park")
HEALTHY = 0                     # class index of 'healthy' in both task label orders
GRID = {"logisticregression__C": [0.01, 0.1, 1, 10]}


def demo_matrix(c, with_iq):
    """age, sex, education (+IQ).  data.DEMO column order is [age, sex, IQ, education]."""
    cols = [0, 1, 3] + ([2] if with_iq else [])
    return c.DEMO[:, cols].astype(np.float64)


def residualize(X, D, y, train, alpha=1.0):
    """Normative model: fit on TRAINING-FOLD HEALTHY CONTROLS only, residualize everyone.

    Two things are load-bearing:

    * healthy-only -- fitting on all subjects absorbs disorder variance into the "normal"
      model and residualizes away the signal being looked for.
    * training-fold-only -- fitting once on all healthy controls leaks test-subject
      information into every feature of every fold, the same class of error as fitting a
      scaler on the full dataset.

    Ridge rather than OLS: ~59 healthy controls per fold against 3-4 correlated predictors
    makes OLS coefficients unstable enough to inject noise into every residual.
    """
    h = train[y[train] == HEALTHY]
    assert len(h) >= 20, f"only {len(h)} healthy controls in this training fold"
    sd, sx = StandardScaler().fit(D[h]), StandardScaler().fit(X[h])
    m = Ridge(alpha=alpha).fit(sd.transform(D[h]), sx.transform(X[h]))
    R = sx.transform(X) - m.predict(sd.transform(D))
    # normative fit quality, scored on held-out healthy controls inside the fold
    r2 = np.nan
    if len(h) > 30:
        k = max(5, len(h) // 5)
        hi, ho = h[:-k], h[-k:]
        m2 = Ridge(alpha=alpha).fit(sd.transform(D[hi]), sx.transform(X[hi]))
        p, t = m2.predict(sd.transform(D[ho])), sx.transform(X[ho])
        ss = ((t - p) ** 2).sum(0) / np.maximum(((t - t.mean(0)) ** 2).sum(0), 1e-12)
        r2 = float(np.mean(1 - ss))
    return R, r2, len(h)


def run(c, task, X, D, seeds, label):
    """Paired CV: demographics alone vs demographics + residualized EEG, identical folds."""
    n, K = len(c.y), len(c.classes)
    oA, oB = np.zeros((n, K)), np.zeros((n, K))
    seen = np.zeros(n, int)
    r2s, nh = [], []
    for tr, te in dev_folds(task, c.y, seeds):
        R, r2, h = residualize(X, D, c.y, tr)
        r2s.append(r2)
        nh.append(h)
        for out, F in ((oA, D), (oB, np.concatenate([D, R], 1))):
            m = make_pipeline(StandardScaler(), LogisticRegression(
                class_weight="balanced", max_iter=5000, C=0.1, random_state=42))
            m.fit(F[tr], c.y[tr])
            out[te] += m.predict_proba(F[te])
        seen[te] += 1
    dev = load(task)[0]
    oA[dev] /= seen[dev][:, None]
    oB[dev] /= seen[dev][:, None]
    d = paired_bootstrap_delta(c.y, oA, oB, dev, c.classes)
    return (metrics(c.y, oA, dev, c.classes, f"demographics only [{label}]"),
            metrics(c.y, oB, dev, c.classes, f"demographics + residualized EEG [{label}]"),
            d, float(np.nanmean(r2s)), int(np.mean(nh)))


def selfcheck():
    """Synthetic: a feature that is exactly 2*age must lose its age signal after residualizing."""
    rng = np.random.RandomState(0)
    c = cohort("4class")
    D = demo_matrix(c, False)
    age = D[:, 0]
    X = np.stack([2 * age + rng.randn(len(age)) * 0.01, rng.randn(len(age))], 1)
    tr = np.arange(len(age))
    R, r2, h = residualize(X, D, c.y, tr)

    # Correlation is the WRONG diagnostic here: ridge shrinkage leaves a residual that is
    # tiny but still exactly proportional to age, so |corr| stays ~1.0 while essentially all
    # the variance has gone.  Variance ratio is what residualization actually guarantees.
    kept = R[:, 0].var() / (X[:, 0].var() / X[:, 0].std() ** 2)      # R is in z-units of X
    assert kept < 0.05, f"age variance not removed: {kept:.4f} of unit variance remains"
    noise_kept = R[:, 1].var()
    assert 0.5 < noise_kept < 2.0, f"pure-noise column was distorted: {noise_kept:.4f}"
    print(f"  synthetic 2*age feature: variance remaining {kept:.5f} (of 1.0); "
          f"pure-noise column retains {noise_kept:.3f}  (R2={r2:.3f}, {h} HC)")
    # pairing sanity: a model bootstrapped against itself must give exactly zero
    o = rng.rand(len(c.y), 4)
    d = paired_bootstrap_delta(c.y, o, o, load("4class")[0], c.classes, n_boot=200)
    assert d["delta_auc"] == 0 and d["ci95"] == [0.0, 0.0], d
    print(f"  paired bootstrap of a model against itself: delta {d['delta_auc']:.1f} CI {d['ci95']}")
    print("all checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="4class", choices=["4class", "binary"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    seeds = (0,) if a.smoke else (0, 1, 2)
    c = cohort(a.task)
    dev = load(a.task)[0]
    X = fc.view(fc.blocks(c), "EEG")
    print(f"{a.task}: n_dev={len(dev)} p_eeg={X.shape[1]} folds={5*len(seeds)}\n")

    out = {"plan": "02", "task": a.task, "n_dev": len(dev), "arms": {}}
    rows = []
    for label, with_iq in (("no IQ", False), ("with IQ", True)):
        t0 = time.time()
        D = demo_matrix(c, with_iq)
        A, Bm, d, r2, nh = run(c, a.task, X, D, seeds, label)
        rows += [A, Bm]
        out["arms"][label] = {"demographics": A, "demographics_plus_residual_eeg": Bm,
                              "delta_auc": d, "normative_r2_heldout_healthy": r2,
                              "healthy_controls_per_fold": nh,
                              "elapsed_sec": round(time.time() - t0, 1)}
        print(f"  [{label}]  normative R2 on held-out HC {r2:+.4f} ({nh} HC/fold)")
        print(f"    demographics            bal {A['balanced_accuracy']:.4f}  AUC {A['macro_auc']:.4f}")
        print(f"    + residualized EEG      bal {Bm['balanced_accuracy']:.4f}  AUC {Bm['macro_auc']:.4f}")
        print(f"    dAUC {d['delta_auc']:+.4f}  95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
              f"  {'EXCLUDES zero' if d['excludes_zero'] else 'includes zero -> null'}\n")

    p = os.path.join(OUT, f"plan02_{a.task}.json")
    json.dump(out, open(p, "w"), indent=2)
    md = [f"# Plan 02 — demographic residualization ({a.task})", "",
          f"Dev only, n={len(dev)}. Normative model fit on training-fold healthy controls only.", "",
          "## The result", "",
          "| arm | ΔAUC | 95% CI | verdict | normative R² |", "|---|---:|---|---|---:|"]
    for k, v in out["arms"].items():
        d = v["delta_auc"]
        md.append(f"| {k} | {d['delta_auc']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                  f"{'increment' if d['excludes_zero'] else '**null**'} | "
                  f"{v['normative_r2_heldout_healthy']:+.4f} |")
    md += ["", "Primary analysis is **no IQ**: IQ was measured after diagnosis (healthy 116.2 vs "
           "89.6–107.8 for patient groups), so it is partly a consequence of illness and "
           "medication — a mediator, not an independent predictor.", "", table(rows), "",
           "Residualizing on age/sex/education does not remove medication status, symptom "
           "severity or time of day. None are in this file. The recording-year probe came back "
           "clean (0.2179 vs 0.2443 majority), which handles drift; the rest stand as limitations."]
    open(os.path.join(ROOT, "OSF_Park_model", f"RESULTS_plan02_{a.task}.md"), "w").write("\n".join(md) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
