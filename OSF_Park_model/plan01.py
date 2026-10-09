"""
Plan 01 -- corrected features, simple model.  Isolates: does the null survive correct
preprocessing?

    .venv/bin/python -u -m OSF_Park_model.plan01 --task 4class --smoke
    .venv/bin/python -u -m OSF_Park_model.plan01 --task 4class

Writes Results/OSF_Park/plan01_<task>.json after every row, and plan01_<task>.md.
Dev-only: the frozen test set is never touched here (see plans/00_protocol.md).
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import features_corrected as fc
from .cv import metrics, nested, table
from .data import ROOT, cohort
from .make_holdout import load

OUT = os.path.join(ROOT, "Results/OSF_Park")

EN_GRID = {"logisticregression__C": [0.01, 0.1, 1, 10],
           "logisticregression__l1_ratio": [0.1, 0.5, 0.9]}
L2_GRID = {"logisticregression__C": [0.01, 0.1, 1, 10]}
SVM_GRID = {"svc__C": [0.1, 1, 10]}


def est(kind):
    if kind == "elasticnet":
        return make_pipeline(StandardScaler(), LogisticRegression(
            penalty="elasticnet", solver="saga", class_weight="balanced",
            max_iter=5000, random_state=42)), EN_GRID
    if kind == "l2":
        return make_pipeline(StandardScaler(), LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight="balanced",
            max_iter=5000, random_state=42)), L2_GRID
    return make_pipeline(StandardScaler(), SVC(          # control: the current pipeline's model
        kernel="rbf", probability=True, class_weight="balanced", random_state=42)), SVM_GRID


# (view, model kind).  Rows 1-5 are the ablation grid; RAW_BASELINE+rbf is the current
# pipeline reproduced on dev, which is what makes the preprocessing change attributable.
ROWS = [("DEMO", "l2"), ("SPECTRAL", "elasticnet"), ("GRAPH", "elasticnet"),
        ("EEG", "elasticnet"), ("EEG+DEMO", "elasticnet"),
        ("EEG", "l2"), ("EEG", "rbf"),
        ("RAW_BASELINE", "rbf")]
# RAW_BASELINE x elasticnet is deliberately absent: saga on 1140 features x a 12-config
# grid x 5 inner x 15 outer folds dominates the whole run's cost, and the plan's row 6
# specifies the RBF SVM anyway -- that is what reproduces the current pipeline and makes
# the preprocessing change attributable.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="4class", choices=["4class", "binary"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seeds = (0,) if a.smoke else (0, 1, 2)

    c = cohort(a.task)
    dev, test = load(a.task)
    B = fc.blocks(c)
    print(f"{a.task}: n={len(c)} dev={len(dev)} test={len(test)} (frozen, untouched) "
          f"folds={5 * len(seeds)}")
    maj = np.bincount(c.y[dev]).max() / len(dev)
    print(f"  chance {1/len(c.classes):.4f} | majority {maj:.4f}\n")

    rows, bar, path = [], None, os.path.join(OUT, f"plan01_{a.task}.json")
    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    for vname, kind in ROWS:
        t0 = time.time()
        X = fc.view(B, vname)
        e, g = est(kind)
        oof, picks = nested(X, c.y, e, g, a.task, seeds)
        label = f"{vname} [{kind}]"
        r = metrics(c.y, oof, dev, c.classes, label, picks, time.time() - t0)
        r["p"] = int(X.shape[1])
        if bar is None:
            bar = r["balanced_accuracy"]                  # DEMO row is first
        r["vs_demo"] = r["balanced_accuracy"] - bar
        rows.append(r)
        np.save(os.path.join(OUT, "preds", f"plan01_{a.task}_{label.replace(' ', '_')}.npy"), oof)
        rec = " ".join(f"{k[:4]} {v:.2f}" for k, v in r["recall_per_class"].items())
        print(f"  {label:28s} p={r['p']:5d} bal {r['balanced_accuracy']:.4f} "
              f"F1 {r['macro_f1']:.4f} AUC {r['macro_auc']:.4f} vs demo {r['vs_demo']:+.4f} "
              f"({r['elapsed_sec']:.0f}s)\n{'':32s}recall: {rec}")
        json.dump({"plan": "01", "task": a.task, "n_dev": len(dev), "seeds": list(seeds),
                   "majority": float(maj), "chance": 1 / len(c.classes),
                   "demo_bar": bar, "results": rows}, open(path, "w"), indent=2)

    md = [f"# Plan 01 — corrected features ({a.task})", "",
          f"Dev only, n={len(dev)} of {len(c)}; test set (n={len(test)}) frozen and untouched.",
          f"{5*len(seeds)} folds, nested hyperparameter search on inner folds.", "",
          f"chance {1/len(c.classes):.4f} · majority {maj:.4f} · **demographics {bar:.4f}**", "",
          table(rows, bar), "",
          "Recall per class (the tell for whether a model predicts every class):", "",
          "| model | " + " | ".join(c.classes) + " |",
          "|---" * (len(c.classes) + 1) + "|"]
    for r in sorted(rows, key=lambda r: -r["balanced_accuracy"]):
        md.append(f"| {r['model']} | " + " | ".join(f"{r['recall_per_class'][k]:.3f}" for k in c.classes) + " |")
    open(os.path.join(ROOT, "OSF_Park_model", f"RESULTS_plan01_{a.task}.md"), "w").write("\n".join(md) + "\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
