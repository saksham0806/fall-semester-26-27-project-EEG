"""
Score the FROZEN test set.  Append-only, one model per plan, run deliberately by a person.

    .venv/bin/python -u -m OSF_Park_model.final_test --plan 01 --view EEG+DEMO --model elasticnet

Appends to Results/OSF_Park/FINAL.md -- never overwrites.  An append-only log makes a second
peek visible rather than deniable.  Scoring the test set twice and reporting the better run
reinstates exactly the selection bias the holdout removes.

n_test is 106 (4class) / 59 (binary), so the accuracy standard error is about +/-0.048 and
+/-0.065.  Report to two decimals and say the interval out loud; do not quote four.
"""
import argparse
import datetime
import hashlib
import json
import os

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import features_corrected as fc
from .cv import metrics
from .data import ROOT, cohort
from .make_holdout import load
from .plan01 import est

LOG = os.path.join(ROOT, "Results/OSF_Park/FINAL.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--task", default="4class", choices=["4class", "binary"])
    ap.add_argument("--view", default="EEG+DEMO", choices=list(fc.VIEWS))
    ap.add_argument("--model", default="elasticnet", choices=["elasticnet", "l2", "rbf"])
    ap.add_argument("--yes", action="store_true", help="required: confirms this is the one shot")
    a = ap.parse_args()
    if not a.yes:
        raise SystemExit("refusing without --yes. The test set is scored ONCE per plan; "
                         "re-running and reporting the better result reinstates selection bias.")

    c = cohort(a.task)
    dev, test = load(a.task)
    X = fc.view(fc.blocks(c), a.view)
    e, g = est(a.model)
    gs = GridSearchCV(clone(e), g, scoring="balanced_accuracy", n_jobs=-1,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0))
    gs.fit(X[dev], c.y[dev])                       # fit on ALL dev, select on inner folds
    proba = np.zeros((len(c.y), len(c.classes)))
    proba[test] = gs.best_estimator_.predict_proba(X[test])
    r = metrics(c.y, proba, test, c.classes, f"{a.view} [{a.model}]")
    se = float(np.sqrt(r["accuracy"] * (1 - r["accuracy"]) / len(test)))

    cfg = hashlib.md5(json.dumps([a.plan, a.task, a.view, a.model, gs.best_params_],
                                 default=str, sort_keys=True).encode()).hexdigest()[:8]
    line = (f"| {datetime.date.today()} | {a.plan} | {a.task} | {a.view} [{a.model}] | "
            f"{gs.best_score_:.4f} | **{r['balanced_accuracy']:.3f}** | {r['accuracy']:.3f} "
            f"±{se:.3f} | {r['macro_f1']:.3f} | {r['macro_auc']:.3f} | {len(test)} | `{cfg}` |")
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if new:
            f.write("# Held-out test results — APPEND ONLY\n\n"
                    "One row per plan. A second row for the same plan means the test set was "
                    "scored twice; the first row is the honest one.\n\n"
                    "| date | plan | task | model | dev bal-acc | **test bal-acc** | test acc | "
                    "macro-F1 | macro-AUC | n | config |\n"
                    "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        f.write(line + "\n")
    print(f"dev(inner) {gs.best_score_:.4f} -> TEST balanced acc {r['balanced_accuracy']:.3f} "
          f"| acc {r['accuracy']:.3f} ±{se:.3f} | n={len(test)}")
    print(f"params {gs.best_params_}\nappended to {LOG}")


if __name__ == "__main__":
    main()
