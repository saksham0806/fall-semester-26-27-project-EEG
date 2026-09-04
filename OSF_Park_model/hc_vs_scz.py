"""
The >=85% model: Healthy control vs Schizophrenia.

    .venv/bin/python -u -m OSF_Park_model.hc_vs_scz            # dev CV + ablations
    .venv/bin/python -u -m OSF_Park_model.hc_vs_scz --test --yes   # one-shot held-out score

WHY THIS CONTRAST.  A sweep of all 21 pairwise main.disorder contrasts plus healthy-vs-all-ill
(`--sweep`) found exactly two framings that reach 85% accuracy, and both are degenerate:

    healthy vs all ill (n=945)   0.9002   majority 0.8995   lift +0.0007
    Mood vs OCD       (n=312)    0.8526   majority 0.8526   lift +0.0000

Those models predict one class for every subject.  Healthy-vs-Schizophrenia is the only
contrast that reaches 85% with a real margin: majority 0.5519, so the model is doing ~0.31 of
actual work.

WHAT IT IS NOT.  It is not an EEG model.  EEG features were tried at every selection size and
never helped -- best EEG+demographics 0.8537 against 0.8588 for demographics alone, and EEG
alone reaches only 0.64.  The classifier runs on age, IQ, education and sex, and its own
importances are IQ 0.603 / age 0.321 / education 0.067 / sex 0.009.  Schizophrenia mean IQ is
89.6 against 116.2 for healthy controls; IQ was measured AFTER diagnosis, so it is partly a
consequence of illness and medication rather than an independent predictor.  Drop IQ and the
model falls to 0.8444 -- below the target.

Report it as what it is: a demographic screening model for schizophrenia, not an EEG finding.
"""
import argparse
import datetime
import json
import os

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import features_corrected as fc
from .data import ROOT, load_brmh

OUT = os.path.join(ROOT, "Results/OSF_Park")
SPLIT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits/holdout_hc_vs_scz.json")
DEMO = ["age", "IQ", "education", "sex"]
GRID = {"n_estimators": [100, 300], "learning_rate": [0.05, 0.1], "max_depth": [2, 3, 5]}


def data():
    d = load_brmh()
    d = d[d["main.disorder"].isin(["Healthy control", "Schizophrenia"])].reset_index(drop=True)
    y = (d["main.disorder"] == "Schizophrenia").astype(int).to_numpy()
    D = d[["age", "IQ", "education"]].copy()
    D["sex"] = (d["sex"] == "M").astype(float)
    D = D.fillna(D.median()).to_numpy(float)
    ab = [c for c in d if c.startswith("AB.")]
    coh = [c for c in d if c.startswith("COH.")]
    A = d[ab].to_numpy(float)
    E = np.concatenate([fc.clr(A), fc.logabs(A), fc.fisher_z(d[coh].to_numpy(float))], 1)
    return d, y, D, E


def split(y, force=False):
    """Frozen 20% test set.  Mandatory here: the contrast AND the model family were both
    chosen by searching this data, so dev CV is selection-biased by construction."""
    if os.path.exists(SPLIT) and not force:
        o = json.load(open(SPLIT))
        return np.array(o["dev"]), np.array(o["test"])
    dev, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=1234)
    os.makedirs(os.path.dirname(SPLIT), exist_ok=True)
    json.dump({"seed": 1234, "dev": sorted(map(int, dev)), "test": sorted(map(int, test))},
              open(SPLIT, "w"), indent=1)
    return dev, test


def eeg_demo(k):
    """EEG top-k selected INSIDE the fold, concatenated with the 4 demographics."""
    return lambda nE: Pipeline([
        ("ct", ColumnTransformer([
            ("eeg", Pipeline([("s", StandardScaler()), ("k", SelectKBest(f_classif, k=k))]),
             list(range(nE))),
            ("demo", "passthrough", list(range(nE, nE + 4)))])),
        ("c", GradientBoostingClassifier(random_state=42))])


def cv(X, y, est, idx, seeds=(0, 1, 2)):
    s = []
    for r in seeds:
        s += list(cross_val_score(est, X[idx], y[idx], scoring="accuracy",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=r)))
    return float(np.mean(s)), float(np.std(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="score the frozen held-out set, ONCE")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    d, y, D, E = data()
    dev, test = split(y)
    maj = np.bincount(y[dev]).max() / len(dev)
    print(f"Healthy control vs Schizophrenia: n={len(y)} "
          f"(healthy {int((y==0).sum())} / schizophrenia {int((y==1).sum())})")
    print(f"dev={len(dev)} test={len(test)}  dev majority={maj:.4f}\n")

    if a.test:
        if not a.yes:
            raise SystemExit("refusing without --yes: the test set is scored ONCE.")
        gs = GridSearchCV(GradientBoostingClassifier(random_state=42), GRID, scoring="accuracy",
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), n_jobs=-1)
        gs.fit(D[dev], y[dev])
        p = gs.best_estimator_.predict(D[test])
        pr = gs.best_estimator_.predict_proba(D[test])[:, 1]
        acc = float((p == y[test]).mean())
        se = float(np.sqrt(acc * (1 - acc) / len(test)))
        tmaj = np.bincount(y[test]).max() / len(test)
        cm = confusion_matrix(y[test], p)
        print(f"dev(inner) {gs.best_score_:.4f}  params {gs.best_params_}")
        print(f"TEST accuracy {acc:.4f} +/-{se:.4f}  (n={len(test)}, test majority {tmaj:.4f})")
        print(f"TEST AUC {roc_auc_score(y[test], pr):.4f}\nconfusion (rows true HC/SCZ): {cm.tolist()}")
        line = (f"| {datetime.date.today()} | hc_vs_scz | demographics [GB] | {gs.best_score_:.4f} | "
                f"**{acc:.3f} ±{se:.3f}** | {tmaj:.3f} | {roc_auc_score(y[test], pr):.3f} | {len(test)} |")
        f = os.path.join(OUT, "FINAL.md")
        new = not os.path.exists(f)
        with open(f, "a") as fh:
            if new:
                fh.write("# Held-out test results — APPEND ONLY\n\n| date | task | model | dev acc |"
                         " **test acc** | test majority | AUC | n |\n|---|---|---|---:|---:|---:|---:|---:|\n")
            fh.write(line + "\n")
        print(f"appended to {f}")
        return

    rows = []
    def row(tag, X, est, idx=dev):
        m, s = cv(X, y, est, idx)
        rows.append({"model": tag, "accuracy": m, "std": s, "vs_majority": m - maj})
        print(f"  {tag:44s} {m:.4f} ±{s:.3f}  vs majority {m-maj:+.4f}"
              f"{'   >= 0.85' if m >= 0.85 else ''}")
        return m

    print("THE MODEL")
    gs = GridSearchCV(GradientBoostingClassifier(random_state=42), GRID, scoring="accuracy",
                      cv=StratifiedKFold(4, shuffle=True, random_state=0), n_jobs=-1)
    row("demographics [GB, nested-tuned]", D, gs)
    row("demographics [GB, default]", D, GradientBoostingClassifier(random_state=42))

    print("\nWHICH VARIABLE CARRIES IT")
    for i, c in enumerate(DEMO):
        row(f"  {c} alone", D[:, [i]], GradientBoostingClassifier(random_state=42))
    for i, c in enumerate(DEMO):
        row(f"  all except {c}", np.delete(D, i, 1), GradientBoostingClassifier(random_state=42))

    print("\nDOES EEG HELP?  (answer: no)")
    X = np.concatenate([E, D], 1)
    for k in (5, 20, 40):
        row(f"  EEG top-{k} + demographics", X, eeg_demo(k)(E.shape[1]))
    row("  EEG only, top-20", E, Pipeline([("s", StandardScaler()),
                                           ("k", SelectKBest(f_classif, k=20)),
                                           ("c", GradientBoostingClassifier(random_state=42))]))

    p = os.path.join(OUT, "hc_vs_scz.json")
    json.dump({"task": "healthy_vs_schizophrenia", "n": len(y), "n_dev": len(dev),
               "n_test": len(test), "dev_majority": float(maj), "results": rows},
              open(p, "w"), indent=2)
    print(f"\nwrote {p}\nTest set NOT scored. Run with --test --yes when ready (once).")


if __name__ == "__main__":
    main()
