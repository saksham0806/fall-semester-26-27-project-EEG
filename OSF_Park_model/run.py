"""
CV harness for the BRMH cohort.

    .venv/bin/python -u -m OSF_Park_model.run --task 4class --smoke   # 1 seed
    .venv/bin/python -u -m OSF_Park_model.run --task 4class
    .venv/bin/python -u -m OSF_Park_model.run --task binary

Writes Results/OSF_Park/<task>.json (after every model, so a killed run keeps finished
rows) plus out-of-fold predictions, and regenerates OSF_Park_model/RESULTS.md.

The three baselines are computed BEFORE any model and printed at the top of every table.
On this cohort a demographics-only model (age/sex/IQ/education) has already been measured
at 0.4328 against a best-EEG 0.4065 -- so an accuracy quoted without those bars next to it
says nothing.  Same discipline as the 0.6267 corpus shortcut in the all3 track.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import xai
from .data import ROOT, cohort, fit_scaler, splits, to_blocks
from .models import build

OUT = os.path.join(ROOT, "Results/OSF_Park")

# (label, model, kwargs).  Ordered cheap -> expensive so a killed run keeps the most rows.
CONFIGS = [
    ("SVM [demographics only]",       "SVM",                dict(key="np_demo")),
    ("SVM [EEG]",                     "SVM",                {}),
    ("SVM [EEG+demographics]",        "SVM",                dict(key="np_both")),
    ("LogisticRegression [EEG]",      "LogisticRegression", {}),
    ("RandomForest [EEG]",            "RandomForest",       {}),
    ("GradientBoosting [EEG]",        "GradientBoosting",   {}),
    ("MLP [EEG]",                     "MLP",                dict(n_in=1140)),
    ("ChannelTransformer [AB]",       "ChannelTransformer", {}),
    ("CohTransformer [AB+COH bias]",  "CohTransformer",     {}),
    ("GatedHybrid [no demographics]", "GatedHybrid",        dict(use_demo=False)),
    ("GatedHybrid [+demographics]",   "GatedHybrid",        dict(use_demo=True)),
]


def baselines(c):
    """chance, majority, and the demographics bar -- the honest reference for this cohort."""
    _, counts = np.unique(c.y, return_counts=True)
    demo = make_pipeline(StandardScaler(), SVC(C=1.0, class_weight="balanced", random_state=42))
    s = cross_val_score(demo, c.DEMO, c.y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return {"chance": 1.0 / len(c.classes),
            "majority": float(counts.max() / counts.sum()),
            "demographics_only": float(s.mean()), "demographics_only_std": float(s.std())}


def year_probe(c):
    """Can recording YEAR be recovered from the features?

    'Same device' is an inference, not a fact: the file has no device/site column and spans
    2011-2018.  If year is highly recoverable this cohort has its own version of the all3
    corpus shortcut, and every row in the table below needs that caveat attached.
    """
    yr = c.year.copy()
    keep = {v for v, n in zip(*np.unique(yr, return_counts=True)) if n >= 20}
    yr = np.array([v if v in keep else -1 for v in yr])
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(200, random_state=42, n_jobs=-1))
    s = cross_val_score(rf, c.X(), yr, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    _, cnt = np.unique(yr, return_counts=True)
    return {"year_probe_acc": float(s.mean()), "year_majority": float(cnt.max() / cnt.sum()),
            "n_year_classes": int(len(cnt))}


def evaluate(c, label, name, kw, seeds, keep_xai, epochs=60):
    t0 = time.time()
    n = len(c.y)
    oof = np.zeros((n, len(c.classes)))
    seen = np.zeros(n, bool)
    per_fold, expl = [], None
    for f, (tr, te) in enumerate(splits(c.y, seeds=seeds)):
        blocks, _ = to_blocks(c, scaler=fit_scaler(c, tr))
        m = build(name, len(c.classes), **kw)
        m.fit({k: blocks[k][tr] for k in m.needs}, c.y[tr], epochs=epochs)
        p = m.predict_proba({k: blocks[k][te] for k in m.needs})
        # 3 seeds x 5 folds means every row is predicted 3 times; average, do not overwrite
        oof[te] += p
        seen[te] = True
        per_fold.append(float((p.argmax(1) == c.y[te]).mean()))
        if keep_xai and f == 0:
            expl = explain(m, c, te, blocks, fit_scaler(c, tr))
    assert seen.all(), "some rows never landed in a test fold"
    oof /= len(seeds)
    pred = oof.argmax(1)

    p_, r_, f_, sup = precision_recall_fscore_support(c.y, pred, labels=range(len(c.classes)),
                                                      zero_division=0)
    res = {"model": label,
           "accuracy": float((pred == c.y).mean()),
           "accuracy_fold_std": float(np.std(per_fold)),
           "balanced_accuracy": float(balanced_accuracy_score(c.y, pred)),
           "macro_f1": float(f1_score(c.y, pred, average="macro")),
           "per_class": {cl: {"precision": float(p_[i]), "recall": float(r_[i]),
                              "f1": float(f_[i]), "n": int(sup[i])}
                         for i, cl in enumerate(c.classes)},
           "confusion": confusion_matrix(c.y, pred).tolist(),
           "elapsed_sec": round(time.time() - t0, 1)}
    if len(c.classes) == 2:
        res["roc_auc"] = float(roc_auc_score(c.y, oof[:, 1]))
    if expl:
        res["xai"] = expl
    np.save(os.path.join(OUT, "preds", label.replace(" ", "_").replace("/", "-") + ".npy"), oof)
    return res


def explain(m, c, te, blocks, sc):
    """XAI on a HELD-OUT fold. Run on training rows it reports memorisation, not evidence."""
    from .data import Cohort
    sub = Cohort(c.df.iloc[te], c.classes)
    e = {}
    try:
        base, imp = xai.permutation(m, sub, sub.y, sc, n_repeats=3)
        e["permutation_balanced_acc"] = base
        e["permutation"] = {k: v for k, v in imp.items()}
    except Exception as err:                       # never let an explanation kill a result
        e["permutation_error"] = repr(err)
    b = {k: blocks[k][te] for k in m.needs}
    for fn, key in ((xai.native, "native"), (xai.attention, "attention"),
                    (xai.gates, "gates"), (xai.grad_input, "grad_input")):
        try:
            if fn is xai.native:
                v = fn(m, c)
            elif fn is xai.gates:
                v = fn(m, b) if m.name == "GatedHybrid" else None
            elif fn is xai.attention:
                v = fn(m, b, c) if m.name in ("ChannelTransformer", "CohTransformer", "GatedHybrid") else None
            else:
                v = fn(m, b, c) if m.name in ("MLP", "ChannelTransformer", "CohTransformer") else None
            if v:
                e[key] = v
        except Exception:
            pass
    return e


def markdown(task, base, probe, rows):
    L = [f"# OSF_Park BRMH — {task}", "",
         f"{rows[0].get('_n', '')}Out-of-fold over {rows[0]['_folds']} folds "
         f"(StratifiedKFold x {rows[0]['_seeds']} seeds). No subject IDs exist in this file, "
         "so grouping is impossible — see README.", "",
         "## Baselines — read every row below against these", "",
         "| baseline | accuracy |", "|---|---:|",
         f"| chance | {base['chance']:.4f} |",
         f"| majority class | {base['majority']:.4f} |",
         f"| **demographics only** (age, sex, IQ, education) | **{base['demographics_only']:.4f}** |", "",
         f"Recording-year probe: **{probe['year_probe_acc']:.4f}** vs {probe['year_majority']:.4f} "
         f"majority over {probe['n_year_classes']} year classes.", "",
         "## Models", "",
         "| model | accuracy | vs demo | balanced acc | macro-F1 | sec |",
         "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["accuracy"]):
        d = r["accuracy"] - base["demographics_only"]
        L.append(f"| {r['model']} | {r['accuracy']:.4f} ±{r['accuracy_fold_std']:.3f} | "
                 f"{d:+.4f} | {r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} | {r['elapsed_sec']:.0f} |")
    best = max(rows, key=lambda r: r["accuracy"])
    L += ["", f"## Per-class — {best['model']}", "",
          "| class | precision | recall | F1 | n |", "|---|---:|---:|---:|---:|"]
    for cl, v in best["per_class"].items():
        L.append(f"| {cl} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['n']} |")
    for r in rows:
        if "xai" in r:
            L += ["", f"## Explainability — {r['model']}"]
            x = r["xai"]
            if "permutation" in x:
                L.append(xai.markdown(f"Permutation importance (balanced-acc drop, base "
                                      f"{x['permutation_balanced_acc']:.3f})", x["permutation"]))
            for k, t in (("native", "Native importance by band"), ("grad_input", "Gradient x input by band"),
                         ("attention", "Attention received per channel"), ("gates", "Gate weight per branch")):
                if k in x:
                    v = x[k]
                    L.append(xai.markdown(t, v.get("by_band", v) if isinstance(v, dict) and "by_band" in v else v))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="4class", choices=["4class", "binary"])
    ap.add_argument("--smoke", action="store_true", help="1 seed, torch models only 8 epochs")
    ap.add_argument("--models", nargs="*", help="substring filter on the model label")
    a = ap.parse_args()

    os.makedirs(os.path.join(OUT, "preds"), exist_ok=True)
    seeds = (0,) if a.smoke else (0, 1, 2)
    c = cohort(a.task)
    print(f"{a.task}: n={len(c)} classes={c.classes} "
          f"counts={np.bincount(c.y).tolist()}  folds={5 * len(seeds)}")

    base, probe = baselines(c), year_probe(c)
    print(f"  chance {base['chance']:.4f} | majority {base['majority']:.4f} | "
          f"DEMOGRAPHICS {base['demographics_only']:.4f} ±{base['demographics_only_std']:.3f}")
    print(f"  year probe {probe['year_probe_acc']:.4f} (majority {probe['year_majority']:.4f}, "
          f"{probe['n_year_classes']} classes)\n")

    configs = [x for x in CONFIGS if not a.models or any(m.lower() in x[0].lower() for m in a.models)]
    rows, path = [], os.path.join(OUT, f"{a.task}.json")
    for label, name, kw in configs:
        r = evaluate(c, label, name, kw, seeds, keep_xai=True, epochs=8 if a.smoke else 60)
        r.update(_folds=5 * len(seeds), _seeds=len(seeds), _n=f"n = {len(c)} recordings. ")
        rows.append(r)
        print(f"  {label:32s} acc {r['accuracy']:.4f}  bal {r['balanced_accuracy']:.4f}  "
              f"F1 {r['macro_f1']:.4f}  vs demo {r['accuracy'] - base['demographics_only']:+.4f}  "
              f"({r['elapsed_sec']:.0f}s)")
        json.dump({"task": a.task, "n": len(c), "classes": c.classes, "seeds": list(seeds),
                   "baselines": base, "year_probe": probe, "results": rows},
                  open(path, "w"), indent=2)

    open(os.path.join(ROOT, "OSF_Park_model", f"RESULTS_{a.task}.md"), "w").write(
        markdown(a.task, base, probe, rows))
    print(f"\nwrote {path} and OSF_Park_model/RESULTS_{a.task}.md")


if __name__ == "__main__":
    main()
