"""
Plan 03 -- hierarchy, stacking, the other 417 subjects, and subgroup confusion.

    .venv/bin/python -u -m OSF_Park_model.hierarchical --smoke
    .venv/bin/python -u -m OSF_Park_model.hierarchical

4-class only; the binary task has no hierarchy to decompose.
"""
import argparse
import json
import os
import time

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features_corrected as fc
from .cv import dev_folds, metrics, table
from .data import ROOT, cohort, ill_cohort
from .make_holdout import load

OUT = os.path.join(ROOT, "Results/OSF_Park")
TASK = "4class"
HEALTHY, DEPRESSION, ANXIETY, STRESS = 0, 1, 2, 3
S2 = {DEPRESSION: 0, ANXIETY: 1, STRESS: 2}                 # ill -> stage-2 label
S2_INV = {v: k for k, v in S2.items()}


def lr(C=0.1):
    return make_pipeline(StandardScaler(), LogisticRegression(
        class_weight="balanced", max_iter=5000, C=C, random_state=42))


# ------------------------------------------------------------------------ A. hierarchy

def _guard(X, y, seed=0, margin=0.02):
    """Use a stage only if it beats a constant predictor in INNER CV.

    Decided on inner folds, never on the outer fold -- deciding it on outer accuracy is
    test-set selection.  Lifted from scripts/all3_hierarchical.py, where it was worth +0.055.

    THE COMPARATOR IS 1/K, NOT THE MAJORITY RATE.  scripts/all3_hierarchical.py scores raw
    accuracy and compares against the majority rate, which is consistent.  This guard scores
    BALANCED accuracy, whose value for any constant predictor is exactly 1/K regardless of
    imbalance -- so the majority rate is the wrong yardstick for it.

    Measured cost of getting this wrong: stage 1 (healthy vs ill) has majority 0.8242 and its
    classifier reaches balanced accuracy 0.7171 -- genuinely good against a 0.5 chance level,
    but it can never exceed 0.8242, so the guard rejected it in 15/15 folds.  Both stages then
    fell back to their majority class, and composing two majority fallbacks predicted
    "depression" for all 421 subjects: composed balanced accuracy exactly 0.2500.
    """
    K = len(np.unique(y))
    maj = np.bincount(y).max() / len(y)
    s = cross_val_score(lr(), X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                        scoring="balanced_accuracy").mean()
    return s > (1.0 / K) + margin, s, maj


def hierarchy(c, X, seeds):
    n = len(c.y)
    pred = np.full(n, -1)
    s1_true, s1_pred, s2_true, s2_pred, raw1 = [], [], [], [], []
    used = {"stage1": 0, "stage2": 0, "folds": 0}
    for tr, te in dev_folds(TASK, c.y, seeds):
        used["folds"] += 1
        y1 = (c.y != HEALTHY).astype(int)                   # stage 1: healthy vs ill
        ok1, _, _ = _guard(X[tr], y1[tr])
        used["stage1"] += int(ok1)
        m1 = clone(lr()).fit(X[tr], y1[tr])
        p1 = m1.predict(X[te]) if ok1 else np.full(len(te), int(np.bincount(y1[tr]).argmax()))

        ill_tr = tr[c.y[tr] != HEALTHY]
        y2 = np.array([S2[v] for v in c.y[ill_tr]])
        ok2, _, _ = _guard(X[ill_tr], y2)
        used["stage2"] += int(ok2)
        m2 = clone(lr()).fit(X[ill_tr], y2)

        raw1 += m1.predict(X[te]).tolist()              # unguarded, for diagnosis
        out = np.where(p1 == 0, HEALTHY, -1)
        idx = np.where(p1 == 1)[0]
        if len(idx):
            p2 = (m2.predict(X[te][idx]) if ok2 else
                  np.full(len(idx), int(np.bincount(y2).argmax())))
            out[idx] = [S2_INV[v] for v in p2]
        pred[te] = out
        s1_true += y1[te].tolist(); s1_pred += p1.tolist()
        ill_te = te[c.y[te] != HEALTHY]
        if len(ill_te):
            s2_true += [S2[v] for v in c.y[ill_te]]
            s2_pred += m2.predict(X[ill_te]).tolist()
    return (pred, (np.array(s1_true), np.array(s1_pred), np.array(raw1)),
            (np.array(s2_true), np.array(s2_pred)), used)


# -------------------------------------------------------------------------- B. stacking

def stack(c, B, seeds):
    """Three base learners -> OUT-OF-FOLD probabilities -> logistic meta-learner.

    The out-of-fold requirement IS the method.  A base learner scoring its own training data
    emits near-one-hot probabilities, the meta learns to trust whichever base overfits
    hardest, and the stack collapses to that single model.  Same lesson as the out-of-fold
    teacher soft targets in DL_Without_Features/.
    """
    views = {"spectral": fc.view(B, "SPECTRAL"), "graph": B["GRAPH"], "demo": B["DEMO"]}
    n, K = len(c.y), len(c.classes)
    oof, seen = np.zeros((n, K)), np.zeros(n, int)
    coefs, gaps = [], {k: [] for k in views}
    for tr, te in dev_folds(TASK, c.y, seeds):
        meta_tr = np.zeros((len(tr), K * len(views)))
        meta_te = np.zeros((len(te), K * len(views)))
        inner = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, c.y[tr]))
        for j, (name, V) in enumerate(views.items()):
            sl = slice(j * K, (j + 1) * K)
            for i_tr, i_te in inner:                        # OOF probabilities, never in-fold
                meta_tr[i_te, sl] = clone(lr()).fit(V[tr][i_tr], c.y[tr][i_tr]).predict_proba(V[tr][i_te])
            full = clone(lr()).fit(V[tr], c.y[tr])
            meta_te[:, sl] = full.predict_proba(V[te])
            gaps[name].append((full.predict(V[tr]) == c.y[tr]).mean() -
                              (meta_tr[:, sl].argmax(1) == c.y[tr]).mean())
        meta = LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0, random_state=42)
        meta.fit(meta_tr, c.y[tr])
        oof[te] += meta.predict_proba(meta_te)
        seen[te] += 1
        coefs.append(np.abs(meta.coef_).mean(0).reshape(len(views), K).sum(1))
    dev = load(TASK)[0]
    oof[dev] /= seen[dev][:, None]
    w = np.mean(coefs, 0)
    return oof, dict(zip(views, (w / w.sum()).tolist())), {k: float(np.mean(v)) for k, v in gaps.items()}


# ---------------------------------------------------------- C. the other 417 ill subjects

def auxiliary(c, X, seeds, use_extra):
    """Stage-2 with an auxiliary 6-class ill representation as extra features.

    use_extra=True  -> auxiliary trained on 850 ill (adds the 417 outside the 4-class cohort)
    use_extra=False -> the REQUIRED control: identical pipeline, 434 in-cohort ill only.
    Without the control a change cannot be attributed to the extra data -- the lesson from
    the DANN run, where only the alpha=0 control revealed the method had not worked.
    """
    ic = ill_cohort()
    Xi = fc.view(fc.blocks(ic), "EEG")
    no_to_row = {int(v): i for i, v in enumerate(ic.df["no."])}
    cohort_no = [int(v) for v in c.df["no."]]
    extra = np.array([i for i, no in enumerate(ic.df["no."]) if int(no) not in set(cohort_no)])

    t, p = [], []
    for tr, te in dev_folds(TASK, c.y, seeds):
        ill_tr = tr[c.y[tr] != HEALTHY]
        aux_rows = [no_to_row[cohort_no[i]] for i in ill_tr]
        if use_extra:
            aux_rows = np.concatenate([aux_rows, extra])    # never includes a held-out subject
        aux = clone(lr()).fit(Xi[aux_rows], ic.y[aux_rows])
        f = lambda idx: np.concatenate([X[idx], aux.predict_proba(
            Xi[[no_to_row[cohort_no[i]] for i in idx]])], 1)
        y2 = np.array([S2[v] for v in c.y[ill_tr]])
        m = clone(lr()).fit(f(ill_tr), y2)
        ill_te = te[c.y[te] != HEALTHY]
        if len(ill_te):
            t += [S2[v] for v in c.y[ill_te]]
            p += m.predict(f(ill_te)).tolist()
    return np.array(t), np.array(p), len(extra)


# ------------------------------------------------------- D. specific.disorder breakdown

def subgroup(c, pred, dev):
    """Regroup existing predictions by specific.disorder -- no training run needed."""
    spec = c.df["specific.disorder"].to_numpy()
    out = {}
    for s in sorted(set(spec[dev])):
        m = dev[spec[dev] == s]
        if len(m) < 10:
            continue
        true = c.y[m][0]
        out[s] = {"n": int(len(m)), "pooled_label": c.classes[true],
                  "recall": float((pred[m] == true).mean()),
                  "se": float(np.sqrt(max((pred[m] == true).mean() * (1 - (pred[m] == true).mean()), 1e-9) / len(m)))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    seeds = (0,) if a.smoke else (0, 1, 2)
    c = cohort(TASK)
    dev = load(TASK)[0]
    B = fc.blocks(c)
    X = fc.view(B, "EEG")
    maj = np.bincount(c.y[dev]).max() / len(dev)
    print(f"4class dev n={len(dev)} majority {maj:.4f} folds={5*len(seeds)}\n")
    out, rows = {"plan": "03", "n_dev": len(dev)}, []

    print("A. hierarchy")
    t0 = time.time()
    pred, (t1, p1, r1), (t2, p2), used = hierarchy(c, X, seeds)
    s1maj = max(np.bincount(t1)) / len(t1)
    s2maj = max(np.bincount(t2)) / len(t2)
    oh = np.eye(4)[pred[dev]]
    full = np.zeros((len(c.y), 4)); full[dev] = oh
    r = metrics(c.y, full, dev, c.classes, "hierarchical (composed)", elapsed=time.time() - t0)
    rows.append(r)
    out["A_hierarchy"] = {
        "composed": r,
        "stage1_healthy_vs_ill": {"balanced_accuracy": float(balanced_accuracy_score(t1, p1)),
                                  "accuracy": float((t1 == p1).mean()), "majority": float(s1maj),
                                  "confusion": confusion_matrix(t1, p1).tolist(),
                                  "unguarded_balanced_accuracy": float(balanced_accuracy_score(t1, r1)),
                                  "unguarded_accuracy": float((t1 == r1).mean()),
                                  "note": ("reported row is the GUARDED output; when the guard "
                                           "fails every fold that is the majority fallback, so "
                                           "read unguarded_* to see what the classifier did")},
        "stage2_mood_anx_trauma": {"balanced_accuracy": float(balanced_accuracy_score(t2, p2)),
                                   "accuracy": float((t2 == p2).mean()), "majority": float(s2maj),
                                   "confusion": confusion_matrix(t2, p2).tolist()},
        "guard_passed": used}
    print(f"   stage1 healthy-vs-ill  guarded bal {balanced_accuracy_score(t1,p1):.4f} | "
          f"UNGUARDED bal {balanced_accuracy_score(t1,r1):.4f} acc {(t1==r1).mean():.4f} "
          f"(majority {s1maj:.4f}) guard passed {used['stage1']}/{used['folds']}")
    print(f"   stage2 mood/anx/trauma acc {(t2==p2).mean():.4f} bal "
          f"{balanced_accuracy_score(t2,p2):.4f} (majority {s2maj:.4f}) "
          f"guard {used['stage2']}/{used['folds']}")
    print(f"   composed 4-class       bal {r['balanced_accuracy']:.4f} F1 {r['macro_f1']:.4f}\n")

    print("B. stacking")
    t0 = time.time()
    so, w, gaps = stack(c, B, seeds)
    rs = metrics(c.y, so, dev, c.classes, "stacked (spectral+graph+demo)", elapsed=time.time() - t0)
    rows.append(rs)
    out["B_stacking"] = {"result": rs, "meta_weight": w, "insample_minus_oof": gaps}
    assert all(v > 0 for v in gaps.values()), "a base learner's OOF matches in-fold -> inner CV is inert"
    print(f"   meta weights {({k: round(v,3) for k,v in w.items()})}")
    print(f"   in-sample minus OOF accuracy {({k: round(v,3) for k,v in gaps.items()})} (must be > 0)")
    print(f"   bal {rs['balanced_accuracy']:.4f} F1 {rs['macro_f1']:.4f}\n")

    print("C. auxiliary task on the other 417")
    t0 = time.time()
    aux = {}
    for flag, name in ((False, "control: 434 in-cohort ill only"), (True, "auxiliary: all 850 ill")):
        t, p, nx = auxiliary(c, X, seeds, flag)
        aux[name] = {"balanced_accuracy": float(balanced_accuracy_score(t, p)),
                     "accuracy": float((t == p).mean()), "n_extra": int(nx)}
        print(f"   {name:32s} stage2 bal {aux[name]['balanced_accuracy']:.4f} "
              f"acc {aux[name]['accuracy']:.4f}")
    d = (aux["auxiliary: all 850 ill"]["balanced_accuracy"]
         - aux["control: 434 in-cohort ill only"]["balanced_accuracy"])
    aux["delta_from_extra_data"] = float(d)
    aux["elapsed_sec"] = round(time.time() - t0, 1)
    out["C_auxiliary"] = aux
    print(f"   delta from the extra 417 subjects: {d:+.4f}\n")

    print("D. specific.disorder breakdown (regrouped from A's predictions, no retraining)")
    sg = subgroup(c, pred, dev)
    out["D_subgroups"] = sg
    for k, v in sorted(sg.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"   {k:32s} n={v['n']:3d} recall {v['recall']:.3f} ±{v['se']:.3f}  [{v['pooled_label']}]")

    out["results"] = rows
    p = os.path.join(OUT, "plan03_4class.json")
    json.dump(out, open(p, "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    A = out["A_hierarchy"]
    md = [f"# Plan 03 — hierarchy, stacking, auxiliary data (4class)", "",
          f"Dev only, n={len(dev)}; majority {maj:.4f}.", "", "## A. Hierarchy", "",
          "| stage | balanced acc | accuracy | majority | guard passed |", "|---|---:|---:|---:|---:|",
          f"| 1 · healthy vs ill (guarded) | {A['stage1_healthy_vs_ill']['balanced_accuracy']:.4f} | "
          f"{A['stage1_healthy_vs_ill']['accuracy']:.4f} | {A['stage1_healthy_vs_ill']['majority']:.4f} | "
          f"{used['stage1']}/{used['folds']} |",
          f"| 1 · healthy vs ill (unguarded classifier) | "
          f"{A['stage1_healthy_vs_ill']['unguarded_balanced_accuracy']:.4f} | "
          f"{A['stage1_healthy_vs_ill']['unguarded_accuracy']:.4f} | "
          f"{A['stage1_healthy_vs_ill']['majority']:.4f} | — |",
          f"| 2 · mood/anxiety/trauma | {A['stage2_mood_anx_trauma']['balanced_accuracy']:.4f} | "
          f"{A['stage2_mood_anx_trauma']['accuracy']:.4f} | {A['stage2_mood_anx_trauma']['majority']:.4f} | "
          f"{used['stage2']}/{used['folds']} |", "",
          "Stage 2 is reported against its own majority (199/434), never against 0.3333.", "",
          "## B. Stacking", "",
          f"Meta weights: {w}. In-sample minus OOF accuracy per base learner: {gaps} "
          "(all > 0, confirming the inner CV is real).", "",
          "## C. The other 417 subjects", "",
          "| arm | stage-2 balanced acc |", "|---|---:|"] + \
         [f"| {k} | {v['balanced_accuracy']:.4f} |" for k, v in aux.items() if isinstance(v, dict)] + \
         ["", f"Delta from the extra data: **{d:+.4f}**. Those 417 add no healthy controls "
          "(all 93 are already in the cohort), so they cannot relieve a stage-1 bottleneck.", "",
          "## D. Specific-disorder breakdown", "",
          "| specific.disorder | pooled as | n | recall | ± |", "|---|---|---:|---:|---:|"] + \
         [f"| {k} | {v['pooled_label']} | {v['n']} | {v['recall']:.3f} | {v['se']:.3f} |"
          for k, v in sorted(sg.items(), key=lambda kv: -kv[1]["recall"])] + \
         ["", "Sub-group n is 30–60, so recall SE is ±0.06–0.09. Read counts, not rates.", "",
          "## Composed", "", table(rows)]
    open(os.path.join(ROOT, "OSF_Park_model", "RESULTS_plan03_4class.md"), "w").write("\n".join(md) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
