"""
all3 hierarchical two-stage classification — route by corpus, then a binary specialist.

WHY. Flat 3-class models lose to the 0.6267 corpus-ID shortcut not for lack of capacity but
because one shared decision boundary cannot suit all three corpora at once. Decomposed:

    corpus   weight  majority  best flat   margin
    MODMA     50.6%   0.5478    0.6586     +0.111
    SAM40     15.3%   0.7500    0.7512     +0.001
    CogLoad   34.1%   0.6886    0.5765     -0.112   <- a third of the data, below majority

A two-stage model can hit the best figure in each corpus, because each gets its own binary
problem:

    stage 1  which corpus recorded this segment?
    stage 2  MODMA   -> healthy vs depression
             SAM40   -> healthy vs stress
             CogLoad -> healthy vs stress

ROUTER AND SPECIALISTS WANT OPPOSITE FEATURE SETS — measured, and the reason this script
takes two feature arguments:
  - the router wants MANY features. v2's 273 route at 0.9701 vs v1's 81 at 0.9161, because
    added features make corpus easier to identify (the same effect that raised the corpus
    probe +0.065 and made v2 useless against the confound).
  - the specialists want FEW. v2's 273 collapsed the MODMA specialist to 0.5280, below its
    own 0.5478 majority — 273 dims against 53 subjects overfits.
Splitting them uses each where it helps.

EVERY CHOICE IS MADE ON INNER CV OVER TRAINING FOLDS ONLY — feature set, C, gamma, and
whether to use a specialist at all. Selecting any of these by outer-fold accuracy would be
test-set selection and would silently invalidate every number here. Majority is entered into
the same search as a candidate, so a specialist is used only if it beats guessing, which makes
0.6267 a floor by construction.

TWO ROUTERS, because the honest claim depends on which:
  oracle      stage 1 reads the true corpus label. Upper bound; consumes ground truth the
              model would not have at inference, so it is not a result.
  corpus-free stage 1 predicts corpus from EEG. The defensible number.

Misrouting has an asymmetric cost: a MODMA segment routed to CogLoad can only be labelled
healthy or stress, so its depression label becomes unreachable.

Run:  .venv/bin/python -u scripts/all3_hierarchical.py                       # v2 router, v1 specialists
      .venv/bin/python -u scripts/all3_hierarchical.py --specialists v1,plv --tune
      .venv/bin/python -u scripts/all3_hierarchical.py --cogload-only        # the diagnostic
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from all3_diagnostic import CORPUS_NAMES, load, report, shortcut_baseline

OUT = os.path.join(PROJECT_ROOT, "Results", "all3")
FEAT_V2 = os.path.join(PROJECT_ROOT, "features_v2", "all3")
N_SPLITS, INNER_SPLITS = 5, 3

# Default is the single config used everywhere else in this repo, so --tune isolates the
# effect of tuning rather than confounding it with a different baseline.
DEFAULT_GRID = [{"C": 1.0, "gamma": "scale"}]
TUNE_GRID = [{"C": c, "gamma": g} for c in (1.0, 10.0) for g in ("scale", 0.01)]


def svm(params=None):
    p = {"C": 1.0, "gamma": "scale"} if params is None else params
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", class_weight="balanced", **p))


def build_features(X):
    """All candidate views, aligned row-for-row with the dataset."""
    from proposed_model.features import extract_all_features
    views = {"v1": extract_all_features(X).astype(np.float32)}
    combined = os.path.join(FEAT_V2, "combined.npy")
    if os.path.exists(combined):
        F2 = np.load(combined)
        assert len(F2) == len(X), f"features_v2 has {len(F2)} rows, dataset has {len(X)}"
        views["v2"] = F2
        idx = json.load(open(os.path.join(FEAT_V2, "features.json")))["blocks"]
        for blk in ("plv", "coherence"):
            if blk in idx and idx[blk].get("cols"):
                views[blk] = F2[:, slice(*idx[blk]["cols"])]
    return views


def select_specialist(views, keys, y, subjects, grid):
    """
    Choose feature view + hyperparameters + specialist-vs-majority, using ONLY the data
    passed in (a training fold restricted to one corpus).

    Majority is seeded as the incumbent, so anything returned has already beaten guessing on
    inner CV. Returns (view_key or None, params or None, inner_score, majority_score).
    """
    maj_acc = float(np.bincount(y).max() / len(y))
    if len(np.unique(y)) < 2 or len(np.unique(subjects)) < INNER_SPLITS:
        return None, None, float("nan"), maj_acc

    best_score, best_key, best_params = maj_acc, None, None
    for key in keys:
        F = views[key]
        for params in grid:
            accs = []
            for itr, ite in GroupKFold(n_splits=INNER_SPLITS).split(F, y, groups=subjects):
                if len(np.unique(y[itr])) < 2:
                    continue
                clf = svm(params).fit(F[itr], y[itr])
                accs.append((clf.predict(F[ite]) == y[ite]).mean())
            if accs and float(np.mean(accs)) > best_score:
                best_score, best_key, best_params = float(np.mean(accs)), key, params
    return best_key, best_params, best_score, maj_acc


def hierarchical_oof(views, router_key, spec_keys, y, corpus, subjects,
                     router="corpus-free", grid=DEFAULT_GRID, verbose=True):
    """Out-of-fold 3-class predictions from the two-stage model."""
    pred = np.full(len(y), -1)
    stats = {"router_accuracy": [], "chosen": {}}

    for tr, te in GroupKFold(n_splits=N_SPLITS).split(views[router_key], y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"

        # --- stage 1 -----------------------------------------------------------------
        if router == "oracle":
            route = corpus[te]
        else:
            FR = views[router_key]
            route = svm().fit(FR[tr], corpus[tr]).predict(FR[te])
            stats["router_accuracy"].append(float((route == corpus[te]).mean()))

        # --- stage 2 -----------------------------------------------------------------
        for c in np.unique(corpus):
            tr_c = tr[corpus[tr] == c]
            te_c = te[route == c]                        # routed here, right or wrong
            if len(te_c) == 0 or len(tr_c) == 0:
                continue
            name = CORPUS_NAMES[int(c)]
            maj = int(np.bincount(y[tr_c]).argmax())

            key, params, inner, maj_acc = select_specialist(
                {k: v[tr_c] for k, v in views.items()}, spec_keys,
                y[tr_c], subjects[tr_c], grid)
            stats["chosen"].setdefault(name, []).append(
                {"view": key, "params": params, "inner_acc": inner, "majority_acc": maj_acc})

            if key is None:
                pred[te_c] = maj                          # guarded fallback
            else:
                F = views[key]
                pred[te_c] = svm(params).fit(F[tr_c], y[tr_c]).predict(F[te_c])

    assert (pred >= 0).all(), "some segments were never routed"
    if verbose:
        for name, ch in stats["chosen"].items():
            used = [d for d in ch if d["view"]]
            if not used:
                print(f"      {name:8s} majority in 5/5 folds (no specialist beat guessing)")
            else:
                vs = {}
                for d in used:
                    vs[d["view"]] = vs.get(d["view"], 0) + 1
                print(f"      {name:8s} specialist in {len(used)}/5 folds, views {vs}")
        if stats["router_accuracy"]:
            print(f"      router accuracy {np.mean(stats['router_accuracy']):.4f}")
    return pred, stats


def cogload_diagnostic(views, y, corpus, subjects):
    """
    Why does CogLoad sit below its own majority in all3, when the binary stress task on
    CogLoad reaches 0.76-0.88 AUC (per CLAUDE.md)? Isolate it: CogLoad alone, healthy vs
    stress, every available view.
    """
    print("\n" + "=" * 78)
    print("CogLoad diagnostic — healthy vs stress, trained on CogLoad alone")
    print("=" * 78)
    m = corpus == 2
    yc, sc = y[m], subjects[m]
    maj = float(np.bincount(yc).max() / len(yc))
    print(f"  {m.sum()} segments, {len(set(sc))} subjects, majority {maj:.4f}\n")
    print(f"  {'view':24s}{'accuracy':>10s}{'vs majority':>13s}{'AUC':>8s}")

    out = {}
    for name, Fall in views.items():
        F = Fall[m]
        pred = np.full(len(yc), -1)
        prob = np.zeros(len(yc))
        for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, yc, groups=sc):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000, class_weight="balanced"))
            clf.fit(F[tr], yc[tr])
            pred[te] = clf.predict(F[te])
            prob[te] = clf.predict_proba(F[te])[:, 1]
        acc = float((pred == yc).mean())
        auc = float(roc_auc_score((yc == 1).astype(int), prob))
        out[name] = {"accuracy": acc, "majority": maj, "margin": acc - maj, "auc": auc}
        print(f"  {name:24s}{acc:10.4f}{acc - maj:+13.4f}{auc:8.4f}"
              f"{'   <- beats majority' if acc > maj else ''}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default="v2", help="view for stage 1 (default v2: routes at 0.970)")
    ap.add_argument("--specialists", default="v1",
                    help="comma-separated views for stage 2; >1 enables per-corpus selection")
    ap.add_argument("--tune", action="store_true", help="nested-CV grid over SVM C and gamma")
    ap.add_argument("--cogload-only", action="store_true")
    ap.add_argument("--tag", default=None, help="suffix for the output json")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    X, y, subjects, corpus = load()
    corpus = corpus.astype(int)
    shortcut = shortcut_baseline(y, corpus)
    print(f"all3: {len(y)} segments, {len(set(subjects))} subjects")
    print(f"corpus-ID shortcut to beat: {shortcut:.4f}")

    views = build_features(X)
    print(f"views available: " + ", ".join(f"{k}({v.shape[1]})" for k, v in views.items()))

    results = {"shortcut_baseline": shortcut}
    if args.cogload_only:
        results["cogload_diagnostic"] = cogload_diagnostic(views, y, corpus, subjects)
        with open(os.path.join(OUT, "cogload_diagnostic.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwritten: {os.path.join(OUT, 'cogload_diagnostic.json')}")
        return

    spec_keys = [k.strip() for k in args.specialists.split(",")]
    for k in [args.router] + spec_keys:
        if k not in views:
            raise SystemExit(f"view '{k}' unavailable; have {sorted(views)}. "
                             "Run scripts/extract_features_v2.py for v2/plv/coherence.")
    grid = TUNE_GRID if args.tune else DEFAULT_GRID
    print(f"router: {args.router}   specialists: {spec_keys}   "
          f"grid: {len(grid)} config(s){' (tuned)' if args.tune else ''}\n")

    for router in ("corpus-free", "oracle"):
        name = (f"Hierarchical {router} / router={args.router} "
                f"spec={'+'.join(spec_keys)}{'+tuned' if args.tune else ''}")
        print("=" * 78)
        print(f"{name}")
        print("=" * 78)
        t0 = time.perf_counter()
        pred, stats = hierarchical_oof(views, args.router, spec_keys, y, corpus, subjects,
                                       router=router, grid=grid)
        results = report(name, y, pred, corpus, shortcut, results,
                         elapsed_sec=time.perf_counter() - t0)
        results[name].update({
            "router": router, "router_view": args.router, "specialist_views": spec_keys,
            "tuned": bool(args.tune),
            "chosen_per_fold": {k: [{kk: vv for kk, vv in d.items()} for d in v]
                                for k, v in stats["chosen"].items()},
        })
        if stats["router_accuracy"]:
            results[name]["router_accuracy"] = float(np.mean(stats["router_accuracy"]))

    tag = args.tag or f"{args.router}_{'-'.join(spec_keys)}{'_tuned' if args.tune else ''}"
    path = os.path.join(OUT, f"hierarchical_{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    for a, k in sorted(((v["accuracy"], k) for k, v in results.items()
                        if isinstance(v, dict) and "accuracy" in v), reverse=True):
        print(f"  {k:58s} {a:.4f}  {a - shortcut:+.4f}"
              f"{'   BEATS SHORTCUT' if a > shortcut else ''}")
    print("\nThe corpus-free row is the defensible number — the oracle row consumes the true\n"
          "corpus label at inference and is an upper bound, not a result.")
    print(f"written: {path}")


if __name__ == "__main__":
    main()
