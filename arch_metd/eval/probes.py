"""Frozen domain probes and the manifold gate — metd_plan.md §8.1, §8.2.

The adversary's own accuracy is NOT a probe (§12 item 9). It is one side of a min-max
game being actively suppressed: its loss rising means it lost, not that z carries no
domain information. So the backbone is frozen and an INDEPENDENT logistic regression is
fitted on z, subject-disjoint.

    python -m arch_metd.eval.probes
"""
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ..contract import BY_DOMAIN, ROOT
from ..data.build import load
from ..models.mtdann import MTDANN
from ..train.folds import TRAIN_DOMAINS, outer_folds

RESULTS = ROOT / "arch_metd/results"
MODES = ("main", "lambda0", "randbb")


@torch.no_grad()
def embed(model, X, batch=512):
    model.eval()
    out = [model.backbone(torch.from_numpy(X[i:i + batch]).unsqueeze(1)).numpy()
           for i in range(0, len(X), batch)]
    return np.concatenate(out)


def _balanced(idx, domain, rng):
    """Equal segments per corpus, so a probe cannot win by predicting the largest."""
    per = min(int((domain[idx] == c).sum()) for c in np.unique(domain[idx]))
    return np.concatenate([rng.choice(idx[domain[idx] == c], per, replace=False)
                           for c in np.unique(domain[idx])])


def probe(z_tr, y_tr, z_te, y_te):
    sc = StandardScaler().fit(z_tr)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(z_tr), y_tr)
    return float((clf.predict(sc.transform(z_te)) == y_te).mean())


def run():
    d = load()
    rng = np.random.default_rng(0)
    keep = np.isin(d["y_domain"], TRAIN_DOMAINS)  # EEGMAT is in no fold
    majority = float(np.bincount(d["y_domain"][keep]).max() / keep.sum())
    n_dom = len(TRAIN_DOMAINS)
    print(f"full-set probe target (majority domain over the trained corpora) {majority:.4f}")
    print(f"anchor probe target (balanced over {n_dom} corpora) {1 / n_dom:.4f}\n")

    out = {"majority_domain": majority, "chance_balanced": 1 / n_dom, "modes": {}}
    for mode in MODES:
        tag = "" if mode == "main" else f"_{mode}"
        anchor_p, full_p, sils = [], [], []
        for i, (train_idx, test_idx) in enumerate(outer_folds(d)):
            model = MTDANN()
            model.load_state_dict(torch.load(
                RESULTS / f"model_fold{i}{tag}.pt", map_location="cpu", weights_only=True))
            tr = train_idx[np.isin(d["y_domain"][train_idx], TRAIN_DOMAINS)]
            te = test_idx[np.isin(d["y_domain"][test_idx], TRAIN_DOMAINS)]
            z_tr, z_te = embed(model, d["X"][tr]), embed(model, d["X"][te])

            full_p.append(probe(z_tr, d["y_domain"][tr], z_te, d["y_domain"][te]))

            # §8.1 anchor probe: balanced, and on the segments where invariance was
            # actually enforced. Running ONLY this reports invariance exactly where you
            # imposed it, which is why the full-set probe runs alongside it.
            a_tr = _balanced(tr[d["anchor"][tr]], d["y_domain"], rng)
            a_te = te[d["anchor"][te]]
            anchor_p.append(probe(z_tr[np.searchsorted(tr, a_tr)], d["y_domain"][a_tr],
                                  z_te[np.searchsorted(te, a_te)], d["y_domain"][a_te]))

            # §8.2 manifold gate, on anchor embeddings of the held-out fold
            za, ya = z_te[np.searchsorted(te, a_te)], d["y_domain"][a_te]
            sub = rng.choice(len(za), min(4000, len(za)), replace=False)
            sils.append(float(silhouette_score(za[sub], ya[sub])))

        out["modes"][mode] = {
            "anchor_probe": float(np.mean(anchor_p)), "anchor_folds": anchor_p,
            "full_probe": float(np.mean(full_p)), "full_folds": full_p,
            "silhouette": float(np.mean(sils)), "silhouette_folds": sils}
        r = out["modes"][mode]
        print(f"{mode:9s} anchor probe {r['anchor_probe']:.4f} (target {1/n_dom:.3f})  "
              f"full-set probe {r['full_probe']:.4f} (target {majority:.3f})  "
              f"silhouette {r['silhouette']:+.4f}")

    # The criterion pre-registered on 2026-09-06 (preregistration 2), replacing §8.2's
    # silhouette — which could not adjudicate, all modes landing within ±0.004 of zero.
    a_main = out["modes"]["main"]["anchor_probe"]
    a_zero = out["modes"]["lambda0"]["anchor_probe"]
    per_fold = np.array(out["modes"]["lambda0"]["anchor_folds"]) - \
        np.array(out["modes"]["main"]["anchor_folds"])
    out["gate"] = {
        "criterion": "anchor_probe(main) < anchor_probe(lambda0) - 0.05",
        "main": a_main, "lambda0": a_zero, "margin": float(a_zero - 0.05 - a_main),
        "passes": bool(a_main < a_zero - 0.05),
        "folds_passing": int((per_fold > 0.05).sum()), "n_folds": len(per_fold),
        "per_fold_margin_mean": float(per_fold.mean()),
        "per_fold_margin_sd": float(per_fold.std()),
        # reported alongside the verdict so a passing margin on a still-separable
        # representation cannot be quoted as invariance
        "anchor_vs_chance": float(a_main - 1 / n_dom),
        "full_vs_majority": float(out["modes"]["main"]["full_probe"] - majority),
        "silhouette_superseded": {m: out["modes"][m]["silhouette"] for m in MODES}}
    g = out["gate"]
    print(f"\ngate (pre-registered): anchor(main) {a_main:.4f} < anchor(lambda0)-0.05 "
          f"{a_zero - 0.05:.4f} -> {'PASS' if g['passes'] else 'FAIL'} "
          f"by {abs(g['margin']):.4f}")
    print(f"  robustness: {g['folds_passing']}/{g['n_folds']} folds pass individually, "
          f"per-fold margin {g['per_fold_margin_mean']:.4f} +- {g['per_fold_margin_sd']:.4f}")
    print(f"  still {g['anchor_vs_chance']:+.4f} above balanced chance and "
          f"{g['full_vs_majority']:+.4f} above the majority domain — not invariance")
    (RESULTS / "probes.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'probes.json'}")


if __name__ == "__main__":
    run()
