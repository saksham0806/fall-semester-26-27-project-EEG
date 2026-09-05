"""Cross-head extrapolation — metd_plan.md §8.4.

Feed MODMA-MDD segments to the STRESS head and CogLoad-task segments to the DEPRESSION
head. Both are out-of-domain by construction: no segment in training carries both labels,
so the joint-high region of the output space is entirely unsupervised.

The comparison is against IN-DOMAIN POSITIVES FROM THE SAME HEAD, never against zero. A
previous run in this project showed every distribution clustered near p ~ 0.4 including
the in-domain case, which cannot distinguish "the heads do not extrapolate" from "the
heads output 0.4 for everything".

    python -m arch_metd.eval.controls
"""
import json

import numpy as np

from ..contract import ROOT
from .aggregate import load_oof

RESULTS = ROOT / "arch_metd/results"


def _stats(p):
    q = np.percentile(p, [25, 50, 75])
    return {"n": int(len(p)), "median": float(q[1]),
            "iqr": [float(q[0]), float(q[2])], "mean": float(p.mean())}


def run(mode="main"):
    d = load_oof(mode)
    out = {}
    for head, in_dom, out_dom, label in (
            ("str", (1, 2), (0, 3), "stress head"),
            ("dep", (0, 3), (1, 2), "depression head")):
        y = d[f"y_{head}"]
        rows = {
            "in-domain positive": _stats(d[head][np.isin(d["y_domain"], in_dom) & (y == 1)]),
            "in-domain negative": _stats(d[head][np.isin(d["y_domain"], in_dom) & (y == 0)]),
            "out-of-domain (unsupervised)": _stats(d[head][np.isin(d["y_domain"], out_dom)]),
        }
        sep = rows["in-domain positive"]["median"] - rows["in-domain negative"]["median"]
        rows["in-domain separation"] = float(sep)
        # If the in-domain classes are barely separated, the head is near-constant and
        # the extrapolation question cannot meaningfully be asked of it. A binary flag
        # hides an order-of-magnitude difference between the two heads here, so grade it.
        rows["verdict"] = ("near-constant" if abs(sep) < 0.02 else
                           "marginal" if abs(sep) < 0.10 else "informative")
        out[label] = rows
        print(f"\n{label}")
        for k in ("in-domain positive", "in-domain negative", "out-of-domain (unsupervised)"):
            r = rows[k]
            print(f"  {k:30s} median {r['median']:.3f}  IQR [{r['iqr'][0]:.3f}, "
                  f"{r['iqr'][1]:.3f}]  n={r['n']}")
        oo = rows["out-of-domain (unsupervised)"]["median"]
        pos = rows["in-domain positive"]["median"]
        print(f"  in-domain separation {sep:+.3f}  -> head {rows['verdict']}")
        print(f"  out-of-domain sits {'ABOVE in-domain positives' if oo > pos else 'below in-domain positives'}"
              f" ({oo:.3f} vs {pos:.3f})")
    (RESULTS / f"crosshead{'' if mode == 'main' else '_' + mode}.json").write_text(
        json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run()
