"""Loaders -> preprocessed arrays -> disk. metd_plan.md §3, §4, §14 step 2.

    python -m arch_metd.data.build --calibrate-k   # sweep K on MODMA, write nothing
    python -m arch_metd.data.build                 # build, needs contract.VAR_GATE_K set

Arrays land in ``datasets/metd/`` — that tree is gitignored wholesale, so a 480 MB
array cannot be staged by accident. Re-runnable: the output is overwritten, never
merged, and no cache is keyed on a filename.
"""
import argparse
import hashlib
import json
from collections import Counter

import numpy as np

from ..contract import CORPORA, ROOT, VAR_GATE_K, validate

ARRAYS = ROOT / "datasets/metd"
RESULTS = ROOT / "arch_metd/results"

META = ("subject_id", "trial_id", "protocol", "condition")


def _loaders():
    """Imported lazily: loading the built arrays must not require mne, pandas or scipy.

    The GPU workstation gets metd.npz (485 MB) and never the 5 GB of raw corpora, so a
    module-level loader import would make `load()` fail there for no reason.
    """
    from .loaders import cogload, eegmat, modma, mumtaz, sam40
    return {0: modma, 1: sam40, 2: cogload, 3: mumtaz, 4: eegmat}


def fingerprint(d):
    """A cheap, order-sensitive identity for a built array set.

    Two boxes rebuilding from the same corpora do not necessarily produce byte-identical
    arrays — gate decisions sit on floating-point boundaries, and row order can differ.
    Results computed against different arrays are not comparable, and the difference is
    invisible in every summary statistic. Record it so a mismatch is one line, not a
    forensic exercise.
    """
    h = hashlib.sha256()
    h.update(str(d["X"].shape).encode())
    for part in (d["X"][:512], d["X"][-512:]):
        h.update(np.ascontiguousarray(part).tobytes())
    for name in ("y_dep", "y_str", "y_domain"):
        h.update(np.ascontiguousarray(d[name]).tobytes())
    return h.hexdigest()[:16]


def three_class(y_dep, y_str):
    """The composed label the system emits: healthy / depression / stress.

    Never a training target — §0 forbids a 3-way softmax. It exists so the corpus dummy
    can be quoted on the same scale as a headline number.
    """
    y = np.full(len(y_dep), -1, np.int8)
    y[(y_dep == 0) | (y_str == 0)] = 0
    y[y_dep == 1] = 1
    y[y_str == 1] = 2
    return y


def baselines(d):
    """Everything a headline number must be quoted against (§8.3, §12 item 10).

    Recomputed from the rebuilt arrays every time. Nothing here is ever hardcoded — the
    old pipeline's 0.6267 was quoted long after the data underneath it had changed.
    """
    dom, y3 = d["y_domain"], three_class(d["y_dep"], d["y_str"])
    out = {"n_segments": int(len(dom)),
           "segments_per_domain": {c.name: int((dom == c.domain).sum()) for c in CORPORA},
           "majority_domain": float(np.bincount(dom).max() / len(dom))}

    for head in ("y_dep", "y_str"):
        y = d[head][d[head] != -1]
        out[head] = {"n": int(len(y)),
                     "majority": float(max(np.mean(y), 1 - np.mean(y)))}

    # a classifier knowing ONLY which corpus a segment came from, with zero EEG
    # understanding: predict that corpus's majority label
    m = y3 != -1
    hit = 0
    for c in CORPORA:
        sel = m & (dom == c.domain)
        if sel.any():
            hit += np.bincount(y3[sel]).max()
    out["corpus_dummy_3class"] = float(hit / m.sum())
    out["majority_3class"] = float(np.bincount(y3[m]).max() / m.sum())
    return out


def rejection_table(rows):
    """§3.2 — publish it. Divergent rates across corpora are a finding, not a detail."""
    head = ("| corpus | recordings | windows | kept | flatline | variance | kept % |\n"
            "|---|---:|---:|---:|---:|---:|---:|\n")
    body = "".join(
        f"| {name} | {t['recordings']} | {t['windows']} | {t['kept']} | "
        f"{t['flatline']} | {t['variance']} | {100 * t['kept'] / max(t['windows'], 1):.1f} |\n"
        for name, t in rows)
    total = Counter()
    for _, t in rows:
        total.update(t)
    body += (f"| **all** | {total['recordings']} | {total['windows']} | {total['kept']} | "
             f"{total['flatline']} | {total['variance']} | "
             f"{100 * total['kept'] / total['windows']:.1f} |\n")
    return head + body


def calibrate_k(candidates=(3, 4, 5, 6, 8, 10, 15, 20, 30, 50)):
    """Sweep K on MODMA and pick the value rejecting 1-2% (§3.2).

    MODMA is the reference because it has zero dropout and zero NaN by measurement.
    K = 5 was used previously and rejects ~9.7%, which is far too aggressive.
    """
    from .preprocess import prepare
    ratios = []
    for x, m in _loaders()[0].load():
        _, keep, ratio = prepare(x, m["fs"])
        ratios.append(ratio[keep])
    ratio = np.concatenate(ratios)
    print(f"\nMODMA variance-ratio distribution over {len(ratio)} windows")
    print("  " + "  ".join(f"p{q}={np.percentile(ratio, q):.2f}"
                           for q in (50, 90, 95, 98, 99, 99.5)))
    print("\n| K | rejected | % |\n|---:|---:|---:|")
    for k in candidates:
        n = int((ratio > k).sum())
        print(f"| {k} | {n} | {100 * n / len(ratio):.2f} |")
    lo, hi = np.percentile(ratio, [98, 99])
    print(f"\ntarget 1-2% rejection -> K between {hi:.2f} and {lo:.2f}")
    return ratio


def build(k):
    assert k is not None, "set contract.VAR_GATE_K first — run --calibrate-k"
    from .preprocess import segment
    loaders = _loaders()
    fields = {name: [] for name in ("X", "y_dep", "y_str", "y_domain", *META)}
    rows = []
    for c in CORPORA:
        totals = Counter()
        for x, m in loaders[c.domain].load():
            w, counts = segment(x, m["fs"], k)
            totals.update(counts)
            totals["recordings"] += 1
            if not len(w):
                continue
            fields["X"].append(w)
            for name in ("y_dep", "y_str"):
                fields[name].append(np.full(len(w), m[name], np.int8))
            fields["y_domain"].append(np.full(len(w), c.domain, np.int8))
            for name in META:
                fields[name].append(np.full(len(w), m[name]))
        rows.append((c.name, totals))
        print(f"  {c.name:8s} {totals['kept']:6d} kept / {totals['windows']:6d} windows")

    d = {name: np.concatenate(parts) for name, parts in fields.items()}
    d["anchor"] = (d["y_dep"] == 0) | (d["y_str"] == 0)
    validate(d)

    # §4: store shuffled. Corpus-ordered arrays have already caused an accidental
    # leave-one-corpus-out via an unshuffled internal split.
    order = np.random.default_rng(42).permutation(len(d["X"]))
    d = {name: v[order] for name, v in d.items()}
    validate(d)

    ARRAYS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.savez(ARRAYS / "metd.npz", **d)
    table = rejection_table(rows)
    (RESULTS / "rejection_table.md").write_text(
        f"# Rejection table (§3.2)\n\nVAR_GATE_K = {k}\n\n{table}")
    base = baselines(d)
    (RESULTS / "baselines.json").write_text(json.dumps(base, indent=2))
    info = {"fingerprint": fingerprint(d), "n_segments": int(len(d["X"])),
            "var_gate_k": k, "segments_per_domain": base["segments_per_domain"]}
    (RESULTS / "build_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nbuild fingerprint {info['fingerprint']}  ({info['n_segments']} segments)")

    print(f"\n{table}")
    print(f"corpus dummy (3-class, knowing only the corpus): {base['corpus_dummy_3class']:.4f}")
    print(f"majority class (3-class):                        {base['majority_3class']:.4f}")
    print(f"majority domain (full-set probe target):         {base['majority_domain']:.4f}")
    print(f"depression n={base['y_dep']['n']:6d} majority {base['y_dep']['majority']:.4f}")
    print(f"stress     n={base['y_str']['n']:6d} majority {base['y_str']['majority']:.4f}")
    print(f"\nwrote {ARRAYS / 'metd.npz'} ({(ARRAYS / 'metd.npz').stat().st_size / 1e6:.0f} MB)")
    return d


def load():
    """The built arrays, validated. No allow_pickle anywhere."""
    with np.load(ARRAYS / "metd.npz") as f:
        d = {name: f[name] for name in f.files}
    validate(d)
    return d


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate-k", action="store_true",
                   help="sweep the variance gate on MODMA and write nothing")
    p.add_argument("--fingerprint", action="store_true",
                   help="identify the built arrays and exit; compare across machines")
    a = p.parse_args()
    if a.calibrate_k:
        calibrate_k()
    elif a.fingerprint:
        d = load()
        print(f"fingerprint {fingerprint(d)} | {len(d['X'])} segments | "
              + " ".join(f"{c.name}={int((d['y_domain'] == c.domain).sum())}"
                         for c in CORPORA))
    else:
        build(VAR_GATE_K)
