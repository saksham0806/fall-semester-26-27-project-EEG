"""Aggregate out-of-fold predictions to the unit the label actually varies over.

metd_plan.md §7: segment-level metrics are NEVER primary. Depression is a per-subject
trait, so segment-level n overstates the effective sample size by ~150x and the interval
that comes with it is meaningless.

Aggregation is by LOGIT MEDIAN, not mean probability: means of sigmoid outputs saturate,
while the median over a subject's ~150 segments resists a handful of bad windows.

    python -m arch_metd.eval.aggregate
"""
import json

import numpy as np

from ..contract import BY_DOMAIN, ROOT
from .metrics import auc_ci, paired_sign_test

RESULTS = ROOT / "arch_metd/results"


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def by_unit(units, score, y):
    """-> (unit ids, median logit per unit, one label per unit). Asserts label purity."""
    ids = np.unique(units)
    out_s, out_y = [], []
    for u in ids:
        m = units == u
        labels = np.unique(y[m])
        assert len(labels) == 1, f"unit {u} spans labels {labels} — wrong aggregation unit"
        out_s.append(np.median(logit(score[m])))
        out_y.append(int(labels[0]))
    return ids, np.asarray(out_s), np.asarray(out_y)


def load_oof(pattern="preds_fold*.npz"):
    """Out-of-fold predictions, concatenated. Every segment appears exactly once."""
    files = sorted(RESULTS.glob(pattern))
    assert files, f"no {pattern} in {RESULTS} — run the training first"
    parts = [dict(np.load(f)) for f in files]
    keys = set(parts[0])
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    if "subject_id" not in d:
        d |= _recover_identity(parts, files)
    if "_identity_ok" in d:
        keep = d.pop("_identity_ok")
        if not keep.all():
            print(f"    dropping {int((~keep).sum())} of {len(keep)} segments whose "
                  f"identity could not be verified\n")
            d = {k: v[keep] for k, v in d.items()}
    return d


def _recover_identity(parts, files):
    """Back-fill identity for predictions saved before subject_id was stored.

    Only trustworthy where it can be PROVEN. A run's arrays can differ in row order from
    the local ones, and a reordering is invisible to any check on a label that is constant
    within a subject. So the test is an exact element-wise match of the label sequence,
    per corpus, per fold: for MODMA's 1,498-row fold at ~45% positives, an exact match
    under a reordering has probability ~0, and the observed match is exact. SAM40 and
    CogLoad instead disagree on ~37% of rows, which is precisely what a random reordering
    of a 25/75 label split produces.

    Corpora that fail are dropped from the aggregate rather than reported approximately —
    a plausible-looking number from a misaligned join is the exact failure this project
    keeps rediscovering.
    """
    from ..data.build import load
    from ..train.folds import outer_folds

    print("predictions carry no subject_id (saved before that fix) — recovering it from "
          "the phase-4 split, per corpus:")
    src = load()
    out = {k: [] for k in ("subject_id", "trial_id", "condition")}
    trusted, verdict = [], {}
    for part, (_, test_idx) in zip(parts, outer_folds(src)):
        n = len(part["dep"])
        assert n == len(test_idx), "fold sizes differ — cannot recover"
        ident = {k: np.empty(n, dtype=src[k].dtype) for k in out}
        ok = np.zeros(n, bool)
        for dom in np.unique(part["y_domain"]):
            a = part["y_domain"] == dom              # positions in PREDICTION order
            b = src["y_domain"][test_idx] == dom     # positions in SPLIT order
            exact = bool(a.sum() == b.sum()
                         and (part["y_dep"][a] == src["y_dep"][test_idx][b]).all()
                         and (part["y_str"][a] == src["y_str"][test_idx][b]).all())
            # scatter, never assign positionally: matching per-corpus subsequences do not
            # imply matching absolute positions, because the corpora can be interleaved
            # differently in the two arrays
            for k in out:
                ident[k][a] = src[k][test_idx][b]
            ok[a] = exact
            verdict.setdefault(BY_DOMAIN[int(dom)].name, []).append(exact)
        trusted.append(ok)
        for k in out:
            out[k].append(ident[k])
    for name, v in sorted(verdict.items()):
        state = ("aligned — usable" if all(v) else
                 f"REORDERED in {sum(1 for x in v if not x)}/{len(v)} folds — excluded")
        print(f"    {name:8s} {state}")
    # a corpus that fails in ANY fold is dropped entirely: mixing verified and unverified
    # folds of one corpus produces a number nobody can interpret
    bad = {BY_DOMAIN[d].domain for d in range(5) if not all(verdict.get(BY_DOMAIN[d].name, [True]))}
    trusted = [ok & ~np.isin(part["y_domain"], list(bad))
               for ok, part in zip(trusted, parts)]
    return {k: np.concatenate(v) for k, v in out.items()} | {
        "_identity_ok": np.concatenate(trusted)}


def report(d):
    rows = {}

    # §7.1 depression — subject level, per corpus. Pooling across two amplifiers inflates
    # if corpus separates the classes, so pooled is secondary and flagged.
    for dom in (0, 3):
        m = (d["y_domain"] == dom) & (d["y_dep"] != -1)
        if not m.any():
            continue
        _, s, y = by_unit(d["subject_id"][m], d["dep"][m], d["y_dep"][m])
        rows[f"depression / {BY_DOMAIN[dom].name} [subject]"] = auc_ci(y, s)

    m = d["y_dep"] != -1
    if m.any() and len(np.unique(d["y_domain"][m])) > 1:
        _, s, y = by_unit(d["subject_id"][m], d["dep"][m], d["y_dep"][m])
        rows["depression / pooled [subject] (secondary)"] = auc_ci(y, s)

    # §7.2 SAM40 — trial level, 480 trials
    m = (d["y_domain"] == 1) & (d["y_str"] != -1)
    if m.any():
        _, s, y = by_unit(d["trial_id"][m], d["str"][m], d["y_str"][m])
        rows["stress / sam40 [trial]"] = auc_ci(y, s)
        within = []
        for sub in np.unique(d["subject_id"][m]):
            sm = m & (d["subject_id"] == sub)
            _, ss, sy = by_unit(d["trial_id"][sm], d["str"][sm], d["y_str"][sm])
            if len(np.unique(sy)) == 2:
                within.append(auc_ci(sy, ss, n_boot=1)["auc"])
        rows["stress / sam40 [within-subject mean] (secondary)"] = {
            "auc": float(np.mean(within)), "n": len(within)}

    # §7.4 CogLoad — within-subject primary. Its pooled recording-level AUC rests on ~15
    # negative recordings in total and cannot separate a working model from a broken one.
    m = (d["y_domain"] == 2) & (d["y_str"] != -1)
    if m.any():
        within = []
        for sub in np.unique(d["subject_id"][m]):
            sm = m & (d["subject_id"] == sub)
            _, ss, sy = by_unit(d["trial_id"][sm], d["str"][sm], d["y_str"][sm])
            if len(np.unique(sy)) == 2:
                within.append(auc_ci(sy, ss, n_boot=1)["auc"])
        rows["stress / cogload [within-subject mean]"] = {
            "auc": float(np.mean(within)), "n": len(within)}
        _, s, y = by_unit(d["trial_id"][m], d["str"][m], d["y_str"][m])
        rows["stress / cogload [recording] (secondary, ~15 negatives)"] = auc_ci(y, s)

    # §7.3 EEGMAT — paired sign test. Only present once a transfer run has been made.
    m = (d["y_domain"] == 4) & (d["y_str"] != -1)
    if m.any():
        ids, s, y = by_unit(d["trial_id"][m], d["str"][m], d["y_str"][m])
        subs = np.array([i.rsplit("_", 1)[0] for i in ids])
        pairs = [(s[(subs == u) & (y == 1)], s[(subs == u) & (y == 0)])
                 for u in np.unique(subs)]
        pairs = [(t[0], b[0]) for t, b in pairs if len(t) and len(b)]
        rows["stress / eegmat [paired sign test]"] = paired_sign_test(
            [t for t, _ in pairs], [b for _, b in pairs])
    return rows


def _selfcheck():
    assert abs(logit(0.5)) < 1e-9 and logit(0.9) > 0 > logit(0.1)
    # logit medians, not mean probabilities: one saturated window must not carry a unit
    units = np.array(["a"] * 4 + ["b"] * 4)
    score = np.array([0.6, 0.6, 0.6, 0.999, 0.4, 0.4, 0.4, 0.001])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    ids, s, yy = by_unit(units, score, y)
    assert list(ids) == ["a", "b"] and list(yy) == [1, 0]
    assert abs(s[0] - logit(0.6)) < 1e-9, "median was pulled by the saturated window"
    assert abs(np.mean(score[:4]) - 0.6) > 0.09, "the mean would have been pulled"

    try:
        by_unit(np.array(["a", "a"]), np.array([0.5, 0.5]), np.array([0, 1]))
    except AssertionError:
        pass
    else:
        raise SystemExit("by_unit accepted a unit spanning two labels")

    r = auc_ci(np.array([0, 0, 1, 1] * 8), np.array([0.1, 0.2, 0.8, 0.9] * 8))
    assert r["auc"] == 1.0 and r["ci_lo"] <= 1.0 and not r["crosses_chance"]
    r = auc_ci(np.array([0, 1] * 16), np.array([0.5] * 32))
    assert r["crosses_chance"], "a chance-level AUC must be flagged"

    r = paired_sign_test([0.9] * 9 + [0.1], [0.5] * 10)
    assert r["n_higher"] == 9 and r["p_value"] < 0.05, r
    print("aggregate selfcheck OK — logit median, unit purity, CI flag, sign test")


if __name__ == "__main__":
    import sys

    _selfcheck()
    if "--selfcheck-only" in sys.argv:
        sys.exit(0)
    d = load_oof()
    rows = report(d)
    print(f"\nout-of-fold, {len(d['dep'])} segments\n")
    for name, r in rows.items():
        if "p_value" in r:
            print(f"  {name:52s} {r['proportion']:.3f}  ({r['n_higher']}/{r['n']}, "
                  f"p={r['p_value']:.4f})")
        elif "ci_lo" in r:
            flag = "  <-- CI crosses chance" if r["crosses_chance"] else ""
            print(f"  {name:52s} {r['auc']:.4f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]  "
                  f"n={r['n']}{flag}")
        else:
            print(f"  {name:52s} {r['auc']:.4f}  n={r['n']}")
    (RESULTS / "aggregate.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {RESULTS / 'aggregate.json'}")
    print("segment-level figures in cv.json are NOT primary (§7, §12 item 7)")
