"""Sections 4 & 5: nested subject-grouped CV, robust aggregation, and the benchmark suite.

    python experiments.py within      # 5-fold joint CV -> OOF logits for every head
    python experiments.py loso        # CogLoad leave-one-subject-out, fixed epoch budget
    python experiments.py cross       # cross-corpus stress transfer (headline)
    python experiments.py legacy      # corpus shortcut + leaky-split diagnostics
    python experiments.py report      # regenerate REPORT.md from results/*.json
    python experiments.py all
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from model import Net, predict, train

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CORPORA = ["MODMA", "SAM40", "CogLoad"]
LAM = 1.0
N_FOLDS = 4


def load():
    p = ROOT / "data"
    d = {k: np.load(p / f"{k}.npy", allow_pickle=False) for k in
         ["X", "y_dep", "y_stress", "y_domain", "anchor"]}
    d["subject"] = np.load(p / "subject.npy")
    d["unit"] = np.load(p / "unit.npy")
    return d


# --------------------------------------------------------------------------- metrics

def boot_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            out.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan,) * 2


def aggregate(units, logits):
    """Robust aggregation: median of segment logits per test unit."""
    u = np.unique(units)
    return u, np.array([np.median(logits[units == k]) for k in u])


def youden(y, s):
    """Threshold maximising J = TPR - FPR, over cutoffs that predict both classes."""
    best, thr = -1, 0.0
    for t in np.unique(s):
        p = s >= t
        if p.all() or not p.any():
            continue
        j = (p & (y == 1)).sum() / max((y == 1).sum(), 1) - (p & (y == 0)).sum() / max((y == 0).sum(), 1)
        if j > best:
            best, thr = j, t
    return thr


def score(name, units, y_unit, s_unit, thr, extra=None):
    r = dict(name=name, n=len(y_unit), auc=float(roc_auc_score(y_unit, s_unit)),
             bal_acc=float(balanced_accuracy_score(y_unit, s_unit >= thr)),
             prevalence=float(y_unit.mean()))
    r["ci95"] = boot_ci(y_unit, s_unit)
    if extra:
        r.update(extra)
    print(f"  {name:<38} AUC {r['auc']:.4f}  [{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]  "
          f"balacc {r['bal_acc']:.4f}  N={r['n']}")
    return r


def unit_labels(d, idx, task):
    """Unit-level truth for a segment subset.  A unit spanning both classes is a label bug."""
    y, u_seg = d[task][idx], d["unit"][idx]
    u = np.unique(u_seg)
    yu = []
    for k in u:
        v = np.unique(y[u_seg == k])
        assert len(v) == 1, f"unit {k} spans labels {v} for {task}"
        yu.append(v[0])
    return u, np.array(yu)


# --------------------------------------------------------------------------- splits

def outer_folds(d, k=5, seed=0):
    """Per-corpus StratifiedGroupKFold on subjects, unioned into k joint folds."""
    fold = np.full(len(d["X"]), -1)
    for c in range(3):
        m = np.flatnonzero(d["y_domain"] == c)
        y = np.where(d["y_dep"][m] >= 0, d["y_dep"][m], d["y_stress"][m])
        sgk = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
        for i, (_, te) in enumerate(sgk.split(m, y, d["subject"][m])):
            fold[m[te]] = i
    assert (fold >= 0).all()
    return fold


def inner_split(d, tr, seed=0):
    """One GroupKFold(4) fold of the training subjects: early stopping + Youden theta."""
    g = d["subject"][tr]
    itr, iva = next(GroupKFold(N_FOLDS).split(tr, groups=g))
    assert not set(g[iva]) & set(g[itr])
    return tr[itr], tr[iva]


# --------------------------------------------------------------------------- experiments

def _cv(d, fold, k, lam):
    """One 5-fold arm at a fixed adversary strength.  Returns OOF logits, latents, thresholds."""
    n = len(d["X"])
    oof = dict(dep=np.full(n, np.nan), stress=np.full(n, np.nan))
    Z = np.zeros((n, 256), np.float32)
    thr = {t: np.zeros(k) for t in ("dep", "stress")}
    probes, best_eps = [], []
    for f in range(k):
        te = np.flatnonzero(fold == f)
        itr, iva = inner_split(d, np.flatnonzero(fold != f))
        m, ep = train(d, itr, iva, lam_max=lam, seed=f)
        best_eps.append(ep)

        pdep, pstr, zte = predict(m, d, te)
        oof["dep"][te], oof["stress"][te], Z[te] = pdep, pstr, zte

        for task, p in zip(("dep", "stress"), predict(m, d, iva)[:2]):
            sel = np.flatnonzero(d[f"y_{task}"][iva] >= 0)
            if len(sel):
                u, yu = unit_labels(d, iva[sel], f"y_{task}")
                _, su = aggregate(d["unit"][iva[sel]], p[sel])
                thr[task][f] = youden(yu, su) if 0 < yu.sum() < len(yu) else 0.0

        probes.append(domain_probe(Z[itr], d, itr, Z[te], te))
        print(f"  lambda={lam} fold {f}: best_epoch={ep}  probe_full={probes[-1]['full']:.3f} "
              f"anchor={probes[-1]['anchor']:.3f}  sil={probes[-1]['silhouette']:.3f}", flush=True)
    return oof, Z, thr, probes, best_eps


def run_within(k=5):
    """Both arms of the adversary.  lambda=0 is not optional: without a method-off control,
    'the GRL cost us nothing' and 'the GRL destroyed the depression head' look identical."""
    d = load()
    fold = outer_folds(d, k)
    t0, arms = time.time(), {}
    for lam, tag in [(LAM, "lam1"), (0.0, "lam0")]:
        oof, Z, thr, probes, best = _cv(d, fold, k, lam)
        print(f"\n[lambda={lam}] within-corpus")
        arms[tag] = dict(lam=lam, best_epochs=best,
                         rows=within_rows(d, oof, thr, f"lambda={lam}", fold, k),
                         probe={m: float(np.mean([p[m] for p in probes])) for m in probes[0]},
                         probe_per_fold=probes, crosshead=crosshead(d, oof))
        if tag == "lam1":
            np.save(RES / "oof_dep.npy", oof["dep"])
            np.save(RES / "oof_stress.npy", oof["stress"])

    out = dict(arms=arms, best_epochs=arms["lam1"]["best_epochs"],
               elapsed_sec=round(time.time() - t0, 1),
               majority_domain=float(max(np.bincount(d["y_domain"])) / len(d["X"])),
               gate_abort=bool(arms["lam1"]["probe"]["silhouette"] > 0.30))
    save("within", out)
    return out


def within_rows(d, oof, thr, tag, fold=None, k=5):
    rows = []
    for c, task, lvl in [(0, "y_dep", "subject"), (1, "y_stress", "trial"), (2, "y_stress", "recording")]:
        idx = np.flatnonzero((d["y_domain"] == c) & (d[task] >= 0) & ~np.isnan(
            oof["dep" if task == "y_dep" else "stress"]))
        if not len(idx):
            continue
        s = oof["dep" if task == "y_dep" else "stress"][idx]
        u, yu = unit_labels(d, idx, task)
        _, su = aggregate(d["unit"][idx], s)
        t = float(np.median(thr["dep" if task == "y_dep" else "stress"]))
        extra = dict(level=lvl, segment_auc=float(roc_auc_score(d[task][idx], s)))
        if c in (1, 2):
            extra["within_subject_auc"] = within_subject_auc(d, idx, s)
        if fold is not None:
            # Each fold's model has its own logit offset, so a single AUC over pooled OOF
            # logits mixes calibration differences into the ranking. Report per fold too.
            pf = []
            for f in range(k):
                sel = fold[idx] == f
                if sel.sum() == 0:
                    continue
                uu, yy = unit_labels(d, idx[sel], task)
                _, ss = aggregate(d["unit"][idx[sel]], s[sel])
                pf.append(float(roc_auc_score(yy, ss)) if 0 < yy.sum() < len(yy) else np.nan)
            extra["per_fold_auc"] = pf
            extra["per_fold_mean"] = float(np.nanmean(pf))
            extra["per_fold_std"] = float(np.nanstd(pf))
        r = score(f"{CORPORA[c]} {task[2:]} ({lvl}-level)", u, yu, su, t, extra)
        rows.append(r)
        if "per_fold_mean" in extra:
            print(f"  {'  per-fold mean':<38} {extra['per_fold_mean']:.4f} "
                  f"+/- {extra['per_fold_std']:.4f}   {np.round(extra['per_fold_auc'], 3)}")
        if c == 2:
            w = extra["within_subject_auc"]
            print(f"  {'CogLoad stress (within-subject, PRIMARY)':<38} AUC {w['mean']:.4f}  "
                  f"[{w['ci95'][0]:.3f},{w['ci95'][1]:.3f}]  N={w['n_subjects']} subjects")
    return rows


def within_subject_auc(d, idx, s, seed=0):
    """Mean per-subject unit-level AUC, bootstrapped over subjects.

    Primary for CogLoad: pooled recording-level AUC there rests on 14 negatives total, so its
    interval is far too wide to separate "works" from "doesn't".  Every subject has both a
    natural baseline and load recordings, so the within-subject contrast is well posed and
    gives 14 independent estimates instead of one fragile pooled one.
    """
    aucs, subs = [], []
    for sub in np.unique(d["subject"][idx]):
        sel = d["subject"][idx] == sub
        u, yu = unit_labels(d, idx[sel], "y_stress")
        _, su = aggregate(d["unit"][idx[sel]], s[sel])
        if 0 < yu.sum() < len(yu):
            aucs.append(roc_auc_score(yu, su))
            subs.append(sub)
    aucs = np.array(aucs)
    rng = np.random.default_rng(seed)
    boot = [aucs[rng.integers(0, len(aucs), len(aucs))].mean() for _ in range(1000)]
    return dict(mean=float(aucs.mean()), std=float(aucs.std()), n_subjects=len(aucs),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                frac_above_half=float((aucs > 0.5).mean()))


def domain_probe(Ztr, d, tr, Zte, te):
    """Frozen backbone: how much corpus identity survives, on anchors vs everything."""
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Ztr, d["y_domain"][tr])
    a = d["anchor"][te]
    full = float((lr.predict(Zte) == d["y_domain"][te]).mean())
    anch = float((lr.predict(Zte[a]) == d["y_domain"][te][a]).mean()) if a.sum() else np.nan
    sil = float(silhouette_score(Zte[a], d["y_domain"][te][a])) if len(np.unique(d["y_domain"][te][a])) > 1 else np.nan
    return dict(full=full, anchor=anch, silhouette=sil)


def crosshead(d, oof):
    """Heads must not fire confidently outside the domain they were supervised on."""
    out = {}
    for nm, head, mask in [
        ("stress_head_on_MODMA_MDD", "stress", (d["y_domain"] == 0) & (d["y_dep"] == 1)),
        ("dep_head_on_CogLoad_task", "dep", (d["y_domain"] == 2) & (d["y_stress"] == 1)),
        ("stress_head_on_SAM40_task", "stress", (d["y_domain"] == 1) & (d["y_stress"] == 1)),
        ("dep_head_on_MODMA_MDD", "dep", (d["y_domain"] == 0) & (d["y_dep"] == 1)),
    ]:
        p = 1 / (1 + np.exp(-oof[head][mask & ~np.isnan(oof[head])]))
        out[nm] = dict(n=int(len(p)), mean=float(p.mean()), median=float(np.median(p)),
                       frac_over_half=float((p > 0.5).mean()))
    return out


def run_loso(fixed_epochs=None, max_subjects=None):
    """CogLoad outer LOSO: fixed epoch budget, no 3-subject inner splits."""
    d = load()
    if fixed_epochs is None:
        fixed_epochs = int(np.median(json.load(open(RES / "within.json"))["best_epochs"]))
    subs = np.unique(d["subject"][d["y_domain"] == 2])[:max_subjects]
    oof = dict(dep=np.full(len(d["X"]), np.nan), stress=np.full(len(d["X"]), np.nan))
    t0 = time.time()
    for i, s in enumerate(subs):
        te = np.flatnonzero(d["subject"] == s)
        tr = np.flatnonzero(d["subject"] != s)
        m, _ = train(d, tr, None, lam_max=LAM, fixed_epochs=fixed_epochs, seed=i)
        oof["stress"][te] = predict(m, d, te)[1]
        print(f"\r  LOSO {i + 1}/{len(subs)} ({s})", end="", flush=True)
    print()
    idx = np.flatnonzero((d["y_domain"] == 2) & (d["y_stress"] >= 0) & np.isfinite(oof["stress"]))
    u, yu = unit_labels(d, idx, "y_stress")
    _, su = aggregate(d["unit"][idx], oof["stress"][idx])
    w = within_subject_auc(d, idx, oof["stress"][idx])
    row = score("CogLoad stress (recording-level, LOSO)", u, yu, su, 0.0,
                dict(level="recording", fixed_epochs=fixed_epochs, within_subject_auc=w,
                     segment_auc=float(roc_auc_score(d["y_stress"][idx], oof["stress"][idx]))))
    print(f"  {'CogLoad LOSO (within-subject, PRIMARY)':<38} AUC {w['mean']:.4f}  "
          f"[{w['ci95'][0]:.3f},{w['ci95'][1]:.3f}]  N={w['n_subjects']} subjects")
    save("loso", dict(rows=[row], elapsed_sec=round(time.time() - t0, 1)))


def run_cross():
    """Headline: zero-shot stress transfer across amplifiers, with the lambda=0 control."""
    d = load()
    rows, t0 = [], time.time()
    for train_c, test_c, lvl in [(1, 2, "recording"), (2, 1, "trial")]:
        tr_all = np.flatnonzero(np.isin(d["y_domain"], [0, train_c]))
        te = np.flatnonzero((d["y_domain"] == test_c) & (d["y_stress"] >= 0))
        itr, iva = inner_split(d, tr_all)
        for lam in (0.0, LAM):
            m, _ = train(d, itr, iva, lam_max=lam, seed=int(lam * 10))
            p = predict(m, d, te)[1]
            u, yu = unit_labels(d, te, "y_stress")
            _, su = aggregate(d["unit"][te], p)
            w = within_subject_auc(d, te, p)
            rows.append(score(f"{CORPORA[train_c]}->{CORPORA[test_c]} stress "
                              f"({lvl}-level, lambda={lam})", u, yu, su, 0.0,
                              dict(level=lvl, lam=lam, train=CORPORA[train_c], test=CORPORA[test_c],
                                   within_subject_auc=w,
                                   segment_auc=float(roc_auc_score(d["y_stress"][te], p)))))
            print(f"    within-subject {w['mean']:.4f} [{w['ci95'][0]:.3f},{w['ci95'][1]:.3f}] "
                  f"over {w['n_subjects']} subjects")
    save("cross", dict(rows=rows, elapsed_sec=round(time.time() - t0, 1)))


def run_legacy():
    """Historical context, not model comparison: corpus shortcut and window-overlap leakage."""
    d = load()
    y3 = np.where(d["y_dep"] == 1, 2, np.where(d["y_stress"] == 1, 1, 0))
    shortcut = sum(np.bincount(y3[d["y_domain"] == c], minlength=3).max() for c in range(3)) / len(y3)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(d["X"]))
    te, tr = perm[:len(perm) // 5], perm[len(perm) // 5:]
    m, _ = train(d, tr[:int(len(tr) * .9)], tr[int(len(tr) * .9):], lam_max=LAM, seed=0)
    pd_, ps_, _ = predict(m, d, te)
    leak = {t: float(roc_auc_score(d[f"y_{t}"][te][d[f"y_{t}"][te] >= 0],
                                   p[d[f"y_{t}"][te] >= 0]))
            for t, p in (("dep", pd_), ("stress", ps_))}

    grouped = {}
    for t in ("dep", "stress"):
        o = np.load(RES / f"oof_{t}.npy")
        m_ = (d[f"y_{t}"] >= 0) & ~np.isnan(o)
        grouped[t] = float(roc_auc_score(d[f"y_{t}"][m_], o[m_]))

    print(f"\n[legacy] corpus shortcut (3-class majority per corpus) = {shortcut:.4f}")
    print(f"  leaky random segment split : dep {leak['dep']:.4f}  stress {leak['stress']:.4f}")
    print(f"  subject-grouped (same model): dep {grouped['dep']:.4f}  stress {grouped['stress']:.4f}")
    save("legacy", dict(corpus_shortcut_3class=float(shortcut),
                        leaky_random_segment_split=leak, subject_grouped=grouped,
                        class_counts=dict(zip(["healthy", "stress", "depression"],
                                              np.bincount(y3, minlength=3).tolist()))))


# --------------------------------------------------------------------------- reporting

def save(name, obj):
    RES.mkdir(exist_ok=True)
    json.dump(obj, open(RES / f"{name}.json", "w"), indent=2, default=float)


def report():
    g = lambda n: json.load(open(RES / f"{n}.json")) if (RES / f"{n}.json").exists() else None
    w, lo, cr, lg = g("within"), g("loso"), g("cross"), g("legacy")
    rej = json.load(open(ROOT / "data" / "rejection_table.json"))
    L = ["# trial_cc_pre — rebuilt pipeline results", "",
         "Subject-grouped throughout. Segment counts after artifact gating.", "",
         "## 1. Segment rejection table", "",
         "| corpus | recordings | windows | kept | flatline | variance | blown channel |",
         "|---|---|---|---|---|---|---|"]
    for k, v in rej.items():
        L.append(f"| {k} | {v['recordings']} | {v['total']} | {v['kept']} | "
                 f"{v['flatline']} | {v['variance']} | {v.get('blown_channel', 0)} |")

    def tbl(rows, title):
        L.extend(["", f"## {title}", "",
                  "| target | level | N | AUC | 95% CI | bal.acc | segment AUC |",
                  "|---|---|---|---|---|---|---|"])
        for r in rows:
            L.append(f"| {r['name']} | {r.get('level', '')} | {r['n']} | **{r['auc']:.4f}** | "
                     f"[{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}] | {r['bal_acc']:.4f} | "
                     f"{r.get('segment_auc', float('nan')):.4f} |")
        ws = [r for r in rows if "within_subject_auc" in r]
        if ws:
            L.extend(["", "Within-subject (mean over subjects, bootstrapped over subjects):", "",
                      "| target | mean AUC | 95% CI | subjects | frac > 0.5 |", "|---|---|---|---|---|"])
            for r in ws:
                s = r["within_subject_auc"]
                L.append(f"| {r['name']} | **{s['mean']:.4f}** | [{s['ci95'][0]:.3f}, "
                         f"{s['ci95'][1]:.3f}] | {s['n_subjects']} | {s['frac_above_half']:.2f} |")

    if w:
        a1, a0 = w["arms"]["lam1"], w["arms"]["lam0"]
        tbl(a1["rows"], "2. Within-corpus, adversary on (lambda=1), 5-fold nested CV")
        L.extend(["", "### 2b. Adversary off (lambda=0) — the method-off control", "",
                  "| target | pooled AUC | per-fold mean ± sd | lambda=1 per-fold mean | delta |",
                  "|---|---|---|---|---|"])
        for r0, r1 in zip(a0["rows"], a1["rows"]):
            L.append(f"| {r0['name']} | {r0['auc']:.4f} | {r0['per_fold_mean']:.4f} ± "
                     f"{r0['per_fold_std']:.4f} | {r1['per_fold_mean']:.4f} | "
                     f"{r1['per_fold_mean'] - r0['per_fold_mean']:+.4f} |")
        L.append("\n**Per-fold mean is the honest summary, not pooled AUC.** Each fold's model "
                 "carries its own logit offset, so ranking pooled out-of-fold logits mixes "
                 "calibration differences between models into the metric.")
        L.append("\n**Read CogLoad from the within-subject row, not the pooled one.** Pooled "
                 "recording-level AUC there rests on 14 negative recordings in total, so its "
                 "interval cannot separate a working model from a broken one. Every subject has "
                 "a natural baseline and load recordings, so within-subject is well posed — but "
                 "at 1 baseline vs ~6 load per subject the per-subject AUC is quantised to 1/6, "
                 "hence the wide bootstrap. MODMA is subject-level; SAM40 is trial-level.")
    if lo:
        tbl(lo["rows"], "3. CogLoad LOSO")
    if cr:
        tbl(cr["rows"], "4. Cross-corpus stress transfer (headline)")
    if w:
        p, c = a1["probe"], a0["probe"]
        L.extend(["", "## 5. Dual frozen domain probes", "",
                  "| probe | lambda=0 | lambda=1 | target |", "|---|---|---|---|",
                  f"| full-set accuracy | {c['full']:.4f} | {p['full']:.4f} | "
                  f"{w['majority_domain']:.3f} (majority domain) |",
                  f"| anchor accuracy | {c['anchor']:.4f} | {p['anchor']:.4f} | 0.333 (chance) |",
                  f"| anchor silhouette S_corpus | {c['silhouette']:.4f} | {p['silhouette']:.4f} | < 0.30 |",
                  "", f"**Manifold gate:** S_corpus = {p['silhouette']:.4f} at lambda=1 -> "
                      f"{'ABORT GRL, use independent backbones' if w['gate_abort'] else 'alignment viable'}.",
                  "", "## 6. Cross-head extrapolation", "",
                  "| head fed out-of-domain | N | mean p | median p | frac p>0.5 |", "|---|---|---|---|---|"])
        for k, v in a1["crosshead"].items():
            L.append(f"| {k} | {v['n']} | {v['mean']:.4f} | {v['median']:.4f} | {v['frac_over_half']:.4f} |")
    if lg:
        L.extend(["", "## 7. Historical diagnostic context (not for model comparison)", "",
                  f"- Corpus shortcut baseline (per-corpus 3-class majority, zero EEG): "
                  f"**{lg['corpus_shortcut_3class']:.4f}**",
                  f"- Leaky random segment split: dep AUC {lg['leaky_random_segment_split']['dep']:.4f}, "
                  f"stress AUC {lg['leaky_random_segment_split']['stress']:.4f}",
                  f"- Same model, subject-grouped: dep AUC {lg['subject_grouped']['dep']:.4f}, "
                  f"stress AUC {lg['subject_grouped']['stress']:.4f}",
                  "", "Depression cannot be transfer-tested: no second depression corpus is present."])
    out = ROOT / ("REPORT.md" if RES.name == "results" else "REPORT_smoke.md")
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


def selfcheck():
    d = dict(unit=np.array(["a", "a", "a", "b", "b"]), y_stress=np.array([1, 1, 1, 0, 0]))
    u, s = aggregate(d["unit"], np.array([0.0, 10.0, 1.0, -5.0, 5.0]))
    assert list(u) == ["a", "b"] and list(s) == [1.0, 0.0], (u, s)   # median, not mean
    assert list(unit_labels(d, np.arange(5), "y_stress")[1]) == [1, 0]
    try:
        unit_labels(dict(unit=np.array(["a", "a"]), y=np.array([0, 1])), np.arange(2), "y")
        raise SystemExit("unit_labels failed to catch a split-label unit")
    except AssertionError:
        pass

    y = np.array([0, 0, 1, 1])
    assert youden(y, np.array([0.1, 0.2, 0.8, 0.9])) == 0.8, "youden picked the wrong cutoff"
    assert youden(y, np.array([1., 1., 1., 1.])) == 0.0, "degenerate cutoff not rejected"
    lo, hi = boot_ci(np.r_[np.zeros(50), np.ones(50)], np.r_[np.random.rand(50), np.random.rand(50) + 1])
    assert 0.9 < lo < hi <= 1.0, (lo, hi)
    print("experiments selfcheck ok")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "selfcheck":
        selfcheck()
        sys.exit()
    smoke = "--smoke" in sys.argv
    if smoke:                       # exercise every code path cheaply before the real run
        import model
        model.EPOCHS = 2
        RES = ROOT / "results_smoke"
    RES.mkdir(exist_ok=True)
    if cmd in ("within", "all"):
        run_within(k=2 if smoke else 5)
    if cmd in ("loso", "all"):
        run_loso(max_subjects=2 if smoke else None)
    if cmd in ("cross", "all"):
        run_cross()
    if cmd in ("legacy", "all"):
        run_legacy()
    report()
