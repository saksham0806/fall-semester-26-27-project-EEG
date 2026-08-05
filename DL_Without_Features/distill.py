"""
Per-corpus teachers -> ONE corpus-blind 3-class student.

    MODMA teacher    (healthy vs depression)  -.
    SAM40 teacher    (healthy vs stress)      -+- soft targets -> student (6ch, 3-way softmax)
    CogLoad teacher  (healthy vs stress)      -'                  no corpus input, ever

Corpus identity is used ONLY at training time, to decide which teacher supervises which
segment. The student takes raw (6, 512) and nothing else, which is the deployable artifact.

Two details that decide whether this means anything:

  * THE KD MASK. Each teacher speaks about two classes only. The KD term is masked to those
    two, and the student's probabilities are renormalized over the same pair. An unmasked KL
    would force P(depression)=0 on every SAM40/CogLoad segment, teaching the student
    "not-MODMA => not depressed" — the corpus shortcut, re-derived from the loss function.
    Verified in core.py: a teacher that never covers class 2 puts exactly zero gradient on
    the class-2 logit.

  * SOFT TARGETS ARE OUT-OF-FOLD. A teacher scoring its own training data emits near-one-hot
    targets and distillation degenerates to the hard labels. Teachers are therefore fitted
    under an inner subject-grouped CV inside each outer training split.

lam=0 is included as the control — a student with no distillation at all. Following the DANN
lesson in CLAUDE.md: a result without its method-off control is uninterpretable.

Run:  .venv/bin/python -u DL_Without_Features/distill.py --smoke     # ~4 min, CPU
      .venv/bin/python -u DL_Without_Features/distill.py             # GPU
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from DL_Without_Features.benchmark import build
from DL_Without_Features.core import DEVICE, EPOCHS, ranked, report, run_cv, save
from DL_Without_Features.core import fit_predict
from DL_Without_Features.data import (CORPUS_CLASSES, CORPUS_NAMES, RESULTS, folds, load,
                                      normalize, shortcut_baseline, subsample)

CONFIGS = [(0.0, 0.0), (2.0, 0.5), (2.0, 1.0), (4.0, 0.5), (4.0, 1.0)]   # (T, lam)


def pick_backbone(path):
    """Highest-accuracy architecture from a finished benchmark run."""
    with open(path) as fh:
        rows = [(v["accuracy"], v["architecture"], v["norm"])
                for v in json.load(fh).values()
                if isinstance(v, dict) and "architecture" in v]
    if not rows:
        raise SystemExit(f"{path} has no completed architecture rows")
    acc, arch, norm = max(rows)
    print(f"backbone auto-selected from benchmark: {arch} [{norm}] at {acc:.4f}")
    return arch, norm


def teacher_soft(arch, X, y, subjects, corpus, n_splits, inner, **kw):
    """
    (n_folds, n, 3) out-of-fold teacher probabilities. -1 marks a class the segment's teacher
    cannot speak about; those entries are what the KD mask switches off.
    """
    soft = np.full((n_splits, len(y), 3), -1.0, dtype=np.float32)
    for f, (tr, _) in enumerate(folds(y, subjects, n_splits)):
        for c, classes in CORPUS_CLASSES.items():
            idx = tr[corpus[tr] == c]
            yb = (y[idx] == classes[1]).astype(int)     # 0 = healthy, 1 = this corpus's illness
            Xc, sc = X[idx], subjects[idx]
            k = min(inner, len(np.unique(sc)))
            assert k >= 2, f"fold {f} corpus {CORPUS_NAMES[c]}: only {k} training subjects"
            for itr, ite in folds(yb, sc, k):
                out = fit_predict(build(arch, n_outputs=2), Xc, yb, sc, itr, ite,
                                  n_outputs=2, **kw)
                soft[f][np.ix_(idx[ite], list(classes))] = out.prob
            print(f"    fold {f + 1} teacher {CORPUS_NAMES[c]:8s} "
                  f"{len(idx)} segments / {len(np.unique(sc))} subjects", flush=True)
    return soft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="auto",
                    help="architecture name, or 'auto' to read benchmark.json")
    ap.add_argument("--norm", default=None, choices=["segment", "subject"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--inner", type=int, default=3, help="inner folds for teacher soft targets")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=os.path.join(RESULTS, "distill.json"))
    args = ap.parse_args()

    bench = os.path.join(RESULTS, "benchmark.json")
    if args.smoke and args.backbone == "auto":
        args.backbone = "EEGNet"          # smoke must not depend on a finished benchmark
    if args.backbone == "auto":
        if not os.path.exists(bench):
            raise SystemExit(f"--backbone auto needs {bench}; run benchmark.py first "
                             f"or pass --backbone <Arch>")
        arch, norm = pick_backbone(bench)
    else:
        arch, norm = args.backbone, "subject"
    norm = args.norm or norm

    X, y, subjects, corpus = load()
    if args.smoke:
        X, y, subjects, corpus = subsample(X, y, subjects, corpus)
        args.epochs, args.folds, args.inner = 2, 2, 2
        args.out = os.path.join(RESULTS, "_smoke", "distill.json")
    out_dir = os.path.dirname(args.out)   # smoke preds/emb stay out of the real ones

    X = normalize(X, subjects, norm)
    shortcut = shortcut_baseline(y, corpus)
    print(f"\nbackbone {arch} [{norm}] | {len(y)} segments, "
          f"{len(np.unique(subjects))} subjects | device {DEVICE}")
    print(f"bar to beat (corpus-ID shortcut): {shortcut:.4f}\n")

    # Teachers depend on backbone/norm/folds only — never on T or lam. Cache and reuse across
    # every config. Cache key carries all three: a name-only key is how stale caches happen.
    tag = f"{arch}_{norm}_{args.folds}f{args.inner}i{'_smoke' if args.smoke else ''}"
    cache = os.path.join(out_dir, f"_teacher_soft_{tag}.npy")
    if os.path.exists(cache):
        print(f"teacher soft targets: reusing {os.path.basename(cache)}")
        soft = np.load(cache)
    else:
        print(f"training per-corpus teachers ({args.folds} outer x 3 corpora x "
              f"{args.inner} inner) ...")
        t0 = time.perf_counter()
        soft = teacher_soft(arch, X, y, subjects, corpus, args.folds, args.inner,
                            epochs=args.epochs)
        os.makedirs(out_dir, exist_ok=True)
        np.save(cache, soft)
        print(f"  teachers done in {time.perf_counter() - t0:.0f}s -> "
              f"{os.path.basename(cache)}")

    covered = (soft[0] >= 0).sum(axis=1)
    print(f"  coverage: {(covered == 2).sum()} training segments carry a 2-class teacher, "
          f"{(covered == 0).sum()} carry none (held-out fold)\n")

    results = {"shortcut_baseline": shortcut, "backbone": arch, "norm": norm,
               "folds": args.folds, "inner_folds": args.inner, "epochs": args.epochs,
               "device": str(DEVICE)}

    for T, lam in CONFIGS:
        label = (f"student {arch} [{norm}] lam=0 (NO distillation, control)" if lam == 0
                 else f"student {arch} [{norm}] T={T:g} lam={lam:g}")
        print(f"\n  training {label} ...", flush=True)
        t0 = time.perf_counter()
        pred, emb, _, hooked = run_cv(build(arch), X, y, subjects,
                                      teacher=None if lam == 0 else soft, T=T, lam=lam,
                                      n_splits=args.folds, epochs=args.epochs)
        report(label, y, pred, corpus, subjects, shortcut, results, emb=emb,
               extra={"architecture": arch, "norm": norm, "T": T, "lam": lam,
                      "embedding": "penultimate" if hooked else "logits"},
               elapsed_sec=time.perf_counter() - t0, out_dir=out_dir)
        save(results, args.out)

    print("\n" + "=" * 78)
    rows = ranked(results, shortcut)
    ctrl = next((a for a, _, k in rows if "lam=0" in k), None)
    best = next(((a, k) for a, _, k in rows if "lam=0" not in k), None)
    if ctrl is not None and best is not None:
        print(f"\ndistillation vs its own no-distillation control: {best[0] - ctrl:+.4f}")
        print("  a positive margin here is the only evidence the teachers contributed "
              "anything;\n  the pooled accuracy alone cannot separate that from the backbone.")
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
