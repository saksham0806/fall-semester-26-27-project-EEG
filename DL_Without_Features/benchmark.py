"""
braindecode raw-EEG architectures on the all3 3-class task. No feature extraction anywhere:
the models consume the (6, 512) windows directly.

Every model runs under two input normalizations, so the invariance lever is a paired
comparison rather than a confounded one:

    --norm segment   pass-through (X.npy is already z-scored per 4 s window)  <- control
    --norm subject   per-channel standardization inside each subject          <- the lever

Each row reports 3-class accuracy against the 0.6267 corpus-ID shortcut AND a corpus probe on
the model's own penultimate embedding. Read them together — accuracy alone cannot tell a
corpus-free model from a device detector.

ANCHOR ROW: braindecode EEGNet under --norm segment should land near 0.4306, what the Keras
EEGNet scored on identical data and folds (Results/all3/models_deep.json). Far off means the
harness is wrong, not the model. Above ~0.85 means the split broke.

Run:  .venv/bin/python -u DL_Without_Features/benchmark.py --smoke     # ~1 min, CPU
      .venv/bin/python -u DL_Without_Features/benchmark.py             # GPU
"""
import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import braindecode.models as bm
import numpy as np

from DL_Without_Features.core import DEVICE, EPOCHS, ranked, report, run_cv, save
from DL_Without_Features.data import RESULTS, load, normalize, shortcut_baseline, subsample

# All verified to construct and forward at n_chans=6, n_times=512, n_outputs=3, sfreq=128.
DEFAULT = ["ShallowFBCSPNet", "Deep4Net", "EEGNet", "EEGNeX",
           "EEGConformer", "ATCNet", "EEGITNet", "SCCNet"]
EXTRA = ["CTNet", "MSVTNet", "EEGInceptionERP", "TIDNet", "EEGTCNet",
         "SincShallowNet", "FBCNet", "EEGSimpleConv", "AttentionBaseNet"]


def build(name, n_outputs=3):
    return lambda: getattr(bm, name)(n_chans=6, n_times=512, n_outputs=n_outputs, sfreq=128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="add the 9 extra architectures")
    ap.add_argument("--models", nargs="+", help="explicit architecture list")
    ap.add_argument("--norm", nargs="+", default=["segment", "subject"],
                    choices=["segment", "subject"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--smoke", action="store_true",
                    help="18 subjects, 2 folds, 2 epochs, 2 models — proves the harness")
    ap.add_argument("--out", default=os.path.join(RESULTS, "benchmark.json"))
    args = ap.parse_args()

    models = args.models or (DEFAULT + EXTRA if args.all else DEFAULT)
    X, y, subjects, corpus = load()
    if args.smoke:
        X, y, subjects, corpus = subsample(X, y, subjects, corpus)
        models, args.epochs, args.folds = models[:2], 2, 2
        args.out = os.path.join(RESULTS, "_smoke", "benchmark.json")
    out_dir = os.path.dirname(args.out)   # smoke preds/emb stay out of the real ones

    shortcut = shortcut_baseline(y, corpus)
    print(f"all3: {len(y)} segments, {len(np.unique(subjects))} subjects, device {DEVICE}")
    print(f"bar to beat (corpus-ID shortcut): {shortcut:.4f}   "
          f"corpus-probe floor: {np.bincount(corpus).max() / len(corpus):.4f}")
    print(f"{len(models)} architectures x {len(args.norm)} normalizations, "
          f"{args.folds}-fold subject-grouped\n")

    results = {"shortcut_baseline": shortcut, "n_segments": len(y),
               "n_subjects": int(len(np.unique(subjects))), "epochs": args.epochs,
               "folds": args.folds, "device": str(DEVICE)}

    for norm in args.norm:
        Xn = normalize(X, subjects, norm)
        print("=" * 78)
        print(f"NORM = {norm}")
        print("=" * 78)
        for name in models:
            label = f"{name} [{norm}]"
            print(f"\n  training {label} ...", flush=True)
            t0 = time.perf_counter()
            try:
                pred, emb, _, hooked = run_cv(build(name), Xn, y, subjects,
                                              n_splits=args.folds, epochs=args.epochs)
            except Exception as exc:                      # one bad arch must not kill the run
                print(f"    FAILED: {type(exc).__name__}: {exc}")
                results[label] = {"error": f"{type(exc).__name__}: {exc}"}
                save(results, args.out)
                continue
            report(label, y, pred, corpus, subjects, shortcut, results, emb=emb,
                   extra={"architecture": name, "norm": norm,
                          "embedding": "penultimate" if hooked else "logits"},
                   elapsed_sec=time.perf_counter() - t0, out_dir=out_dir)
            save(results, args.out)   # after every model: a killed session keeps its rows

        del Xn

    print("\n" + "=" * 78)
    ranked(results, shortcut)
    print(f"\nwritten: {args.out}")
    print("\nA row beating 0.6267 with a probe near the floor is the result you want.\n"
          "A row beating 0.6267 with a probe near 0.97 is the device shortcut in a new costume.")


if __name__ == "__main__":
    main()
