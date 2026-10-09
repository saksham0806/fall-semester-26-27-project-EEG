"""
Build features_v2/all3/ — corrected preprocessing + the feature families v1 lacks.

v1 divides band power by total power over the full 0-64 Hz while the bands span only
0.5-45 Hz. Measured on this dataset, that unused 45-64 Hz gap holds 98.2% of MODMA's power
(a 50 Hz mains spike), 74.7% of CogLoad's, and 34.4% of SAM40's — corpus-specific
contamination sitting in the denominator of every relative-power feature.

v2 bandpasses 0.5-45 Hz zero-phase first, then computes 273 handcrafted columns in 9 named
blocks plus two Riemannian views. `src/proposed_model/features.py` is untouched, so v1 stays
reproducible and the two can be compared directly.

THE PAYOFF is the corpus probe printed at the end: can a classifier still tell which machine
recorded each segment? v1 scores 0.9161. A material drop means the spectral defect was
driving the all3 confound; no drop means the device signature lives elsewhere. Both outcomes
are informative.

Run:  .venv/bin/python -u scripts/extract_features_v2.py
      .venv/bin/python -u scripts/extract_features_v2.py --smoke   # 500 segments, ~1 min
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.mean import mean_riemann
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Import the module directly, not via `proposed_model`, whose __init__ pulls in TensorFlow.
from proposed_model.features_v2 import (BANDS, BLOCKS, TOTAL_BAND, bandpass, block_names,
                                        extract_all_features_v2, gap_power_fraction,
                                        preprocess)

SRC = os.path.join(PROJECT_ROOT, "datasets", "all3")
OUT = os.path.join(PROJECT_ROOT, "features_v2", "all3")
CORPUS_NAMES = {0: "MODMA", 1: "SAM40", 2: "CogLoad"}
V1_CORPUS_PROBE = 0.9161426767221054   # Results/all3/diagnostic.json
N_SPLITS = 5


def riemannian(Xf, groups):
    """
    Per-band covariance -> per-group re-centering -> tangent space. (N, 105).

    Same construction as scripts/all3_diagnostic.py:61, but fed the zero-phase filtered
    signal. Note this view was already immune to the 45-64 Hz gap (it band-filters first);
    what changes here is only lfilter -> sosfiltfilt.
    """
    out = []
    for lo, hi in BANDS.values():
        covs = Covariances(estimator="oas").fit_transform(bandpass(Xf, lo, hi))
        rc = np.empty_like(covs)
        for g in np.unique(groups):
            m = groups == g
            W = invsqrtm(mean_riemann(covs[m]))
            rc[m] = W @ covs[m] @ W
        out.append(TangentSpace(metric="riemann").fit_transform(rc))
    return np.hstack(out).astype(np.float32)


def corpus_probe(F, corpus, subjects):
    """How recoverable is the recording device from these features? v1 scores 0.9161."""
    return float(cross_val_score(
        make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", class_weight="balanced")),
        F, corpus, cv=GroupKFold(n_splits=N_SPLITS), groups=subjects,
        scoring="accuracy", n_jobs=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="first 500 segments only; validates the pipeline, not the science")
    ap.add_argument("--skip-probe", action="store_true", help="skip the corpus probe")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    t_start = time.perf_counter()

    X = np.load(os.path.join(SRC, "X.npy"))
    y = np.load(os.path.join(SRC, "y.npy")).astype(int)
    corpus = np.load(os.path.join(SRC, "corpus.npy")).astype(int)
    subjects = np.load(os.path.join(SRC, "subjects.npy"), allow_pickle=True)

    if args.smoke:
        keep = np.sort(np.concatenate([np.where(corpus == c)[0][:200]
                                       for c in np.unique(corpus)]))
        X, y, corpus, subjects = X[keep], y[keep], corpus[keep], subjects[keep]
        print(f"SMOKE MODE — {len(X)} segments, results are not meaningful\n")

    print(f"loaded {X.shape} from datasets/all3/")

    # --- 1. preprocessing, with the before/after that proves it worked ------------------
    print(f"\nbandpass {TOTAL_BAND[0]}-{TOTAL_BAND[1]} Hz, zero-phase (sosfiltfilt)")
    print(f"  {'corpus':10s} {'>45Hz before':>14s} {'after':>10s}")
    before = {}
    for c in np.unique(corpus):
        before[int(c)] = gap_power_fraction(X[corpus == c][:1500])

    t0 = time.perf_counter()
    Xf = preprocess(X)
    filt_sec = time.perf_counter() - t0

    after = {}
    for c in np.unique(corpus):
        after[int(c)] = gap_power_fraction(Xf[corpus == c][:1500])
        print(f"  {CORPUS_NAMES[int(c)]:10s} {before[int(c)]:13.3f} {after[int(c)]:10.5f}")

    worst = max(after.values())
    assert worst < 0.02, (f"bandpass failed — {worst:.4f} of power still above 45 Hz. "
                          "Features would still carry the mains signature.")
    print(f"  filtered in {filt_sec:.1f}s — gap content removed")

    # --- 2. blocks ----------------------------------------------------------------------
    print(f"\nextracting {len(BLOCKS)} blocks")
    t0 = time.perf_counter()
    feats = extract_all_features_v2(Xf, preprocessed=True, verbose=True)
    blocks_sec = time.perf_counter() - t0

    index, combined, names_all = {}, [], []
    for name, F in feats.items():
        cols = block_names(name)
        assert F.shape == (len(X), len(cols)), f"{name}: {F.shape} vs {len(cols)} names"
        assert np.isfinite(F).all(), f"{name}: non-finite values"
        np.save(os.path.join(OUT, f"{name}.npy"), F)
        index[name] = {"file": f"{name}.npy", "dim": int(F.shape[1]),
                       "cols": [len(names_all), len(names_all) + len(cols)], "names": cols}
        combined.append(F)
        names_all += cols
        print(f"    {name:16s} {F.shape[1]:4d} cols")

    C = np.hstack(combined).astype(np.float32)
    np.save(os.path.join(OUT, "combined.npy"), C)
    print(f"  combined         {C.shape[1]:4d} cols  -> combined.npy")

    # --- 3. Riemannian views -------------------------------------------------------------
    print("\nriemannian tangent space (zero-phase filtered)")
    t0 = time.perf_counter()
    for kind, groups in (("percorpus", corpus), ("global", np.zeros(len(X), int))):
        R = riemannian(Xf, groups)
        np.save(os.path.join(OUT, f"riemannian_{kind}.npy"), R)
        assert np.isfinite(R).all(), f"riemannian_{kind}: non-finite values"
        index[f"riemannian_{kind}"] = {"file": f"riemannian_{kind}.npy", "dim": int(R.shape[1]),
                                       "cols": None, "names": None}
        print(f"    riemannian_{kind:10s} {R.shape[1]:4d} cols")
    riem_sec = time.perf_counter() - t0

    # --- 4. labels + manifest -------------------------------------------------------------
    for nm, arr in (("y", y), ("corpus", corpus), ("subjects", subjects)):
        np.save(os.path.join(OUT, f"{nm}.npy"), arr)

    meta = {
        "version": 2,
        "n_segments": int(len(X)),
        "n_handcrafted": int(C.shape[1]),
        "source": "datasets/all3/",
        "smoke_mode": bool(args.smoke),
        "preprocessing": {
            "bandpass_hz": list(TOTAL_BAND),
            "filter": "butterworth order 4, SOS, zero-phase (sosfiltfilt)",
            "renormalised_after_filtering": False,
            "gap_power_above_45hz": {CORPUS_NAMES[c]: {"before": before[c], "after": after[c]}
                                     for c in before},
        },
        "blocks": index,
        "combined_column_names": names_all,
        "classes": {"0": "healthy", "1": "stress", "2": "depression"},
        "vs_v1": ("v1 (features/all3/, 81 cols) divides band power by total power over "
                  "0-64 Hz while bands span 0.5-45 Hz. v2 bandpasses to 0.5-45 first and "
                  "uses a matching denominator. v1 is unmodified and still reproducible."),
        "NOTE": ("3-class accuracy is only interpretable against the corpus-ID shortcut "
                 "0.6267, never against chance. Always split on subjects."),
    }

    # --- 5. the payoff: is the device still recoverable? -----------------------------------
    if not args.skip_probe:
        print(f"\ncorpus probe — can a model still identify the recording device?")
        t0 = time.perf_counter()

        # PAIRED control. v1 is re-measured on the SAME segments rather than compared to the
        # stored full-dataset 0.9161: probe accuracy depends strongly on sample and subject
        # count, so an unpaired delta overstates the effect (measured at -0.207 unpaired vs
        # -0.124 paired on a 600-segment smoke run). Only the paired delta is a real result.
        from proposed_model.features import extract_all_features
        v1_here = corpus_probe(extract_all_features(X), corpus, subjects)
        v2_here = corpus_probe(C, corpus, subjects)
        delta = v2_here - v1_here

        meta["corpus_probe"] = {
            "v1_same_segments": v1_here, "v2_same_segments": v2_here, "delta_paired": delta,
            "v1_full_dataset_reference": V1_CORPUS_PROBE,
            "paired": True, "n_segments": int(len(X)), "n_subjects": int(len(set(subjects))),
        }
        print(f"  paired on the same {len(X)} segments / {len(set(subjects))} subjects:")
        print(f"    v1   81 features, unfiltered  {v1_here:.4f}")
        print(f"    v2  {C.shape[1]:3d} features, filtered    {v2_here:.4f}   "
              f"delta {delta:+.4f}")
        print(f"  (reference: v1 on the full 15728-segment set = {V1_CORPUS_PROBE:.4f})")
        if delta < -0.05:
            print("  -> the spectral defect was a real driver of the corpus confound.\n"
                  "     Every all3 conclusion should be revisited on v2 features.")
        else:
            print("  -> device signature survives the fix; it does not live in the\n"
                  "     45-64 Hz gap. The confound has another source.")
        print(f"  ({time.perf_counter() - t0:.0f}s)")

    meta["elapsed_sec"] = {"filter": round(filt_sec, 1), "blocks": round(blocks_sec, 1),
                           "riemannian": round(riem_sec, 1),
                           "total": round(time.perf_counter() - t_start, 1)}
    with open(os.path.join(OUT, "features.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {OUT}  ({C.shape[0]} x {C.shape[1]} handcrafted + 2 x 105 riemannian)")
    print(f"total {meta['elapsed_sec']['total']:.0f}s")
    if args.smoke:
        print("\nSMOKE MODE — rerun without --smoke for the real matrices.")


if __name__ == "__main__":
    main()
