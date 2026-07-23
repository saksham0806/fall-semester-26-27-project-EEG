"""
Extract the feature matrices for datasets/all3/ into features/all3/, ready for ML.

Two views, both already defined in this repo — nothing new is invented here:
  handcrafted (81) — proposed_model.features.extract_all_features
  riemannian (105) — per-band covariance -> per-corpus re-centering -> tangent space
                     (all3_diagnostic.riemannian_features; reused from the existing
                      datasets/all3/_cache_riemann_*.npy when present)

Labels/corpus/subjects are copied alongside so features/all3/ is self-contained.

Run:  .venv/bin/python -u scripts/extract_all3_features.py
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from proposed_model.features import (ASYMMETRY_PAIRS, BANDS, CHANNEL_MAP,
                                     extract_all_features)

SRC = os.path.join(PROJECT_ROOT, "datasets", "all3")
OUT = os.path.join(PROJECT_ROOT, "features", "all3")
CHANNELS = list(CHANNEL_MAP)


def handcrafted_names():
    """Must match the hstack order in extract_all_features()."""
    names = [f"relpsd_{b}_{c}" for b in BANDS for c in CHANNELS]
    names += [f"de_{b}_{c}" for b in BANDS for c in CHANNELS]
    names += [f"faa_{r}_{l}" for r, l in ASYMMETRY_PAIRS]
    names += [f"specent_{c}" for c in CHANNELS]
    names += [f"std_{c}" for c in CHANNELS]
    names += [f"rms_{c}" for c in CHANNELS]
    return names


def riemann(kind, X, corpus):
    """Reuse the cache all3_models.py already wrote; compute only if missing."""
    cache = os.path.join(SRC, f"_cache_riemann_{kind}.npy")
    if os.path.exists(cache):
        print(f"  riemannian/{kind}: reusing {os.path.relpath(cache, PROJECT_ROOT)}")
        return np.load(cache)
    from all3_diagnostic import riemannian_features
    print(f"  riemannian/{kind}: computing (slow)")
    # global = one re-centering group for every segment
    F = riemannian_features(X, corpus if kind == "percorpus" else np.zeros(len(X), int))
    np.save(cache, F)
    return F


def main():
    os.makedirs(OUT, exist_ok=True)
    X = np.load(os.path.join(SRC, "X.npy"))
    y = np.load(os.path.join(SRC, "y.npy"))
    corpus = np.load(os.path.join(SRC, "corpus.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy"), allow_pickle=True)
    print(f"loaded {X.shape} segments")

    names = handcrafted_names()
    print("  handcrafted: computing")
    H = extract_all_features(X, fs=128).astype(np.float32)
    assert H.shape == (len(X), len(names)), (H.shape, len(names))
    assert np.isfinite(H).all(), "non-finite handcrafted features"
    np.save(os.path.join(OUT, "handcrafted.npy"), H)

    views = {"handcrafted": {"file": "handcrafted.npy", "dim": H.shape[1],
                             "names": names}}
    for kind in ("percorpus", "global"):
        R = riemann(kind, X, corpus).astype(np.float32)
        assert np.isfinite(R).all(), f"non-finite riemannian/{kind}"
        np.save(os.path.join(OUT, f"riemannian_{kind}.npy"), R)
        views[f"riemannian_{kind}"] = {"file": f"riemannian_{kind}.npy",
                                       "dim": R.shape[1]}

    for name, arr in (("y", y), ("corpus", corpus), ("subjects", subjects)):
        np.save(os.path.join(OUT, f"{name}.npy"), arr)

    with open(os.path.join(OUT, "features.json"), "w") as f:
        json.dump({
            "n_segments": int(len(X)),
            "source": "datasets/all3/",
            "fs": 128, "channels": CHANNELS, "window_samples": 512,
            "views": views,
            "classes": {"0": "healthy", "1": "stress", "2": "depression"},
            "baseline_to_beat": 0.6267,
            "NOTE": ("Class is confounded with corpus; 3-class accuracy means nothing "
                     "except against corpus_id_only = 0.6267. riemannian_percorpus has "
                     "the device signature removed and drops to chance — that is the "
                     "honest view, not a bug. See datasets/all3/README.md."),
        }, f, indent=2)

    for k, v in views.items():
        print(f"  {k}: ({len(X)}, {v['dim']})")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
