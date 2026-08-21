"""
Plan 01 features: gain-corrected spectral + graph-reduced connectivity.

Parallel view alongside data.py's AB/COH/DEMO -- those are NEVER modified, same discipline
as features.py vs features_v2.py in src/proposed_model/.

    .venv/bin/python -m OSF_Park_model.features_corrected            # self-check
    .venv/bin/python -m OSF_Park_model.features_corrected --build    # build + cache graphs

| block  | cols | what                                                        |
|--------|-----:|-------------------------------------------------------------|
| CLR    |  114 | centered log-ratio of band power, per channel                |
| LOGABS |  114 | log10 absolute power (secondary view)                        |
| FAA    |    3 | frontal alpha asymmetry, FP2/FP1 F4/F3 F8/F7                 |
| TBR    |   19 | log theta/beta ratio per channel                             |
| GRAPH  |  138 | 6 bands x (19 nodal strengths + clustering, L, E_glob, Q)    |
"""
import argparse
import hashlib
import json
import os
import time

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree

from .data import BANDS, DROP_NO, ROOT, cohort

CACHE = os.path.join(ROOT, "features/osf_park")
DENSITY = 0.20          # fixed, NOT swept -- see plans/01 section 4
EPS = 1e-12


# ------------------------------------------------------------------ spectral transforms

def clr(AB):
    """Centered log-ratio of the 6 band powers within each channel.

    Removes the per-channel gain factor EXACTLY: scale one channel's six bands by any k and
    log(kP) - mean(log(kP)) is unchanged.  That factor is skull thickness / impedance /
    electrode contact -- the nuisance that dominates absolute uV^2.

    Relative power must be computed on LINEAR power and transformed afterwards; taking
    ratios of already-logged values is a different (and wrong) quantity.
    """
    AB = np.asarray(AB, np.float64)
    assert AB.min() > 0, "band power must be strictly positive for a log-ratio"
    P = np.log(AB.reshape(len(AB), 6, 19).transpose(0, 2, 1))         # (n, 19, 6) band-major
    C = P - P.mean(axis=-1, keepdims=True)
    return C.transpose(0, 2, 1).reshape(len(AB), 114)                 # back to band-major


def logabs(AB):
    return np.log10(np.asarray(AB, np.float64))


def _band(AB, band):
    """(n, 19) linear power for one band."""
    return np.asarray(AB, np.float64).reshape(len(AB), 6, 19)[:, BANDS.index(band), :]


def faa(AB, channels):
    """Frontal alpha asymmetry: log(right) - log(left).

    Computed on log ABSOLUTE alpha, not on CLR.  Asymmetry is a between-channel contrast and
    CLR is a within-channel normalisation -- taking FAA on CLR would cancel the effect being
    measured.
    """
    a = np.log(_band(AB, "alpha"))
    out = [a[:, channels.index(r)] - a[:, channels.index(l)]
           for l, r in (("FP1", "FP2"), ("F3", "F4"), ("F7", "F8"))]
    return np.stack(out, 1)


def tbr(AB):
    """log theta/beta ratio per channel -- ratio on linear power, then logged."""
    return np.log(_band(AB, "theta") / _band(AB, "beta"))


def fisher_z(COH):
    """arctanh(COH/100) with a MANDATORY clip.

    24 cells in this file are exactly 100.0 and arctanh(1.0) is inf.  One inf destroys the
    fold's mean and std for every subject -- silently, since sklearn fits on it happily.
    """
    Z = np.arctanh(np.clip(COH / 100.0, 0.0, 1.0 - 1e-6))
    assert np.isfinite(Z).all(), "non-finite after Fisher-z -- the clip is not working"
    return Z


# ---------------------------------------------------------------------- graph reduction

def _threshold(W, d):
    """MST-union thresholding: minimum spanning tree of the distance graph, unioned with the
    top-d fraction of edges by weight.

    Plain proportional thresholding DISCONNECTS these graphs -- measured on all 527 subjects,
    94% of subject-band graphs fragment at d=0.20 and 22% still fragment at d=0.50, because
    coherence has hub structure and the strongest edges cluster.  A fragmented graph has
    infinite characteristic path length, so the metric is undefined for that subject.

    The MST (18 edges on 19 nodes) is computed on distance = 1/w, so it keeps the STRONGEST
    connecting edges, and guarantees connectivity by construction.  Standard practice in
    network neuroscience for exactly this reason.  Deterministic; no seed needed.
    """
    iu = np.triu_indices(19, 1)
    w = W[iu]
    k = max(1, int(round(d * len(w))))
    cut = np.partition(w, -k)[-k]
    T = np.where(W >= cut, W, 0.0)
    np.fill_diagonal(T, 0.0)

    dist = np.where(W > 0, 1.0 / np.maximum(W, EPS), 0.0)
    mst = minimum_spanning_tree(csr_matrix(np.triu(dist, 1))).toarray()
    ii, jj = np.nonzero(mst)
    T[ii, jj] = W[ii, jj]
    T[jj, ii] = W[jj, ii]
    return T


def _one_graph(W):
    """23 metrics for one 19x19 weighted matrix: 19 strengths + C, L, E_glob, Q."""
    strength = W.sum(1)
    D = np.where(W > 0, 1.0 / np.maximum(W, EPS), np.inf)   # distance = inverse weight
    np.fill_diagonal(D, 0.0)
    sp = dijkstra(np.where(np.isinf(D), 0.0, D), directed=False, unweighted=False)
    off = ~np.eye(19, dtype=bool)
    finite = np.isfinite(sp) & off
    L = sp[finite].mean() if finite.any() else np.inf
    Eg = (1.0 / sp[off][np.isfinite(sp[off])]).sum() / (19 * 18)
    G = nx.from_numpy_array(W)
    C = nx.average_clustering(G, weight="weight")
    try:
        comm = nx.community.greedy_modularity_communities(G, weight="weight")
        Q = nx.community.modularity(G, comm, weight="weight")
    except Exception:
        Q = 0.0
    return np.concatenate([strength, [C, L, Eg, Q]]), np.isfinite(sp[off]).all()


def graph_features(cohmat, d=DENSITY, verbose=False):
    """(n, 6, 19, 19) -> (n, 138).  Also reports how many subject-band graphs fragmented."""
    n = len(cohmat)
    out = np.zeros((n, 6, 23), np.float64)
    frag = 0
    for i in range(n):
        for b in range(6):
            v, ok = _one_graph(_threshold(cohmat[i, b], d))
            out[i, b] = v
            frag += not ok
        if verbose and (i + 1) % 100 == 0:
            print(f"    graphs {i + 1}/{n}", flush=True)
    return out.reshape(n, 138), frag


def _key(c, d):
    """Key on the COHORT'S ACTUAL ROWS, not just density.

    Keying on density + DROP_NO alone was a bug: cohort("binary") (292 rows) and ill_cohort()
    (850 rows) both hit the 4-class 527-row cache and crashed on concatenate.  A shape
    mismatch is the lucky failure -- a same-shaped cohort would have loaded silently wrong.
    """
    ids = ",".join(str(int(v)) for v in c.df["no."])
    return hashlib.md5(f"{d}|mstunion|{ids}".encode()).hexdigest()[:10]


def _cache_meta(c, d=DENSITY):
    return os.path.join(CACHE, f"graph_{_key(c, d)}.json")


def cached_graph(c, d=DENSITY, rebuild=False):
    """Deterministic given raw COH -- cache it.  STALE-CACHE HAZARD: the filename encodes
    density and DROP_NO, so changing the METRIC SET means deleting the file by hand."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"graph_{_key(c, d)}.npy")
    if os.path.exists(p) and not rebuild:
        g = np.load(p)
        assert len(g) == len(c), f"cache {p} has {len(g)} rows, cohort has {len(c)}"
        return g
    t0 = time.time()
    print(f"  building graph features (d={d}, n={len(c)}) ...", flush=True)
    g, frag = graph_features(c.coh_matrix() / 100.0, d, verbose=True)
    if frag:
        print(f"  WARNING: {frag} subject-band graphs fragmented at d={d}; "
              f"path length is inf for those. Raise the density.")
    np.save(p, g)
    json.dump({"density": d, "drop_no": sorted(DROP_NO), "n": len(c), "fragmented": int(frag),
               "first_no": int(c.df["no."].iloc[0]), "last_no": int(c.df["no."].iloc[-1]),
               "cols": 138, "seconds": round(time.time() - t0, 1)},
              open(p.replace(".npy", ".json"), "w"), indent=1)
    print(f"  cached {p}  ({time.time() - t0:.0f}s, {frag} fragmented)")
    return g


# ------------------------------------------------------------------------------- blocks

def blocks(c, rebuild=False):
    """All plan-01 feature blocks, unscaled.  Standardize inside the fold."""
    return {"CLR": clr(c.AB), "LOGABS": logabs(c.AB), "FAA": faa(c.AB, c.channels),
            "TBR": tbr(c.AB), "GRAPH": cached_graph(c, rebuild=rebuild),
            "DEMO": c.DEMO.astype(np.float64), "COHZ": fisher_z(c.COH)}


VIEWS = {
    "DEMO":          ["DEMO"],
    "SPECTRAL":      ["CLR", "FAA", "TBR"],
    "GRAPH":         ["GRAPH"],
    "EEG":           ["CLR", "FAA", "TBR", "GRAPH"],
    "EEG+DEMO":      ["CLR", "FAA", "TBR", "GRAPH", "DEMO"],
    "RAW_BASELINE":  ["LOGABS", "COHZ"],
}


def view(B, name):
    return np.concatenate([B[k] for k in VIEWS[name]], 1)


def _selfcheck():
    c = cohort("4class")
    B = blocks(c)
    n = len(c)
    assert B["CLR"].shape == (n, 114) and B["LOGABS"].shape == (n, 114)
    assert B["FAA"].shape == (n, 3) and B["TBR"].shape == (n, 19)
    assert B["GRAPH"].shape == (n, 138) and B["COHZ"].shape == (n, 1026)
    for k, v in B.items():
        assert np.isfinite(v).all(), f"{k} non-finite"

    # CLR sums to zero within each channel
    C = B["CLR"].reshape(n, 6, 19).transpose(0, 2, 1)
    assert np.abs(C.sum(-1)).max() < 1e-5, "CLR does not sum to zero per channel"

    # CLR is invariant to per-channel rescaling -- the whole point of the transform
    AB2 = c.AB.astype(np.float64).reshape(n, 6, 19).copy()
    AB2[:, :, 7] *= 7.3                                  # scale channel T3's six bands
    assert np.allclose(clr(AB2.reshape(n, 114)), B["CLR"], atol=1e-10), "CLR is not gain-invariant"
    # ... and log-absolute power is NOT invariant, confirming the two views differ
    assert not np.allclose(logabs(AB2.reshape(n, 114)), B["LOGABS"])

    # Fisher-z clip
    assert fisher_z(np.array([[100.0, 0.0]])).max() < 20 and np.isfinite(fisher_z(np.array([[100.0]]))).all()

    # graph block: strengths positive, no fragmentation
    S = B["GRAPH"].reshape(n, 6, 23)
    assert (S[:, :, :19] > 0).all(), "a nodal strength is <= 0"
    assert np.isfinite(S[:, :, 20]).all(), "characteristic path length is inf -- graph fragmented"
    assert json.load(open(_cache_meta(c)))["fragmented"] == 0, "MST-union failed to guarantee connectivity"
    from .data import cohort as _co, ill_cohort as _ic
    for other in (_co("binary"), _ic()):
        g = cached_graph(other)
        assert len(g) == len(other), f"cache leaked across cohorts: {len(g)} vs {len(other)}"
        assert _key(other, DENSITY) != _key(c, DENSITY), "cache key is not cohort-specific"
    print(f"  cache keys distinct across cohorts (4class/binary/ill: "
          f"{_key(c, DENSITY)} {_key(_co('binary'), DENSITY)} {_key(_ic(), DENSITY)})")
    for name in VIEWS:
        v = view(B, name)
        print(f"  view {name:14s} p={v.shape[1]:5d}")
    assert view(B, "EEG").shape[1] == 274 and view(B, "EEG+DEMO").shape[1] == 278
    print(f"n={n}  CLR skew {np.median(np.abs(_skew(B['CLR']))):.2f} "
          f"vs LOGABS {np.median(np.abs(_skew(B['LOGABS']))):.2f} "
          f"vs raw {np.median(np.abs(_skew(c.AB))):.2f}")
    print("all checks passed")


def _skew(X):
    m, s = X.mean(0), X.std(0) + EPS
    return (((X - m) / s) ** 3).mean(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="force rebuild of the graph cache")
    a = ap.parse_args()
    if a.build:
        blocks(cohort("4class"), rebuild=True)
    _selfcheck()
