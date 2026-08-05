"""
datasets/all3 loader for the raw-EEG track. Pure numpy + sklearn — no torch, no TensorFlow.

Deliberately does NOT import `src/`: `src/proposed_model/__init__.py` pulls in the entire
TensorFlow stack, and this track is torch-only.

Self-check:  .venv/bin/python -m DL_Without_Features.data
"""
import os

import numpy as np
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(PROJECT_ROOT, "datasets", "all3")
RESULTS = os.path.join(PROJECT_ROOT, "Results", "DL_Without_Features")

N_SPLITS = 5
FS = 128
CLASS_NAMES = {0: "healthy", 1: "stress", 2: "depression"}
CORPUS_NAMES = {0: "MODMA", 1: "SAM40", 2: "CogLoad"}

# Which two of the three classes each corpus can actually supply. This is the whole problem:
# no corpus holds all three, so `depression` is a boolean identity with `corpus == MODMA`.
CORPUS_CLASSES = {0: (0, 2), 1: (0, 1), 2: (0, 1)}


def load():
    """X (n, 6, 512) float32, y int, subjects, corpus. Already global-per-segment z-scored."""
    return (np.load(os.path.join(DATA, "X.npy")),
            np.load(os.path.join(DATA, "y.npy")).astype(int),
            np.load(os.path.join(DATA, "subjects.npy"), allow_pickle=True),
            np.load(os.path.join(DATA, "corpus.npy")).astype(int))


def normalize(X, subjects, mode):
    """
    "segment" — pass through. X.npy is already z-scored globally per 4 s window by
                combine_datasets.py:25. This is the control arm.
    "subject" — per-channel standardization *within each subject* on top of that, which
                removes residual per-amplifier gain and offset. Label-free and computed
                inside one subject, so it needs no train/test fit and applies unchanged to
                a new recording at inference time.
    """
    if mode == "segment":
        return X
    if mode != "subject":
        raise ValueError(f"unknown norm mode {mode!r}, expected 'segment' or 'subject'")
    out = np.empty_like(X)
    for s in np.unique(subjects):
        m = subjects == s
        seg = X[m]                                        # (k, 6, 512)
        mu = seg.mean(axis=(0, 2), keepdims=True)         # (1, 6, 1) — per channel
        sd = seg.std(axis=(0, 2), keepdims=True)
        out[m] = (seg - mu) / np.maximum(sd, 1e-8)
    return out


def folds(y, subjects, n_splits=N_SPLITS):
    """
    Subject-grouped, always. Never StratifiedKFold: segments are 4 s windows at 50% overlap,
    so a random split puts overlapping windows of the same person in train and test and the
    same model jumps from 0.6256 to 0.9393 on pure leakage.
    """
    for tr, te in GroupKFold(n_splits=n_splits).split(y, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        yield tr, te


def shortcut_baseline(y, corpus):
    """Accuracy of a model that knows ONLY the recording device. The bar to beat: 0.6267."""
    return float(sum(np.bincount(y[corpus == c]).max() for c in np.unique(corpus)) / len(y))


def subsample(X, y, subjects, corpus, per_corpus=6, per_subject=64, seed=42):
    """--smoke slice: a few whole subjects per corpus, capped in segments. Keeps every fold
    subject-disjoint and every class present, small enough to run on local CPU."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(corpus):
        for s in rng.permutation(np.unique(subjects[corpus == c]))[:per_corpus]:
            idx = np.flatnonzero(subjects == s)
            keep.append(rng.permutation(idx)[:per_subject])
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep], subjects[keep], corpus[keep]


def _selfcheck():
    X, y, subjects, corpus = load()
    assert X.ndim == 3 and X.shape[1:] == (6, 512), X.shape
    assert len(X) == len(y) == len(subjects) == len(corpus)
    assert set(np.unique(y)) == {0, 1, 2}
    assert not np.isnan(X).any()

    sc = shortcut_baseline(y, corpus)
    assert abs(sc - 0.6267) < 1e-3, f"shortcut moved: {sc:.4f} (expected 0.6267)"

    # depression is EXACTLY corpus == MODMA — assert the identity rather than trusting it
    assert set(np.unique(corpus[y == 2])) == {0} and not (y[corpus != 0] == 2).any()

    for c, classes in CORPUS_CLASSES.items():
        assert set(np.unique(y[corpus == c])) == set(classes), c

    Xs = normalize(X[:2000], subjects[:2000], "subject")
    for s in np.unique(subjects[:2000]):
        m = subjects[:2000] == s
        assert np.abs(Xs[m].mean(axis=(0, 2))).max() < 1e-4
        assert np.abs(Xs[m].std(axis=(0, 2)) - 1).max() < 1e-3
    head = X[:10]
    assert normalize(head, subjects[:10], "segment") is head

    n = 0
    for tr, te in folds(y, subjects):
        n += len(te)
        assert set(np.unique(y[tr])) == {0, 1, 2}, "a fold lost a class"
    assert n == len(y), "folds do not partition the data"

    Xa, ya, sa, ca = subsample(X, y, subjects, corpus)
    assert set(np.unique(ya)) == {0, 1, 2} and len(np.unique(sa)) == 18

    print(f"data.py OK — {len(y)} segments, {len(np.unique(subjects))} subjects, "
          f"shortcut {sc:.4f}, smoke slice {len(ya)} segments / {len(np.unique(sa))} subjects")


if __name__ == "__main__":
    _selfcheck()
