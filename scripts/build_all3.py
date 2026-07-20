"""
Build datasets/all3/ — a single 3-class dataset (healthy / stress / depression).

`datasets/combined_data/` carries two *separate* binary label arrays with a -1
"not applicable" sentinel, and zero of its 55,157 segments carry both labels. That
structure cannot express a 3-class target. This derives one.

Derived from combined_data/*.npy rather than the raw corpora: those segments are
already windowed, resampled to 128 Hz and normalised identically by
scripts/combine_datasets.py, so re-reading ~14 GB of .mat/.txt would spend ~15
minutes producing byte-identical output.

No label *value* is changed. The two original binary arrays are carried through, so
the 3-class label is derived rather than destructive and the binary tasks remain
reproducible from this folder alone.

READ THE CONFOUND WARNING in the generated metadata.json before using this for
anything. Class is heavily confounded with recording corpus, and a classifier that
knows only which corpus a segment came from — with zero EEG understanding — scores
0.6267. Any 3-class accuracy at or below that demonstrates nothing.

Run:  .venv/bin/python scripts/build_all3.py
"""
import json
import os

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
OUT = os.path.join(PROJECT_ROOT, "datasets", "all3")

MODMA, SAM40, COGLOAD, AMIGOS = 0, 1, 2, 3
TASK_CORPORA = (SAM40, COGLOAD)
CLASS_NAMES = {0: "healthy", 1: "stress", 2: "depression"}
CORPUS_NAMES = {MODMA: "MODMA", SAM40: "SAM40", COGLOAD: "CogLoad", AMIGOS: "AMIGOS"}


def derive_labels(corpus, y_dep, y_str):
    """
    healthy(0)    = MODMA healthy controls + task-corpus relax segments
    stress(1)     = task-corpus active segments
    depression(2) = MODMA MDD

    healthy is deliberately drawn from ALL corpora: it is the only class that spans
    them, and therefore the only thing preventing corpus identity from being a
    complete answer to the 3-class question.
    """
    y = np.full(len(corpus), -1, dtype=np.int8)
    y[(corpus == MODMA) & (y_dep == 0)] = 0
    y[np.isin(corpus, TASK_CORPORA) & (y_str == 0)] = 0
    y[np.isin(corpus, TASK_CORPORA) & (y_str == 1)] = 1
    y[(corpus == MODMA) & (y_dep == 1)] = 2
    return y


def corpus_shortcut_accuracy(y, corpus):
    """
    Accuracy of the best constant prediction *within each corpus* — i.e. what a model
    scores knowing only which device recorded the segment, with no EEG understanding.
    This is the real baseline, not chance and not the majority class.
    """
    correct = sum(np.bincount(y[corpus == c]).max()
                  for c in np.unique(corpus) if (corpus == c).any())
    return float(correct / len(y))


def main():
    os.makedirs(OUT, exist_ok=True)

    X = np.load(os.path.join(SRC, "X.npy"))
    y_dep = np.load(os.path.join(SRC, "y_depression.npy"))
    y_str = np.load(os.path.join(SRC, "y_stress.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy"), allow_pickle=True)
    corpus = np.load(os.path.join(SRC, "datasets.npy"))
    n_total = len(X)
    print(f"source: {n_total} segments from combined_data/")

    y = derive_labels(corpus, y_dep, y_str)
    labelled = y != -1
    unlabelled = corpus == AMIGOS

    # The labelled set and the AMIGOS set must exactly partition the source.
    assert not (labelled & unlabelled).any(), "AMIGOS leaked into the labelled set"
    assert (labelled | unlabelled).all(), "some segments are neither labelled nor AMIGOS"
    assert set(np.unique(corpus[labelled])) == {MODMA, SAM40, COGLOAD}

    Xl, yl, sl, cl = X[labelled], y[labelled], subjects[labelled], corpus[labelled]
    shortcut = corpus_shortcut_accuracy(yl, cl)

    print(f"\nlabelled: {labelled.sum()}   unlabelled (AMIGOS): {unlabelled.sum()}")
    print(f"\n{'class':12s} {'n':>6s}  corpus breakdown")
    composition = {}
    for k, name in CLASS_NAMES.items():
        m = yl == k
        brk = {CORPUS_NAMES[c]: int((m & (cl == c)).sum())
               for c in np.unique(cl) if (m & (cl == c)).any()}
        composition[name] = {"n": int(m.sum()), "by_corpus": brk,
                             "n_subjects": int(len(set(sl[m])))}
        print(f"{name:12s} {m.sum():6d}  " + " · ".join(f"{a} {b}" for a, b in brk.items()))

    np.save(os.path.join(OUT, "X.npy"), Xl)
    np.save(os.path.join(OUT, "y.npy"), yl)
    np.save(os.path.join(OUT, "subjects.npy"), sl)
    np.save(os.path.join(OUT, "corpus.npy"), cl)
    # Originals carried through unchanged so the binary tasks stay reproducible here.
    np.save(os.path.join(OUT, "y_depression.npy"), y_dep[labelled])
    np.save(os.path.join(OUT, "y_stress.npy"), y_str[labelled])
    np.save(os.path.join(OUT, "X_unlabelled.npy"), X[unlabelled])

    meta = {
        "WARNING": (
            "Class is heavily confounded with recording corpus. MODMA contains only "
            "{healthy, depression}; SAM40/CogLoad contain only {healthy, stress}. Each "
            "used a different amplifier, so corpus identity is trivially recoverable "
            "and collapses this 3-class problem to 2 classes for free. A classifier "
            f"knowing ONLY the corpus, with zero EEG understanding, scores {shortcut:.4f}. "
            "Always report 3-class accuracy against that baseline, never alone. Prefer "
            "the within-corpus accuracies from scripts/all3_diagnostic.py."
        ),
        "classes": {str(k): v for k, v in CLASS_NAMES.items()},
        "n_labelled": int(labelled.sum()),
        "n_unlabelled_amigos": int(unlabelled.sum()),
        "composition": composition,
        "baselines": {
            "chance": 1 / 3,
            "majority_class": float(np.bincount(yl).max() / len(yl)),
            "corpus_id_only": shortcut,
        },
        "n_subjects_total": int(len(set(sl))),
        "n_subjects_spanning_multiple_classes": int(sum(
            len(np.unique(yl[sl == s])) > 1 for s in np.unique(sl))),
        "derived_from": "datasets/combined_data/ (segments unchanged, labels remapped only)",
    }
    with open(os.path.join(OUT, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    b = meta["baselines"]
    print(f"\nbaselines:  chance {b['chance']:.4f}   majority {b['majority_class']:.4f}"
          f"   CORPUS-ID-ONLY {b['corpus_id_only']:.4f}  <-- beat this, not the others")
    print(f"\nwritten to {OUT}/")


if __name__ == "__main__":
    main()
