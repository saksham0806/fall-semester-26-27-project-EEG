"""
Train deployable healthy / stressed / depressed classifiers and serialise them.

Three approaches, trained and evaluated identically so they can be compared, then refit on
ALL data and written to models/ as self-contained joblib artifacts that predict.py loads.

  A  binary-pair    depression model (MODMA) + stress model (SAM40+CogLoad) run in parallel;
                    healthy = neither fires. These two contrasts are the project's only
                    scientifically sound results (0.7716 / 0.7560 AUC) because each is a
                    WITHIN-corpus comparison rather than a between-device one.
  B  flat-3class    one SVM, 81 features, three classes. Simplest to deploy. Its CV score sits
                    BELOW the corpus-ID baseline, so much of it is device signature.
  C  hierarchical   stage 1 predicts recording CONTEXT (resting vs task) instead of corpus
                    identity, so the router means something for a new recording;
                    stage 2 routes to the matching binary specialist.

READ THIS BEFORE QUOTING ANY NUMBER THIS SCRIPT PRINTS.
Every score here is 5-fold SUBJECT-grouped CV — subject-disjoint but NOT device-disjoint. All
training data comes from three amplifiers, corpus identity is ~97% recoverable from the
features, and removing the device signature collapses 3-class accuracy to 0.3255 (chance).
So these numbers are an UPPER BOUND for new hardware, not a prediction. Approach A is the most
likely to survive a device change because both its models are within-corpus contrasts;
B and C both lean on the between-corpus cue to some degree.

Run:  .venv/bin/python -u scripts/train_deployable.py
"""
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from proposed_model.features import extract_all_features

DATA = os.path.join(PROJECT_ROOT, "datasets", "all3")
MODELS = os.path.join(PROJECT_ROOT, "models")
CLASSES = {0: "healthy", 1: "stressed", 2: "depressed"}
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "F4", "F8"]
FS, WINDOW = 128, 512
N_SPLITS = 5


def svm_proba():
    """
    SVC + probability calibration. SVC(probability=True) is deprecated in sklearn 1.9.

    The internal CV must be SHUFFLED: the default cv=3 is a non-shuffled KFold and X.npy is
    ordered by corpus, so the folds land on three different amplifiers and the calibrator
    learns an inverted mapping (measured: AUC 0.4410 vs 0.6527 shuffled).
    """
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", class_weight="balanced"),
                               ensemble=False,
                               cv=StratifiedKFold(3, shuffle=True, random_state=42)))


def load():
    return (np.load(os.path.join(DATA, "X.npy")),
            np.load(os.path.join(DATA, "y.npy")).astype(int),
            np.load(os.path.join(DATA, "subjects.npy"), allow_pickle=True),
            np.load(os.path.join(DATA, "corpus.npy")).astype(int))


# ------------------------------------------------------------------ approach A: binary pair

def compose_binary(p_dep, p_str):
    """
    Two independent binary posteriors -> one 3-class distribution.

    healthy is the "neither" outcome, so it is the product of both negatives. Row-normalised,
    which keeps the ordering of the two positive scores intact.
    """
    scores = np.column_stack([(1 - p_dep) * (1 - p_str), p_str, p_dep])
    return scores / scores.sum(axis=1, keepdims=True)


def eval_binary_pair(F, y, subjects, corpus):
    """CV the composed pair. Each specialist is trained only on the corpora that define it."""
    proba = np.zeros((len(y), 3))
    dep_mask = corpus == 0                       # MODMA: healthy vs depression
    str_mask = np.isin(corpus, [1, 2])           # SAM40 + CogLoad: healthy vs stress

    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        tr_d = tr[dep_mask[tr]]
        tr_s = tr[str_mask[tr]]
        d = svm_proba().fit(F[tr_d], (y[tr_d] == 2).astype(int))
        s = svm_proba().fit(F[tr_s], (y[tr_s] == 1).astype(int))
        proba[te] = compose_binary(d.predict_proba(F[te])[:, 1],
                                   s.predict_proba(F[te])[:, 1])
    return proba.argmax(axis=1), proba


def fit_binary_pair(F, y, corpus):
    dep_mask = corpus == 0
    str_mask = np.isin(corpus, [1, 2])
    return {"depression": svm_proba().fit(F[dep_mask], (y[dep_mask] == 2).astype(int)),
            "stress": svm_proba().fit(F[str_mask], (y[str_mask] == 1).astype(int))}


# ------------------------------------------------------------------- approach B: flat 3-class

def eval_flat(F, y, subjects):
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        pred[te] = svm_proba().fit(F[tr], y[tr]).predict(F[te])
    return pred, None


# ----------------------------------------------------------------- approach C: hierarchical

def eval_hierarchical(F, y, subjects, corpus):
    """
    Stage 1 predicts CONTEXT (0 = resting, 1 = task), not corpus identity — a distinction that
    still means something for a recording from unseen hardware. Stage 2 picks the specialist.
    """
    context = (corpus != 0).astype(int)          # MODMA resting vs SAM40/CogLoad task
    pred = np.full(len(y), -1)
    router_acc = []

    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        router = make_pipeline(StandardScaler(),
                               SVC(C=1.0, kernel="rbf", class_weight="balanced"))
        router.fit(F[tr], context[tr])
        route = router.predict(F[te])
        router_acc.append(float((route == context[te]).mean()))

        for ctx, pos in ((0, 2), (1, 1)):        # resting->depression, task->stress
            tr_c = tr[context[tr] == ctx]
            te_c = te[route == ctx]
            if len(te_c) == 0 or len(tr_c) == 0:
                continue
            clf = make_pipeline(StandardScaler(),
                                SVC(C=1.0, kernel="rbf", class_weight="balanced"))
            clf.fit(F[tr_c], (y[tr_c] == pos).astype(int))
            pred[te_c] = np.where(clf.predict(F[te_c]) == 1, pos, 0)

    assert (pred >= 0).all()
    return pred, float(np.mean(router_acc))


def fit_hierarchical(F, y, corpus):
    context = (corpus != 0).astype(int)
    router = make_pipeline(StandardScaler(),
                           SVC(C=1.0, kernel="rbf", class_weight="balanced")).fit(F, context)
    specialists = {}
    for ctx, pos in ((0, 2), (1, 1)):
        m = context == ctx
        specialists[ctx] = (svm_proba().fit(F[m], (y[m] == pos).astype(int)), pos)
    return {"router": router, "specialists": specialists}


# --------------------------------------------------------------------------------- reporting

def score(name, y, pred, corpus, extra=None):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    acc = float((pred == y).mean())
    out = {"accuracy": acc,
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}
    if extra:
        out.update(extra)
    print(f"\n  {name}")
    print(f"    accuracy {acc:.4f}   balanced {out['balanced_accuracy']:.4f}   "
          f"macro-F1 {out['macro_f1']:.4f}")
    for c, nm in ((0, "MODMA"), (1, "SAM40"), (2, "CogLoad")):
        m = corpus == c
        print(f"    within {nm:8s} {float((pred[m] == y[m]).mean()):.4f}")
    return out


def main():
    os.makedirs(MODELS, exist_ok=True)
    X, y, subjects, corpus = load()
    print(f"all3: {len(y)} segments, {len(set(subjects))} subjects")
    print("extracting 81 features ...")
    F = extract_all_features(X).astype(np.float32)

    shortcut = float(sum(np.bincount(y[corpus == c]).max() for c in np.unique(corpus)) / len(y))
    print(f"corpus-ID baseline (context for the numbers below): {shortcut:.4f}\n")
    print("=" * 78)
    print("CROSS-VALIDATED PERFORMANCE  (5-fold subject-grouped; NOT device-disjoint)")
    print("=" * 78)

    results = {"corpus_id_baseline": shortcut, "n_segments": int(len(y)),
               "n_subjects": int(len(set(subjects)))}

    t0 = time.perf_counter()
    pred_a, _ = eval_binary_pair(F, y, subjects, corpus)
    results["A_binary_pair"] = score("A  binary pair (depression + stress composed)",
                                     y, pred_a, corpus)
    results["A_binary_pair"]["elapsed_sec"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    pred_b, _ = eval_flat(F, y, subjects)
    results["B_flat_3class"] = score("B  flat 3-class SVM", y, pred_b, corpus)
    results["B_flat_3class"]["elapsed_sec"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    pred_c, racc = eval_hierarchical(F, y, subjects, corpus)
    results["C_hierarchical"] = score("C  hierarchical (context router -> specialist)",
                                      y, pred_c, corpus, {"router_accuracy": racc})
    results["C_hierarchical"]["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    print(f"    context router (resting vs task) accuracy {racc:.4f}")

    # ------------------------------------------------------------------ refit on ALL data
    print("\n" + "=" * 78)
    print("REFITTING ON ALL DATA AND SERIALISING")
    print("=" * 78)

    meta = {"classes": CLASSES, "channels": CHANNELS, "fs": FS, "window_samples": WINDOW,
            "feature_version": "v1-81", "n_features": int(F.shape[1]),
            "trained_on": "datasets/all3 (MODMA + SAM40 + CogLoad), 15728 segments",
            "corpus_id_baseline": shortcut,
            "WARNING": ("Validated subject-disjoint but NOT device-disjoint. All training data "
                        "comes from 3 amplifiers; corpus identity is ~97% recoverable and "
                        "removing it drops 3-class accuracy to chance (0.3255). CV scores are "
                        "an upper bound for new hardware, not a prediction.")}

    artifacts = {
        "A_binary_pair": {"approach": "binary_pair", "models": fit_binary_pair(F, y, corpus),
                          "cv": results["A_binary_pair"]},
        "B_flat_3class": {"approach": "flat_3class", "models": svm_proba().fit(F, y),
                          "cv": results["B_flat_3class"]},
        "C_hierarchical": {"approach": "hierarchical", "models": fit_hierarchical(F, y, corpus),
                           "cv": results["C_hierarchical"]},
    }
    for name, art in artifacts.items():
        art["meta"] = meta
        path = os.path.join(MODELS, f"{name}.joblib")
        joblib.dump(art, path, compress=3)
        print(f"  {name:16s} -> {os.path.relpath(path, PROJECT_ROOT)}  "
              f"({os.path.getsize(path) / 1e6:.1f} MB, CV acc {art['cv']['accuracy']:.4f})")

    with open(os.path.join(MODELS, "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    for k in ("A_binary_pair", "B_flat_3class", "C_hierarchical"):
        a = results[k]["accuracy"]
        print(f"  {k:16s} {a:.4f}   {a - shortcut:+.4f} vs corpus-ID baseline")
    print("\nAll three are upper bounds. Approach A is the most likely to survive a change of\n"
          "recording hardware — both its models are within-corpus contrasts, so neither needs\n"
          "the between-device cue that inflates B and C.")
    print(f"\nUse: .venv/bin/python predict.py --model models/A_binary_pair.joblib <eeg.npy>")


if __name__ == "__main__":
    main()
