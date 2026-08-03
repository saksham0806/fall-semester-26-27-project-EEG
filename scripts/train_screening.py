"""
Train the three screening models that make up the defensible deliverable.

DESIGN. Stress and depression are NOT distinguishable from this EEG — every depression segment
is MODMA and every stress segment is not, so `depression` is a boolean identity with
`corpus == MODMA`. Any stress-vs-depression classifier is a device detector wearing a clinical
label. In real use the distinction is set by the recording protocol, not inferred:

    screen    healthy vs ill          any recording; the only task whose two classes both
                                      appear in all three corpora, so it learns from
                                      within-device contrast rather than amplifier signature
    resting   healthy vs depressed    resting-state assessment  (trained on MODMA)
    task      healthy vs stressed     cognitive-load protocol   (trained on SAM40 + CogLoad)

Each model is binary, so each is measured against its own majority-class baseline — a screening
tool that always answers "healthy" is useless however accurate, which is why the threshold
search below refuses degenerate single-class cutoffs.

Protocol: 5-fold subject-grouped CV. The decision threshold is tuned by NESTED inner CV inside
each training fold and only then applied to the held-out fold; tuning it on the test fold would
be selection on the test set.

HONEST BOUNDS. Subject-disjoint but NOT device-disjoint. On unseen hardware these models are
worth roughly chance (measured twice: 0.3198 and 0.3255). Research signal, not diagnosis.

Run:  .venv/bin/python -u scripts/train_screening.py
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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import (GroupKFold, StratifiedKFold, cross_val_predict)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from proposed_model.features import extract_all_features

DATA = os.path.join(PROJECT_ROOT, "datasets", "all3")
MODELS = os.path.join(PROJECT_ROOT, "models")
CORPUS_NAMES = {0: "MODMA", 1: "SAM40", 2: "CogLoad"}
N_SPLITS, INNER_SPLITS = 5, 3


MIN_RATE = 0.15   # a usable screen must act on BOTH classes, not just the majority


def clf():
    """
    SVC + probability calibration.

    The internal CV MUST be shuffled. CalibratedClassifierCV's default `cv=3` is a
    non-shuffled KFold, and datasets/all3/X.npy is ordered by corpus — so the three internal
    folds land on three different amplifiers, the calibrator is fit on a cross-corpus transfer
    it will never see at predict time, and it learns an INVERTED mapping. Measured on
    healthy-vs-ill: AUC 0.4410 with the default, 0.6527 shuffled, 0.6553 for the raw
    decision_function. Scores were negatively correlated (-0.25) with the SVM's own ranking.
    """
    return make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", class_weight="balanced"),
                               ensemble=False,
                               cv=StratifiedKFold(3, shuffle=True, random_state=42)))


def tune_threshold(y_true, scores):
    """
    Cutoff maximising raw accuracy among cutoffs that actually discriminate.

    Two guards, both load-bearing on an imbalanced screening task:
      - skip cutoffs predicting a single class (run_models.py has the same rule);
      - require sensitivity AND specificity >= MIN_RATE. Without this second one the search
        happily returns 99% sensitivity / 1% specificity, which scores well on accuracy and is
        clinically worthless — it is "answer ill for everyone" with a rounding error.

    Returns (threshold, usable). usable=False means no cutoff discriminates and the caller
    should treat the model as not fit for screening, whatever its accuracy says.
    """
    best_t, best_a, usable = 0.5, -1.0, False
    for t in np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 97))):
        pred = (scores >= t).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        tp = float(((pred == 1) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        sens = tp / max((y_true == 1).sum(), 1)
        spec = tn / max((y_true == 0).sum(), 1)
        if min(sens, spec) < MIN_RATE:
            continue
        a = float((pred == y_true).mean())
        if a > best_a:
            best_a, best_t, usable = a, float(t), True
    return best_t, usable


def evaluate(name, F, y, subjects, corpus, description):
    """Nested-CV accuracy, AUC and the operating point, plus a per-corpus breakdown."""
    proba = np.zeros(len(y))
    pred = np.full(len(y), -1)
    thresholds, usable = [], []

    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        # inner CV on the training fold only -> threshold; never touches the test fold
        inner = cross_val_predict(clf(), F[tr], y[tr],
                                  cv=GroupKFold(n_splits=INNER_SPLITS),
                                  groups=subjects[tr], method="predict_proba")[:, 1]
        t, ok = tune_threshold(y[tr], inner)
        thresholds.append(t)
        usable.append(ok)

        m = clf().fit(F[tr], y[tr])
        proba[te] = m.predict_proba(F[te])[:, 1]
        pred[te] = (proba[te] >= t).astype(int)

    maj = float(np.bincount(y).max() / len(y))
    acc = float((pred == y).mean())
    out = {
        "description": description,
        "accuracy": acc, "majority_baseline": maj, "margin": acc - maj,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)),
        "threshold_mean": float(np.mean(thresholds)),
        "threshold_per_fold": [round(t, 4) for t in thresholds],
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "usable_folds": int(sum(usable)),
        "n_segments": int(len(y)), "n_subjects": int(len(set(subjects))),
    }

    print(f"\n  {name}  —  {description}")
    print(f"    accuracy {acc:.4f}   majority {maj:.4f}   margin {acc - maj:+.4f}"
          f"{'   BEATS majority' if acc > maj else '   <-- no better than guessing'}")
    print(f"    AUC {out['auc']:.4f}   balanced {out['balanced_accuracy']:.4f}   "
          f"threshold {out['threshold_mean']:.3f}")
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    out["sensitivity"], out["specificity"] = float(sens), float(spec)
    print(f"    sensitivity {sens:.4f}   specificity {spec:.4f}")

    if len(np.unique(corpus)) > 1:
        out["within_corpus"] = {}
        for c in np.unique(corpus):
            m = corpus == c
            if len(np.unique(y[m])) < 2:
                continue
            a = float((pred[m] == y[m]).mean())
            b = float(np.bincount(y[m]).max() / m.sum())
            out["within_corpus"][CORPUS_NAMES[int(c)]] = {
                "accuracy": a, "majority": b, "margin": a - b,
                "auc": float(roc_auc_score(y[m], proba[m]))}
            print(f"    within {CORPUS_NAMES[int(c)]:8s} {a:.4f}  "
                  f"(majority {b:.4f}, margin {a - b:+.4f})")
    return out, float(np.mean(thresholds))


def main():
    os.makedirs(MODELS, exist_ok=True)
    X = np.load(os.path.join(DATA, "X.npy"))
    y3 = np.load(os.path.join(DATA, "y.npy")).astype(int)
    subjects = np.load(os.path.join(DATA, "subjects.npy"), allow_pickle=True)
    corpus = np.load(os.path.join(DATA, "corpus.npy")).astype(int)

    print(f"all3: {len(y3)} segments, {len(set(subjects))} subjects")
    print("extracting 81 features ...")
    F = extract_all_features(X).astype(np.float32)

    resting = corpus == 0                      # MODMA, resting-state
    task = np.isin(corpus, [1, 2])             # SAM40 + CogLoad, cognitive load

    specs = {
        "screen_healthy_vs_ill": dict(
            mask=np.ones(len(y3), bool), y=(y3 != 0).astype(int),
            description="healthy vs ill — any recording",
            positive="ill", negative="healthy"),
        "resting_healthy_vs_depressed": dict(
            mask=resting, y=(y3[resting] == 2).astype(int),
            description="healthy vs depressed — resting-state protocol (MODMA)",
            positive="depressed", negative="healthy"),
        "task_healthy_vs_stressed": dict(
            mask=task, y=(y3[task] == 1).astype(int),
            description="healthy vs stressed — cognitive-load protocol (SAM40 + CogLoad)",
            positive="stressed", negative="healthy"),
    }

    print("\n" + "=" * 78)
    print("CROSS-VALIDATED PERFORMANCE  (5-fold subject-grouped, threshold by nested inner CV)")
    print("=" * 78)

    results, thresholds = {}, {}
    for name, sp in specs.items():
        m = sp["mask"]
        t0 = time.perf_counter()
        results[name], thresholds[name] = evaluate(
            name, F[m], sp["y"], subjects[m], corpus[m], sp["description"])
        results[name]["elapsed_sec"] = round(time.perf_counter() - t0, 1)

    print("\n" + "=" * 78)
    print("REFITTING ON ALL DATA AND SERIALISING")
    print("=" * 78)
    for name, sp in specs.items():
        m = sp["mask"]
        art = {
            "task": name, "description": sp["description"],
            "classes": {0: sp["negative"], 1: sp["positive"]},
            "model": clf().fit(F[m], sp["y"]),
            "threshold": thresholds[name],
            "cv": results[name],
            "meta": {
                "channels": ["Fp1", "Fp2", "F7", "F3", "F4", "F8"],
                "fs": 128, "window_samples": 512, "feature_version": "v1-81",
                "trained_on": sorted({CORPUS_NAMES[int(c)] for c in np.unique(corpus[m])}),
                "WARNING": ("Subject-disjoint but NOT device-disjoint validation. On unseen "
                            "recording hardware these models are worth roughly chance "
                            "(measured 0.3198 and 0.3255). Research signal, not a diagnosis."),
            },
        }
        path = os.path.join(MODELS, f"{name}.joblib")
        joblib.dump(art, path, compress=3)
        print(f"  {name:30s} -> models/{name}.joblib  "
              f"({os.path.getsize(path) / 1e6:.1f} MB, acc {results[name]['accuracy']:.4f})")

    with open(os.path.join(MODELS, "screening_cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    print(f"  {'model':32s}{'accuracy':>10s}{'majority':>10s}{'margin':>9s}{'AUC':>8s}")
    for name in specs:
        r = results[name]
        print(f"  {name:32s}{r['accuracy']:10.4f}{r['majority_baseline']:10.4f}"
              f"{r['margin']:+9.4f}{r['auc']:8.4f}")
    print("\nUse:")
    print("  .venv/bin/python predict.py rec.npy --protocol screen")
    print("  .venv/bin/python predict.py rec.npy --protocol resting")
    print("  .venv/bin/python predict.py rec.npy --protocol task")


if __name__ == "__main__":
    main()
