"""
Phase 1.3 + 1.6 — depression under subject-grouped 5-fold CV, with subject-level metrics.

Phase 0 left depression on a single 10-subject test split, where ROC-AUC swung by
up to +/-0.19 between identical runs (EEGTransformer +0.188, CNN-LSTM -0.166) purely
from training stochasticity. No single-run figure from that setup is reportable.

This replaces it with 5-fold subject-grouped CV over all 53 MODMA subjects and
accumulates out-of-fold predictions, so every subject is predicted exactly once by a
model that never saw them. Two metrics:

  segment-level — AUC over all out-of-fold segments.
  subject-level — mean probability per subject, then AUC over the 53 subject scores.
                  This is the clinically meaningful unit: every segment from a
                  subject shares one label, so the real question is whether the
                  subject is MDD, not whether a 4-second window is.

Run:  .venv/bin/python -u scripts/phase1_depression_cv.py
"""
import gc
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from proposed_model import (
    extract_all_features, get_svm_pipeline, get_random_forest_pipeline,
    get_xgboost_pipeline, build_1d_cnn, build_cnn_lstm, EEGNet, TSception,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase1")
MODMA_ID = 0
N_SPLITS, SEED = 5, 42
EPOCHS, BATCH_SIZE, LR = 25, 128, 5e-4


def load_modma():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_depression.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    datasets = np.load(os.path.join(DATA_DIR, "datasets.npy"))
    valid = ~np.isnan(X).any(axis=(1, 2))
    X, y, subjects, datasets = X[valid], y[valid], subjects[valid], datasets[valid]
    m = (datasets == MODMA_ID) & (y != -1)
    return X[m], y[m].astype(int), subjects[m]


def subject_level(y, prob, subjects):
    """Mean probability per subject -> AUC and balanced accuracy over subjects."""
    subs = np.unique(subjects)
    s_true = np.array([y[subjects == s][0] for s in subs])       # constant within subject
    s_prob = np.array([prob[subjects == s].mean() for s in subs])
    thr = np.quantile(s_prob, 1.0 - s_true.mean())
    return {
        "n_subjects": int(len(subs)),
        "auc": float(roc_auc_score(s_true, s_prob)),
        "balanced_acc": float(balanced_accuracy_score(s_true, (s_prob >= thr).astype(int))),
    }


def oof_predictions(fit_predict, X, y, subjects, splits):
    """Out-of-fold probabilities: every subject predicted once, by a model blind to them."""
    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in splits:
        prob = fit_predict(tr, te)
        oof[te] = prob
        per_fold.append(float(roc_auc_score(y[te], prob)))
    return oof, per_fold


def make_classical(name, get_pipe, feats, y):
    def fit_predict(tr, te):
        if name == "XGBoost":
            neg, pos = (y[tr] == 0).sum(), (y[tr] == 1).sum()
            pipe = get_pipe(scale_pos_weight=float(neg / pos) if pos else 1.0)
        else:
            pipe = get_pipe()
        pipe.fit(feats[tr], y[tr])
        return pipe.predict_proba(feats[te])[:, 1]
    return fit_predict


def make_deep(build_fn, X, y, subjects):
    Xdl = X[..., np.newaxis]

    def fit_predict(tr, te):
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(SEED)
        model = build_fn()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                      loss="binary_crossentropy", metrics=["accuracy"])
        # Inner validation split by subject so early stopping never sees the held-out fold.
        inner = np.unique(subjects[tr])
        rng = np.random.default_rng(SEED)
        val_subs = set(rng.permutation(inner)[: max(1, len(inner) // 8)])
        vm = np.array([s in val_subs for s in subjects[tr]])
        itr, ival = tr[~vm], tr[vm]
        pos, neg = (y[itr] == 1).sum(), (y[itr] == 0).sum()
        cw = {0: len(y[itr]) / (2 * neg), 1: len(y[itr]) / (2 * pos)}
        model.fit(Xdl[itr], y[itr], validation_data=(Xdl[ival], y[ival]),
                  epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=cw,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=5, restore_best_weights=True)],
                  verbose=0)
        return model.predict(Xdl[te], verbose=0).flatten()
    return fit_predict


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    X, y, subjects = load_modma()
    print(f"MODMA: {X.shape[0]} segments, {len(np.unique(subjects))} subjects, "
          f"{(y == 1).mean() * 100:.1f}% MDD")
    feats = extract_all_features(X)
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=subjects))
    print(f"{N_SPLITS}-fold subject-grouped CV, "
          f"~{len(np.unique(subjects[splits[0][1]]))} held-out subjects per fold\n")

    jobs = [("SVM", make_classical("SVM", get_svm_pipeline, feats, y)),
            ("Random Forest", make_classical("RF", get_random_forest_pipeline, feats, y)),
            ("XGBoost", make_classical("XGBoost", get_xgboost_pipeline, feats, y)),
            ("1D CNN", make_deep(lambda: build_1d_cnn(nb_classes=1), X, y, subjects)),
            ("CNN-LSTM", make_deep(lambda: build_cnn_lstm(nb_classes=1), X, y, subjects)),
            ("EEGNet", make_deep(lambda: EEGNet(nb_classes=1), X, y, subjects)),
            ("TSception", make_deep(lambda: TSception(nb_classes=1), X, y, subjects))]

    results = {}
    for name, fp in jobs:
        t0 = time.time()
        oof, per_fold = oof_predictions(fp, X, y, subjects, splits)
        results[name] = {
            "auc_segment_oof": float(roc_auc_score(y, oof)),
            "auc_per_fold_mean": float(np.mean(per_fold)),
            "auc_per_fold_std": float(np.std(per_fold)),
            "subject_level": subject_level(y, oof, subjects),
        }
        print(f"  {name:16s} done in {time.time() - t0:6.1f}s")

    print(f"\n{'=' * 84}")
    print(f"PHASE 1.3/1.6 — depression, {N_SPLITS}-fold subject-grouped CV over all "
          f"{len(np.unique(subjects))} subjects")
    print(f"{'=' * 84}")
    print(f"{'model':16s} {'segment AUC':>13s} {'fold AUC (mean+/-std)':>24s} "
          f"{'subject AUC':>13s} {'subj bal-acc':>13s}")
    print("-" * 84)
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["subject_level"]["auc"]):
        print(f"{name:16s} {r['auc_segment_oof']:13.4f} "
              f"{r['auc_per_fold_mean']:14.4f}+/-{r['auc_per_fold_std']:5.4f} "
              f"{r['subject_level']['auc']:13.4f} {r['subject_level']['balanced_acc']:13.4f}")
    print("-" * 84)
    best = max(results.items(), key=lambda kv: kv[1]["subject_level"]["auc"])
    print(f"best subject-level AUC: {best[0]} at {best[1]['subject_level']['auc']:.4f} "
          f"over {best[1]['subject_level']['n_subjects']} subjects")
    print("\nCompare against Phase 0 single-split figures, which swung +/-0.19 between\n"
          "identical runs. These are out-of-fold over every subject and are the\n"
          "numbers that should be reported.")

    with open(os.path.join(OUT_DIR, "depression_cv.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'depression_cv.json')}")


if __name__ == "__main__":
    main()
