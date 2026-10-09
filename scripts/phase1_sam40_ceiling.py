"""
Phase 1.1 — SAM40-only stress ceiling, with Phase 1.3 subject-grouped 5-fold CV.

SAM40 is the only corpus with a genuine stress-vs-relax contrast, native frontal
channels, and a balanced design (40 subjects x 60 segments, every subject exactly
75% stress). Phase 0 showed the *pooled* stress task sits at chance with inverted
AUC; this establishes whether stress is decodable at all on clean data, and is the
go/no-go gate for the whole stress track.

Two AUCs are reported and they answer different questions:

  pooled AUC  — rank all held-out segments together. Sensitive to per-subject
                shifts in the model's output range (the pathology that broke
                threshold transfer in Phase 0).
  within-subj — AUC computed inside each subject, then averaged. Asks "given one
                person, can we separate their stressed segments from their own
                relaxed ones?" Immune to inter-subject calibration. This is the
                fair question for SAM40's design and is the headline number.

Run:  .venv/bin/python scripts/phase1_sam40_ceiling.py
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
    extract_all_features,
    get_svm_pipeline,
    get_random_forest_pipeline,
    get_xgboost_pipeline,
    build_1d_cnn,
    EEGNet,
    TSception,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase1")
SAM40_ID = 1
N_SPLITS = 5
SEED = 42
EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 5e-4


def load_sam40():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_stress.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    datasets = np.load(os.path.join(DATA_DIR, "datasets.npy"))

    # Same NaN handling as run_models.py, applied before subsetting.
    valid = ~np.isnan(X).any(axis=(1, 2))
    X, y, subjects, datasets = X[valid], y[valid], subjects[valid], datasets[valid]

    m = (datasets == SAM40_ID) & (y != -1)
    return X[m], y[m].astype(int), subjects[m]


def within_subject_auc(y, prob, subjects):
    """Mean per-subject AUC. Skips subjects lacking both classes."""
    scores = []
    for s in np.unique(subjects):
        m = subjects == s
        if len(np.unique(y[m])) < 2:
            continue
        scores.append(roc_auc_score(y[m], prob[m]))
    return float(np.mean(scores)) if scores else float("nan")


def score_fold(y_test, prob_test, subjects_test, train_pos_rate):
    """Prevalence-matched threshold (Phase 0 quantile transfer), plus both AUCs."""
    t = float(np.quantile(prob_test, 1.0 - train_pos_rate))
    pred = (prob_test >= t).astype(int)
    counts = np.bincount(y_test, minlength=2)
    return {
        "accuracy": float((pred == y_test).mean()),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
        "majority_baseline": float(counts.max() / counts.sum()),
        "auc_pooled": float(roc_auc_score(y_test, prob_test)),
        "auc_within_subject": within_subject_auc(y_test, prob_test, subjects_test),
    }


def run_classical(name, get_pipeline, feats, y, subjects, splits):
    folds = []
    for tr, te in splits:
        if name == "XGBoost":
            neg, pos = (y[tr] == 0).sum(), (y[tr] == 1).sum()
            pipe = get_pipeline(scale_pos_weight=float(neg / pos) if pos else 1.0)
        else:
            pipe = get_pipeline()
        pipe.fit(feats[tr], y[tr])
        prob = pipe.predict_proba(feats[te])[:, 1]
        folds.append(score_fold(y[te], prob, subjects[te], float((y[tr] == 1).mean())))
    return folds


def run_deep(name, build_fn, X, y, subjects, splits):
    Xdl = X[..., np.newaxis]
    folds = []
    for tr, te in splits:
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(SEED)

        model = build_fn()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        # Inner validation split by subject, so early stopping never sees a
        # training subject and never sees the held-out fold.
        inner_subs = np.unique(subjects[tr])
        rng = np.random.default_rng(SEED)
        val_subs = set(rng.permutation(inner_subs)[: max(1, len(inner_subs) // 8)])
        vmask = np.array([s in val_subs for s in subjects[tr]])
        inner_tr, inner_val = tr[~vmask], tr[vmask]

        pos = (y[inner_tr] == 1).sum()
        neg = (y[inner_tr] == 0).sum()
        cw = {0: len(y[inner_tr]) / (2 * neg), 1: len(y[inner_tr]) / (2 * pos)}

        model.fit(
            Xdl[inner_tr], y[inner_tr],
            validation_data=(Xdl[inner_val], y[inner_val]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=cw,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0,
        )
        prob = model.predict(Xdl[te], verbose=0).flatten()
        folds.append(score_fold(y[te], prob, subjects[te], float((y[inner_tr] == 1).mean())))
    return folds


def summarize(folds):
    keys = folds[0].keys()
    return {k: (float(np.mean([f[k] for f in folds])),
                float(np.std([f[k] for f in folds]))) for k in keys}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    X, y, subjects = load_sam40()
    print(f"SAM40: {X.shape[0]} segments, {len(np.unique(subjects))} subjects, "
          f"{(y == 1).mean() * 100:.1f}% positive")

    feats = extract_all_features(X)
    splits = list(GroupKFold(n_splits=N_SPLITS).split(X, y, groups=subjects))
    print(f"{N_SPLITS}-fold subject-grouped CV — "
          f"{len(np.unique(subjects[splits[0][1]]))} held-out subjects per fold\n")

    results = {}
    for name, fn in [("SVM", get_svm_pipeline),
                     ("Random Forest", get_random_forest_pipeline),
                     ("XGBoost", get_xgboost_pipeline)]:
        t0 = time.time()
        results[name] = summarize(run_classical(name, fn, feats, y, subjects, splits))
        print(f"  {name:16s} done in {time.time() - t0:5.1f}s")

    for name, fn in [("1D CNN", lambda: build_1d_cnn(nb_classes=1)),
                     ("EEGNet", lambda: EEGNet(nb_classes=1)),
                     ("TSception", lambda: TSception(nb_classes=1))]:
        t0 = time.time()
        results[name] = summarize(run_deep(name, fn, X, y, subjects, splits))
        print(f"  {name:16s} done in {time.time() - t0:5.1f}s")

    base = results["SVM"]["majority_baseline"][0]
    print(f"\n{'=' * 78}\nPHASE 1.1 — SAM40-only stress, {N_SPLITS}-fold subject-grouped CV")
    print(f"majority baseline = {base:.4f}   (mean +/- std across folds)\n")
    print(f"{'model':16s} {'balanced acc':>16s} {'AUC pooled':>16s} {'AUC within-subj':>18s}")
    print("-" * 78)
    for name, r in results.items():
        print(f"{name:16s} {r['balanced_acc'][0]:8.4f}+/-{r['balanced_acc'][1]:5.4f} "
              f"{r['auc_pooled'][0]:8.4f}+/-{r['auc_pooled'][1]:5.4f} "
              f"{r['auc_within_subject'][0]:10.4f}+/-{r['auc_within_subject'][1]:5.4f}")

    best = max(results.items(), key=lambda kv: kv[1]["auc_within_subject"][0])
    print("-" * 78)
    print(f"best within-subject AUC: {best[0]} at {best[1]['auc_within_subject'][0]:.4f}")
    print("VERDICT: stress IS decodable on clean data — pooled task is the problem"
          if best[1]["auc_within_subject"][0] > 0.60 else
          "VERDICT: no signal even on SAM40 — 6 frontal channels @4s may not carry stress")

    with open(os.path.join(OUT_DIR, "sam40_ceiling.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'sam40_ceiling.json')}")


if __name__ == "__main__":
    main()
