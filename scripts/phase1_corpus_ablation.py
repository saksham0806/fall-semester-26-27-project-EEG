"""
Phase 1.4 — corpus ablation: what does each stress corpus contribute?

Phase 1.1 showed stress is decodable on SAM40 alone (within-subject AUC 0.67)
while the pooled task sits at 0.40 with inverted AUC. This isolates the cause by
holding the *test* set fixed — always SAM40 held-out subjects, same folds as 1.1 —
and varying only what is added to the training set.

If adding AMIGOS drops SAM40 performance, it is contributing anti-signal rather
than merely diluting, and the fix is to drop or relabel it, not to domain-adapt it.

Run:  .venv/bin/python -u scripts/phase1_corpus_ablation.py
"""
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from proposed_model import extract_all_features, get_svm_pipeline, get_xgboost_pipeline

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase1")
SAM40, COGLOAD, AMIGOS = 1, 2, 3
N_SPLITS = 5
SEED = 42
SVM_CAP = 10000  # SVM is O(n^2)-ish; cap added corpora to keep runtime sane


def load():
    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_stress.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    datasets = np.load(os.path.join(DATA_DIR, "datasets.npy"))
    valid = ~np.isnan(X).any(axis=(1, 2))
    X, y, subjects, datasets = X[valid], y[valid], subjects[valid], datasets[valid]
    m = y != -1
    return X[m], y[m].astype(int), subjects[m], datasets[m]


def within_subject_auc(y, prob, subjects):
    scores = []
    for s in np.unique(subjects):
        k = subjects == s
        if len(np.unique(y[k])) > 1:
            scores.append(roc_auc_score(y[k], prob[k]))
    return float(np.mean(scores)) if scores else float("nan")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    X, y, subjects, datasets = load()
    feats = extract_all_features(X)
    print(f"loaded {X.shape[0]} stress segments across "
          f"{len(np.unique(datasets))} corpora\n")

    sam_idx = np.where(datasets == SAM40)[0]
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        sam_idx, y[sam_idx], groups=subjects[sam_idx]))

    combos = {
        "SAM40 only": [],
        "SAM40 + CogLoad": [COGLOAD],
        "SAM40 + AMIGOS": [AMIGOS],
        "SAM40 + both (= pooled)": [COGLOAD, AMIGOS],
    }

    results = {}
    for model_name, get_pipe in [("SVM", get_svm_pipeline),
                                 ("XGBoost", get_xgboost_pipeline)]:
        results[model_name] = {}
        for label, extra_ids in combos.items():
            t0 = time.time()
            per_fold = []
            for tr_rel, te_rel in folds:
                tr = sam_idx[tr_rel]
                te = sam_idx[te_rel]

                add = np.where(np.isin(datasets, extra_ids))[0] if extra_ids else np.array([], int)
                if model_name == "SVM" and len(add) > SVM_CAP:
                    add = rng.choice(add, SVM_CAP, replace=False)
                train_idx = np.concatenate([tr, add]) if len(add) else tr

                if model_name == "XGBoost":
                    neg = (y[train_idx] == 0).sum()
                    pos = (y[train_idx] == 1).sum()
                    pipe = get_pipe(scale_pos_weight=float(neg / pos) if pos else 1.0)
                else:
                    pipe = get_pipe()
                pipe.fit(feats[train_idx], y[train_idx])
                prob = pipe.predict_proba(feats[te])[:, 1]

                thr = np.quantile(prob, 1.0 - float((y[train_idx] == 1).mean()))
                per_fold.append({
                    "auc_within": within_subject_auc(y[te], prob, subjects[te]),
                    "auc_pooled": float(roc_auc_score(y[te], prob)),
                    "bal_acc": float(balanced_accuracy_score(y[te], (prob >= thr).astype(int))),
                    "n_train": int(len(train_idx)),
                })
            agg = {k: (float(np.mean([f[k] for f in per_fold])),
                       float(np.std([f[k] for f in per_fold]))) for k in per_fold[0]}
            results[model_name][label] = agg
            print(f"  {model_name:8s} {label:26s} "
                  f"within-AUC={agg['auc_within'][0]:.4f} ({time.time() - t0:.0f}s)")

    print(f"\n{'=' * 82}")
    print("PHASE 1.4 — corpus ablation. Test set is always SAM40 held-out subjects.")
    print(f"{'=' * 82}")
    for model_name, res in results.items():
        base = res["SAM40 only"]["auc_within"][0]
        print(f"\n{model_name}")
        print(f"  {'training data':26s} {'n_train':>8s} {'within-subj AUC':>18s} {'vs SAM40-only':>15s}")
        print("  " + "-" * 74)
        for label, agg in res.items():
            d = agg["auc_within"][0] - base
            mark = "" if label == "SAM40 only" else f"{d:+.4f}"
            print(f"  {label:26s} {agg['n_train'][0]:8.0f} "
                  f"{agg['auc_within'][0]:10.4f}+/-{agg['auc_within'][1]:5.4f} {mark:>15s}")

    with open(os.path.join(OUT_DIR, "corpus_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'corpus_ablation.json')}")


if __name__ == "__main__":
    main()
