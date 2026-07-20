"""
Phase 1.2 — is the AMIGOS arousal *label* the problem, or the AMIGOS *data*?

Phase 1.4 showed AMIGOS is neutral-to-helpful as training data for SAM40, yet the
pooled stress task (dominated by AMIGOS at test time) sits at AUC 0.40. That points
at the label, not the signal.

The current label is a global cut, `arousal >= 5.0`. Self-report scales carry strong
per-subject bias: one person rates everything 6-7, another 3-4, so a global cut
partly encodes *who* the subject is rather than *what state they were in*. A
per-subject median split removes that offset by construction.

This regenerates per-segment arousal by replaying the AMIGOS loader, then compares
label schemes under subject-grouped 5-fold CV on AMIGOS alone.

Correctness check: replaying the global >= 5.0 rule must reproduce the stored
y_stress for AMIGOS exactly, which validates segment alignment before anything else.

Run:  .venv/bin/python -u scripts/phase1_amigos_relabel.py
"""
import glob
import json
import os
import sys

import numpy as np
import scipy.io

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from proposed_model import extract_all_features, get_xgboost_pipeline

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
AMIGOS_DIR = os.path.join(PROJECT_ROOT, "datasets", "AMIGOS DATASET")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase1")
AMIGOS_ID = 3
WINDOW, STRIDE = 512, 256
CHANNELS = [0, 13, 1, 2, 11, 12]
N_SPLITS, SEED = 5, 42


def replay_amigos_arousal():
    """Per-segment raw arousal, in the exact order combine_datasets.py emits."""
    arousal, subj = [], []
    for f in glob.glob(os.path.join(AMIGOS_DIR, "Data_Preprocessed_P*/Data_Preprocessed_P*.mat")):
        sub_id = os.path.basename(f).replace("Data_Preprocessed_", "").replace(".mat", "")
        try:
            mat = scipy.io.loadmat(f)
            joined, labels = mat["joined_data"], mat["labels_selfassessment"]
            for t in range(joined.shape[1]):
                sig, lab = joined[0, t], labels[0, t]
                if not isinstance(sig, np.ndarray) or sig.size == 0:
                    continue
                if not isinstance(lab, np.ndarray) or lab.size == 0:
                    continue
                n = sig.T[CHANNELS, :].shape[1]
                n_seg = len(range(0, n - WINDOW + 1, STRIDE))
                arousal.extend([float(lab[0, 0])] * n_seg)
                subj.extend([f"amigos_{sub_id}"] * n_seg)
        except Exception as e:
            print(f"  skip {f}: {e}")
    return np.array(arousal), np.array(subj, dtype=object)


def per_subject_median_labels(arousal, subjects):
    """1 if a trial's arousal is above that subject's own median, else 0."""
    y = np.zeros(len(arousal), dtype=int)
    for s in np.unique(subjects):
        m = subjects == s
        y[m] = (arousal[m] > np.median(arousal[m])).astype(int)
    return y


def evaluate(feats, y, subjects, tag):
    """Subject-grouped 5-fold CV; reports pooled and within-subject AUC."""
    aucs, within = [], []
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(feats, y, groups=subjects):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        neg, pos = (y[tr] == 0).sum(), (y[tr] == 1).sum()
        pipe = get_xgboost_pipeline(scale_pos_weight=float(neg / pos) if pos else 1.0)
        pipe.fit(feats[tr], y[tr])
        prob = pipe.predict_proba(feats[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        ws = [roc_auc_score(y[te][subjects[te] == s], prob[subjects[te] == s])
              for s in np.unique(subjects[te])
              if len(np.unique(y[te][subjects[te] == s])) > 1]
        within.append(np.mean(ws) if ws else np.nan)
    r = {"auc_pooled": (float(np.mean(aucs)), float(np.std(aucs))),
         "auc_within_subject": (float(np.nanmean(within)), float(np.nanstd(within))),
         "positive_rate": float(y.mean())}
    print(f"  {tag:28s} pooled={r['auc_pooled'][0]:.4f}+/-{r['auc_pooled'][1]:.4f}  "
          f"within-subj={r['auc_within_subject'][0]:.4f}+/-{r['auc_within_subject'][1]:.4f}  "
          f"pos={r['positive_rate'] * 100:.1f}%")
    return r


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y_stored = np.load(os.path.join(DATA_DIR, "y_stress.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    datasets = np.load(os.path.join(DATA_DIR, "datasets.npy"))

    print("replaying AMIGOS loader to recover per-segment arousal...")
    arousal, subj_replay = replay_amigos_arousal()
    am = datasets == AMIGOS_ID
    print(f"  replayed {len(arousal)} segments; stored AMIGOS segments: {am.sum()}")

    assert len(arousal) == am.sum(), "segment count mismatch — alignment failed"
    assert (subj_replay == subjects[am]).all(), "subject order mismatch — alignment failed"
    reproduced = (arousal >= 5.0).astype(int)
    match = (reproduced == y_stored[am]).mean()
    print(f"  global>=5.0 rule reproduces stored labels: {match * 100:.2f}%")
    assert match == 1.0, "replay does not reproduce stored labels — do not trust results"
    print("  alignment verified\n")

    # NaN handling must match run_models.py, applied to the AMIGOS block only.
    valid = ~np.isnan(X[am]).any(axis=(1, 2))
    Xa = X[am][valid]
    arousal, subj_a = arousal[valid], subj_replay[valid]
    feats = extract_all_features(Xa)
    print(f"AMIGOS after NaN drop: {Xa.shape[0]} segments, "
          f"{len(np.unique(subj_a))} subjects\n")

    print("subject-grouped 5-fold CV on AMIGOS alone (XGBoost on 81 features):")
    results = {
        "global >= 5.0 (current)": evaluate(feats, (arousal >= 5.0).astype(int), subj_a,
                                            "global >= 5.0 (current)"),
        "per-subject median split": evaluate(feats, per_subject_median_labels(arousal, subj_a),
                                             subj_a, "per-subject median split"),
    }

    a = results["global >= 5.0 (current)"]["auc_within_subject"][0]
    b = results["per-subject median split"]["auc_within_subject"][0]
    print(f"\n{'=' * 78}")
    print(f"within-subject AUC: global={a:.4f}  per-subject median={b:.4f}  delta={b - a:+.4f}")
    if b > 0.60:
        print("VERDICT: per-subject relabelling RECOVERS a decodable AMIGOS target.")
    elif b > a + 0.03:
        print("VERDICT: relabelling helps but AMIGOS arousal remains weakly decodable.")
    else:
        print("VERDICT: relabelling does NOT help — AMIGOS arousal is not EEG-decodable\n"
              "         from 6 frontal channels. Use AMIGOS as training data only,\n"
              "         and drop it as an evaluation target.")

    with open(os.path.join(OUT_DIR, "amigos_relabel.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: {os.path.join(OUT_DIR, 'amigos_relabel.json')}")


if __name__ == "__main__":
    main()
