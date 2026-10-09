"""
Which model should be the TEACHER? Measure, do not assume.

A student can only inherit what its teacher knows, so the teacher must be the strongest model
available at high density — not merely a deep one. Two candidates are already measured on MODMA
depression under the same 5-fold subject-grouped protocol:

    Riemannian tangent @32ch   0.6753   Results/phase2/channel_density.json
    deep EEGNet       @32ch   0.7371   Results/distilled/premise_gate.json
    handcrafted SVM   @ 6ch   0.7716   Results/phase1/depression_cv.json  <- best in project

The gap that matters is the third row: the project's best model uses SIX channels. This script
measures handcrafted-SVM at 6 / 16 / 32 / 64 channels to settle whether it improves with density.

    if handcrafted-SVM improves with channels -> it is the teacher, and KD proceeds
    if it peaks at 6 channels                 -> the strongest model needs no extra channels,
                                                 and distillation is pointless for a reason the
                                                 premise gate could not see (the gate only
                                                 compared deep models to each other)

That second outcome is a real finding, not a failure: it would mean density helps deep models
catch up to a classical baseline that never needed density in the first place.

Features are channel-generic (distilled/data.py:handcrafted) — 13 per channel, i.e. the v1
families minus frontal alpha asymmetry, which needs named left/right pairs and does not
generalise past a fixed montage.

Run:  .venv/bin/python -u distilled/scripts/01_teacher_bakeoff.py
"""
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from distilled.data import channel_subset, handcrafted, load_modma, subject_auc
from distilled.montages import MODMA

OUT = os.path.join(PROJECT_ROOT, "Results", "distilled")
CHANNEL_SETS = [6, 16, 32, 64]
N_SPLITS = 5

# Reference points, all MODMA depression / subject-level AUC / 5-fold subject-grouped.
KNOWN = {
    "handcrafted SVM @6ch (phase1)": 0.7716,
    "Riemannian @32ch (channel_density)": 0.6753,
    "deep EEGNet @32ch (premise gate)": 0.7371,
    "deep EEGNet @6ch (premise gate)": 0.6667,
}


def oof_proba(F, y, subjects):
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(F, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        clf = make_pipeline(StandardScaler(),
                            SVC(C=1.0, kernel="rbf", class_weight="balanced",
                                probability=False))
        clf.fit(F[tr], y[tr])
        prob[te] = clf.decision_function(F[te])
    return prob


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, subjects = load_modma()
    print(f"MODMA {X.shape}, {len(set(subjects))} subjects\n")

    print("=" * 74)
    print("TEACHER BAKEOFF — does the classical model improve with channel density?")
    print("=" * 74)
    print(f"  {'chans':>6s}{'features':>10s}{'subject AUC':>14s}{'segment AUC':>14s}{'min':>7s}")

    results = {"known_reference_points": KNOWN}
    for n in CHANNEL_SETS:
        chans = channel_subset(n, MODMA.student_idx)
        t0 = time.perf_counter()
        F = handcrafted(np.asarray(X[:, chans]))
        prob = oof_proba(F, y, subjects)
        sa, ga = subject_auc(y, prob, subjects), float(roc_auc_score(y, prob))
        mins = (time.perf_counter() - t0) / 60
        results[f"handcrafted_svm_{n}ch"] = {
            "n_chans": len(chans), "n_features": int(F.shape[1]),
            "subject_auc": sa, "segment_auc": ga, "minutes": round(mins, 1)}
        print(f"  {n:>6d}{F.shape[1]:>10d}{sa:>14.4f}{ga:>14.4f}{mins:>7.1f}")

    rows = [(results[f"handcrafted_svm_{n}ch"]["subject_auc"], n) for n in CHANNEL_SETS]
    best_auc, best_n = max(rows)
    at_6 = results["handcrafted_svm_6ch"]["subject_auc"]

    print("\n" + "=" * 74)
    print("  reference points (same task, same protocol):")
    for k, v in KNOWN.items():
        print(f"    {k:38s} {v:.4f}")
    print("=" * 74)
    print(f"  handcrafted-SVM best: {best_auc:.4f} at {best_n} channels "
          f"(6ch gives {at_6:.4f}, delta {best_auc - at_6:+.4f})")

    density_helps = best_n > 6 and (best_auc - at_6) >= 0.02
    beats_deep = best_auc > KNOWN["deep EEGNet @32ch (premise gate)"]
    results["verdict"] = {"best_auc": best_auc, "best_n_chans": best_n,
                          "auc_at_6ch": at_6, "gain_over_6ch": best_auc - at_6,
                          "density_helps_classical": bool(density_helps),
                          "beats_deep_teacher": bool(beats_deep)}

    if density_helps and beats_deep:
        print(f"\n  -> TEACHER = handcrafted SVM @{best_n}ch. Strongest candidate measured;\n"
              f"     distil from it, not from the deep model.")
    elif density_helps:
        print(f"\n  -> classical improves with density but the deep teacher is still stronger.\n"
              f"     TEACHER = deep EEGNet @32ch (0.7371).")
    else:
        print(f"\n  -> classical does NOT improve with density (best {best_auc:.4f} at {best_n}ch\n"
              f"     vs {at_6:.4f} at 6ch). The project's strongest model needs no extra\n"
              f"     channels, so a high-density teacher has nothing the student cannot already\n"
              f"     reach. Distillation is not justified for THIS task — record it and\n"
              f"     reconsider before building distill.py.")

    with open(os.path.join(OUT, "teacher_bakeoff.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: Results/distilled/teacher_bakeoff.json")


if __name__ == "__main__":
    main()
