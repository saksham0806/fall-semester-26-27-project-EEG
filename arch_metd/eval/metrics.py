"""Metrics — metd_plan.md §7.1, §7.3.

Every headline AUC ships with an interval. At n = 53 subjects the CI is roughly +/-0.13
even at AUC 0.80, so a point estimate on its own is not interpretable.
"""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score


def auc(y, score):
    """Point estimate, no interval. For per-unit means where a CI is not meaningful."""
    return float(roc_auc_score(y, score))


def auc_ci(y, score, n_boot=1000, seed=0):
    """AUC with a bootstrap 95% CI over the resampled unit (subject, trial, recording)."""
    y, score = np.asarray(y), np.asarray(score)
    auc = float(roc_auc_score(y, score))
    rng = np.random.default_rng(seed)
    boots = [roc_auc_score(y[i], score[i])
             for i in rng.integers(0, len(y), (n_boot, len(y)))
             if len(np.unique(y[i])) == 2]
    # a resample can come back single-class; with few units they all can
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return {"auc": auc, "ci_lo": float(lo), "ci_hi": float(hi), "n": int(len(y)),
            "n_pos": int(np.sum(y)), "crosses_chance": bool(lo <= 0.5 <= hi)}


def paired_sign_test(task_score, baseline_score):
    """§7.3: EEGMAT gives exactly one baseline and one task recording per subject.

    A within-subject AUC is degenerate on one pair, so ask the paired question instead —
    is p_str(task) > p_str(baseline)? — and test the proportion against 0.5 exactly.
    """
    higher = np.asarray(task_score) > np.asarray(baseline_score)
    k, n = int(higher.sum()), len(higher)
    return {"n": n, "n_higher": k, "proportion": k / n if n else float("nan"),
            "p_value": float(binomtest(k, n, 0.5).pvalue) if n else float("nan")}
