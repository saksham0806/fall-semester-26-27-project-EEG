"""
THE GATE. Does a high-density teacher actually know more than a 6-channel student?

The whole distillation premise is that 128 channels carry discriminative structure the frontal
6 do not. `Results/phase2/channel_density.json` argues against it: under a Riemannian pipeline
MODMA went 6->16 ch for +0.023 subject AUC and 6->64 ch for +0.000. If that also holds for a
deep model there is nothing to distil and `distilled/` should be redesigned, not built.

That earlier sweep used tangent-space features, whose dimensionality grows O(ch^2) and starves
against 53 subjects — which is a plausible reason it saw no gain. A deep model does not have
that failure mode, so this is a fair re-test rather than a repeat.

DECISION RULE
    teacher(128ch) subject AUC >= student(6ch) subject AUC + 0.05   ->  PROCEED
    otherwise                                                      ->  STOP, redesign

Two details that make the comparison honest:
  * the 6 student channels are a strict SUBSET of the 64/128 sets, so the teacher sees
    everything the student sees plus more. phase2_channel_density.py:119 used np.linspace sets
    that do NOT contain the frontal six, so its 6-vs-N comparison was confounded by WHICH
    channels changed, not just how many.
  * identical windows, folds, seed and architecture across channel counts. Only n_chans varies.

Run:  .venv/bin/python -u distilled/scripts/00_validate_premise.py
"""
import json
import os
import re
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.signal import resample
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from distilled.montages import MODMA

MODMA_DIR = os.path.join(PROJECT_ROOT, "datasets", "MODMA",
                         "EEG_128channels_resting_lanzhou_2015")
INFO_XLSX = os.path.join(MODMA_DIR,
                         "subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx")
OUT = os.path.join(PROJECT_ROOT, "Results", "distilled")

NATIVE_FS, FS, WINDOW, STRIDE = 250, 128, 512, 256
N_SPLITS, SEED = 5, 42
EPOCHS, BATCH, LR = 20, 64, 1e-3
CHANNEL_SETS = [6, 32, 128]
MARGIN = 0.05


def z_score_normalize(seg):
    """Verbatim from scripts/combine_datasets.py:25 — keep identical or nothing compares."""
    med = np.median(seg)
    mad = np.median(np.abs(seg - med)) + 1e-8
    clipped = np.clip(seg, med - 5 * mad, med + 5 * mad)
    return (clipped - np.mean(clipped)) / (np.std(clipped) + 1e-8)


def channel_subset(n):
    """
    n channels including the frontal six, so smaller sets nest inside larger ones.

    Without the nesting the comparison answers "do these OTHER channels help", not "does MORE
    help" — which is the question the gate is asking.
    """
    frontal = list(MODMA.student_idx)
    if n <= len(frontal):
        return sorted(frontal)
    rest = [i for i in np.linspace(0, 127, n).astype(int).tolist() if i not in frontal]
    return sorted(frontal + rest[: n - len(frontal)])


def load_modma():
    """All 128 channels, windowed once. Channel subsets are sliced from this array."""
    import pandas as pd
    info = pd.read_excel(INFO_XLSX)
    idcol = next(c for c in info.columns if "subject" in str(c).lower() and "id" in str(c).lower())
    typecol = next(c for c in info.columns if "type" in str(c).lower())
    label = {str(r[idcol]).strip(): (1 if "mdd" in str(r[typecol]).lower() else 0)
             for _, r in info.iterrows()}

    X, y, subj = [], [], []
    files = sorted(f for f in os.listdir(MODMA_DIR) if f.endswith(".mat"))
    # Filenames are inconsistent: '02010002rest 20150416 1017..mat', '02010008_rest ....mat',
    # '02010021 20150805 1730.mat.mat'. Splitting on 'rest' or '_' matches only 37/53. The
    # leading zero-padded digit run is the subject id in every case -> 53/53.
    for i, fn in enumerate(files):
        m = re.match(r"0*(\d+)", fn)
        sid = m.group(1) if m else None
        if sid not in label:
            print(f"\n  WARNING: no label for {fn!r} (parsed id {sid!r}) — skipped")
            continue
        mat = loadmat(os.path.join(MODMA_DIR, fn))
        # First 2-D array with 128/129 rows. Impedances_0 is (129,1) and also matches, so
        # require a real time axis too (combine_datasets.py:114 does not, and gets away with it
        # only because dict order happens to favour the data key).
        arr = next(v for k, v in mat.items()
                   if isinstance(v, np.ndarray) and v.ndim == 2
                   and v.shape[0] in (128, 129) and v.shape[1] > 1000)
        arr = arr[:128].astype(np.float64)
        arr = resample(arr, int(arr.shape[1] * FS / NATIVE_FS), axis=1)
        for s in range(0, arr.shape[1] - WINDOW + 1, STRIDE):
            X.append(z_score_normalize(arr[:, s:s + WINDOW]).astype(np.float32))
            y.append(label[sid])
            subj.append(sid)
        print(f"\r  loaded {i + 1}/{len(files)} subjects", end="", flush=True)
    print()
    return np.stack(X), np.array(y), np.array(subj)


def make_model(n_chans):
    from braindecode.models import EEGNet
    return EEGNet(n_chans=n_chans, n_outputs=2, n_times=WINDOW, sfreq=FS)


def subject_auc(y, prob, subjects):
    """One score per subject (depression is a subject-level trait), then AUC over subjects."""
    su = np.unique(subjects)
    sy = np.array([y[subjects == s][0] for s in su])
    sp = np.array([prob[subjects == s].mean() for s in su])
    return float(roc_auc_score(sy, sp))


def run(X, y, subjects, chans):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prob = np.zeros(len(y))

    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        model = make_model(len(chans))
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()

        Xtr = torch.from_numpy(X[tr][:, chans]); ytr = torch.from_numpy(y[tr]).long()
        Xte = torch.from_numpy(X[te][:, chans])

        model.train()
        for _ in range(EPOCHS):
            perm = torch.randperm(len(Xtr))
            for b in range(0, len(perm), BATCH):
                idx = perm[b:b + BATCH]
                opt.zero_grad()
                loss = lossf(model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            out = []
            for b in range(0, len(Xte), 256):
                out.append(torch.softmax(model(Xte[b:b + 256]), dim=1)[:, 1].numpy())
        prob[te] = np.concatenate(out)
        del model
    return prob


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    print(f"loading MODMA at 128 channels from {os.path.relpath(MODMA_DIR, PROJECT_ROOT)}")
    X, y, subjects = load_modma()
    print(f"  {X.shape} windows, {len(set(subjects))} subjects, "
          f"{int(y.sum())} MDD / {int((y == 0).sum())} HC segments "
          f"({X.nbytes / 1e9:.2f} GB)\n")

    results = {}
    print("=" * 74)
    print("PREMISE GATE — does more channels help a DEEP model on MODMA?")
    print("=" * 74)
    print(f"  {'chans':>6s}{'subject AUC':>14s}{'segment AUC':>14s}{'minutes':>10s}")

    for n in CHANNEL_SETS:
        chans = channel_subset(n)
        assert set(MODMA.student_idx).issubset(set(chans)), "frontal 6 must nest inside"
        t0 = time.perf_counter()
        prob = run(X, y, subjects, chans)
        mins = (time.perf_counter() - t0) / 60
        sa = subject_auc(y, prob, subjects)
        ga = float(roc_auc_score(y, prob))
        results[n] = {"subject_auc": sa, "segment_auc": ga, "n_chans": len(chans),
                      "minutes": round(mins, 1), "channels": chans}
        print(f"  {n:>6d}{sa:>14.4f}{ga:>14.4f}{mins:>10.1f}")

    base = results[CHANNEL_SETS[0]]["subject_auc"]
    best_n = max(CHANNEL_SETS[1:], key=lambda n: results[n]["subject_auc"])
    best = results[best_n]["subject_auc"]
    gain = best - base
    passed = gain >= MARGIN

    results["verdict"] = {"baseline_6ch": base, "best_highdensity": best,
                          "best_n_chans": best_n, "gain": gain,
                          "margin_required": MARGIN, "passed": bool(passed)}

    print("\n" + "=" * 74)
    print(f"  6-channel baseline      {base:.4f}")
    print(f"  best high-density ({best_n:3d}ch) {best:.4f}")
    print(f"  gain                    {gain:+.4f}   (need >= +{MARGIN:.2f})")
    print("=" * 74)
    if passed:
        print("  PASS — the teacher knows more. Distillation has something to transfer.\n"
              "         Proceed to 01_build_cache.py.")
    else:
        print("  FAIL — a high-density deep model is NOT materially better than 6 channels.\n"
              "         Distillation has nothing to transfer; do not build the KD pipeline.\n"
              "         This confirms Results/phase2/channel_density.json under a second,\n"
              "         independent model family — the 6-channel bottleneck is not what\n"
              "         limits this dataset. That is a real finding; record it and redesign.")

    with open(os.path.join(OUT, "premise_gate.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten: Results/distilled/premise_gate.json")


if __name__ == "__main__":
    main()
