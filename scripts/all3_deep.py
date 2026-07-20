"""
all3 Tier 2 (deep baselines) + Tier 3 F3 (gated fusion).

Tier 2 runs the six existing architectures at nb_classes=3. This is a parameter change,
not new code: _get_classification_layer (src/proposed_model/eeg_models.py:4-9) already
branches to a softmax head with sparse_categorical_crossentropy when nb_classes > 1, and
that path had never been called in this project.

F3 is DSFFN's gated fusion — a learned sigmoid gate blending a TSception stream with the
81 handcrafted features. build_dsffn now takes nb_classes for the 3-class head; passing
nothing preserves its original dual-sigmoid multi-task outputs.

Marked low-priority in plans/all3_plan.md: three independent lines of evidence say this family
loses on this data. Run for completeness of the comparison, not in expectation of a win.

Run:  ./run.sh is not used; export CUDA paths first or this falls back to CPU.
      .venv/bin/python -u scripts/all3_deep.py
"""
import gc
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
sys.path.insert(0, PROJECT_ROOT)  # dsffn uses relative imports -> needs src as a package

import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

from proposed_model import (build_1d_cnn, build_lstm, build_cnn_lstm,
                            EEGNet, TSception, EEGTransformer)
from proposed_model.features import extract_all_features
from all3_diagnostic import load, report, shortcut_baseline
from src.dsffn.model import build_dsffn  # not `dsffn.model`: relative imports need the package root

OUT = os.path.join(PROJECT_ROOT, "Results", "all3")
N_SPLITS, SEED = 5, 42
EPOCHS, BATCH, LR = 25, 128, 5e-4


def oof_deep(build_fn, X, y, subjects, extra=None):
    """
    Out-of-fold 3-class predictions. `extra` is a second input array (the 81 features)
    for the dual-input fusion model; None for single-input architectures.
    """
    pred = np.full(len(y), -1)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(SEED)

        model = build_fn()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        # Inner validation by subject, so early stopping never sees the held-out fold.
        inner = np.unique(subjects[tr])
        rng = np.random.default_rng(SEED)
        val_subs = set(rng.permutation(inner)[: max(1, len(inner) // 8)])
        vm = np.array([s in val_subs for s in subjects[tr]])
        itr, ival = tr[~vm], tr[vm]

        cls = np.unique(y[itr])
        cw = dict(zip(cls, compute_class_weight("balanced", classes=cls, y=y[itr])))

        Xi = [X[itr], extra[itr]] if extra is not None else X[itr]
        Xv = [X[ival], extra[ival]] if extra is not None else X[ival]
        Xt = [X[te], extra[te]] if extra is not None else X[te]

        model.fit(Xi, y[itr], validation_data=(Xv, y[ival]),
                  epochs=EPOCHS, batch_size=BATCH, class_weight=cw, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=5, restore_best_weights=True)])
        pred[te] = model.predict(Xt, verbose=0).argmax(axis=1)
        del model
    return pred


def main():
    os.makedirs(OUT, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    X, y, subjects, corpus = load()
    shortcut = shortcut_baseline(y, corpus)
    Xdl = X[..., np.newaxis]
    print(f"all3: {len(y)} segments, {len(set(subjects))} subjects")
    print(f"shortcut to beat (corpus-ID only): {shortcut:.4f}")
    print(f"GPU: {'yes' if tf.config.list_physical_devices('GPU') else 'NO — running on CPU'}\n")

    results = {"shortcut_baseline": shortcut}

    print("=" * 78)
    print("TIER 2 — deep baselines at nb_classes=3")
    print("=" * 78)
    for name, fn in [("D1 1D CNN", lambda: build_1d_cnn(nb_classes=3)),
                     ("D2 LSTM", lambda: build_lstm(nb_classes=3)),
                     ("D3 CNN-LSTM", lambda: build_cnn_lstm(nb_classes=3)),
                     ("D4 EEGNet", lambda: EEGNet(nb_classes=3)),
                     ("D5 TSception", lambda: TSception(nb_classes=3)),
                     ("D6 EEGTransformer", lambda: EEGTransformer(nb_classes=3))]:
        results = report(name, y, oof_deep(fn, Xdl, y, subjects), corpus, shortcut, results)

    print("\n" + "=" * 78)
    print("TIER 3 — F3 gated fusion (DSFFN: TSception stream + 81 features)")
    print("=" * 78)
    F81 = extract_all_features(X).astype("float32")
    results = report("F3 DSFFN gated fusion", y,
                     oof_deep(lambda: build_dsffn(nb_classes=3), Xdl, y, subjects, extra=F81),
                     corpus, shortcut, results)

    with open(os.path.join(OUT, "models_deep.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 78)
    ranked = sorted(((v["accuracy"], k) for k, v in results.items() if isinstance(v, dict)),
                    reverse=True)
    print(f"{'RANKED':44s} {'acc':>8s} {'vs 0.6267':>11s}")
    for acc, k in ranked:
        print(f"  {k:42s} {acc:8.4f} {acc - shortcut:+11.4f}")
    beat = [k for a, k in ranked if a > shortcut + 0.02]
    print(f"\nmodels clearing the shortcut by >0.02: {beat if beat else 'NONE'}")
    print(f"written: {os.path.join(OUT, 'models_deep.json')}")


if __name__ == "__main__":
    main()
