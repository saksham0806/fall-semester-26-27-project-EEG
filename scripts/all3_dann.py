"""
all3 domain-adversarial models — the one architecture that attacks the confound instead of
exploiting it.

Class is confounded with recording corpus: MODMA holds only {healthy, depression}, SAM40 and
CogLoad only {healthy, stress}, each on a different amplifier. Corpus identity is 91.6%
recoverable from the 81 features, and a classifier knowing ONLY the corpus scores 0.6267.
Every one of the 18 models run so far sits at or below that bar.

DANN puts a gradient reversal layer between the shared backbone and a corpus classifier, so
the backbone is pushed toward features that predict class but NOT corpus. Both the GRL and
the model are already implemented (src/proposed_model/multitask_dann_models.py); this script
only runs them at nb_classes=3, nb_domains=3.

EXPECTED OUTCOME, stated up front so it stays falsifiable: pooled accuracy should FALL toward
0.33, because the adversary is deliberately destroying the signature that produces 0.6267.
That is the measurement, not a failure. Success is judged on Track A only:

    CogLoad within-corpus accuracy > 0.6886

which no model has ever reached — the best so far is 0.5765, i.e. 0.11 BELOW a constant
predictor. That is the single real gap in this dataset.

The alpha=0.0 rows are controls, not filler: without them a change cannot be attributed to
the adversary rather than to the backbone.

Watch the domain-head accuracy. It should fall from ~0.92 toward 0.33 as alpha rises. If it
stays flat the GRL is not biting and the task numbers mean nothing.

Run:  export the CUDA paths from run.sh first, or this silently falls back to CPU.
      .venv/bin/python -u scripts/all3_dann.py
"""
import gc
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import tensorflow as tf
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight

from proposed_model.multitask_dann_models import build_dann_eeg_model
from all3_diagnostic import load, report, shortcut_baseline

OUT = os.path.join(PROJECT_ROOT, "Results", "all3")
N_SPLITS, SEED = 5, 42
EPOCHS, BATCH, LR = 25, 128, 5e-4

# 2 backbones x 4 alphas. alpha=0.0 disables the adversary -> plain 3-class backbone.
BACKBONES = ["TSception", "EEGTransformer"]
ALPHAS = [0.0, 0.1, 0.3, 1.0]


def oof_dann(backbone, alpha, X, y, subjects, corpus):
    """
    Out-of-fold 3-class predictions plus out-of-fold domain-head predictions.

    Protocol is copied from all3_deep.oof_deep so the numbers stay directly comparable:
    5-fold GroupKFold on subjects, inner validation split by subject, balanced class weights.
    """
    pred = np.full(len(y), -1)
    dom_pred = np.full(len(y), -1)

    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, groups=subjects):
        assert not (set(subjects[tr]) & set(subjects[te])), "subject leakage"
        tf.keras.backend.clear_session()
        gc.collect()
        tf.keras.utils.set_random_seed(SEED)

        model = build_dann_eeg_model(backbone_type=backbone, nb_classes=3,
                                     nb_domains=len(np.unique(corpus)), alpha=alpha)
        # The builder self-compiles with optimizer='adam' at Keras' default LR; recompile to
        # match the LR every other all3 script uses.
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
            loss={"task": "sparse_categorical_crossentropy",
                  "domain": "sparse_categorical_crossentropy"},
            loss_weights={"task": 1.0, "domain": 1.0},
            metrics={"task": "accuracy", "domain": "accuracy"})

        # Inner validation by subject, so early stopping never sees the held-out fold.
        inner = np.unique(subjects[tr])
        rng = np.random.default_rng(SEED)
        val_subs = set(rng.permutation(inner)[: max(1, len(inner) // 8)])
        vm = np.array([s in val_subs for s in subjects[tr]])
        itr, ival = tr[~vm], tr[vm]

        cls = np.unique(y[itr])
        cw = dict(zip(cls, compute_class_weight("balanced", classes=cls, y=y[itr])))

        # class_weight is ambiguous on a two-output model, so balance the task head with an
        # explicit per-sample weight and leave the domain head unweighted — the adversary
        # should see the corpus distribution as it actually is.
        model.fit(X[itr], {"task": y[itr], "domain": corpus[itr]},
                  validation_data=(X[ival], {"task": y[ival], "domain": corpus[ival]}),
                  epochs=EPOCHS, batch_size=BATCH, verbose=0,
                  sample_weight={"task": np.array([cw[c] for c in y[itr]], dtype="float32"),
                                 "domain": np.ones(len(itr), dtype="float32")},
                  callbacks=[tf.keras.callbacks.EarlyStopping(
                      monitor="val_loss", patience=5, restore_best_weights=True)])

        task_p, dom_p = model.predict(X[te], verbose=0)
        pred[te] = task_p.argmax(axis=1)
        dom_pred[te] = dom_p.argmax(axis=1)
        del model

    return pred, dom_pred


def main():
    os.makedirs(OUT, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)

    X, y, subjects, corpus = load()
    shortcut = shortcut_baseline(y, corpus)
    Xdl = X[..., np.newaxis]
    corpus = corpus.astype(int)

    print(f"all3: {len(y)} segments, {len(set(subjects))} subjects, "
          f"{len(np.unique(corpus))} corpora")
    print(f"shortcut to beat (corpus-ID only): {shortcut:.4f}")
    print(f"CogLoad within-corpus target:      0.6886 (no model has ever cleared it)")
    print(f"GPU: {'yes' if tf.config.list_physical_devices('GPU') else 'NO — running on CPU'}\n")

    results = {"shortcut_baseline": shortcut}

    for backbone in BACKBONES:
        print("=" * 78)
        print(f"DANN — {backbone} backbone, corpus as domain label")
        print("=" * 78)
        for alpha in ALPHAS:
            tag = "control, adversary off" if alpha == 0.0 else f"adversarial"
            name = f"DANN {backbone} alpha={alpha}"
            print(f"\n[{name}]  ({tag})")

            t0 = time.perf_counter()
            pred, dom_pred = oof_dann(backbone, alpha, Xdl, y, subjects, corpus)
            elapsed = time.perf_counter() - t0

            results = report(name, y, pred, corpus, shortcut, results, elapsed_sec=elapsed)

            # The diagnostic that says whether the adversary actually did anything.
            dom_acc = float((dom_pred == corpus).mean())
            results[name]["domain_accuracy"] = dom_acc
            results[name]["alpha"] = alpha
            results[name]["backbone"] = backbone
            print(f"    domain-head accuracy  {dom_acc:.4f}  "
                  f"(chance {1/len(np.unique(corpus)):.4f}; "
                  f"{'adversary is working' if dom_acc < 0.60 else 'corpus still recoverable'})")

    with open(os.path.join(OUT, "dann.json"), "w") as f:
        json.dump(results, f, indent=2)

    # --- summary against the criteria that actually matter -------------------------
    rows = [(k, v) for k, v in results.items() if isinstance(v, dict)]

    print("\n" + "=" * 78)
    print("TRACK A — within-corpus (the honest measure)")
    print("=" * 78)
    print(f"{'model':34s} {'MODMA':>16s} {'SAM40':>16s} {'CogLoad':>16s}")
    print(f"{'  (majority baseline)':34s} {0.5478:>16.4f} {0.7500:>16.4f} {0.6886:>16.4f}")
    for k, v in sorted(rows, key=lambda kv: -kv[1]["within_corpus"]["CogLoad"]["accuracy"]):
        w = v["within_corpus"]
        print(f"{k:34s}" + "".join(
            f"{w[c]['accuracy']:10.4f} {w[c]['margin']:+5.3f}" for c in ("MODMA", "SAM40", "CogLoad")))

    print("\n" + "=" * 78)
    print("TRACK B — pooled, expected to FALL as alpha rises")
    print("=" * 78)
    print(f"{'model':34s} {'acc':>8s} {'vs 0.6267':>11s} {'domain acc':>12s}")
    for k, v in sorted(rows, key=lambda kv: -kv[1]["accuracy"]):
        print(f"{k:34s} {v['accuracy']:8.4f} {v['margin']:+11.4f} {v['domain_accuracy']:12.4f}")

    won = [k for k, v in rows if v["within_corpus"]["CogLoad"]["accuracy"] > 0.6886]
    print(f"\nCRITERION 1 — CogLoad within-corpus > 0.6886: {won if won else 'NONE'}")
    if not won:
        print("  Clean negative result. No corpus contains more than two of the three classes,\n"
              "  so this is a data-collection blocker, not a modelling one. Record it in\n"
              "  datasets/all3/README.md and stop adding models to all3.")
    print(f"\nwritten: {os.path.join(OUT, 'dann.json')}")


if __name__ == "__main__":
    main()
