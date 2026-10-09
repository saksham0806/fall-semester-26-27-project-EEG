"""
Phase 2 M3 — self-supervised masked-reconstruction pretraining on the pooled corpus.

M2 showed a foundation model pretrained on thousands of hours (CBraMod) gives only
0.5716 as a frozen probe, against 0.7560 for M1's Riemannian features. That is not
automatically fatal for SSL: CBraMod was pretrained on a *different* montage and task
distribution (clinical TUH, many channels), whereas pretraining on this project's own
segments is domain-matched by construction. This tests the domain-matched version.

Labels are never used during pretraining, so AMIGOS is an unambiguous asset here — it
is 71% of the corpus and its arousal label (the part Phase 1 showed is undecodable) is
simply not involved.

Protocol matches M1/M2 exactly: frozen encoder -> logistic regression -> 5-fold
subject-grouped CV on SAM40 -> within-subject AUC.

Run:  ./run.sh is not needed; .venv/bin/python -u scripts/phase2_ssl_pretrain.py
"""
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(PROJECT_ROOT, "datasets", "combined_data")
OUT_DIR = os.path.join(PROJECT_ROOT, "Results", "phase2")
SAM40, SEED = 1, 42
PRETRAIN_N = 20000        # cap: the box is memory-tight, 20k is ample for this encoder
MASK_FRAC, EPOCHS, BATCH = 0.3, 15, 128
M1_BASELINE = 0.7560


def build_autoencoder(chans=6, samples=512, latent=128):
    """Conv encoder -> latent -> deconv decoder. Encoder is reused as the frozen probe."""
    inp = tf.keras.Input(shape=(samples, chans))
    x = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPooling1D(4)(x)
    x = tf.keras.layers.Conv1D(128, 7, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D(4)(x)
    emb = tf.keras.layers.GlobalAveragePooling1D(name="embedding")(x)

    d = tf.keras.layers.Dense((samples // 16) * 128, activation="relu")(emb)
    d = tf.keras.layers.Reshape((samples // 16, 128))(d)
    d = tf.keras.layers.UpSampling1D(4)(d)
    d = tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu")(d)
    d = tf.keras.layers.UpSampling1D(4)(d)
    out = tf.keras.layers.Conv1D(chans, 7, padding="same")(d)

    return tf.keras.Model(inp, out), tf.keras.Model(inp, emb)


def mask_timesteps(X, frac, rng):
    """Zero out contiguous-ish random timesteps; the target stays the clean signal."""
    Xm = X.copy()
    n_mask = int(X.shape[1] * frac)
    for i in range(len(Xm)):
        idx = rng.choice(X.shape[1], n_mask, replace=False)
        Xm[i, idx, :] = 0.0
    return Xm


def within_subject_auc(y, prob, subjects):
    s = [roc_auc_score(y[subjects == k], prob[subjects == k]) for k in np.unique(subjects)
         if len(np.unique(y[subjects == k])) > 1]
    return float(np.mean(s)) if s else float("nan")


def probe(F, y, subj):
    scores = []
    for tr, te in GroupKFold(n_splits=5).split(F, y, groups=subj):
        sc = StandardScaler().fit(F[tr])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(sc.transform(F[tr]), y[tr])
        scores.append(within_subject_auc(y[te], clf.predict_proba(sc.transform(F[te]))[:, 1], subj[te]))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)

    X = np.load(os.path.join(DATA_DIR, "X.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_stress.npy"))
    subj = np.load(os.path.join(DATA_DIR, "subjects.npy"), allow_pickle=True)
    ds = np.load(os.path.join(DATA_DIR, "datasets.npy"))
    v = ~np.isnan(X).any(axis=(1, 2))
    X, y, subj, ds = X[v], y[v], subj[v], ds[v]

    sam = (ds == SAM40) & (y != -1)
    # Pretrain on everything EXCEPT the SAM40 evaluation corpus, so the probe is never
    # fitted on data the encoder memorised. Labels are unused throughout.
    pool = np.where(~sam)[0]
    pool = rng.choice(pool, min(PRETRAIN_N, len(pool)), replace=False)
    Xp = np.transpose(X[pool], (0, 2, 1))  # (N, 512, 6)
    print(f"pretraining on {len(Xp)} unlabelled segments (SAM40 held out entirely)")

    ae, encoder = build_autoencoder()
    ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    ae.fit(mask_timesteps(Xp, MASK_FRAC, rng), Xp,
           epochs=EPOCHS, batch_size=BATCH, validation_split=0.1, verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                       restore_best_weights=True)])
    print(f"  reconstruction MSE: {ae.evaluate(mask_timesteps(Xp, MASK_FRAC, rng), Xp, verbose=0):.5f}")

    Xs = np.transpose(X[sam], (0, 2, 1))
    ys, ss = y[sam].astype(int), subj[sam]
    F_ssl = encoder.predict(Xs, verbose=0)

    # Control: identical architecture, no pretraining. Isolates what SSL bought.
    tf.keras.utils.set_random_seed(SEED + 1)
    _, enc_rand = build_autoencoder()
    F_rand = enc_rand.predict(Xs, verbose=0)

    results = {}
    for tag, F in [("ssl pretrained", F_ssl), ("random init (control)", F_rand)]:
        m, s = probe(F, ys, ss)
        results[tag] = {"auc_within": [m, s], "dims": int(F.shape[1])}
        print(f"  {tag:22s} within-subj AUC = {m:.4f} +/- {s:.4f}   "
              f"(M1 {M1_BASELINE:.4f}, delta {m - M1_BASELINE:+.4f})")

    gain = results["ssl pretrained"]["auc_within"][0] - results["random init (control)"]["auc_within"][0]
    print(f"\nSSL gain over random init: {gain:+.4f}")

    with open(os.path.join(OUT_DIR, "ssl_pretrain.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"written: {os.path.join(OUT_DIR, 'ssl_pretrain.json')}")


if __name__ == "__main__":
    main()
