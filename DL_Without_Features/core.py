"""
Shared torch machinery for the raw-EEG track: fold trainer, embedding hook, masked knowledge
distillation loss, corpus probe, and the reporter.

Nothing here imports `src/` — that would drag in TensorFlow. The JSON schema written by
`report()` is kept identical to `scripts/all3_diagnostic.py:report()` so the two tracks stay
comparable and `scripts/all3_tracks.py` could consume it unchanged.

Self-check:  .venv/bin/python -m DL_Without_Features.core
"""
import json
import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from .data import (CLASS_NAMES, CORPUS_NAMES, N_SPLITS, RESULTS, folds)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, LR, PATIENCE, SEED = 50, 128, 1e-3, 5, 42

FoldOut = namedtuple("FoldOut", "pred emb prob hooked")


# --------------------------------------------------------------------------------------
# embedding extraction
# --------------------------------------------------------------------------------------

def attach_embedding_hook(model):
    """
    Capture the input to the classifier so the corpus probe can be run on what the model
    actually learned, not on its 3 output logits.

    braindecode 1.x standardizes the classifier module name to `final_layer`. Verified on
    ShallowFBCSPNet (40x28x1), Deep4Net (200), EEGNet (16x1x16), EEGConformer (32) — the
    shapes differ, so flatten.

    ponytail: ATCNet's `final_layer` is a ModuleList that is never called directly, so the
    hook never fires and the caller falls back to the logits (flagged in the JSON). A fix
    would be a per-model layer table; not worth it for 1 architecture in 17.
    """
    cap = {}
    mod = dict(model.named_modules()).get("final_layer")
    if mod is None:
        return cap, None
    handle = mod.register_forward_hook(
        lambda m, inp, out: cap.__setitem__("emb", inp[0].detach().flatten(1)))
    return cap, handle


# --------------------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------------------

def kd_loss(logits, y, teacher, T, lam, weight):
    """
    Cross-entropy on the hard 3-class label, plus a knowledge-distillation term masked to
    the two classes the segment's teacher can actually speak about.

    `teacher` is (B, 3) with -1 marking a class that teacher does not cover.

    THE MASK IS LOAD-BEARING. An unmasked KL forces P(depression)=0 on every SAM40/CogLoad
    segment, which teaches the student "not-MODMA => not depressed" — the corpus shortcut,
    re-derived from the loss function. Masking, plus renormalizing the student over the same
    pair, leaves the absent class free and keeps the student corpus-blind.
    """
    ce = F.cross_entropy(logits, y, weight=weight)
    if teacher is None or lam == 0:
        return ce

    mask = teacher >= 0                                        # (B, 3)
    # -1e9 rather than -inf: logsumexp of -inf backpropagates NaN. A large finite fill gives
    # a softmax weight of exactly 0 on the masked class, so its gradient is exactly 0.
    z = (logits / T).masked_fill(~mask, -1e9)
    logp = z - torch.logsumexp(z, dim=1, keepdim=True)          # renormalized over the pair

    t = teacher.clamp_min(0)
    t = t / t.sum(dim=1, keepdim=True).clamp_min(1e-8)
    kd = -(t * logp * mask).sum(dim=1).mean()
    return ce + lam * (T ** 2) * kd


# --------------------------------------------------------------------------------------
# fold trainer
# --------------------------------------------------------------------------------------

def _inner_split(tr, subjects, seed=SEED):
    """Hold out 1/8 of the TRAINING subjects for early stopping, so it never sees the fold."""
    inner = np.unique(subjects[tr])
    if len(inner) < 3:
        # Holding one out would leave 0-1 training subjects. Bites the per-corpus teachers in
        # distill.py (CogLoad has only 15 subjects) and every --smoke run. Early stopping then
        # monitors training loss, which is weak but does not crash or leak.
        return tr, tr
    rng = np.random.default_rng(seed)
    val_subs = set(rng.permutation(inner)[: max(1, len(inner) // 8)])
    vm = np.array([s in val_subs for s in subjects[tr]])
    return tr[~vm], tr[vm]


def fit_predict(build_fn, X, y, subjects, tr, te, n_outputs=3, teacher=None,
                T=2.0, lam=0.0, epochs=EPOCHS, batch=BATCH, lr=LR, patience=PATIENCE,
                seed=SEED):
    """
    Train one model on `tr`, return out-of-fold predictions and embeddings for `te`.
    Mirrors scripts/all3_deep.py:46-82 (inner subject split, balanced class weights, Adam,
    early stopping on val loss with best-weight restore) so results stay comparable.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    itr, ival = _inner_split(tr, subjects, seed)

    present = np.unique(y[itr])
    weight = torch.zeros(n_outputs)
    weight[present] = torch.tensor(
        compute_class_weight("balanced", classes=present, y=y[itr]), dtype=torch.float32)
    weight = weight.to(DEVICE)

    def loader(idx, shuffle):
        parts = [torch.from_numpy(np.ascontiguousarray(X[idx])).float(),
                 torch.from_numpy(y[idx]).long()]
        if teacher is not None:
            parts.append(torch.from_numpy(teacher[idx]).float())
        return DataLoader(TensorDataset(*parts), batch_size=batch, shuffle=shuffle)

    dl_tr, dl_va = loader(itr, True), loader(ival, False)

    model = build_fn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        for b in dl_tr:
            b = [t.to(DEVICE) for t in b]
            opt.zero_grad(set_to_none=True)
            kd_loss(model(b[0]), b[1], b[2] if teacher is not None else None,
                    T, lam, weight).backward()
            opt.step()

        model.eval()
        tot = k = 0
        with torch.no_grad():
            for b in dl_va:
                b = [t.to(DEVICE) for t in b]
                loss = kd_loss(model(b[0]), b[1], b[2] if teacher is not None else None,
                               T, lam, weight)
                tot += float(loss) * len(b[1])
                k += len(b[1])
        val = tot / max(k, 1)

        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    cap, handle = attach_embedding_hook(model)
    model.eval()
    probs, embs = [], []
    with torch.no_grad():
        for i in range(0, len(te), batch):
            xb = torch.from_numpy(np.ascontiguousarray(X[te[i:i + batch]])).float().to(DEVICE)
            out = model(xb)
            probs.append(F.softmax(out, dim=1).cpu().numpy())
            embs.append(cap.get("emb", out.detach()).cpu().numpy())
    if handle is not None:
        handle.remove()
    hooked = "emb" in cap

    del model, best_state
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    prob = np.concatenate(probs)
    return FoldOut(prob.argmax(1), np.concatenate(embs).astype(np.float16), prob, hooked)


def run_cv(build_fn, X, y, subjects, n_outputs=3, teacher=None, T=2.0, lam=0.0,
           n_splits=N_SPLITS, **kw):
    """Out-of-fold predictions + embeddings over subject-grouped folds.

    `teacher` may be (n, 3) or (n_folds, n, 3) — the latter when soft targets are produced
    per outer fold, which is what distill.py does.
    """
    pred = np.full(len(y), -1)
    prob = np.zeros((len(y), n_outputs), dtype=np.float32)
    emb, hooked = None, True
    for f, (tr, te) in enumerate(folds(y, subjects, n_splits)):
        t = teacher[f] if (teacher is not None and teacher.ndim == 3) else teacher
        out = fit_predict(build_fn, X, y, subjects, tr, te, n_outputs, t, T, lam, **kw)
        if emb is None:
            emb = np.zeros((len(y), out.emb.shape[1]), dtype=np.float16)
        pred[te], prob[te], emb[te] = out.pred, out.prob, out.emb
        hooked &= out.hooked
        print(f"      fold {f + 1}/{n_splits}  acc {(out.pred == y[te]).mean():.4f}", flush=True)
    assert (pred >= 0).all()
    return pred, emb, prob, hooked


# --------------------------------------------------------------------------------------
# corpus probe + reporting
# --------------------------------------------------------------------------------------

def corpus_probe(emb, corpus, subjects, n_splits=N_SPLITS):
    """
    How recoverable is the recording device from what the model learned? Majority is 0.4217;
    an SVM on the 81 handcrafted features reaches 0.9161. A model that beats the 0.6267
    accuracy bar with a probe near 0.97 is the shortcut in a new costume.
    """
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return float(cross_val_score(pipe, emb.astype(np.float32), corpus,
                                 cv=GroupKFold(n_splits=n_splits), groups=subjects,
                                 n_jobs=-1).mean())


def verdict(margin, probe, probe_floor):
    if margin <= 0.02:
        return "no better than device detection"
    return ("REAL SIGNAL — and corpus-free" if probe < probe_floor + 0.15
            else "beats the bar but rides the device signature")


def report(name, y, pred, corpus, subjects, shortcut, results, emb=None, extra=None,
           elapsed_sec=None, out_dir=RESULTS):
    """Same JSON schema as scripts/all3_diagnostic.py:report(), plus `corpus_probe`."""
    acc = float((pred == y).mean())
    bal = float(balanced_accuracy_score(y, pred))
    macro_f1 = float(f1_score(y, pred, average="macro"))
    margin = acc - shortcut
    probe_floor = float(np.bincount(corpus).max() / len(corpus))

    print(f"\n  {name}")
    print(f"    3-class accuracy      {acc:.4f}   balanced {bal:.4f}   macro-F1 {macro_f1:.4f}")
    print(f"    vs corpus-ID shortcut {shortcut:.4f}   margin {margin:+.4f}")

    p, r, f, s = precision_recall_fscore_support(y, pred, labels=sorted(CLASS_NAMES),
                                                 zero_division=0)
    print("      " + "".join(f"{CLASS_NAMES[i]:>12s}" for i in sorted(CLASS_NAMES)))
    for label, vals in (("precision", p), ("recall", r), ("f1", f)):
        print(f"    {label:9s}" + "".join(f"{v:12.4f}" for v in vals))

    within = {}
    for c in np.unique(corpus):
        m = corpus == c
        a = float((pred[m] == y[m]).mean())
        base = float(np.bincount(y[m]).max() / m.sum())
        within[CORPUS_NAMES[c]] = {"accuracy": a, "majority_within": base, "margin": a - base,
                                   "macro_f1": float(f1_score(y[m], pred[m], average="macro"))}
        print(f"    within {CORPUS_NAMES[c]:8s} {a:.4f}  (majority {base:.4f}, "
              f"margin {a - base:+.4f})")

    row = {"accuracy": acc, "balanced_accuracy": bal, "macro_f1": macro_f1,
           "shortcut_baseline": shortcut, "margin": margin,
           "per_class": {CLASS_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                          "f1": float(f[i]), "support": int(s[i])}
                         for i in sorted(CLASS_NAMES)},
           "confusion_matrix": confusion_matrix(y, pred,
                                                labels=sorted(CLASS_NAMES)).tolist(),
           "within_corpus": within}

    if emb is not None:
        probe = corpus_probe(emb, corpus, subjects)
        row["corpus_probe"] = probe
        row["corpus_probe_floor"] = probe_floor
        row["embedding_dim"] = int(emb.shape[1])
        print(f"    corpus probe on {emb.shape[1]}-d embedding  {probe:.4f}  "
              f"(floor {probe_floor:.4f})")
        row["verdict"] = verdict(margin, probe, probe_floor)
        print(f"    --> {row['verdict']}")

    if extra:
        row.update(extra)
    if elapsed_sec is not None:
        row["elapsed_sec"] = round(float(elapsed_sec), 1)
        print(f"    elapsed {elapsed_sec:.1f}s")

    results[name] = row

    # Persist out-of-fold arrays so a newly wanted metric is a recompute, not a retrain.
    key = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
    for sub, arr in (("preds", pred), ("emb", emb)):
        if arr is None:
            continue
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        np.save(os.path.join(out_dir, sub, f"{key}.npy"), arr)
    return results


def save(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)


def ranked(results, shortcut):
    rows = sorted(((v["accuracy"], v.get("corpus_probe"), k)
                   for k, v in results.items() if isinstance(v, dict)), reverse=True)
    print(f"\n{'RANKED':42s} {'acc':>8s} {'vs ' + format(shortcut, '.4f'):>11s} {'probe':>8s}")
    for acc, probe, k in rows:
        pr = f"{probe:8.4f}" if probe is not None else "       -"
        print(f"  {k:40s} {acc:8.4f} {acc - shortcut:+11.4f} {pr}")
    return rows


# --------------------------------------------------------------------------------------

def _selfcheck():
    torch.manual_seed(0)
    w = torch.ones(3)
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.0, 1.0, 0.5]])
    y = torch.tensor([0, 1])

    # 1. lam=0 reduces exactly to weighted cross-entropy
    teach = torch.tensor([[0.7, -1.0, 0.3], [0.4, 0.6, -1.0]])
    assert torch.allclose(kd_loss(logits, y, teach, 2.0, 0.0, w),
                          F.cross_entropy(logits, y, weight=w))
    assert torch.allclose(kd_loss(logits, y, None, 2.0, 1.0, w),
                          F.cross_entropy(logits, y, weight=w))

    # 2. masked KD matches a hand computation over the covered pair only
    T, lam = 2.0, 1.0
    exp = []
    for i, cover in enumerate([[0, 2], [0, 1]]):
        z = (logits[i] / T)[cover]
        lp = z - torch.logsumexp(z, 0)
        t = teach[i][cover]
        exp.append(-(t / t.sum() * lp).sum())
    want = F.cross_entropy(logits, y, weight=w) + lam * T ** 2 * torch.stack(exp).mean()
    assert torch.allclose(kd_loss(logits, y, teach, T, lam, w), want, atol=1e-6), "masked KD"

    # 3. THE property the mask exists for: a teacher that never covers class 2 must put
    #    zero KD gradient on the class-2 logit, so the student is never taught
    #    "not-MODMA => not depressed".
    only01 = torch.tensor([[0.6, 0.4, -1.0], [0.3, 0.7, -1.0]])
    grads = []
    for l in (0.0, 1.0):
        lg = torch.tensor([[0.3, 0.2, 0.1], [0.1, 0.4, 0.2]], requires_grad=True)
        kd_loss(lg, torch.tensor([0, 1]), only01, 2.0, l, w).backward()
        grads.append(lg.grad.clone())
    assert torch.isfinite(grads[1]).all(), grads[1]
    assert torch.allclose(grads[1][:, 2], grads[0][:, 2], atol=1e-7), (grads[0], grads[1])
    assert not torch.allclose(grads[1][:, :2], grads[0][:, :2]), "KD term is inert"

    # 4. the embedding hook fires and returns 2-D for the default architectures
    import warnings
    warnings.filterwarnings("ignore")
    import braindecode.models as bm
    x = torch.randn(4, 6, 512)
    fired = {}
    for n in ["ShallowFBCSPNet", "Deep4Net", "EEGNet", "EEGNeX", "EEGConformer",
              "ATCNet", "EEGITNet", "SCCNet"]:
        m = getattr(bm, n)(n_chans=6, n_times=512, n_outputs=3, sfreq=128)
        cap, h = attach_embedding_hook(m)
        m.eval()
        with torch.no_grad():
            m(x)
        if h is not None:
            h.remove()
        fired[n] = tuple(cap["emb"].shape) if "emb" in cap else None
        assert fired[n] is None or (len(fired[n]) == 2 and fired[n][0] == 4), (n, fired[n])
    assert fired["ATCNet"] is None, "ATCNet unexpectedly hooked — drop the logits fallback"
    assert all(v is not None for k, v in fired.items() if k != "ATCNet"), fired

    print("core.py OK — 9 asserts passed")
    print(f"  device: {DEVICE}")
    for n, shp in fired.items():
        print(f"  {n:18s} embedding {shp[1] if shp else 'LOGITS FALLBACK (3)'}")


if __name__ == "__main__":
    _selfcheck()
