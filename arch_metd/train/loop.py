"""Training loop — metd_plan.md §6.3, §6.4.

    python -m arch_metd.train.loop --smoke     # 1 outer fold, few epochs, CPU
    python -m arch_metd.train.loop             # full nested CV, GPU box

Local torch is CPU-only (2.13.0+cpu), so the real run belongs on the GPU workstation per
GPU_Run.md. ``--smoke`` exists to prove the loop runs before it gets there.
"""
import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ..contract import ROOT
from ..data.build import load
from ..models.grl import lambda_at
from ..models.heads import anchor_weights
from ..models.mtdann import MTDANN
from .folds import inner_folds, outer_folds

RESULTS = ROOT / "arch_metd/results"
BATCH = 64
LR, WEIGHT_DECAY = 1e-3, 1e-2
PATIENCE = 15
EPOCHS = 100


def _tensors(d, idx):
    return TensorDataset(
        torch.from_numpy(d["X"][idx]).unsqueeze(1),
        torch.from_numpy(d["y_dep"][idx]), torch.from_numpy(d["y_str"][idx]),
        torch.from_numpy(d["y_domain"][idx]), torch.from_numpy(d["anchor"][idx]))


def _loader(d, idx, train):
    """§6.3: corpus-balanced sampling, so the largest corpus does not set the objective.

    Mumtaz alone is 44% of all segments; unbalanced, the shared backbone spends most of
    its gradient on one amplifier.
    """
    ds = _tensors(d, idx)
    if not train:
        return DataLoader(ds, batch_size=256)
    counts = np.bincount(d["y_domain"][idx], minlength=5)
    w = np.where(counts[d["y_domain"][idx]] > 0, 1.0 / counts[d["y_domain"][idx]], 0.0)
    sampler = WeightedRandomSampler(torch.from_numpy(w).double(), len(idx), replacement=True)
    return DataLoader(ds, batch_size=BATCH, sampler=sampler, drop_last=True)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = {"dep": [], "str": [], "domain": [], "y_dep": [], "y_str": [], "y_domain": []}
    for x, y_dep, y_str, y_domain, _ in loader:
        _, dep, stress, dom = model(x.to(device))
        out["dep"].append(torch.sigmoid(dep).squeeze(-1).cpu())
        out["str"].append(torch.sigmoid(stress).squeeze(-1).cpu())
        out["domain"].append(dom.argmax(-1).cpu())
        out["y_dep"].append(y_dep); out["y_str"].append(y_str); out["y_domain"].append(y_domain)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def valid_auc(p):
    """Sum of the two heads' AUCs over the samples that carry each label (§6.3)."""
    total, parts = 0.0, {}
    for head in ("dep", "str"):
        y = p[f"y_{head}"]
        m = y != -1
        if m.sum() and len(np.unique(y[m])) == 2:
            parts[head] = float(roc_auc_score(y[m], p[head][m]))
            total += parts[head]
    return total, parts


def youden(y, score):
    """§6.4: threshold maximizing TPR - FPR, chosen on the inner fold only."""
    order = np.argsort(-score)
    y = y[order]
    tpr = np.cumsum(y) / max(y.sum(), 1)
    fpr = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.sort(score)[::-1][np.argmax(tpr - fpr)])


def train_fold(d, train_idx, val_idx, device, epochs=EPOCHS, patience=PATIENCE, seed=0):
    torch.manual_seed(seed)
    model = MTDANN().to(device)
    weights = anchor_weights(d["y_domain"][train_idx], d["anchor"][train_idx])
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingWarmRestarts(opt, T_0=max(epochs // 4, 1))
    train_loader, val_loader = _loader(d, train_idx, True), _loader(d, val_idx, False)

    best = {"score": -np.inf, "epoch": -1, "state": None, "lambda": 0.0}
    for epoch in range(epochs):
        model.train()
        lambd = lambda_at(epoch / max(epochs - 1, 1))
        for batch in train_loader:
            batch = [t.to(device) for t in batch]
            loss, _ = model.loss(batch, weights, lambd)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.backbone.apply_max_norm()
        sched.step()

        score, parts = valid_auc(predict(model, val_loader, device))
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "lambda": lambd, "parts": parts,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        # §5.3: early stopping is LOCKED until lambda >= 0.5. Without the lock, patience
        # can fire while lambda is still near zero and you ship the un-adapted model by
        # accident — the adversary would never have been switched on.
        if lambd >= 0.5 and epoch - best["epoch"] >= patience:
            break

    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "lambda_at_stop": round(lambd, 4),
                   "lambda_at_best": round(best["lambda"], 4),
                   "stopped_epoch": epoch, "inner_auc": best["parts"]}


def run(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load()
    epochs = 3 if smoke else EPOCHS
    results = []
    for i, (train_idx, test_idx) in enumerate(outer_folds(d)):
        if smoke:
            train_idx = train_idx[::20]
        inner_train, inner_val = next(inner_folds(d, train_idx))
        t0 = time.time()
        model, info = train_fold(d, inner_train, inner_val, device,
                                 epochs=epochs, patience=1 if smoke else PATIENCE)
        # the outer test fold is touched exactly once, here, after everything is fixed
        p = predict(model, _loader(d, test_idx, False), device)
        score, parts = valid_auc(p)
        # thresholds come from the inner fold, never from the outer test fold
        pv = predict(model, _loader(d, inner_val, False), device)
        thresholds = {}
        for head in ("dep", "str"):
            m = pv[f"y_{head}"] != -1
            if m.sum() and len(np.unique(pv[f"y_{head}"][m])) == 2:
                thresholds[head] = youden(pv[f"y_{head}"][m], pv[head][m])
        info |= {"fold": i, "test_auc": parts, "thresholds": thresholds,
                 "domain_acc": float((p["domain"] == p["y_domain"]).mean()),
                 "elapsed_sec": round(time.time() - t0, 1)}
        results.append(info)
        print(f"  fold {i}: {info['test_auc']} | lambda@stop {info['lambda_at_stop']} "
              f"| epoch {info['best_epoch']}/{info['stopped_epoch']} "
              f"| {info['elapsed_sec']}s")
        np.savez(RESULTS / f"preds_fold{i}{'_smoke' if smoke else ''}.npz", **p)

    out = RESULTS / ("cv_smoke.json" if smoke else "cv.json")
    out.write_text(json.dumps({"device": device, "epochs": epochs, "folds": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1/20 of the data, 3 epochs, CPU")
    a = ap.parse_args()
    if not a.smoke:
        # §8.6: a prior run returned MODMA subject-level AUC 0.3434 with one fold at
        # exactly 0.0000 — inversion, not chance, from a broken label join. Verify the
        # join before spending a GPU queue slot on it.
        from ..data.verify import check_labels
        check_labels(load())
    run(smoke=a.smoke)
