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

from ..contract import CORPORA, ROOT
from ..data.build import fingerprint, load
from ..models.grl import lambda_at
from ..models.heads import anchor_weights, positive_weight
from ..models.mtdann import MTDANN
from .folds import TRANSFERS, inner_folds, outer_folds, transfer

RESULTS = ROOT / "arch_metd/results"
BATCH = 64
LR, WEIGHT_DECAY = 1e-3, 1e-2
PATIENCE = 15
EPOCHS = 100
LAMBDA_FLOOR = 0.5


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


def permute_within_corpus(d, seed=0):
    """§8.3 negative control: shuffle labels within each corpus. AUC must collapse to 0.50.

    Permuted at the AGGREGATION UNIT, not the segment — subject for the depression
    corpora, trial for the stress ones. Shuffling segment labels inside a subject would
    leave each subject's mean label intact, so a subject-level metric would barely move
    and the control would pass while testing nothing.
    """
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in d.items()}
    for c in CORPORA:
        m = d["y_domain"] == c.domain
        head = "y_dep" if c.task == "depression" else "y_str"
        unit = d["subject_id"] if c.task == "depression" else d["trial_id"]
        ids = np.unique(unit[m])
        labels = np.array([d[head][m & (unit == u)][0] for u in ids])
        for u, y in zip(ids, rng.permutation(labels)):
            out[head][m & (unit == u)] = y
        after = np.array([out[head][m & (unit == u)][0] for u in ids])
        # the UNIT-level label multiset is the invariant, not the segment-level one:
        # units differ wildly in length (Mumtaz 93-336 segments), so swapping two units
        # of opposite class legitimately moves the segment-level balance
        assert np.array_equal(np.sort(labels), np.sort(after)), c.name
        assert (out[head][m] != -1).sum() == (d[head][m] != -1).sum(), c.name
    return out


def train_fold(d, train_idx, val_idx, device, epochs=EPOCHS, patience=PATIENCE, seed=0,
               lambda_fn=lambda_at, floor=LAMBDA_FLOOR, random_backbone=False):
    torch.manual_seed(seed)
    model = MTDANN().to(device)
    if random_backbone:
        # §8.3: a frozen random feature extractor with trained heads. eval() as well as
        # requires_grad_(False), or dropout keeps resampling a representation that is
        # supposed to be fixed.
        model.backbone.requires_grad_(False)
    weights = anchor_weights(d["y_domain"][train_idx], d["anchor"][train_idx])
    # from the training split only — computing it over all data would leak the test
    # fold's class balance into the loss
    pos_weight = {"dep": positive_weight(d["y_dep"][train_idx]),
                  "str": positive_weight(d["y_str"][train_idx])}
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = CosineAnnealingWarmRestarts(opt, T_0=max(epochs // 4, 1))
    train_loader, val_loader = _loader(d, train_idx, True), _loader(d, val_idx, False)

    best = {"score": -np.inf, "epoch": -1, "state": None, "lambda": 0.0}
    fallback = None
    for epoch in range(epochs):
        model.train()
        if random_backbone:
            model.backbone.eval()
        lambd = lambda_fn(epoch / max(epochs - 1, 1))
        for batch in train_loader:
            batch = [t.to(device) for t in batch]
            loss, _ = model.loss(batch, weights, lambd, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if not random_backbone:
                model.backbone.apply_max_norm()
        sched.step()

        score, parts = valid_auc(predict(model, val_loader, device))
        snapshot = {"score": score, "epoch": epoch, "lambda": lambd, "parts": parts,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        fallback = snapshot
        # Only epochs with the adversary actually on are eligible to be SHIPPED. The
        # lock below delays stopping, but validation AUC peaks early — measured, the
        # best epoch landed at 4-8 with lambda 0.20-0.38 in 3 of 5 folds, so the
        # selected checkpoint was un-adapted while lambda_at_stop read a reassuring
        # 0.74-0.82. Gating selection, not just stopping, is what closes that.
        if lambd >= floor and score > best["score"]:
            best = snapshot
        # §5.3: early stopping is LOCKED until lambda >= 0.5. Without the lock, patience
        # can fire while lambda is still near zero and you ship the un-adapted model by
        # accident — the adversary would never have been switched on.
        if lambd >= floor and epoch - best["epoch"] >= patience:
            break

    if best["state"] is None:
        # no epoch ever reached the floor (too short a budget); ship the last epoch and
        # say so rather than silently returning an untrained model
        best = fallback
        best["below_floor"] = True
    model.load_state_dict(best["state"])
    return model, {"best_epoch": best["epoch"], "lambda_at_stop": round(lambd, 4),
                   "pos_weight": {k: None if v is None else round(float(v), 4)
                                  for k, v in pos_weight.items()},
                   "lambda_at_best": round(best["lambda"], 4),
                   "selected_below_lambda_floor": bool(best.get("below_floor", False)),
                   "stopped_epoch": epoch, "inner_auc": best["parts"]}


MODES = {
    "main": {},
    # the method-off control. §12 item 9 and the DANN post-mortem in CLAUDE.md: without
    # it, "the adversary helped" is unfalsifiable.
    "lambda0": {"lambda_fn": lambda p: 0.0, "floor": -np.inf},
    "permuted": {"permute": True},
    "randbb": {"random_backbone": True},
}


def run_transfer(name, mode, d, device, epochs, patience):
    """One cross-corpus run (§6.2, §7.5). Train and test corpora are disjoint.

    This is what the architecture actually claims. Within-corpus CV measures decodability
    only: corpus identity is present in the test fold too, so a model can ride the device
    signature and no accuracy column will say so. Only transfer tests whether the
    representation survives a hardware change.
    """
    train_idx, test_idx = transfer(d, name)
    cfg = dict(MODES[mode])
    cfg.pop("permute", None)
    inner_train, inner_val = next(inner_folds(d, train_idx))
    t0 = time.time()
    model, info = train_fold(d, inner_train, inner_val, device, epochs=epochs,
                             patience=patience, **cfg)
    p = predict(model, _loader(d, test_idx, False), device)
    pv = predict(model, _loader(d, inner_val, False), device)
    thresholds = {}
    for h in ("dep", "str"):
        m = pv[f"y_{h}"] != -1
        if m.sum() and len(np.unique(pv[f"y_{h}"][m])) == 2:
            thresholds[h] = youden(pv[f"y_{h}"][m], pv[h][m])
    n_train_domains = len(np.unique(d["y_domain"][train_idx]))
    info |= {"transfer": name, "mode": mode, "thresholds": thresholds,
             "train_domains": np.unique(d["y_domain"][train_idx]).tolist(),
             "test_domains": np.unique(d["y_domain"][test_idx]).tolist(),
             "n_train": len(train_idx), "n_test": len(test_idx),
             # With one training corpus there is nothing to align against, so the
             # adversary is disabled at source (heads.domain_loss) and main is identical
             # to lambda0 by construction. Three of the four §6.2 transfers are in that
             # position, which means only the primary stress direction can test domain
             # adaptation at all.
             "adversary_operative": n_train_domains > 1,
             "elapsed_sec": round(time.time() - t0, 1)}
    for k in ("subject_id", "trial_id", "condition"):
        p[k] = d[k][test_idx]
    tag = f"transfer_{name}" + ("" if mode == "main" else f"_{mode}")
    np.savez(RESULTS / f"preds_{tag}.npz", **p)
    torch.save(model.state_dict(), RESULTS / f"model_{tag}.pt")
    print(f"  {name:28s} {mode:8s} lambda@stop {info['lambda_at_stop']:.3f} "
          f"| adversary {'on' if info['adversary_operative'] else 'OFF (1 train corpus, nothing to align)'} "
          f"| {info['elapsed_sec']}s")
    return info


def run_all_transfers(smoke=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load()
    epochs = 3 if smoke else EPOCHS
    out = []
    for name in TRANSFERS:
        for mode in ("main", "lambda0"):  # every transfer with its own method-off control
            out.append(run_transfer(name, mode, d, device, epochs,
                                    1 if smoke else PATIENCE))
    path = RESULTS / f"transfer{'_smoke' if smoke else ''}.json"
    path.write_text(json.dumps({"device": device, "epochs": epochs,
                                "fingerprint": fingerprint(d), "runs": out}, indent=2))
    print(f"wrote {path}")


def run(smoke=False, mode="main"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = load()
    cfg = dict(MODES[mode])
    if cfg.pop("permute", False):
        d = permute_within_corpus(d)
    tag = ("" if mode == "main" else f"_{mode}") + ("_smoke" if smoke else "")
    epochs = 3 if smoke else EPOCHS
    results = []
    for i, (train_idx, test_idx) in enumerate(outer_folds(d)):
        if smoke:
            train_idx = train_idx[::20]
        inner_train, inner_val = next(inner_folds(d, train_idx))
        t0 = time.time()
        model, info = train_fold(d, inner_train, inner_val, device, epochs=epochs,
                                 patience=1 if smoke else PATIENCE, **cfg)
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
        for name in ("subject_id", "trial_id", "condition"):
            p[name] = d[name][test_idx]
        np.savez(RESULTS / f"preds_fold{i}{tag}.npz", **p)
        # §8.1 and §8.2 freeze a TRAINED backbone and fit an independent probe on z.
        # Holding the weights only in memory made the diagnostics impossible without
        # retraining, which is how the first run left phase 6 blocked.
        torch.save(model.state_dict(), RESULTS / f"model_fold{i}{tag}.pt")

    out = RESULTS / f"cv{tag}.json"
    # stamp which arrays produced this, so a later aggregate can refuse a mismatch
    out.write_text(json.dumps({"device": device, "epochs": epochs, "mode": mode,
                               "fingerprint": fingerprint(d),
                               "n_segments": int(len(d["X"])),
                               "folds": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1/20 of the data, 3 epochs, CPU")
    ap.add_argument("--mode", default="main", choices=list(MODES),
                    help="main | lambda0 (method-off) | permuted | randbb (§8.3 controls)")
    ap.add_argument("--all-modes", action="store_true",
                    help="main plus all three controls, in one session")
    ap.add_argument("--transfer", action="store_true",
                    help="§6.2 cross-corpus runs instead of CV, each with its lambda=0 control")
    a = ap.parse_args()
    if not a.smoke:
        # §8.6: a prior run returned MODMA subject-level AUC 0.3434 with one fold at
        # exactly 0.0000 — inversion, not chance, from a broken label join. Verify the
        # join before spending a GPU queue slot on it.
        from ..data.verify import check_labels
        check_labels(load())
    if a.transfer:
        print("\n===== cross-corpus transfer (§6.2, §7.5) =====")
        run_all_transfers(smoke=a.smoke)
    else:
        for mode in (list(MODES) if a.all_modes else [a.mode]):
            print(f"\n===== mode: {mode} =====")
            run(smoke=a.smoke, mode=mode)
