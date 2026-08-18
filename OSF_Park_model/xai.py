"""
Explainability, aggregated to the 19 channels and 6 bands rather than 1140 raw columns.

No new dependency: shap and xgboost are NOT installed in .venv.  Grouped permutation
importance is ~15 lines, is model-agnostic by construction, and answers the question the
1140-column version does not ("which channel / which band"), so it is what is used here.
"""
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from .data import AB_BAND_COLS, AB_CHANNEL_COLS, BANDS, to_blocks


def _score(model, blocks, y):
    return balanced_accuracy_score(y, model.predict_proba(blocks).argmax(1))


def permutation(model, c, y, scaler, n_repeats=5, seed=0):
    """Drop in balanced accuracy when a feature group is shuffled.

    Permutes the raw AB block and rebuilds every view through to_blocks(), so the flat,
    channel-tensor and coherence-bias inputs all move together -- permuting only the flat
    copy would leave the transformer reading unshuffled data and report zero importance.
    """
    rng = np.random.RandomState(seed)
    base = _score(model, to_blocks(c, scaler=scaler)[0], y)
    out = {}
    groups = [(ch, AB_CHANNEL_COLS(i)) for i, ch in enumerate(c.channels)]
    groups += [(b, AB_BAND_COLS(i)) for i, b in enumerate(BANDS)]
    groups += [("COH block", None), ("DEMO block", "demo")]
    for name, cols in groups:
        drops = []
        for _ in range(n_repeats):
            AB, COH, DEMO = c.AB.copy(), c.COH.copy(), c.DEMO.copy()
            perm = rng.permutation(len(y))
            if cols is None:
                COH = COH[perm]
            elif cols == "demo":
                DEMO = DEMO[perm]
            else:
                AB[:, cols] = AB[perm][:, cols]
            drops.append(base - _score(model, to_blocks(c, AB, COH, DEMO, scaler)[0], y))
        out[name] = (float(np.mean(drops)), float(np.std(drops)))
    return base, out


def native(model, c):
    """Built-in importances, folded down to channel and band over the AB block."""
    est = getattr(model, "pipe", None)
    if est is None:
        return None
    est = est[-1]
    imp = getattr(est, "feature_importances_", None)
    if imp is None:
        if not hasattr(est, "coef_"):
            return None                                  # SVC(rbf) has no coefficients
        imp = np.abs(est.coef_).mean(0)
    if len(imp) < 114:
        return None                                      # demographics-only view, no AB block
    ab = imp[:114]
    return {"by_channel": {ch: float(ab[AB_CHANNEL_COLS(i)].sum()) for i, ch in enumerate(c.channels)},
            "by_band": {b: float(ab[AB_BAND_COLS(i)].sum()) for i, b in enumerate(BANDS)},
            "AB_total": float(ab.sum()), "COH_total": float(imp[114:].sum())}


def grad_input(model, blocks, c):
    """|gradient x input| for a torch model, summed to channel and band."""
    m = model.model
    m.eval()
    d = {k: blocks[k].clone().requires_grad_(k in model.needs) for k in model.needs}
    out = m(**d)
    out.gather(1, out.argmax(1, keepdim=True)).sum().backward()
    key = "chan" if "chan" in model.needs else "flat"
    g = (d[key].grad * d[key]).abs().detach().numpy()
    g = g[:, :114].reshape(len(g), 6, 19).transpose(0, 2, 1) if key == "flat" else g
    g = g.mean(0)                                        # (19, 6)
    return {"by_channel": dict(zip(c.channels, g.sum(1).tolist())),
            "by_band": dict(zip(BANDS, g.sum(0).tolist()))}


def attention(model, blocks, c):
    """Mean attention received per channel token, averaged over layers and subjects.

    Mean-received attention (column mean of the attention matrix) rather than CLS attention
    -- these models mean-pool, so there is no CLS token to read.
    """
    m = model.model
    trunk = m.a if hasattr(m, "a") else m
    m.eval()
    with torch.no_grad():
        model.predict_proba(blocks)
    w = torch.stack(trunk.attn_w).mean(0).mean(0).mean(0).numpy()
    return dict(zip(c.channels, (w / w.sum()).tolist()))


def gates(model, blocks):
    """Mean gate weight per branch -- the hybrid's own account of which view it used."""
    with torch.no_grad():
        model.predict_proba(blocks)
    g = model.model.gate_w.mean(0).numpy()
    names = ["AB transformer", "COH projection", "demographics"][:len(g)]
    return dict(zip(names, g.tolist()))


def markdown(title, dd, top=8):
    rows = [f"\n**{title}**\n", "| feature | importance |", "|---|---:|"]
    items = sorted(dd.items(), key=lambda kv: -(kv[1][0] if isinstance(kv[1], tuple) else kv[1]))
    for k, v in items[:top]:
        s = f"{v[0]:+.4f} ±{v[1]:.4f}" if isinstance(v, tuple) else f"{v:.4f}"
        rows.append(f"| {k} | {s} |")
    return "\n".join(rows)
