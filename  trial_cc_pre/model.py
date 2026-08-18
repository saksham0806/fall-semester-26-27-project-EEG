"""Section 3: shared EEGNet-style backbone, two masked task heads, anchor-only GRL adversary."""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS, BATCH, LR, PATIENCE = 60, 128, 1e-3, 8


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, n_dom=3, drop=0.25):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 8, (1, 64), bias=False), nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, (6, 1), groups=8, bias=False), nn.BatchNorm2d(16),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(16, 16, (1, 16), groups=16, bias=False), nn.Conv2d(16, 16, 1, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop),
            nn.Flatten(),
        )
        self.dep = nn.Sequential(nn.Linear(256, 32), nn.ELU(), nn.Linear(32, 1))
        self.stress = nn.Sequential(nn.Linear(256, 32), nn.ELU(), nn.Linear(32, 1))
        self.adv = nn.Sequential(nn.Linear(256, 64), nn.ELU(), nn.Linear(64, n_dom))

    def forward(self, x, lam=0.0):
        z = self.backbone(x)
        return self.dep(z).squeeze(1), self.stress(z).squeeze(1), self.adv(GRL.apply(z, lam)), z


def masked_bce(logit, y, pos_weight):
    m = y >= 0
    if not m.any():
        return logit.sum() * 0
    return F.binary_cross_entropy_with_logits(logit[m], y[m].float(), pos_weight=pos_weight)


def anchor_ce(dom_logit, y_dom, anchor, w):
    if not anchor.any():
        return dom_logit.sum() * 0
    return F.cross_entropy(dom_logit[anchor], y_dom[anchor].long(), weight=w)


def _pos_weight(y):
    y = y[y >= 0]
    if len(y) == 0 or y.sum() in (0, len(y)):
        return None
    return torch.tensor(float((y == 0).sum()) / float((y == 1).sum()), device=DEV)


def _anchor_weights(dom, anchor, n_dom=3):
    a = dom[anchor]
    n = np.array([(a == c).sum() for c in range(n_dom)], float)
    w = np.where(n > 0, len(a) / (n_dom * np.maximum(n, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32, device=DEV)


def _tensors(d, idx):
    return (torch.tensor(d["X"][idx]).unsqueeze(1),
            torch.tensor(d["y_dep"][idx].astype(np.int64)),
            torch.tensor(d["y_stress"][idx].astype(np.int64)),
            torch.tensor(d["y_domain"][idx].astype(np.int64)),
            torch.tensor(d["anchor"][idx]))


@torch.no_grad()
def predict(model, d, idx, bs=512):
    model.eval()
    dep, st, z = [], [], []
    X = torch.tensor(d["X"][idx]).unsqueeze(1)
    for i in range(0, len(X), bs):
        a, b, _, c = model(X[i:i + bs].to(DEV))
        dep.append(a.cpu()), st.append(b.cpu()), z.append(c.cpu())
    return (torch.cat(dep).numpy(), torch.cat(st).numpy(), torch.cat(z).numpy())


def _auc(y, s):
    m = y >= 0
    return roc_auc_score(y[m], s[m]) if m.sum() and 0 < y[m].sum() < m.sum() else np.nan


def train(d, tr, va, lam_max=1.0, epochs=None, fixed_epochs=None, seed=0, verbose=False):
    """Train the multi-task net.  `va` drives early stopping unless fixed_epochs is set."""
    epochs = epochs or EPOCHS          # read at call time so --smoke can lower it
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    X, yd, ys, dom, anc = _tensors(d, tr)
    pw_d, pw_s = _pos_weight(d["y_dep"][tr]), _pos_weight(d["y_stress"][tr])
    w_adv = _anchor_weights(d["y_domain"][tr], d["anchor"][tr])

    n_ep = fixed_epochs or epochs
    steps = max(1, (len(X) + BATCH - 1) // BATCH) * n_ep
    best, best_ep, best_state, step = -np.inf, n_ep, None, 0

    for ep in range(1, n_ep + 1):
        model.train()
        for i in torch.randperm(len(X)).split(BATCH):
            # DANN ramp: a fixed lambda fights the task objective from step one.
            lam = lam_max * (2 / (1 + np.exp(-10 * step / steps)) - 1)
            step += 1
            l_dep, l_str, l_dom, _ = model(X[i].to(DEV), lam)
            loss = (masked_bce(l_dep, yd[i].to(DEV), pw_d)
                    + masked_bce(l_str, ys[i].to(DEV), pw_s)
                    + anchor_ce(l_dom, dom[i].to(DEV), anc[i].to(DEV), w_adv))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if fixed_epochs is None:
            pd_, ps_, _ = predict(model, d, va)
            sc = np.nanmean([_auc(d["y_dep"][va], pd_), _auc(d["y_stress"][va], ps_)])
            if verbose:
                print(f"  ep{ep:>3} val={sc:.4f}")
            if sc > best:
                best, best_ep = sc, ep
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            elif ep - best_ep >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep


def selfcheck():
    rng = np.random.default_rng(0)
    n = 400
    d = dict(X=rng.normal(size=(n, 6, 512)).astype(np.float32),
             y_dep=np.where(np.arange(n) < 200, rng.integers(0, 2, n), -1).astype(np.int8),
             y_stress=np.where(np.arange(n) >= 200, rng.integers(0, 2, n), -1).astype(np.int8),
             y_domain=(np.arange(n) // 134).astype(np.int8))
    d["anchor"] = (d["y_dep"] == 0) | (d["y_stress"] == 0)

    m = Net().to(DEV)
    a, b, c, z = m(torch.tensor(d["X"][:8]).unsqueeze(1).to(DEV))
    assert a.shape == (8,) and b.shape == (8,) and c.shape == (8, 3) and z.shape == (8, 256)

    # GRL flips the sign of the gradient reaching the backbone, and lam=0 blocks it.
    m.eval()                                   # dropout/BN would make the two passes differ
    g = []
    for lam in (1.0, -1.0, 0.0):
        m.zero_grad()
        _, _, dom, _ = m(torch.tensor(d["X"][:16]).unsqueeze(1).to(DEV), lam)
        F.cross_entropy(dom, torch.zeros(16, dtype=torch.long, device=DEV)).backward()
        g.append(m.backbone[1].weight.grad.clone())
    assert torch.allclose(g[0], -g[1], atol=1e-6), "GRL did not reverse"
    assert g[2].abs().max() == 0, "lam=0 leaked gradient into the backbone"

    # masked loss ignores -1 and puts zero gradient on the head it never supervises
    m.zero_grad()
    dep, _, _, _ = m(torch.tensor(d["X"][200:216]).unsqueeze(1).to(DEV))
    masked_bce(dep, torch.tensor(d["y_dep"][200:216].astype(np.int64)).to(DEV), None).backward()
    assert m.dep[0].weight.grad.abs().max() == 0, "masked loss trained on -1 labels"

    w = _anchor_weights(d["y_domain"], d["anchor"])
    na = np.array([(d["y_domain"][d["anchor"]] == c).sum() for c in range(3)])
    assert np.allclose((w.cpu().numpy() * na).sum(), d["anchor"].sum(), rtol=1e-4), "anchor weights"

    tr, va = np.arange(0, n, 2), np.arange(1, n, 2)
    _, ep = train(d, tr, va, epochs=2, seed=0)
    assert ep <= 2
    _, ep = train(d, tr, va, fixed_epochs=1)
    assert ep == 1, "fixed_epochs ignored"
    print("model selfcheck ok  (device:", DEV, ")")


if __name__ == "__main__":
    selfcheck() if "--selfcheck" in sys.argv or len(sys.argv) == 1 else None
