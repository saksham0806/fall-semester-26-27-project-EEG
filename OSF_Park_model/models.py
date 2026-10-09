"""
Five model families for the BRMH cohort.  Tabular/graph, NOT time-series -- this dataset
starts at band power, so nothing from the project's (6, 512) window pipeline applies.

    .venv/bin/python -m OSF_Park_model.models     # self-check

Every model takes a dict of feature blocks and exposes fit / predict_proba, so run.py
treats sklearn pipelines and torch modules identically.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
DEV = "cpu"          # 528 x 1140 is trivial; this box is memory-tight and GPU-free here


# --------------------------------------------------------------------------- classical
# Shape copied from src/proposed_model/classical_models.py rather than imported: importing
# src.proposed_model drags in TensorFlow *and* torch via its __init__.  Boosting is labelled
# GradientBoosting, not XGBoost -- xgboost is in requirements.txt but NOT installed here.

def classical(name):
    clf = {
        "SVM": SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=-1),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, max_features="sqrt", random_state=SEED),
    }[name]
    return make_pipeline(StandardScaler(), clf)


class Sk:
    """sklearn pipeline against the block-dict interface."""
    needs = ("flat",)

    def __init__(self, name, key="np_flat", **_):
        self.name, self.key, self.pipe = name, key, classical(name)
        self.needs = (key,)

    def fit(self, d, y, **_):
        self.pipe.fit(d[self.key], y)
        return self

    def predict_proba(self, d):
        return self.pipe.predict_proba(d[self.key])


# ------------------------------------------------------------------------------- torch

class Block(nn.Module):
    """One encoder block.  Written out rather than using nn.TransformerEncoderLayer so a
    per-sample attention bias can be injected and the attention weights read back out."""

    def __init__(self, d, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, bias=None):
        h = self.n1(x)
        a, w = self.attn(h, h, h, attn_mask=bias, need_weights=True, average_attn_weights=True)
        x = x + a
        return x + self.ff(self.n2(x)), w


class ChannelTransformer(nn.Module):
    """19 channels as tokens, each a 6-band power vector.

    use_coh=True adds a per-head additive attention bias projected from the 6 coherence
    matrices -- i.e. coherence enters as *graph structure* over the channel tokens rather
    than as 1026 flat numbers.
    """

    def __init__(self, n_classes, d=32, heads=4, layers=2, use_coh=False, head=True):
        super().__init__()
        self.use_coh, self.heads = use_coh, heads
        self.embed = nn.Linear(6, d)
        self.pos = nn.Parameter(torch.zeros(1, 19, d))       # learned channel identity
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        if use_coh:
            self.coh_proj = nn.Linear(6, heads, bias=False)
            nn.init.zeros_(self.coh_proj.weight)             # start as a plain transformer
        self.out = nn.Linear(d, n_classes) if head else None

    def trunk(self, chan, coh=None):
        x = self.embed(chan) + self.pos
        bias = None
        if self.use_coh:
            b = self.coh_proj(coh.permute(0, 2, 3, 1))       # (n,19,19,6) -> (n,19,19,H)
            bias = b.permute(0, 3, 1, 2).reshape(-1, 19, 19)  # (n*H,19,19), added to scores
        self.attn_w = []
        for blk in self.blocks:
            x, w = blk(x, bias)
            self.attn_w.append(w)
        return self.norm(x).mean(1)                          # mean-pool over channels

    def forward(self, chan, coh=None, **_):
        return self.out(self.trunk(chan, coh))


class MLP(nn.Module):
    def __init__(self, n_in, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_classes))

    def forward(self, flat, **_):
        return self.net(flat)


class GatedHybrid(nn.Module):
    """The proposed architecture: three views, one learned gate.

    A  coherence-biased channel transformer over band power   (structure)
    B  a linear projection of the raw 1026 coherences         (bulk connectivity)
    C  demographics                                           (the known confound)

    C is ablatable, which is the point: run with and without it and the difference IS the
    measurement of whether EEG contributes anything beyond age/sex/IQ on this cohort.
    The per-sample gate doubles as the explanation -- see xai.py.
    """

    def __init__(self, n_classes, d=32, use_demo=True):
        super().__init__()
        self.use_demo = use_demo
        self.a = ChannelTransformer(n_classes, d=d, use_coh=True, head=False)
        self.b = nn.Sequential(nn.Linear(1026, d), nn.GELU(), nn.Dropout(0.5), nn.LayerNorm(d))
        if use_demo:
            self.c = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.LayerNorm(d))
        self.k = 3 if use_demo else 2
        self.gate = nn.Linear(d * self.k, self.k)
        self.out = nn.Linear(d, n_classes)

    def forward(self, chan, coh, cohflat, demo=None, **_):
        v = [self.a.trunk(chan, coh), self.b(cohflat)]
        if self.use_demo:
            v.append(self.c(demo))
        v = torch.stack(v, 1)                                  # (n, k, d)
        self.gate_w = torch.softmax(self.gate(v.flatten(1)), -1)
        return self.out((v * self.gate_w.unsqueeze(-1)).sum(1))


class Torch:
    """Fixed budget for every torch model: AdamW, lr 1e-3, wd 1e-2, batch 64, <=60 epochs,
    early stop on an inner stratified split.  Nothing here is tuned -- same caveat CLAUDE.md
    attaches to 'deep models lose here'."""

    def __init__(self, name, n_classes, n_in=None, **kw):
        self.name, self.n_classes, self.n_in, self.kw = name, n_classes, n_in, kw
        self.needs = {"MLP": ("flat",),
                      "ChannelTransformer": ("chan",),
                      "CohTransformer": ("chan", "coh"),
                      "GatedHybrid": ("chan", "coh", "cohflat", "demo")}[name]

    def _build(self):
        torch.manual_seed(SEED)
        if self.name == "MLP":
            return MLP(self.n_in, self.n_classes)
        if self.name == "GatedHybrid":
            return GatedHybrid(self.n_classes, **self.kw)
        return ChannelTransformer(self.n_classes, use_coh=(self.name == "CohTransformer"))

    def fit(self, d, y, epochs=60, patience=10):
        self.model = self._build().to(DEV)
        tr, va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=y, random_state=SEED)
        w = compute_class_weight("balanced", classes=np.unique(y), y=y)
        lossf = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2)
        yt = torch.tensor(y)
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator().manual_seed(SEED)
        for _ in range(epochs):
            self.model.train()
            for i in torch.randperm(len(tr), generator=g).split(64):
                b = tr[i.numpy()]
                opt.zero_grad()
                lossf(self.model(**{k: d[k][b] for k in self.needs}), yt[b]).backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                vl = lossf(self.model(**{k: d[k][va] for k in self.needs}), yt[va]).item()
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, d):
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(**{k: d[k] for k in self.needs}), -1).numpy()


# ---------------------------------------------------------------------------- registry

def build(name, n_classes, n_in=None, key="np_flat", **kw):
    if name in ("SVM", "RandomForest", "LogisticRegression", "GradientBoosting"):
        return Sk(name, key=key)
    return Torch(name, n_classes, n_in=n_in, **kw)


def _selfcheck():
    from . import data
    c = data.cohort("4class")
    n = 64
    d = {"flat": torch.randn(n, 1140), "chan": torch.randn(n, 19, 6),
         "coh": torch.randn(n, 6, 19, 19), "cohflat": torch.randn(n, 1026),
         "demo": torch.randn(n, 4)}
    y = np.random.RandomState(0).randint(0, 4, n)
    for name in ("MLP", "ChannelTransformer", "CohTransformer", "GatedHybrid"):
        m = build(name, 4, n_in=1140)
        p = m.fit(d, y, epochs=2, patience=99).predict_proba(d)
        assert p.shape == (n, 4) and np.allclose(p.sum(1), 1, atol=1e-5), name
        npar = sum(q.numel() for q in m.model.parameters())
        print(f"{name:20s} ok  params={npar:,}")

    h = build("GatedHybrid", 4).fit(d, y, epochs=1, patience=99)
    h.predict_proba(d)
    assert h.model.gate_w.shape == (n, 3) and torch.allclose(h.model.gate_w.sum(1), torch.ones(n))
    assert len(h.model.a.attn_w) == 2 and h.model.a.attn_w[0].shape == (n, 19, 19)
    assert build("GatedHybrid", 4, use_demo=False).fit(d, y, epochs=1, patience=99).model.k == 2

    # the coherence bias must actually change the forward pass once its projection is nonzero
    m = ChannelTransformer(4, use_coh=True).eval()
    with torch.no_grad():
        a = m(d["chan"], d["coh"])
        nn.init.normal_(m.coh_proj.weight, std=1.0)
        b = m(d["chan"], d["coh"])
    assert not torch.allclose(a, b), "coherence bias is not wired into attention"

    for name in ("SVM", "RandomForest", "LogisticRegression", "GradientBoosting"):
        p = build(name, 4).fit({"np_flat": c.X()[:120]}, c.y[:120]).predict_proba({"np_flat": c.X()[:40]})
        assert p.shape[0] == 40
        print(f"{name:20s} ok")
    print("all checks passed")


if __name__ == "__main__":
    _selfcheck()
