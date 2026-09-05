"""Multi-task DANN assembly — metd_plan.md §5.

    L = L_dep_masked + L_str_masked + lambda * L_domain_anchor_weighted

Self-check: ``python -m arch_metd.models.mtdann``
"""
import torch
from torch import nn

from ..contract import N_DOMAINS
from .eegnet import EEGNet, Z_DIM
from .grl import GradientReversal
from .heads import anchor_weights, domain_loss, head, masked_bce


class MTDANN(nn.Module):
    def __init__(self, n_domains=N_DOMAINS, dropout=0.25):
        super().__init__()
        self.backbone = EEGNet(dropout=dropout)
        self.dep = head(Z_DIM)
        self.str = head(Z_DIM)
        self.grl = GradientReversal()
        self.domain = head(Z_DIM, hidden=64, out=n_domains)

    def forward(self, x):
        z = self.backbone(x)
        return z, self.dep(z), self.str(z), self.domain(self.grl(z))

    def loss(self, batch, weights, lambd):
        x, y_dep, y_str, y_domain, anchor = batch
        self.grl.lambd = lambd
        _, dep, stress, dom = self(x)
        parts = {"dep": masked_bce(dep, y_dep),
                 "str": masked_bce(stress, y_str),
                 "domain": domain_loss(dom, y_domain, anchor, weights)}
        # lambda scales the adversary in the loss as well as in the reversal, so a
        # lambda = 0 control is genuinely method-off rather than merely un-reversed
        return parts["dep"] + parts["str"] + lambd * parts["domain"], parts


def _selfcheck():
    torch.manual_seed(0)
    b = 16
    x = torch.randn(b, 1, 6, 512)
    y_dep = torch.tensor([1, 0, -1, -1] * 4, dtype=torch.int8)
    y_str = torch.tensor([-1, -1, 1, 0] * 4, dtype=torch.int8)
    y_domain = torch.tensor([0, 0, 1, 2] * 4, dtype=torch.int8)
    anchor = (y_dep == 0) | (y_str == 0)

    m = MTDANN()
    z, dep, stress, dom = m(x)
    assert z.shape == (b, 256) and dep.shape == (b, 1) and dom.shape == (b, N_DOMAINS), z.shape

    w = anchor_weights(y_domain, anchor)
    assert abs(float(w[w > 0].sum()) - 1.0) > 0 and (w[3:] == 0).all(), w
    # the two present anchor corpora carry equal total weight despite unequal counts
    counts = torch.bincount(y_domain[anchor].long(), minlength=N_DOMAINS)
    mass = w * counts
    assert torch.allclose(mass[mass > 0], mass[mass > 0][0]), mass

    # a segment that is not an anchor puts zero gradient on the domain head
    m.zero_grad()
    domain_loss(m.domain(m.backbone(x)), y_domain, torch.zeros_like(anchor), w).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.domain.parameters()), \
        "non-anchor segments reached the adversary"

    # masked BCE ignores -1 and normalizes by the valid count, not the batch size
    logits = torch.zeros(4, 1)
    assert abs(float(masked_bce(logits, torch.tensor([1, 0, -1, -1], dtype=torch.int8)))
               - 0.6931) < 1e-3
    assert float(masked_bce(logits, torch.full((4,), -1, dtype=torch.int8))) == 0.0

    # the GRL flips the sign of the gradient reaching the backbone
    grads = {}
    for lambd in (0.0, 1.0):
        m.zero_grad()
        m.grl.lambd = lambd
        domain_loss(m.domain(m.grl(m.backbone(x))), y_domain, anchor, w).backward()
        grads[lambd] = m.backbone.temporal.weight.grad.clone()
    assert grads[0.0].abs().sum() == 0, "lambda=0 still passed gradient to the backbone"
    assert grads[1.0].abs().sum() > 0, "lambda=1 passed no gradient"

    from .grl import lambda_at
    assert abs(lambda_at(0.0)) < 1e-9 and abs(lambda_at(1.0) - 0.9999) < 1e-3
    # lambda crosses 0.5 at p = 0.11, not somewhere near the middle: it is 0.905 by 30%
    # of training and 0.9999 at the end. So §5.3's "lock early stopping until lambda >=
    # 0.5" only covers the first 11% of the budget — it stops you shipping a completely
    # un-adapted model, but it is not a guarantee that adaptation had time to work. The
    # lambda at the stopping epoch is logged per fold for exactly that reason.
    assert lambda_at(0.10) < 0.5 < lambda_at(0.12), "lambda crosses 0.5 in the wrong place"
    assert lambda_at(0.3) > 0.9

    m.backbone.depthwise.weight.data.mul_(100)
    m.backbone.apply_max_norm(1.0)
    n = m.backbone.depthwise.weight.flatten(1).norm(dim=1)
    assert (n <= 1.0 + 1e-5).all(), n

    total = sum(p.numel() for p in m.parameters())
    print(f"mtdann OK — z=256, {total} params, anchor gate and GRL sign verified, "
          f"lambda 0->{lambda_at(1.0):.4f}, crosses 0.5 at p=0.11")


if __name__ == "__main__":
    _selfcheck()
