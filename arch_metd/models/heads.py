"""Masked binary heads and the anchor-gated adversary — metd_plan.md §5.1, §5.2."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..contract import N_DOMAINS


def head(z_dim, hidden=32, out=1):
    return nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, out))


def masked_bce(logits, y):
    """BCE over the samples where ``y != -1``, divided by the count of VALID samples.

    Dividing by the batch size instead would scale each head's gradient by how much of
    the batch happened to carry its label, which changes with corpus-balanced sampling.
    Returns 0 when the batch carries no valid sample for this head.
    """
    valid = y != -1
    n = valid.sum()
    if n == 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(
        logits[valid].squeeze(-1), y[valid].float(), reduction="sum")
    return loss / n


def anchor_weights(y_domain, anchor):
    """Inverse anchor frequency, w_c = N_anchor / (5 * N_anchor_c) (§5.1).

    Anchor counts are badly skewed — MODMA's healthy controls alone are thousands of
    segments against SAM40's few hundred relax windows. An unweighted adversary wins by
    always predicting the largest corpus and applies no alignment pressure to the rest.
    """
    counts = np.bincount(np.asarray(y_domain)[np.asarray(anchor)], minlength=N_DOMAINS)
    total = counts.sum()
    w = np.zeros(N_DOMAINS, dtype=np.float32)
    present = counts > 0
    w[present] = total / (present.sum() * counts[present])
    return torch.from_numpy(w)


def domain_loss(logits, y_domain, anchor, weights):
    """Cross-entropy over ANCHOR segments only, class-weighted.

    §5.1 is the load-bearing decision in this architecture. A marginal adversary over all
    segments would be asked to erase the difference between MODMA-MDD and SAM40-task —
    different classes in different domains — so erasing the domain would erase the label.
    Restricting to the healthy/baseline anchor aligns recording physics and leaves
    class-conditional structure alone.
    """
    if anchor.sum() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[anchor], y_domain[anchor].long(),
                           weight=weights.to(logits.device))
