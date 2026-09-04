"""Gradient reversal and the lambda schedule — metd_plan.md §5, §5.3."""
import math

import torch
from torch import nn


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


class GradientReversal(nn.Module):
    """Identity forwards, sign-flipped and scaled backwards.

    ``lambd`` is set by the training loop each step, never learned.
    """

    def __init__(self, lambd=0.0):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return _Reverse.apply(x, self.lambd)


def lambda_at(progress):
    """§5.3: 2 / (1 + exp(-10p)) - 1, p in [0, 1] = training progress.

    Ramped, never fixed. The previous DANN in this repo held alpha constant from step
    one, so the adversary fought the task objective before the backbone had learned
    anything — its domain head came out non-monotonic in alpha and worst at full
    strength, which is training instability rather than domain removal.
    """
    p = min(max(progress, 0.0), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
