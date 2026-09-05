"""EEGNet-8,2 backbone — metd_plan.md §5. Input (B, 1, 6, 512) -> z in R^256."""
import torch
from torch import nn

from ..contract import CHANNELS, WINDOW

F1, D, F2 = 8, 2, 16
Z_DIM = F2 * (WINDOW // 4 // 8)  # 16 filters x 16 time steps


class EEGNet(nn.Module):
    """§6.3 prefers GroupNorm over BatchNorm, and the held-out corpora are why.

    A leave-one-corpus-out test batch is single-domain by construction, so BatchNorm
    running statistics estimated across five amplifiers are evaluated on one. GroupNorm
    normalizes within a sample and has no running statistics to mismatch.
    """

    def __init__(self, n_chans=len(CHANNELS), dropout=0.25):
        super().__init__()
        # temporal kernel 64 at 128 Hz = a 0.5 s receptive field, i.e. down to 2 Hz
        self.temporal = nn.Conv2d(1, F1, (1, 64), padding="same", bias=False)
        self.norm1 = nn.GroupNorm(F1, F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False)
        self.norm2 = nn.GroupNorm(F1 * D, F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), groups=F1 * D,
                                   padding="same", bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.norm3 = nn.GroupNorm(F2, F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, x):
        x = self.norm1(self.temporal(x))
        x = self.drop(self.pool1(self.act(self.norm2(self.depthwise(x)))))
        x = self.sep_point(self.sep_depth(x))
        x = self.drop(self.pool2(self.act(self.norm3(x))))
        return x.flatten(1)

    @torch.no_grad()
    def apply_max_norm(self, max_val=1.0):
        """Keras' max_norm=1.0 on the depthwise kernel — call after optimizer.step()."""
        w = self.depthwise.weight
        norm = w.flatten(1).norm(dim=1, keepdim=True).clamp(min=1e-8)
        w.mul_((max_val / norm).clamp(max=1.0).view(-1, 1, 1, 1))
