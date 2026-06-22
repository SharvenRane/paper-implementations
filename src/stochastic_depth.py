"""Stochastic depth / drop path (Huang et al. 2016, "Deep Networks with
Stochastic Depth").

During training a residual branch is dropped for a whole sample with
probability drop_prob. Surviving branches are scaled up by 1 / keep_prob so
the expected value of the activation is unchanged (inverted dropout). At
evaluation time the branch is always kept and nothing is scaled.
"""

import torch
import torch.nn as nn


def drop_path(x, drop_prob=0.0, training=False, generator=None):
    if drop_prob == 0.0 or not training:
        return x
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError("drop_prob must be in [0, 1)")

    keep_prob = 1.0 - drop_prob
    # Shape (batch, 1, 1, ...) so the same mask value applies per sample.
    shape = (x.shape[0],) + (1,) * (x.dim() - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device)
    mask = mask.bernoulli_(keep_prob, generator=generator)
    return x * mask / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"
