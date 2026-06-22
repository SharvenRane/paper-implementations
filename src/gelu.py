"""Gaussian Error Linear Unit (Hendrycks, Gimpel 2016).

GELU(x) = x * Phi(x), where Phi is the standard normal cumulative
distribution function. The exact form uses the error function:

    GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

A tanh approximation is also provided, matching the form used in the
original GPT and BERT implementations.
"""

import math

import torch
import torch.nn as nn


def gelu(x, approximate="none"):
    if approximate == "none":
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    elif approximate == "tanh":
        c = math.sqrt(2.0 / math.pi)
        inner = c * (x + 0.044715 * torch.pow(x, 3))
        return 0.5 * x * (1.0 + torch.tanh(inner))
    else:
        raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")


class GELU(nn.Module):
    def __init__(self, approximate="none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x):
        return gelu(x, approximate=self.approximate)

    def extra_repr(self):
        return f"approximate={self.approximate!r}"
