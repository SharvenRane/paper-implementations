"""Layer Normalization (Ba, Kiros, Hinton 2016).

Normalizes the activations over the last D dimensions of the input, then
applies a learned per-feature scale (gamma) and shift (beta). This mirrors
the behaviour of torch.nn.LayerNorm.
"""

import numbers

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))
        mean = x.mean(dim=dims, keepdim=True)
        # Biased variance (population variance), matching torch.nn.LayerNorm.
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x_hat = x_hat * self.weight + self.bias
        return x_hat

    def extra_repr(self):
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
