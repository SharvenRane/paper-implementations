"""MixUp data augmentation (Zhang et al. 2018, "mixup: Beyond Empirical
Risk Minimization").

Each training example is replaced by a convex combination of two examples,
and the target becomes the same convex combination of the two labels. The
mixing coefficient lambda is drawn from a Beta(alpha, alpha) distribution.
"""

import numpy as np
import torch


def mixup_batch(x, y, alpha=1.0, num_classes=None, generator=None):
    """Mix a batch of inputs and one hot encoded targets.

    Returns the mixed inputs, the mixed soft targets, and the lambda used.
    A single lambda is shared across the batch, following the reference
    implementation.
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, generator=generator, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[perm]

    if num_classes is None:
        if y.dim() == 1:
            raise ValueError("num_classes is required when y holds class indices")
        num_classes = y.size(-1)

    if y.dim() == 1:
        y_onehot = torch.zeros(
            batch_size, num_classes, device=x.device, dtype=mixed_x.dtype
        )
        y_onehot.scatter_(1, y.unsqueeze(1), 1.0)
    else:
        y_onehot = y.to(mixed_x.dtype)

    mixed_y = lam * y_onehot + (1.0 - lam) * y_onehot[perm]
    return mixed_x, mixed_y, lam


def mixup_data(x, y, alpha=1.0, generator=None):
    """Mix inputs and return both label sets plus lambda, matching the loss
    form lam * loss(pred, y_a) + (1 - lam) * loss(pred, y_b) from the paper.
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    perm = torch.randperm(batch_size, generator=generator, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[perm]
    y_a, y_b = y, y[perm]
    return mixed_x, y_a, y_b, lam
