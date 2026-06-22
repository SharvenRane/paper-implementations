"""Label smoothing cross entropy (Szegedy et al. 2016, "Rethinking the
Inception Architecture").

Instead of training against a one hot target, the target distribution puts
mass (1 - smoothing) on the true class and spreads the remaining smoothing
mass uniformly across all classes. This discourages the model from becoming
overconfident and tends to improve calibration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction="mean"):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum' or 'none'")
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits, target):
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Negative log likelihood of the true class for each sample.
        nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        # Mean negative log probability across all classes (smooth part).
        smooth = -log_probs.mean(dim=-1)

        loss = (1.0 - self.smoothing) * nll + self.smoothing * smooth

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
