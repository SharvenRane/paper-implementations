import torch
import torch.nn.functional as F

from src.label_smoothing import LabelSmoothingCrossEntropy


def test_zero_smoothing_equals_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(16, 10)
    target = torch.randint(0, 10, (16,))
    mine = LabelSmoothingCrossEntropy(smoothing=0.0)
    ref = F.cross_entropy(logits, target)
    assert torch.allclose(mine(logits, target), ref, atol=1e-6)


def test_smoothing_reduces_confidence():
    # With a confident correct prediction, smoothing raises the loss because
    # the target no longer wants probability 1 on the true class.
    logits = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    hard = LabelSmoothingCrossEntropy(smoothing=0.0)(logits, target)
    soft = LabelSmoothingCrossEntropy(smoothing=0.2)(logits, target)
    assert float(soft) > float(hard)


def test_matches_manual_smoothed_target():
    torch.manual_seed(2)
    logits = torch.randn(8, 5)
    target = torch.randint(0, 5, (8,))
    smoothing = 0.1
    num_classes = 5

    log_probs = F.log_softmax(logits, dim=-1)
    true = F.one_hot(target, num_classes).float()
    smooth_target = (1 - smoothing) * true + smoothing / num_classes
    expected = -(smooth_target * log_probs).sum(dim=-1).mean()

    mine = LabelSmoothingCrossEntropy(smoothing=smoothing)(logits, target)
    assert torch.allclose(mine, expected, atol=1e-6)


def test_reduction_none_shape():
    logits = torch.randn(7, 3)
    target = torch.randint(0, 3, (7,))
    out = LabelSmoothingCrossEntropy(smoothing=0.1, reduction="none")(logits, target)
    assert out.shape == (7,)


def test_invalid_smoothing_raises():
    try:
        LabelSmoothingCrossEntropy(smoothing=1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
