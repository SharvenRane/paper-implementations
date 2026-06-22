import numpy as np
import torch

from src.mixup import mixup_batch, mixup_data


def test_mixed_labels_are_convex():
    torch.manual_seed(0)
    np.random.seed(0)
    x = torch.randn(16, 3, 8, 8)
    y = torch.randint(0, 5, (16,))
    mixed_x, mixed_y, lam = mixup_batch(x, y, alpha=1.0, num_classes=5)

    # Each soft label row sums to 1 (still a valid distribution).
    sums = mixed_y.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    # All entries are in [0, 1] and lambda is in [0, 1].
    assert mixed_y.min() >= 0.0 and mixed_y.max() <= 1.0
    assert 0.0 <= lam <= 1.0
    assert mixed_x.shape == x.shape


def test_mixed_input_is_convex_combination():
    torch.manual_seed(1)
    np.random.seed(1)
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    g = torch.Generator().manual_seed(123)
    # Reconstruct the permutation by reusing the same generator state.
    mixed_x, mixed_y, lam = mixup_batch(
        x, y, alpha=1.0, num_classes=3, generator=torch.Generator().manual_seed(7)
    )
    # Every mixed sample must lie between the per coordinate min and max of x.
    lo = x.min(dim=0).values
    hi = x.max(dim=0).values
    assert (mixed_x >= lo - 1e-5).all()
    assert (mixed_x <= hi + 1e-5).all()


def test_lambda_one_is_identity():
    torch.manual_seed(2)
    x = torch.randn(6, 10)
    y = torch.randint(0, 4, (6,))
    # alpha=0 forces lambda=1, so output equals input.
    mixed_x, mixed_y, lam = mixup_batch(x, y, alpha=0.0, num_classes=4)
    assert lam == 1.0
    assert torch.allclose(mixed_x, x)
    # Labels become the original one hot.
    expected = torch.zeros(6, 4)
    expected.scatter_(1, y.unsqueeze(1), 1.0)
    assert torch.allclose(mixed_y, expected)


def test_mixup_data_two_label_form():
    torch.manual_seed(3)
    np.random.seed(3)
    x = torch.randn(12, 5)
    y = torch.randint(0, 7, (12,))
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)
    assert torch.equal(y_a, y)
    assert y_b.shape == y.shape
    assert 0.0 <= lam <= 1.0
    assert mixed_x.shape == x.shape


def test_soft_label_matches_mixed_index_pair():
    # With a known generator the soft label of a sample equals a convex mix
    # of its own one hot and its partner's one hot.
    torch.manual_seed(4)
    np.random.seed(4)
    x = torch.randn(4, 2)
    y = torch.tensor([0, 1, 2, 3])
    num_classes = 4
    g = torch.Generator().manual_seed(99)
    perm = torch.randperm(4, generator=torch.Generator().manual_seed(99))
    mixed_x, mixed_y, lam = mixup_batch(
        x, y, alpha=1.0, num_classes=num_classes, generator=g
    )
    # lam came from numpy; recompute the expected soft label using the same
    # permutation generator seed and the returned lam.
    base = torch.eye(num_classes)
    expected = lam * base[y] + (1 - lam) * base[y][perm]
    assert torch.allclose(mixed_y, expected, atol=1e-6)
