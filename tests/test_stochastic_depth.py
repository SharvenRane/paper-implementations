import torch

from src.stochastic_depth import drop_path, DropPath


def test_eval_is_identity():
    x = torch.randn(8, 16)
    out = drop_path(x, drop_prob=0.5, training=False)
    assert torch.allclose(out, x)


def test_zero_prob_is_identity():
    x = torch.randn(8, 16)
    out = drop_path(x, drop_prob=0.0, training=True)
    assert torch.allclose(out, x)


def test_drops_whole_samples_and_scales():
    torch.manual_seed(0)
    x = torch.ones(1000, 4)
    drop_prob = 0.3
    keep_prob = 1.0 - drop_prob
    out = drop_path(x, drop_prob=drop_prob, training=True)

    # Each sample is either fully zero or fully scaled by 1 / keep_prob.
    per_sample = out.reshape(out.shape[0], -1)
    is_zero = (per_sample == 0).all(dim=1)
    scaled_val = 1.0 / keep_prob
    is_scaled = torch.isclose(
        per_sample, torch.full_like(per_sample, scaled_val)
    ).all(dim=1)
    assert (is_zero | is_scaled).all()

    # Roughly drop_prob fraction dropped.
    frac_dropped = is_zero.float().mean().item()
    assert abs(frac_dropped - drop_prob) < 0.05


def test_expected_value_preserved():
    torch.manual_seed(1)
    x = torch.randn(20000, 8)
    out = drop_path(x, drop_prob=0.4, training=True)
    # Inverted dropout keeps the expected activation equal to the input mean.
    assert abs(out.mean().item() - x.mean().item()) < 0.05


def test_module_train_eval_modes():
    torch.manual_seed(2)
    x = torch.randn(64, 10)
    m = DropPath(drop_prob=0.5)
    m.eval()
    assert torch.allclose(m(x), x)
    m.train()
    out = m(x)
    # In train mode with high prob, some samples should be zeroed.
    zeroed = (out.reshape(64, -1) == 0).all(dim=1).any()
    assert zeroed
