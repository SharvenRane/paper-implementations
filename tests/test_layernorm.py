import torch

from src.layernorm import LayerNorm


def test_matches_torch_layernorm_1d():
    torch.manual_seed(0)
    x = torch.randn(8, 16)
    ref = torch.nn.LayerNorm(16)
    mine = LayerNorm(16)
    # Same default init (weight ones, bias zeros), so outputs must match.
    out_ref = ref(x)
    out_mine = mine(x)
    assert torch.allclose(out_ref, out_mine, atol=1e-6)


def test_matches_torch_layernorm_multidim():
    torch.manual_seed(1)
    x = torch.randn(4, 5, 6, 7)
    normalized_shape = (6, 7)
    ref = torch.nn.LayerNorm(normalized_shape)
    mine = LayerNorm(normalized_shape)
    # Randomize affine params and copy into both to test scale and shift.
    with torch.no_grad():
        w = torch.randn(normalized_shape)
        b = torch.randn(normalized_shape)
        ref.weight.copy_(w)
        ref.bias.copy_(b)
        mine.weight.copy_(w)
        mine.bias.copy_(b)
    assert torch.allclose(ref(x), mine(x), atol=1e-5)


def test_output_is_normalized():
    torch.manual_seed(2)
    x = torch.randn(32, 64) * 5 + 3
    mine = LayerNorm(64, elementwise_affine=False)
    out = mine(x)
    mean = out.mean(dim=-1)
    std = out.std(dim=-1, unbiased=False)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-3)


def test_gradients_flow():
    x = torch.randn(3, 10, requires_grad=True)
    mine = LayerNorm(10)
    loss = mine(x).pow(2).sum()
    loss.backward()
    assert x.grad is not None
    assert mine.weight.grad is not None
