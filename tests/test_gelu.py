import math

import torch

from src.gelu import gelu, GELU


def test_matches_exact_formula():
    x = torch.linspace(-5, 5, 100)
    expected = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    assert torch.allclose(gelu(x), expected, atol=1e-7)


def test_matches_torch_gelu_exact():
    torch.manual_seed(0)
    x = torch.randn(50)
    ref = torch.nn.functional.gelu(x, approximate="none")
    assert torch.allclose(gelu(x, approximate="none"), ref, atol=1e-6)


def test_matches_torch_gelu_tanh():
    torch.manual_seed(1)
    x = torch.randn(50)
    ref = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(gelu(x, approximate="tanh"), ref, atol=1e-6)


def test_known_values():
    # GELU(0) = 0, and GELU is bounded below near small negatives.
    assert abs(float(gelu(torch.tensor(0.0)))) < 1e-7
    # For large positive x, GELU(x) approaches x.
    big = torch.tensor(10.0)
    assert abs(float(gelu(big) - big)) < 1e-4


def test_module_matches_function():
    x = torch.randn(20)
    assert torch.allclose(GELU()(x), gelu(x))


def test_invalid_approximate_raises():
    try:
        gelu(torch.tensor(1.0), approximate="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
