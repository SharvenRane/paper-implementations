# paper-implementations

Clean, tested reimplementations of small building blocks that show up again and again in deep learning papers. The point is not to replace what already ships in PyTorch. The point is to write each block from the equation in the paper, then prove with tests that it behaves the way the paper says it should.

Everything here runs on CPU in about a second. No downloads, no pretrained weights, no GPU.

## What is included

**LayerNorm** (Ba, Kiros, Hinton 2016). Normalizes activations over the last few dimensions, then applies a learned scale and shift. Implemented with the biased (population) variance so it lines up with `torch.nn.LayerNorm`.

**GELU** (Hendrycks, Gimpel 2016). The Gaussian error linear unit, `x * Phi(x)`. Both the exact `erf` form and the tanh approximation used in the original GPT and BERT code are provided.

**Label smoothing cross entropy** (Szegedy et al. 2016). Replaces the one hot target with a softened distribution that puts most of the mass on the true class and spreads the rest uniformly. This pulls the model away from overconfident predictions.

**MixUp** (Zhang et al. 2018). Trains on convex combinations of pairs of examples and the matching convex combination of their labels. The mixing coefficient is drawn from a Beta distribution.

**Stochastic depth / drop path** (Huang et al. 2016). During training a whole residual branch is dropped per sample with some probability, and the survivors are scaled up so the expected activation stays put. At evaluation nothing is dropped.

## Layout

```
src/        the implementations
tests/      pytest behavior and property checks
```

## How the tests check correctness

Each test is a behavior check, not a snapshot of numbers I made up:

- LayerNorm output is compared directly against `torch.nn.LayerNorm` on random tensors, including the multi dimensional normalized shape case with randomized affine parameters.
- GELU is compared against its closed form `0.5 * x * (1 + erf(x / sqrt(2)))` and against `torch.nn.functional.gelu` for both the exact and tanh variants.
- Label smoothing with smoothing set to zero reduces exactly to cross entropy, and a positive smoothing value raises the loss on a confidently correct prediction. There is also a check against a manually constructed smoothed target.
- MixUp soft labels stay valid probability distributions that sum to one, mixed inputs land inside the convex hull of the batch, and lambda equal to one recovers the identity.
- Drop path is the identity at evaluation, drops or scales whole samples during training, and preserves the expected activation thanks to inverted scaling.

## Running

```
pip install -r requirements.txt
pytest tests/ -q
```

On my machine this reports 25 passed in about one second on CPU.

## Usage

```python
import torch
from src import LayerNorm, gelu, LabelSmoothingCrossEntropy, mixup_batch, DropPath

x = torch.randn(4, 32)
norm = LayerNorm(32)
h = gelu(norm(x))

logits = torch.randn(4, 10)
target = torch.randint(0, 10, (4,))
loss = LabelSmoothingCrossEntropy(smoothing=0.1)(logits, target)

images = torch.randn(4, 3, 8, 8)
mixed_x, mixed_y, lam = mixup_batch(images, target, alpha=0.2, num_classes=10)

drop = DropPath(drop_prob=0.1)
y = drop(h)
```
