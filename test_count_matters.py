#!/usr/bin/env python3
"""
Quick check: for the reconstruction loss, how much does COUNT matter
vs content adaptation?

If the upstream can minimize loss equally well at any count, then
reconstruction loss is NOT a test of count-optimization — the model
has no reason to change count.
"""

import torch

T, D = 40, 32


def common(boundaries):
    boundaries = boundaries.clone()
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if n_segments == 0:
        return None
    tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(
        start=0, end=n_segments, device=boundaries.device)
    hh1 = boundaries.cumsum(1)
    hh1 -= boundaries
    return tmp - hh1.unsqueeze(-1)


def recon_loss_at_count(hidden, n_boundaries):
    """Reconstruction loss with n evenly-spaced boundaries (no learning)."""
    B = hidden.shape[0]
    if n_boundaries == 0:
        n_boundaries = 1
    spacing = max(T // n_boundaries, 1)
    boundaries = torch.zeros(B, T)
    for i in range(0, T, spacing):
        boundaries[:, i] = 1.0
    boundaries[:, -1] = 1.0

    foo = common(boundaries)
    if foo is None:
        return float('inf')
    out = (foo.detach() != 0)
    sm = torch.where(out, torch.zeros_like(foo), 1.0 - foo)
    c = sm.sum(dim=1).clamp(min=1e-8)
    pooled = torch.bmm(sm.transpose(1, 2), hidden) / c.unsqueeze(-1)

    seg_ids = (boundaries.cumsum(1) - boundaries).long().clamp(0, pooled.shape[1] - 1)
    recon = torch.gather(pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
    return (recon - hidden).pow(2).mean().item()


torch.manual_seed(42)

# Simple: piecewise constant (5 segments)
simple = torch.zeros(4, T, D)
for i in range(4):
    for s in range(5):
        val = torch.randn(D) * 2.0
        start = s * (T // 5)
        end = (s + 1) * (T // 5) if s < 4 else T
        simple[i, start:end] = val

# Complex: random
complex_ = torch.randn(4, T, D)

print("Reconstruction loss vs boundary count (no learning, evenly spaced):")
print()
print(f"  {'count':>6}  {'simple':>10}  {'complex':>10}  {'ratio':>8}")
print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*8}")

for n in [1, 2, 3, 5, 8, 10, 15, 20, 30, 39]:
    ls = recon_loss_at_count(simple, n)
    lc = recon_loss_at_count(complex_, n)
    ratio = lc / ls if ls > 0.001 else float('inf')
    print(f"  {n:>6}  {ls:>10.4f}  {lc:>10.4f}  {ratio:>8.2f}")

print()
print("  Key question: at the count the model converges to (typically 0-5),")
print("  is the loss already near-minimal for both classes?")
print("  If yes, the model has no incentive to change count.")
