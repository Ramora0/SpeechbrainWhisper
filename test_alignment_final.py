#!/usr/bin/env python3
"""
Final definitive test: across many configurations, does the continuous
gradient predict the discrete benefit of adding a boundary?

Test separately:
  1. Single segment (only sentinel exists) - adding first boundary
  2. Multi-segment (2+ segments exist) - adding another boundary

Use reconstruction loss since that's the one that clearly benefits
from more boundaries.
"""

import torch
import torch.nn as nn


def common(boundaries):
    boundaries = boundaries.clone()
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if n_segments == 0:
        return None
    tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(
        start=0, end=n_segments, device=boundaries.device)
    hh1 = boundaries.cumsum(1)
    hh1 -= boundaries
    foo = tmp - hh1.unsqueeze(-1)
    return foo


def pool_and_recon_loss(boundaries, hidden):
    """Mean pool with soft mask, then compute reconstruction loss."""
    B, T, D = hidden.shape
    foo = common(boundaries)
    if foo is None:
        return torch.tensor(float('inf'))

    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
    counts = soft_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
    pooled = seg_sum / counts.unsqueeze(-1)

    seg_ids = (boundaries.cumsum(1) - boundaries).long().clamp(0, max(pooled.shape[1]-1, 0))
    recon = torch.gather(pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
    return (recon - hidden).pow(2).mean()


def run_alignment_test(n_existing_boundaries, n_trials=500):
    """Test gradient alignment for adding a boundary, given n existing boundaries."""
    T, D = 40, 16

    aligned = 0
    misaligned = 0
    zero_grad = 0
    grad_magnitudes = []

    for trial in range(n_trials):
        torch.manual_seed(trial * 100 + n_existing_boundaries)
        hidden = torch.randn(1, T, D)

        # Place existing boundaries evenly
        existing_positions = []
        if n_existing_boundaries > 0:
            spacing = T // (n_existing_boundaries + 1)
            for i in range(n_existing_boundaries):
                existing_positions.append((i + 1) * spacing)

        # New boundary position: midpoint of a random existing segment
        if len(existing_positions) == 0:
            new_pos = T // 2
        else:
            all_bounds = [0] + sorted(existing_positions) + [T-1]
            # Pick the largest segment to split
            max_gap = 0
            best_mid = T // 2
            for i in range(len(all_bounds) - 1):
                gap = all_bounds[i+1] - all_bounds[i]
                if gap > max_gap:
                    max_gap = gap
                    best_mid = (all_bounds[i] + all_bounds[i+1]) // 2
            new_pos = best_mid

        # Continuous gradient
        prob = torch.tensor(0.3, requires_grad=True)
        boundaries = torch.zeros(1, T)
        for ep in existing_positions:
            boundaries[0, ep] = 1.0
        hard = (prob > 0.5).float()
        boundaries[0, new_pos] = hard - prob.detach() + prob
        boundaries[0, -1] = 1.0

        loss = pool_and_recon_loss(boundaries, hidden)
        if loss.requires_grad:
            loss.backward()
            grad = prob.grad.item()
        else:
            grad = 0.0

        # Discrete comparison: loss with vs without new boundary
        b_with = torch.zeros(1, T)
        for ep in existing_positions:
            b_with[0, ep] = 1.0
        b_with[0, new_pos] = 1.0
        b_with[0, -1] = 1.0
        loss_with = pool_and_recon_loss(b_with, hidden).item()

        b_without = torch.zeros(1, T)
        for ep in existing_positions:
            b_without[0, ep] = 1.0
        b_without[0, -1] = 1.0
        loss_without = pool_and_recon_loss(b_without, hidden).item()

        adding_helps = loss_with < loss_without
        # Adding always helps for reconstruction (more segments = better fit)

        if abs(grad) < 1e-10:
            zero_grad += 1
        elif (grad < 0) == adding_helps:  # negative grad = prob should increase = add boundary
            aligned += 1
        else:
            misaligned += 1

        grad_magnitudes.append(abs(grad))

    total_nonzero = aligned + misaligned
    pct = 100 * aligned / total_nonzero if total_nonzero > 0 else 0

    return aligned, misaligned, zero_grad, sum(grad_magnitudes) / len(grad_magnitudes)


def main():
    print("=" * 70)
    print("DEFINITIVE TEST: Does continuous gradient predict discrete benefit?")
    print("=" * 70)
    print(f"\nReconstruction loss (always benefits from more boundaries)")
    print(f"T=40, D=16, 500 trials per configuration\n")

    print(f"  {'Config':>30}  {'Aligned':>8}  {'Misaligned':>10}  {'Zero':>6}  "
          f"{'Accuracy':>9}  {'Avg |grad|':>11}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*11}")

    for n_existing in [1, 2, 3, 5, 8, 12]:
        a, m, z, avg_g = run_alignment_test(n_existing)
        total = a + m
        pct = 100 * a / total if total > 0 else 0
        print(f"  {n_existing:>2} existing boundaries         "
              f"{a:>8}  {m:>10}  {z:>6}  {pct:>8.1f}%  {avg_g:>11.2e}")

    print(f"\n  50% accuracy = random chance (gradient carries NO useful information)")
    print(f"  >80% = gradient meaningfully predicts benefit of adding boundary")
    print(f"  <50% = gradient is ANTI-correlated (actively pushes wrong direction)")

    # Also test: does gradient for REMOVING work better?
    print(f"\n\n  Now testing REMOVE direction (loss=pooled.pow(2).mean, wants fewer):")
    print(f"  {'Config':>30}  {'Aligned':>8}  {'Misaligned':>10}  {'Zero':>6}  "
          f"{'Accuracy':>9}")
    print(f"  {'-'*30}  {'-'*8}  {'-'*10}  {'-'*6}  {'-'*9}")

    for n_existing in [2, 3, 5, 8]:
        a, m, z = test_remove_alignment(n_existing)
        total = a + m
        pct = 100 * a / total if total > 0 else 0
        print(f"  {n_existing:>2} existing (remove one)        "
              f"{a:>8}  {m:>10}  {z:>6}  {pct:>8.1f}%")


def test_remove_alignment(n_boundaries, n_trials=500):
    """Test if gradient correctly identifies which boundary to REMOVE."""
    T, D = 40, 16
    aligned = 0
    misaligned = 0
    zero_grad = 0

    for trial in range(n_trials):
        torch.manual_seed(trial * 100 + n_boundaries + 1000)
        hidden = torch.randn(1, T, D)

        # Place boundaries evenly
        spacing = T // (n_boundaries + 1)
        positions = [(i + 1) * spacing for i in range(n_boundaries)]

        # Pick one to make removable
        remove_idx = trial % n_boundaries
        remove_pos = positions[remove_idx]

        # With the boundary (prob > 0.5)
        prob = torch.tensor(0.7, requires_grad=True)
        boundaries = torch.zeros(1, T)
        for i, ep in enumerate(positions):
            if i == remove_idx:
                hard = (prob > 0.5).float()
                boundaries[0, ep] = hard - prob.detach() + prob
            else:
                boundaries[0, ep] = 1.0
        boundaries[0, -1] = 1.0

        foo = common(boundaries)
        if foo is None:
            continue
        out = (foo.detach() != 0)
        sm = torch.where(out, torch.zeros_like(foo), 1.0 - foo)
        c = sm.sum(dim=1).clamp(min=1e-8)
        pooled = torch.bmm(sm.transpose(1, 2), hidden) / c.unsqueeze(-1)
        loss = pooled.pow(2).mean()
        loss.backward()
        grad = prob.grad.item()

        # Discrete: loss with vs without this boundary
        b_with = torch.zeros(1, T)
        for ep in positions:
            b_with[0, ep] = 1.0
        b_with[0, -1] = 1.0
        loss_with = pool_and_recon_loss(b_with, hidden).item()  # Wrong loss for this test

        # Actually use the same loss (pooled.pow(2))
        foo_w = common(b_with)
        out_w = (foo_w.detach() != 0)
        sm_w = torch.where(out_w, torch.zeros_like(foo_w), 1.0 - foo_w)
        c_w = sm_w.sum(dim=1).clamp(min=1e-8)
        p_w = torch.bmm(sm_w.transpose(1, 2), hidden) / c_w.unsqueeze(-1)
        loss_with = p_w.pow(2).mean().item()

        b_without = torch.zeros(1, T)
        for i, ep in enumerate(positions):
            if i != remove_idx:
                b_without[0, ep] = 1.0
        b_without[0, -1] = 1.0
        foo_wo = common(b_without)
        out_wo = (foo_wo.detach() != 0)
        sm_wo = torch.where(out_wo, torch.zeros_like(foo_wo), 1.0 - foo_wo)
        c_wo = sm_wo.sum(dim=1).clamp(min=1e-8)
        p_wo = torch.bmm(sm_wo.transpose(1, 2), hidden) / c_wo.unsqueeze(-1)
        loss_without = p_wo.pow(2).mean().item()

        removing_helps = loss_without < loss_with  # True if loss drops when we remove

        if abs(grad) < 1e-10:
            zero_grad += 1
        elif (grad > 0) == removing_helps:  # positive grad = decrease prob = remove
            aligned += 1
        else:
            misaligned += 1

    return aligned, misaligned, zero_grad


if __name__ == "__main__":
    main()
