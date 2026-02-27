#!/usr/bin/env python3
"""
Follow-up: many gradients were exactly 0.0 in the previous test.
Why? Is the gradient dead when a boundary is inactive?

The chain is:
  loss → pooled → soft_mask → (1 - foo) → foo → cumsum → boundary_i → STE → prob

When boundary_i is inactive (prob=0.3, hard=0):
  STE value: hard - soft.detach() + soft = 0 - 0.3 + 0.3 = 0.3
  Wait — no. In test.py, hard = (prob > 0.5).float() = 0.0
  STE: boundary_val = 0.0 - prob.detach() + prob

  But prob.detach() = 0.3, so boundary_val = 0.0 - 0.3 + 0.3 = 0.0
  ... with gradient 1.0 w.r.t. prob.

  Hmm actually: 0.0 - 0.3 + 0.3 = 0.0, yes, but the gradient of
  (0.0 - detach(prob) + prob) w.r.t. prob is 1.0 (only the non-detached prob contributes).

  So boundary value = 0.0, gradient = 1.0 w.r.t. prob.

  cumsum([..., 0.0, ...]) at boundary_i position:
  hh1_j = sum(boundaries[0..j]) - boundaries[j]
  For j > i: hh1_j includes boundary_i = 0.0
  d(hh1_j)/d(boundary_i) = 1 for j > i

  foo = tmp - hh1
  d(foo_j,s)/d(boundary_i) = -1 for j > i

  soft_mask at in-segment positions = 1 - foo
  d(soft_mask_j,s)/d(boundary_i) = +1 for j > i (in-segment only)

  BUT: the out_of_segment mask is computed from foo.detach()
  When boundary_i = 0, foo values are the SAME as without boundary_i.
  So the in-segment/out-of-segment classification is the same.

  The gradient reaches soft_mask at in-segment positions.
  These positions already have soft_mask = 1.0 (since foo = 0).
  The gradient says: increasing boundary_i by epsilon makes soft_mask = 1 + epsilon.

  In the MEAN POOL: pooled = (soft_mask^T @ hidden) / count
  count = soft_mask.sum(dim=1)

  If all in-segment soft_mask values increase by epsilon equally:
  - numerator: each position contributes hidden_j * (1 + epsilon) instead of hidden_j * 1
  - denominator: count increases by (num_positions * epsilon)
  - pooled = (sum_j hidden_j * (1+eps)) / (N * (1+eps)) = (sum_j hidden_j) / N = same!

  THE GRADIENT IS ZERO BECAUSE MEAN POOLING IS INVARIANT TO UNIFORM SCALING!
  If all soft_mask values in a segment are shifted by the same amount,
  the mean doesn't change.

  This is the fundamental issue: the cumsum gradient is +1 for ALL downstream
  positions uniformly. Mean pooling divides by count. A uniform shift cancels
  in numerator/denominator. The gradient is EXACTLY ZERO.
"""

import torch


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


def test_uniform_shift_cancellation():
    """
    Prove: when a boundary is inactive and there's 1 segment,
    the cumsum gradient provides a UNIFORM shift to all soft_mask values,
    which cancels perfectly in mean pooling (numerator / denominator).
    """
    print("=" * 70)
    print("TEST: Uniform shift cancellation in mean pooling")
    print("=" * 70)

    T, D = 8, 4
    torch.manual_seed(42)
    hidden = torch.randn(1, T, D)

    # Case 1: Single-segment, inactive boundary
    prob = torch.tensor(0.3, requires_grad=True)
    boundaries = torch.zeros(1, T)
    hard = (prob > 0.5).float()
    boundaries[0, 3] = hard - prob.detach() + prob
    boundaries[0, -1] = 1.0

    foo = common(boundaries)
    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)

    # Check: are all in-segment soft_mask values affected equally?
    print(f"\n  boundaries: {boundaries[0].tolist()}")
    print(f"  foo[:, seg=0]: {foo[0, :, 0].tolist()}")
    print(f"  soft_mask[:, seg=0]: {soft_mask[0, :, 0].tolist()}")

    counts = soft_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
    pooled = seg_sum / counts.unsqueeze(-1)

    loss = pooled.pow(2).mean()
    loss.backward()

    print(f"\n  d(loss)/d(prob) = {prob.grad.item():+.10f}")
    if abs(prob.grad.item()) < 1e-8:
        print(f"  CONFIRMED: Gradient is ZERO (or nearly zero)")
        print(f"\n  Reason: cumsum shifts ALL in-segment soft_mask values by +1 uniformly")
        print(f"  Mean pooling = sum(w_i * h_i) / sum(w_i)")
        print(f"  If all w_i shift by same epsilon: sum((w+eps)*h) / sum(w+eps)")
        print(f"  = (sum(w*h) + eps*sum(h)) / (sum(w) + N*eps)")
        print(f"  At eps=0, derivative = (sum(h)*sum(w) - sum(w*h)*N) / sum(w)^2")
        print(f"  = (mean(h) - mean(h)) * ... = 0 only if weights are already uniform")
        print(f"  Actually let me check if weights are uniform...")
        print(f"  All soft_mask in segment: {soft_mask[0, :, 0].tolist()}")
        print(f"  They're all 1.0 (uniform!) since foo=0 at all in-segment positions.")
        print(f"  So yes: uniform weights + uniform shift = zero gradient.")


def test_multi_segment_gradient():
    """
    When there are MULTIPLE segments, does an inactive boundary
    between them get nonzero gradient?

    With 2 existing segments, an inactive boundary at the border
    affects positions in segment 2 but not segment 1.
    The shift is no longer uniform across ALL soft_mask values
    (only segment 2 positions shift), so the gradient might be nonzero.
    """
    print("\n" + "=" * 70)
    print("TEST: Multi-segment case — inactive boundary gets gradient?")
    print("=" * 70)

    T, D = 12, 4
    torch.manual_seed(42)
    hidden = torch.randn(1, T, D)

    # Two existing segments (boundaries at 5 and 11), try adding one at 8
    prob = torch.tensor(0.3, requires_grad=True)
    boundaries = torch.zeros(1, T)
    boundaries[0, 5] = 1.0  # existing
    hard = (prob > 0.5).float()
    boundaries[0, 8] = hard - prob.detach() + prob  # potential new one
    boundaries[0, -1] = 1.0  # existing sentinel

    foo = common(boundaries)
    print(f"\n  boundaries: {boundaries[0].tolist()}")
    print(f"  n_segments: {foo.shape[2]}")
    for s in range(foo.shape[2]):
        print(f"  foo[:, seg={s}]: {[f'{v:.1f}' for v in foo[0, :, s].tolist()]}")

    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)

    print(f"\n  soft_mask (which positions contribute to which segment):")
    for s in range(soft_mask.shape[2]):
        print(f"  soft_mask[:, seg={s}]: {[f'{v:.2f}' for v in soft_mask[0, :, s].tolist()]}")

    counts = soft_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
    pooled = seg_sum / counts.unsqueeze(-1)

    loss = pooled.pow(2).mean()
    loss.backward()

    grad = prob.grad.item()
    print(f"\n  d(loss)/d(prob) = {grad:+.10f}")

    if abs(grad) > 1e-8:
        print(f"  NONZERO! In multi-segment case, inactive boundary gets gradient.")
        print(f"  Reason: cumsum shift affects segment 2 positions but not segment 1,")
        print(f"  breaking the uniform-shift cancellation within segment 2.")
    else:
        print(f"  Still zero — even multi-segment doesn't help.")

    # Check: what about a RECONSTRUCTION loss?
    seg_ids = (boundaries.cumsum(1) - boundaries).long().clamp(0, pooled.shape[1]-1)
    recon = torch.gather(pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
    loss2 = (recon - hidden).pow(2).mean()

    prob2 = torch.tensor(0.3, requires_grad=True)
    boundaries2 = torch.zeros(1, T)
    boundaries2[0, 5] = 1.0
    hard2 = (prob2 > 0.5).float()
    boundaries2[0, 8] = hard2 - prob2.detach() + prob2
    boundaries2[0, -1] = 1.0

    foo2 = common(boundaries2)
    out2 = (foo2.detach() != 0)
    sm2 = torch.where(out2, torch.zeros_like(foo2), 1.0 - foo2)
    c2 = sm2.sum(dim=1).clamp(min=1e-8)
    p2 = torch.bmm(sm2.transpose(1, 2), hidden) / c2.unsqueeze(-1)

    si2 = (boundaries2.cumsum(1) - boundaries2).long().clamp(0, p2.shape[1]-1)
    r2 = torch.gather(p2, 1, si2.unsqueeze(-1).expand(-1, -1, D))
    loss_r2 = (r2 - hidden).pow(2).mean()
    loss_r2.backward()
    grad2 = prob2.grad.item()

    print(f"\n  With reconstruction loss: d(loss)/d(prob) = {grad2:+.10f}")

    # Discrete comparison
    boundaries_3seg = torch.zeros(1, T)
    boundaries_3seg[0, 5] = 1.0
    boundaries_3seg[0, 8] = 1.0
    boundaries_3seg[0, -1] = 1.0
    foo_3 = common(boundaries_3seg)
    out_3 = (foo_3.detach() != 0)
    sm_3 = torch.where(out_3, torch.zeros_like(foo_3), 1.0 - foo_3)
    c_3 = sm_3.sum(dim=1).clamp(min=1e-8)
    p_3 = torch.bmm(sm_3.transpose(1, 2), hidden) / c_3.unsqueeze(-1)
    si_3 = (boundaries_3seg.cumsum(1) - boundaries_3seg).long().clamp(0, p_3.shape[1]-1)
    r_3 = torch.gather(p_3, 1, si_3.unsqueeze(-1).expand(-1, -1, D))
    loss_3seg = (r_3 - hidden).pow(2).mean().item()

    print(f"  Loss 2 seg: {loss_r2.item():.6f}")
    print(f"  Loss 3 seg: {loss_3seg:.6f}")
    print(f"  Adding boundary {'HELPS' if loss_3seg < loss_r2.item() else 'HURTS'} by {abs(loss_r2.item() - loss_3seg):.6f}")
    print(f"  Gradient says: {'add' if grad2 < 0 else 'do NOT add'}")


def test_within_segment_vs_between_segment():
    """
    The cumsum gradient shifts ALL positions after boundary_i.
    But the soft_mask is only nonzero at in-segment positions.

    For a boundary WITHIN an existing segment (between two existing boundaries):
    - Positions after the new boundary but within the same segment get shifted
    - Positions in the NEXT segment also get shifted
    - This breaks uniformity

    For a boundary that would create a new segment from nothing (only 1 existing):
    - ALL positions after are in the same segment
    - Shift is uniform -> gradient = 0
    """
    print("\n" + "=" * 70)
    print("TEST: Gradient depends on existing segment structure")
    print("=" * 70)

    T, D = 16, 4
    torch.manual_seed(42)
    hidden = torch.randn(1, T, D)

    configs = [
        ("1 existing segment",
         [],  # existing boundaries (besides sentinel)
         7),  # potential new boundary position
        ("2 existing segments",
         [7],
         3),
        ("3 existing segments",
         [4, 9],
         7),
        ("4 existing segments",
         [3, 7, 11],
         5),
    ]

    for name, existing, new_pos in configs:
        prob = torch.tensor(0.3, requires_grad=True)
        boundaries = torch.zeros(1, T)
        for ep in existing:
            boundaries[0, ep] = 1.0
        hard = (prob > 0.5).float()
        boundaries[0, new_pos] = hard - prob.detach() + prob
        boundaries[0, -1] = 1.0

        foo = common(boundaries)
        if foo is None:
            print(f"  {name}: no segments (skip)")
            continue

        out = (foo.detach() != 0)
        sm = torch.where(out, torch.zeros_like(foo), 1.0 - foo)
        c = sm.sum(dim=1).clamp(min=1e-8)
        p = torch.bmm(sm.transpose(1, 2), hidden) / c.unsqueeze(-1)

        loss = p.pow(2).mean()
        loss.backward()

        grad = prob.grad.item()
        print(f"  {name:30s}  new_at={new_pos:2d}  n_seg={foo.shape[2]}  "
              f"grad={grad:+.8f}  {'ZERO' if abs(grad) < 1e-8 else 'NONZERO'}")


if __name__ == "__main__":
    test_uniform_shift_cancellation()
    test_multi_segment_gradient()
    test_within_segment_vs_between_segment()
