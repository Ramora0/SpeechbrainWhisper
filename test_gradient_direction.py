#!/usr/bin/env python3
"""
Diagnostic: WHY does the gradient through mean-pooling always push
boundary probs DOWN, regardless of whether the downstream loss wants
more or fewer boundaries?

Hypothesis: The gradient through the soft_mask (1 - foo) in the
mean-pooling has a structural bias. Let's trace the exact gradient
direction for a single boundary probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def common(boundaries):
    """Arithmetic segment-distance (gradient-preserving)."""
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


def trace_single_boundary_gradient():
    """
    Minimal 1D case: 6 positions, 1 boundary somewhere in the middle.

    We'll make the boundary probability a leaf variable and trace
    d(loss)/d(prob) for different losses.
    """
    print("=" * 70)
    print("TRACE: Gradient of a single boundary probability")
    print("=" * 70)

    T = 6
    D = 4

    # Fixed hidden states (not learnable here - we isolate the boundary prob)
    torch.manual_seed(42)
    hidden = torch.randn(1, T, D)

    for boundary_pos in [2, 3, 4]:
        print(f"\n--- Boundary at position {boundary_pos} ---")

        # Create a boundary tensor where position `boundary_pos` has a learnable prob
        # and the last position is always 1 (required sentinel)
        prob = torch.tensor(0.5, requires_grad=True)

        # STE: hard boundary from prob
        hard = (prob > 0.5).float()
        boundary_val = hard - prob.detach() + prob  # STE

        # Build boundary vector: 0s everywhere except boundary_pos and last
        boundaries = torch.zeros(1, T)
        boundaries[0, boundary_pos] = boundary_val
        boundaries[0, -1] = 1.0  # sentinel

        foo = common(boundaries)
        if foo is None:
            print("  No segments (unexpected)")
            continue

        # Build soft mask (same as BoundaryPredictor2)
        out_of_segment = (foo.detach() != 0)
        soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)

        # Mean pool
        counts = soft_mask.sum(dim=1).clamp(min=1e-8)
        seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
        pooled = seg_sum / counts.unsqueeze(-1)

        n_segments = pooled.shape[1]
        print(f"  Segments: {n_segments}")
        print(f"  Segment sizes: {counts[0].tolist()}")

        # Loss A: minimize pooled magnitude (wants fewer boundaries)
        loss_a = pooled.pow(2).mean()
        loss_a.backward(retain_graph=True)
        grad_a = prob.grad.item()
        prob.grad = None

        # Loss B: maximize pooled magnitude (wants more boundaries)
        loss_b = -pooled.pow(2).mean()
        loss_b.backward(retain_graph=True)
        grad_b = prob.grad.item()
        prob.grad = None

        # Loss C: maximize number of segments directly
        # (if more boundaries = more segments, gradient should push prob UP)
        loss_c = -boundaries.sum()
        loss_c.backward(retain_graph=True)
        grad_c = prob.grad.item()
        prob.grad = None

        print(f"  Loss A (minimize pooled): grad = {grad_a:+.6f}  "
              f"({'push DOWN' if grad_a > 0 else 'push UP'})")
        print(f"  Loss B (maximize pooled): grad = {grad_b:+.6f}  "
              f"({'push DOWN' if grad_b > 0 else 'push UP'})")
        print(f"  Loss C (maximize count):  grad = {grad_c:+.6f}  "
              f"({'push DOWN' if grad_c > 0 else 'push UP'})")


def trace_pooling_gradient_mechanism():
    """
    The key question: when we change a boundary from 0->1 (splitting a segment),
    what happens to the pooled output, and which direction does the gradient push?

    Theory:
    - Adding a boundary splits one segment into two
    - The pooled representations of the two halves are generally DIFFERENT from the original
    - Any downstream loss that depends on the CONTENT of pooled vectors
      (not just count) will have a complex gradient that may systematically
      favor fewer boundaries
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: What happens to pooled output when we add/remove a boundary?")
    print("=" * 70)

    T = 8
    D = 4
    torch.manual_seed(42)
    hidden = torch.randn(1, T, D)

    # Case 1: No middle boundary (1 segment + sentinel)
    boundaries_1seg = torch.zeros(1, T)
    boundaries_1seg[0, -1] = 1.0

    foo_1 = common(boundaries_1seg)
    out_1 = (foo_1.detach() != 0)
    mask_1 = torch.where(out_1, torch.zeros_like(foo_1), 1.0 - foo_1)
    counts_1 = mask_1.sum(dim=1).clamp(min=1e-8)
    pooled_1 = torch.bmm(mask_1.transpose(1, 2), hidden) / counts_1.unsqueeze(-1)

    # Case 2: Middle boundary at position 4 (2 segments + sentinel)
    boundaries_2seg = torch.zeros(1, T)
    boundaries_2seg[0, 3] = 1.0
    boundaries_2seg[0, -1] = 1.0

    foo_2 = common(boundaries_2seg)
    out_2 = (foo_2.detach() != 0)
    mask_2 = torch.where(out_2, torch.zeros_like(foo_2), 1.0 - foo_2)
    counts_2 = mask_2.sum(dim=1).clamp(min=1e-8)
    pooled_2 = torch.bmm(mask_2.transpose(1, 2), hidden) / counts_2.unsqueeze(-1)

    print(f"\n  1 segment:  pooled norm = {pooled_1.norm().item():.4f}")
    print(f"  2 segments: pooled norm = {pooled_2.norm().item():.4f}")
    print(f"  1 segment mean:  {pooled_1[0, 0, :].tolist()}")
    print(f"  2 segments half1: {pooled_2[0, 0, :].tolist()}")
    print(f"  2 segments half2: {pooled_2[0, 1, :].tolist()}")

    # For RECONSTRUCTION loss, more segments should help
    # Reconstruct: repeat pooled per position
    # 1 segment case
    reconstructed_1 = pooled_1.expand(1, T, D)
    recon_loss_1 = (reconstructed_1 - hidden).pow(2).mean().item()

    # 2 segment case
    seg_ids_2 = (boundaries_2seg.cumsum(dim=1) - boundaries_2seg).long()
    seg_ids_2_clamped = seg_ids_2.clamp(0, pooled_2.shape[1] - 1)
    reconstructed_2 = torch.gather(
        pooled_2, 1,
        seg_ids_2_clamped.unsqueeze(-1).expand(-1, -1, D)
    )
    recon_loss_2 = (reconstructed_2 - hidden).pow(2).mean().item()

    print(f"\n  Reconstruction loss (1 seg): {recon_loss_1:.6f}")
    print(f"  Reconstruction loss (2 seg): {recon_loss_2:.6f}")
    print(f"  More segments {'HELPS' if recon_loss_2 < recon_loss_1 else 'HURTS'}")


def analyze_gradient_through_cumsum():
    """
    The STE + cumsum gradient path:

    loss -> pooled -> soft_mask -> (1 - foo) -> foo -> cumsum -> boundaries -> STE -> probs

    foo = tmp - hh1.unsqueeze(-1)
    hh1 = boundaries.cumsum(1) - boundaries

    So: d(foo)/d(boundary_i) = -d(hh1)/d(boundary_i)
    And: hh1 = cumsum - identity, so d(hh1)/d(boundary_i) = 1 for all positions j >= i, minus 1 at position i
    Which means d(hh1_j)/d(boundary_i) = 1 if j > i, 0 if j <= i

    This means increasing boundary_i DECREASES foo for all subsequent positions.

    Now soft_mask = 1 - foo (at in-segment positions)
    So d(soft_mask)/d(boundary_i) = -d(foo)/d(boundary_i) = +d(hh1)/d(boundary_i)

    For positions j > i: d(soft_mask_j)/d(boundary_i) = +1

    BUT WAIT: soft_mask is only nonzero where detached foo == 0.
    The positions where foo == 0 are exactly the in-segment positions.

    When we increase boundary_i:
    - It splits a segment, changing which positions are "in-segment"
    - But the gradient doesn't know about the SPLIT because the mask uses detached foo
    - The gradient only sees: "increasing boundary makes soft_mask values larger at downstream positions"

    Let's verify this numerically.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: Gradient through cumsum path")
    print("=" * 70)

    T = 8

    # Make boundary probabilities differentiable
    probs = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], requires_grad=True)

    # STE
    hard = (probs > 0.5).float()
    boundaries = hard - probs.detach() + probs
    boundaries = boundaries.unsqueeze(0)  # [1, T]

    # Compute foo
    foo = common(boundaries)
    if foo is None:
        print("  No boundaries")
        return

    print(f"  boundaries: {boundaries[0].tolist()}")
    print(f"  foo shape: {foo.shape}")
    print(f"  foo[0,:,0]: {foo[0,:,0].tolist()}")  # distance to segment 0
    if foo.shape[2] > 1:
        print(f"  foo[0,:,1]: {foo[0,:,1].tolist()}")  # distance to segment 1

    # Gradient of foo w.r.t. prob at position 3
    for seg in range(foo.shape[2]):
        for pos in range(T):
            if foo[0, pos, seg].requires_grad:
                foo[0, pos, seg].backward(retain_graph=True)
                if probs.grad is not None:
                    g = probs.grad[3].item()
                    probs.grad.zero_()
                    if abs(g) > 1e-10:
                        print(f"  d(foo[pos={pos},seg={seg}])/d(prob[3]) = {g:+.4f}")


def test_soft_mask_gradient_bias():
    """
    KEY TEST: Does the soft_mask gradient have an inherent directional bias?

    The soft_mask = where(detached_oob, 0, 1 - foo)

    At in-segment positions: soft_mask = 1 - foo = 1 - 0 = 1 (value)
    But the gradient is: d(soft_mask)/d(foo) = -1
    And d(foo)/d(boundary) = -1 for positions after the boundary (via cumsum)

    So: d(soft_mask)/d(boundary) = (-1) * (-1) = +1 for downstream positions

    This means: increasing a boundary probability INCREASES the soft_mask
    values at in-segment positions that come AFTER the boundary.

    But the soft_mask is used in: seg_sum = soft_mask^T @ hidden

    So increasing soft_mask at those positions adds more hidden-state content
    to the segment sum. Then pooled = seg_sum / count.

    The count also increases! So the net effect on pooled is ambiguous...

    BUT: the gradient doesn't flow through count properly because count
    uses sum of soft_mask, and the denominator gradient has a specific sign.

    Let me check: is there a bias in d(loss)/d(prob) when loss = f(pooled)?
    """
    print("\n" + "=" * 70)
    print("KEY TEST: Direction of d(loss)/d(boundary_prob) for various losses")
    print("=" * 70)

    torch.manual_seed(42)
    T = 10
    D = 8
    hidden = torch.randn(1, T, D)

    # Test many different downstream losses
    losses_wanting_more = 0
    losses_wanting_fewer = 0

    for trial in range(50):
        torch.manual_seed(trial)
        hidden = torch.randn(1, T, D)

        # Prob at position 4, sentinel at end
        prob = torch.tensor(0.3, requires_grad=True)

        boundaries = torch.zeros(1, T)
        hard = (prob > 0.5).float()
        boundaries[0, 4] = hard - prob.detach() + prob  # STE
        boundaries[0, -1] = 1.0

        foo = common(boundaries)
        if foo is None:
            continue

        out_of_segment = (foo.detach() != 0)
        soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
        counts = soft_mask.sum(dim=1).clamp(min=1e-8)
        seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
        pooled = seg_sum / counts.unsqueeze(-1)

        # Random downstream linear combination (simulates arbitrary loss)
        target = torch.randn_like(pooled)
        loss = (pooled - target).pow(2).mean()
        loss.backward()

        grad = prob.grad.item()
        if grad > 0:
            losses_wanting_fewer += 1  # positive grad -> descent decreases prob
        else:
            losses_wanting_more += 1

    print(f"\n  Over 50 random trials (random hidden, random target):")
    print(f"  Gradient pushes boundary DOWN: {losses_wanting_fewer}/50")
    print(f"  Gradient pushes boundary UP:   {losses_wanting_more}/50")

    if losses_wanting_fewer > 35:
        print(f"\n  STRONG BIAS: Gradient almost always pushes boundaries DOWN!")
        print(f"  This is a STRUCTURAL problem in the gradient path.")
    elif losses_wanting_fewer > 25:
        print(f"\n  MODERATE BIAS toward fewer boundaries.")
    else:
        print(f"\n  No strong directional bias detected.")


def test_gradient_components():
    """
    Decompose the gradient into numerator and denominator contributions.

    pooled = seg_sum / count
    d(pooled)/d(soft_mask) has two terms:
      1. d(seg_sum)/d(soft_mask) / count  (numerator term)
      2. -seg_sum / count^2 * d(count)/d(soft_mask)  (denominator term)

    These pull in opposite directions. Let's see which wins.
    """
    print("\n" + "=" * 70)
    print("DECOMPOSITION: Numerator vs Denominator gradient contributions")
    print("=" * 70)

    torch.manual_seed(42)
    T = 8
    D = 4
    hidden = torch.randn(1, T, D)

    prob = torch.tensor(0.3, requires_grad=True)

    boundaries = torch.zeros(1, T)
    hard = (prob > 0.5).float()
    boundaries[0, 3] = hard - prob.detach() + prob
    boundaries[0, -1] = 1.0

    foo = common(boundaries)
    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)

    counts = soft_mask.sum(dim=1).clamp(min=1e-8)  # [1, S]
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)  # [1, S, D]

    # Compute pooled as two separate terms to trace gradients
    # pooled = seg_sum / counts
    pooled = seg_sum / counts.unsqueeze(-1)

    # Use a simple loss
    loss = pooled.pow(2).mean()
    loss.backward()

    grad_total = prob.grad.item()
    print(f"\n  Total gradient on boundary prob: {grad_total:+.8f}")

    # Now compute numerator-only gradient (freeze denominator)
    prob2 = torch.tensor(0.3, requires_grad=True)
    boundaries2 = torch.zeros(1, T)
    hard2 = (prob2 > 0.5).float()
    boundaries2[0, 3] = hard2 - prob2.detach() + prob2
    boundaries2[0, -1] = 1.0

    foo2 = common(boundaries2)
    out_of_segment2 = (foo2.detach() != 0)
    soft_mask2 = torch.where(out_of_segment2, torch.zeros_like(foo2), 1.0 - foo2)

    counts_frozen = soft_mask2.detach().sum(dim=1).clamp(min=1e-8)
    seg_sum2 = torch.bmm(soft_mask2.transpose(1, 2), hidden)
    pooled_num_only = seg_sum2 / counts_frozen.unsqueeze(-1)

    loss2 = pooled_num_only.pow(2).mean()
    loss2.backward()
    grad_num = prob2.grad.item()

    # Denominator-only gradient (freeze numerator)
    prob3 = torch.tensor(0.3, requires_grad=True)
    boundaries3 = torch.zeros(1, T)
    hard3 = (prob3 > 0.5).float()
    boundaries3[0, 3] = hard3 - prob3.detach() + prob3
    boundaries3[0, -1] = 1.0

    foo3 = common(boundaries3)
    out_of_segment3 = (foo3.detach() != 0)
    soft_mask3 = torch.where(out_of_segment3, torch.zeros_like(foo3), 1.0 - foo3)

    counts3 = soft_mask3.sum(dim=1).clamp(min=1e-8)
    seg_sum_frozen = torch.bmm(soft_mask3.detach().transpose(1, 2), hidden)
    pooled_den_only = seg_sum_frozen / counts3.unsqueeze(-1)

    loss3 = pooled_den_only.pow(2).mean()
    loss3.backward()
    grad_den = prob3.grad.item()

    print(f"  Numerator-only gradient:        {grad_num:+.8f}  "
          f"({'DOWN' if grad_num > 0 else 'UP'})")
    print(f"  Denominator-only gradient:       {grad_den:+.8f}  "
          f"({'DOWN' if grad_den > 0 else 'UP'})")
    print(f"  Sum of components:               {grad_num + grad_den:+.8f}")

    if grad_num > 0 and grad_den > 0:
        print(f"\n  BOTH components push boundaries DOWN!")
    elif grad_num < 0 and grad_den < 0:
        print(f"\n  BOTH components push boundaries UP!")
    else:
        winner = "NUMERATOR" if abs(grad_num) > abs(grad_den) else "DENOMINATOR"
        direction = "DOWN" if (grad_num + grad_den) > 0 else "UP"
        print(f"\n  Components disagree. {winner} wins -> pushes {direction}")


def test_cumsum_gradient_direction():
    """
    The fundamental issue: cumsum gradient.

    hh1 = cumsum(boundaries) - boundaries
    foo = segment_indices - hh1

    d(foo_j,s)/d(boundary_i) = -d(hh1_j)/d(boundary_i)

    For cumsum: d(cumsum_j)/d(boundary_i) = 1 if j >= i, else 0
    For hh1 = cumsum - identity: d(hh1_j)/d(boundary_i) = 1 if j > i, 0 if j == i, 0 if j < i

    So d(foo_j,s)/d(boundary_i) = -1 for all j > i, 0 otherwise.

    This means increasing boundary_i DECREASES foo at ALL subsequent positions
    for ALL segments. The gradient is inherently one-sided.

    In soft_mask = 1 - foo: increasing boundary_i INCREASES soft_mask at all j > i.

    But soft_mask is zeroed where detached foo != 0. So only in-segment positions matter.

    The result is:
    - All positions AFTER boundary i in the SAME segment get +1 gradient
    - All positions in OTHER segments also get shifted, but they're masked out

    Net effect: increasing boundary_i adds more weight to positions after i
    in the numerator AND increases the denominator count.

    Since pooled = sum/count, this is like adding the average of post-boundary
    positions to an already-computed average. This ALWAYS changes the pooled
    value, and the direction of loss change depends on whether those positions'
    hidden states help or hurt. But...

    The cumsum structure means the gradient has NO way to say "add a boundary HERE
    to help". It can only say "all boundaries get a uniform push".
    """
    print("\n" + "=" * 70)
    print("CUMSUM GRADIENT ANALYSIS")
    print("=" * 70)

    T = 8

    # Create differentiable boundaries
    probs = torch.tensor([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                         requires_grad=True)
    hard = (probs > 0.5).float()
    boundaries = (hard - probs.detach() + probs).unsqueeze(0)

    # cumsum
    hh1 = boundaries.cumsum(1) - boundaries

    print(f"  boundaries:    {boundaries[0].tolist()}")
    print(f"  cumsum:        {boundaries.cumsum(1)[0].tolist()}")
    print(f"  hh1 (seg_ids): {hh1[0].tolist()}")

    # Check gradient of hh1 at each position w.r.t. prob at position 3
    print(f"\n  d(hh1[pos])/d(prob[3]):")
    for pos in range(T):
        hh1[0, pos].backward(retain_graph=True)
        g = probs.grad[3].item()
        probs.grad.zero_()
        print(f"    pos {pos}: {g:+.4f}")

    print(f"\n  Key observation: d(hh1)/d(prob) is +1 for ALL positions after the boundary")
    print(f"  This means the cumsum gradient treats ALL downstream positions uniformly.")
    print(f"  There's no position-specific information - just 'push everything after me'.")


if __name__ == "__main__":
    trace_single_boundary_gradient()
    trace_pooling_gradient_mechanism()
    test_soft_mask_gradient_bias()
    test_gradient_components()
    test_cumsum_gradient_direction()
    # analyze_gradient_through_cumsum()  # uncomment for detailed foo gradients
