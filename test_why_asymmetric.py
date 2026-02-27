#!/usr/bin/env python3
"""
WHY can the downstream loss reduce boundaries but never increase them?

The gradient path is:
  downstream_loss → pooled → soft_mask/log_bias → foo → cumsum → boundaries → STE → probs → MLP

Hypothesis: The continuous gradient (STE) can only encode position-reweighting
within EXISTING segments. It cannot represent the benefit of CREATING a new segment,
because segment creation is a discrete structural change (n_segments jumps by 1,
a new column appears in foo, a new pooled vector is produced).

- Removing a boundary: continuous effect (reduce soft_mask weight) is ALIGNED
  with discrete effect (merge two segments into one). The gradient correctly
  previews what will happen.

- Adding a boundary: continuous effect (increase soft_mask weight at some positions)
  is NOT ALIGNED with discrete effect (split one segment into two, producing a
  new pooled vector). The gradient cannot preview this benefit.

This test verifies this hypothesis directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def mean_pool(boundaries, hidden):
    B, T, D = hidden.shape
    foo = common(boundaries)
    if foo is None:
        return torch.zeros(B, 1, D, device=hidden.device)
    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
    counts = soft_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
    return seg_sum / counts.unsqueeze(-1)


def test_continuous_vs_discrete():
    """
    Show that the CONTINUOUS gradient around a boundary doesn't predict
    the DISCRETE effect of adding/removing that boundary.

    Setup: 8 positions, boundary at position 4.
    Measure: how does loss change when we continuously vary the boundary
    value vs when we discretely flip it 0↔1?
    """
    print("=" * 70)
    print("TEST: Continuous gradient vs discrete effect of a boundary")
    print("=" * 70)

    torch.manual_seed(42)
    T, D = 8, 4
    hidden = torch.randn(1, T, D)

    # Target: we want the pooled output to reconstruct the hidden states well.
    # More segments = better reconstruction. So the discrete benefit of adding
    # a boundary is clear.

    def reconstruction_loss(pooled, boundaries, hidden):
        """Loss that benefits from more segments."""
        seg_ids = (boundaries.cumsum(dim=1) - boundaries).long()
        seg_ids = seg_ids.clamp(0, max(pooled.shape[1] - 1, 0))
        if pooled.shape[1] == 0:
            return torch.tensor(10.0)
        recon = torch.gather(pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
        return (recon - hidden).pow(2).mean()

    print(f"\n  Hidden states (8 positions, 4 dims):")
    for i in range(T):
        print(f"    pos {i}: {hidden[0, i].tolist()}")

    # Case A: 1 boundary at end (1 segment = global average)
    boundaries_1seg = torch.zeros(1, T)
    boundaries_1seg[0, -1] = 1.0
    pooled_1seg = mean_pool(boundaries_1seg, hidden)
    loss_1seg = reconstruction_loss(pooled_1seg, boundaries_1seg, hidden)

    # Case B: 2 boundaries (2 segments: [0..3] and [4..7])
    boundaries_2seg = torch.zeros(1, T)
    boundaries_2seg[0, 3] = 1.0
    boundaries_2seg[0, -1] = 1.0
    pooled_2seg = mean_pool(boundaries_2seg, hidden)
    loss_2seg = reconstruction_loss(pooled_2seg, boundaries_2seg, hidden)

    print(f"\n  DISCRETE comparison:")
    print(f"    1 segment (no middle boundary): loss = {loss_1seg.item():.6f}")
    print(f"    2 segments (boundary at pos 3): loss = {loss_2seg.item():.6f}")
    print(f"    Discrete benefit of adding boundary: {loss_1seg.item() - loss_2seg.item():+.6f}")

    # Now: what does the CONTINUOUS gradient say?
    # Start with 1 segment, compute gradient w.r.t. a potential boundary at pos 3
    prob = torch.tensor(0.3, requires_grad=True)  # below 0.5 → no boundary
    boundaries = torch.zeros(1, T)
    hard = (prob > 0.5).float()  # = 0
    boundaries[0, 3] = hard - prob.detach() + prob  # STE: value=0, grad flows through prob
    boundaries[0, -1] = 1.0

    pooled = mean_pool(boundaries, hidden)
    loss = reconstruction_loss(pooled, boundaries, hidden)
    loss.backward()

    grad = prob.grad.item()
    print(f"\n  CONTINUOUS gradient (STE):")
    print(f"    prob = 0.3 (below threshold, no boundary active)")
    print(f"    d(loss)/d(prob) = {grad:+.8f}")
    print(f"    Gradient says: {'INCREASE prob (add boundary)' if grad < 0 else 'DECREASE prob (keep no boundary)'}")

    discrete_says = "ADD boundary (loss drops)"
    print(f"\n  MISMATCH:")
    print(f"    Discrete truth: {discrete_says} (loss drops by {loss_1seg.item() - loss_2seg.item():.4f})")
    print(f"    Continuous gradient: {'agrees' if grad < 0 else 'DISAGREES — pushes prob DOWN'}")

    if grad >= 0:
        print(f"\n  ** The gradient CANNOT SEE the benefit of adding the boundary! **")
        print(f"  The STE gradient only measures local reweighting effects,")
        print(f"  not the structural change of creating a new segment.")


def test_remove_vs_add_symmetry():
    """
    Show the asymmetry directly: gradient for removing an existing boundary
    IS aligned with the discrete effect, but gradient for adding one is NOT.
    """
    print("\n" + "=" * 70)
    print("TEST: Gradient alignment for ADD vs REMOVE boundary")
    print("=" * 70)

    torch.manual_seed(42)
    T, D = 12, 8
    hidden = torch.randn(1, T, D)

    results = []

    for boundary_pos in [3, 5, 7, 9]:
        # ---- REMOVE test: boundary exists, should we remove it? ----
        prob_remove = torch.tensor(0.7, requires_grad=True)  # above 0.5 → boundary active
        boundaries_r = torch.zeros(1, T)
        hard_r = (prob_remove > 0.5).float()  # = 1
        boundaries_r[0, boundary_pos] = hard_r - prob_remove.detach() + prob_remove
        boundaries_r[0, -1] = 1.0

        pooled_r = mean_pool(boundaries_r, hidden)
        loss_r = pooled_r.pow(2).mean()
        loss_r.backward()
        grad_remove = prob_remove.grad.item()

        # Discrete: loss with vs without the boundary
        b_with = torch.zeros(1, T)
        b_with[0, boundary_pos] = 1.0
        b_with[0, -1] = 1.0
        loss_with = mean_pool(b_with, hidden).pow(2).mean().item()

        b_without = torch.zeros(1, T)
        b_without[0, -1] = 1.0
        loss_without = mean_pool(b_without, hidden).pow(2).mean().item()

        discrete_remove = loss_without - loss_with  # negative = removing helps

        # ---- ADD test: boundary doesn't exist, should we add it? ----
        prob_add = torch.tensor(0.3, requires_grad=True)  # below 0.5 → no boundary
        boundaries_a = torch.zeros(1, T)
        hard_a = (prob_add > 0.5).float()  # = 0
        boundaries_a[0, boundary_pos] = hard_a - prob_add.detach() + prob_add
        boundaries_a[0, -1] = 1.0

        pooled_a = mean_pool(boundaries_a, hidden)
        loss_a = pooled_a.pow(2).mean()
        loss_a.backward()
        grad_add = prob_add.grad.item()

        discrete_add = loss_with - loss_without  # positive = adding hurts (for this loss)

        results.append((boundary_pos, grad_remove, discrete_remove, grad_add, discrete_add))

    print(f"\n  Loss = pooled.pow(2).mean()  (wants fewer boundaries)")
    print(f"\n  {'pos':>4}  {'grad(remove)':>13}  {'discrete(remove)':>17}  {'aligned?':>9}  "
          f"{'grad(add)':>13}  {'discrete(add)':>17}  {'aligned?':>9}")
    print(f"  {'-'*4}  {'-'*13}  {'-'*17}  {'-'*9}  {'-'*13}  {'-'*17}  {'-'*9}")

    for pos, gr, dr, ga, da in results:
        # For REMOVE: gradient > 0 means "decrease prob" = remove boundary
        # Discrete < 0 means removing helps (loss drops)
        remove_aligned = (gr > 0) == (dr < 0)

        # For ADD: gradient < 0 means "increase prob" = add boundary
        # Discrete > 0 means adding hurts
        # So gradient > 0 (don't add) should align with discrete > 0 (adding hurts)
        add_aligned = (ga > 0) == (da > 0)

        print(f"  {pos:>4}  {gr:>+13.6f}  {dr:>+17.6f}  {'YES' if remove_aligned else 'NO':>9}  "
              f"{ga:>+13.6f}  {da:>+17.6f}  {'YES' if add_aligned else 'NO':>9}")


def test_what_STE_gradient_actually_represents():
    """
    Decompose what the STE gradient through cumsum actually measures.

    When boundary_i = 0 (no boundary, prob below 0.5):
      The STE value is: 0 - prob.detach() + prob = prob (since hard=0)
      So boundaries[i] has value prob (small, ~0.3)

      foo = segment_indices - (cumsum(boundaries) - boundaries)
      At positions after i: cumsum increases by prob (not 1!)
      So foo changes by -prob at all downstream positions

      In the soft_mask: positions that are in-segment get soft_mask = 1 - foo
      The gradient chain: d(loss)/d(prob) via d(foo)/d(prob) = -1 for downstream positions

    The key: when the boundary is inactive (prob < 0.5), the STE still passes
    through a continuous value. But this continuous value represents a
    FRACTIONAL boundary — it shifts segment IDs by a fraction.

    A fractional boundary doesn't split a segment. It just shifts all downstream
    positions' segment IDs by a tiny amount, which changes soft_mask values.
    This is pure content reweighting, not structural.
    """
    print("\n" + "=" * 70)
    print("TEST: What does STE gradient measure for inactive boundaries?")
    print("=" * 70)

    T = 8

    # Inactive boundary at position 3 (prob=0.3, hard=0, STE value=0.3)
    prob = torch.tensor(0.3, requires_grad=True)
    boundaries = torch.zeros(1, T)
    hard = (prob > 0.5).float()  # 0
    boundaries[0, 3] = hard - prob.detach() + prob  # = 0 - 0.3 + 0.3 = 0.3 with grad

    # Wait, that's wrong. Let me reconsider.
    # hard = 0, soft_boundaries from RelaxedBernoulli would be some value.
    # STE: hard - soft.detach() + soft
    # In test.py, they don't use RelaxedBernoulli, they just do:
    # hard = (prob > 0.5).float() = 0
    # boundary_val = hard - prob.detach() + prob = 0 - 0.3 + 0.3 = 0.3
    # So boundary value is 0.3 (not 0!)

    # With RelaxedBernoulli at temp=0.5 and prob=0.3:
    # soft sample might be e.g. 0.15 (below 0.5) -> hard = 0
    # STE: 0 - 0.15 + 0.15 = 0.15 (value), but gradient flows through the 0.15

    boundaries[0, -1] = 1.0

    print(f"  boundaries[0] = {boundaries[0].tolist()}")
    print(f"  Note: position 3 has value {boundaries[0, 3].item():.1f} (fractional!)")

    # cumsum
    cs = boundaries.cumsum(1)
    hh1 = cs - boundaries
    print(f"  cumsum        = {cs[0].tolist()}")
    print(f"  hh1 (seg_ids) = {hh1[0].tolist()}")

    foo = common(boundaries)
    if foo is not None:
        print(f"\n  foo shape: {foo.shape} (n_segments = {foo.shape[2]})")
        for s in range(foo.shape[2]):
            print(f"  foo[:, seg={s}] = {foo[0, :, s].tolist()}")

        out_of_segment = (foo.detach() != 0)
        soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
        print(f"\n  soft_mask (detached out-of-segment zeroing):")
        for s in range(soft_mask.shape[2]):
            print(f"  soft_mask[:, seg={s}] = {[f'{v:.2f}' for v in soft_mask[0, :, s].tolist()]}")

        print(f"\n  Key insight: the soft_mask values at in-segment positions")
        print(f"  are NOT 1.0 — they're 1.0 + fractional offset from the")
        print(f"  inactive boundary. This is what the gradient 'sees'.")
        print(f"  It's a tiny perturbation to averaging weights, NOT a segment split.")


def test_n_segments_bottleneck():
    """
    The ultimate bottleneck: n_segments = int(boundaries.sum(-1).max().item())

    This line determines the SIZE of the foo matrix (and thus the number of
    pooled output vectors). It's completely non-differentiable.

    Even if the gradient successfully increases a boundary prob past 0.5,
    the NEXT forward pass will have n_segments+1, producing one more pooled
    vector. But the gradient from the CURRENT pass couldn't have anticipated
    this because n_segments was fixed.

    This is a chicken-and-egg problem:
    - To know the benefit of adding a segment, you need to see the loss with the extra segment
    - To see the loss with the extra segment, you need to add the segment first
    - The continuous gradient approximation doesn't add segments

    For REMOVING a segment, this is less of a problem because:
    - The segment that's about to be removed is already small/weak
    - Its contribution to the loss is visible in the current pass
    - The gradient correctly says "this segment's contribution is unhelpful"
    """
    print("\n" + "=" * 70)
    print("TEST: The n_segments bottleneck")
    print("=" * 70)

    torch.manual_seed(42)
    T, D = 20, 8
    hidden = torch.randn(1, T, D)

    # Start with 2 boundaries (3 segments)
    # What does the gradient say about adding a 3rd boundary?
    prob_new = torch.tensor(0.3, requires_grad=True)

    boundaries = torch.zeros(1, T)
    boundaries[0, 6] = 1.0   # existing boundary
    hard = (prob_new > 0.5).float()
    boundaries[0, 13] = hard - prob_new.detach() + prob_new  # potential new boundary
    boundaries[0, -1] = 1.0  # sentinel

    foo = common(boundaries)
    n_seg = foo.shape[2] if foo is not None else 0

    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
    counts = soft_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
    pooled = seg_sum / counts.unsqueeze(-1)

    print(f"\n  Current: {n_seg} segments (boundaries at 6 and {T-1})")
    print(f"  Potential new boundary at position 13 (prob=0.3, inactive)")
    print(f"  pooled shape: {pooled.shape}")

    # Reconstruction loss
    seg_ids = (boundaries.cumsum(dim=1) - boundaries).long().clamp(0, n_seg - 1)
    recon = torch.gather(pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
    loss_current = (recon - hidden).pow(2).mean()
    loss_current.backward()
    grad_continuous = prob_new.grad.item()

    # Now actually add the boundary and measure true effect
    boundaries_3seg = torch.zeros(1, T)
    boundaries_3seg[0, 6] = 1.0
    boundaries_3seg[0, 13] = 1.0
    boundaries_3seg[0, -1] = 1.0

    foo_3 = common(boundaries_3seg)
    out_3 = (foo_3.detach() != 0)
    mask_3 = torch.where(out_3, torch.zeros_like(foo_3), 1.0 - foo_3)
    counts_3 = mask_3.sum(dim=1).clamp(min=1e-8)
    pooled_3 = torch.bmm(mask_3.transpose(1, 2), hidden) / counts_3.unsqueeze(-1)

    seg_ids_3 = (boundaries_3seg.cumsum(dim=1) - boundaries_3seg).long().clamp(0, pooled_3.shape[1] - 1)
    recon_3 = torch.gather(pooled_3, 1, seg_ids_3.unsqueeze(-1).expand(-1, -1, D))
    loss_3seg = (recon_3 - hidden).pow(2).mean()

    print(f"\n  Loss with 2 segments: {loss_current.item():.6f}")
    print(f"  Loss with 3 segments: {loss_3seg.item():.6f}")
    print(f"  Discrete benefit:     {loss_current.item() - loss_3seg.item():+.6f}")
    print(f"  Continuous gradient:  {grad_continuous:+.8f}")
    print(f"  Gradient says:        {'add boundary' if grad_continuous < 0 else 'do NOT add boundary'}")

    if grad_continuous >= 0 and loss_3seg.item() < loss_current.item():
        print(f"\n  ** CONFIRMED: Gradient says 'no' but adding helps! **")
        print(f"  The gradient measures in-segment reweighting effects")
        print(f"  (how pooled[seg_1] changes if we shift weights)")
        print(f"  It CANNOT measure the benefit of creating pooled[seg_2]")
        print(f"  because that output vector doesn't exist in this forward pass.")


def test_many_positions():
    """
    Run the add-vs-remove alignment test across many random hidden states
    and boundary positions to get statistics.
    """
    print("\n" + "=" * 70)
    print("TEST: Alignment statistics across many random configurations")
    print("=" * 70)

    T, D = 20, 16
    add_aligned = 0
    add_misaligned = 0
    remove_aligned = 0
    remove_misaligned = 0

    for trial in range(200):
        torch.manual_seed(trial)
        hidden = torch.randn(1, T, D)
        bp = trial % (T - 2) + 1  # boundary position

        # ADD test: no boundary, should adding one help?
        prob = torch.tensor(0.3, requires_grad=True)
        boundaries = torch.zeros(1, T)
        hard = (prob > 0.5).float()
        boundaries[0, bp] = hard - prob.detach() + prob
        boundaries[0, -1] = 1.0

        pooled = mean_pool(boundaries, hidden)
        # Reconstruction loss (wants more segments)
        seg_ids = (boundaries.cumsum(1) - boundaries).long().clamp(0, max(pooled.shape[1]-1, 0))
        recon = torch.gather(pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
        loss = (recon - hidden).pow(2).mean()
        loss.backward()
        grad_add = prob.grad.item()

        # Discrete effect
        b_with = torch.zeros(1, T)
        b_with[0, bp] = 1.0
        b_with[0, -1] = 1.0
        p_with = mean_pool(b_with, hidden)
        si_with = (b_with.cumsum(1) - b_with).long().clamp(0, max(p_with.shape[1]-1, 0))
        r_with = torch.gather(p_with, 1, si_with.unsqueeze(-1).expand(-1, -1, D))
        loss_with = (r_with - hidden).pow(2).mean().item()

        b_without = torch.zeros(1, T)
        b_without[0, -1] = 1.0
        p_without = mean_pool(b_without, hidden)
        si_without = (b_without.cumsum(1) - b_without).long().clamp(0, max(p_without.shape[1]-1, 0))
        r_without = torch.gather(p_without, 1, si_without.unsqueeze(-1).expand(-1, -1, D))
        loss_without = (r_without - hidden).pow(2).mean().item()

        # Adding helps if loss_with < loss_without
        adding_helps = loss_with < loss_without
        grad_says_add = grad_add < 0

        if adding_helps == grad_says_add:
            add_aligned += 1
        else:
            add_misaligned += 1

    print(f"\n  Reconstruction loss (benefits from more boundaries)")
    print(f"  200 trials: random hidden states, various boundary positions")
    print(f"\n  ADD boundary:")
    print(f"    Gradient agrees with discrete benefit: {add_aligned}/200 ({100*add_aligned/200:.0f}%)")
    print(f"    Gradient DISAGREES:                    {add_misaligned}/200 ({100*add_misaligned/200:.0f}%)")

    if add_misaligned > add_aligned:
        print(f"\n  ** The gradient is WRONG more often than right for adding boundaries! **")
        print(f"  ** This is why downstream loss cannot increase boundary count. **")
    elif add_aligned > add_misaligned * 2:
        print(f"\n  Gradient is mostly aligned — the issue may be elsewhere.")
    else:
        print(f"\n  Gradient is barely better than random for adding boundaries.")


if __name__ == "__main__":
    test_continuous_vs_discrete()
    test_remove_vs_add_symmetry()
    test_what_STE_gradient_actually_represents()
    test_n_segments_bottleneck()
    test_many_positions()
