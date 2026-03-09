#!/usr/bin/env python3
"""
Test whether BoundaryPredictor2's boundary-decision parameters receive
gradients from a downstream loss applied to the pooled output.

The gradient path in question:
  downstream_loss → pooled → _attention_pooling → segment_mask → foo
    → common() → hard_boundaries → STE → soft_boundaries → probs
    → cos_sim → boundary_mlp / q_proj / k_proj / similarity_bias

The concern: common() uses boolean comparisons (==) and _attention_pooling
uses (foo == 0).float() for segment_mask — both non-differentiable — which
may block gradient flow entirely.
"""

import torch
import torch.nn as nn

# Suppress diagnostic prints during test
import flags
flags.PRINT_FLOW = False
flags.PRINT_DATA = False
flags.PRINT_BP_LOSS_CHECKS = False
flags.PRINT_NAN_INF = False

from BoundaryPredictor import BoundaryPredictor2


def make_bp2(D=32, prior=0.2, temp=0.5):
    """Create a small BP2 for testing."""
    return BoundaryPredictor2(input_dim=D, prior=prior, temp=temp)


def make_inputs(B=2, T=20, D=32):
    """Create synthetic inputs that require grad so we can trace the graph."""
    hidden = torch.randn(B, T, D, requires_grad=True)
    lengths = torch.tensor([1.0, 0.8])  # one full-length, one shorter
    target_counts = torch.tensor([4.0, 3.0])
    return hidden, lengths, target_counts


# ---------------------------------------------------------------------------
# Test 1: Which parameters get gradients from downstream-only loss?
# ---------------------------------------------------------------------------
def test_downstream_gradient_reaches_boundary_params():
    """
    Apply a loss ONLY to the pooled output (simulating CTC/seq2seq loss).
    Check whether boundary-decision parameters receive nonzero gradients.
    """
    print("=" * 70)
    print("TEST 1: Downstream loss → boundary-decision parameters?")
    print("=" * 70)

    D = 32
    bp = make_bp2(D=D)
    bp.train()

    hidden, lengths, target_counts = make_inputs(D=D)

    pooled, bp_loss, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)

    # Downstream-only loss: ignore bp_loss, just use pooled output
    downstream_loss = pooled.pow(2).mean()
    downstream_loss.backward()

    # Categorize parameters
    boundary_params = {
        "boundary_mlp.0.weight": bp.boundary_mlp[0].weight,
        "boundary_mlp.0.bias":   bp.boundary_mlp[0].bias,
        "boundary_mlp.2.weight": bp.boundary_mlp[2].weight,
        "boundary_mlp.2.bias":   bp.boundary_mlp[2].bias,
        "q_proj_layer.weight":   bp.q_proj_layer.weight,
        "k_proj_layer.weight":   bp.k_proj_layer.weight,
        "similarity_bias":       bp.similarity_bias,
    }

    pooling_params = {
        "pool_key.weight":       bp.pool_key.weight,
        "pool_value.weight":     bp.pool_value.weight,
        "pool_output.weight":    bp.pool_output.weight,
        "learned_query":         bp.learned_query,
    }

    print("\n  Boundary-decision parameters (should these get gradient?):")
    any_boundary_grad = False
    for name, param in boundary_params.items():
        has_grad = param.grad is not None and param.grad.abs().max().item() > 0
        mag = param.grad.abs().max().item() if has_grad else 0.0
        status = "GRADIENT" if has_grad else "zero/None"
        print(f"    {name:30s}  {status:10s}  max|grad|={mag:.2e}")
        if has_grad:
            any_boundary_grad = True

    print("\n  Pooling parameters (should get gradient):")
    any_pooling_grad = False
    for name, param in pooling_params.items():
        has_grad = param.grad is not None and param.grad.abs().max().item() > 0
        mag = param.grad.abs().max().item() if has_grad else 0.0
        status = "GRADIENT" if has_grad else "zero/None"
        print(f"    {name:30s}  {status:10s}  max|grad|={mag:.2e}")
        if has_grad:
            any_pooling_grad = True

    print()
    if any_boundary_grad:
        print("  RESULT: Downstream loss DOES reach boundary-decision parameters.")
    else:
        print("  RESULT: Downstream loss does NOT reach boundary-decision parameters.")
        print("  The gradient is blocked by non-differentiable ops in common()/_attention_pooling.")

    if any_pooling_grad:
        print("  (Pooling parameters DO get gradient — the loss is connected to the graph.)")
    else:
        print("  WARNING: Even pooling parameters got no gradient — something is wrong.")

    return any_boundary_grad, any_pooling_grad


# ---------------------------------------------------------------------------
# Test 2: Do boundary params get gradient from boundary loss?
# ---------------------------------------------------------------------------
def test_boundary_loss_gradient():
    """
    Sanity check: the boundary loss SHOULD provide gradients to boundary params.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Boundary loss → boundary-decision parameters? (sanity check)")
    print("=" * 70)

    D = 32
    bp = make_bp2(D=D)
    bp.train()

    hidden, lengths, target_counts = make_inputs(D=D)

    pooled, bp_loss, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)

    # Use ONLY the boundary loss
    bp_loss.backward()

    boundary_params = {
        "boundary_mlp.0.weight": bp.boundary_mlp[0].weight,
        "boundary_mlp.0.bias":   bp.boundary_mlp[0].bias,
        "boundary_mlp.2.weight": bp.boundary_mlp[2].weight,
        "boundary_mlp.2.bias":   bp.boundary_mlp[2].bias,
        "q_proj_layer.weight":   bp.q_proj_layer.weight,
        "k_proj_layer.weight":   bp.k_proj_layer.weight,
        "similarity_bias":       bp.similarity_bias,
    }

    print("\n  Boundary-decision parameters:")
    any_grad = False
    for name, param in boundary_params.items():
        has_grad = param.grad is not None and param.grad.abs().max().item() > 0
        mag = param.grad.abs().max().item() if has_grad else 0.0
        status = "GRADIENT" if has_grad else "zero/None"
        print(f"    {name:30s}  {status:10s}  max|grad|={mag:.2e}")
        if has_grad:
            any_grad = True

    print()
    if any_grad:
        print("  RESULT: Boundary loss DOES reach boundary-decision params (expected).")
    else:
        print("  RESULT: Boundary loss does NOT reach boundary-decision params!")
        print("  This would mean boundary params are only trained by boundary loss,")
        print("  and even THAT doesn't work — a serious bug.")

    return any_grad


# ---------------------------------------------------------------------------
# Test 3: Combined loss — does downstream component contribute?
# ---------------------------------------------------------------------------
def test_combined_loss_gradient_attribution():
    """
    With the combined loss (downstream + boundary), measure how much of the
    gradient on boundary params comes from downstream vs boundary loss.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Gradient attribution — downstream vs boundary loss")
    print("=" * 70)

    D = 32
    bp = make_bp2(D=D)
    bp.train()

    hidden, lengths, target_counts = make_inputs(D=D)

    # Run 1: downstream-only gradient
    pooled1, bp_loss1, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)
    downstream_loss1 = pooled1.pow(2).mean()
    downstream_loss1.backward()

    downstream_grads = {}
    for name, param in bp.named_parameters():
        if param.grad is not None:
            downstream_grads[name] = param.grad.clone()
        else:
            downstream_grads[name] = torch.zeros_like(param)

    bp.zero_grad()

    # Run 2: boundary-only gradient
    pooled2, bp_loss2, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)
    bp_loss2.backward()

    boundary_grads = {}
    for name, param in bp.named_parameters():
        if param.grad is not None:
            boundary_grads[name] = param.grad.clone()
        else:
            boundary_grads[name] = torch.zeros_like(param)

    # Compare
    print(f"\n  {'Parameter':40s}  {'|downstream|':>14s}  {'|boundary|':>14s}  {'ratio':>8s}")
    print(f"  {'-'*40}  {'-'*14}  {'-'*14}  {'-'*8}")

    for name in sorted(downstream_grads.keys()):
        d_norm = downstream_grads[name].norm().item()
        b_norm = boundary_grads[name].norm().item()
        if b_norm > 0 and d_norm > 0:
            ratio = f"{d_norm/b_norm:.4f}"
        elif d_norm > 0:
            ratio = "inf"
        elif b_norm > 0:
            ratio = "0"
        else:
            ratio = "n/a"
        print(f"  {name:40s}  {d_norm:14.2e}  {b_norm:14.2e}  {ratio:>8s}")


# ---------------------------------------------------------------------------
# Test 4: Trace the exact gradient-blocking point
# ---------------------------------------------------------------------------
def test_gradient_blocking_point():
    """
    Identify exactly where in the computation graph the gradient is blocked.
    Trace each intermediate tensor in _attention_pooling.
    """
    print("\n" + "=" * 70)
    print("TEST 4: Where exactly is the gradient blocked?")
    print("=" * 70)

    from utils import common

    D = 32
    B, T = 2, 20

    torch.manual_seed(42)
    hidden = torch.randn(B, T, D)
    lengths = torch.tensor([1.0, 0.8])

    # Create boundaries with STE (mimicking BP2's forward)
    soft_boundaries = torch.rand(B, T) * 0.5 + 0.25  # uniform in [0.25, 0.75]
    soft_boundaries.requires_grad_(True)
    hard_samples = (soft_boundaries > 0.5).float()
    hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

    # Force some boundaries to be set so we have segments
    hard_boundaries = hard_boundaries.clone()
    hard_boundaries[:, -1] = 1.0
    hard_boundaries[:, T//3] = 1.0
    hard_boundaries[:, 2*T//3] = 1.0

    # Step through common()
    foo = common(hard_boundaries)
    foo_has_grad = foo is not None and foo.requires_grad
    print(f"\n  1. foo = common(hard_boundaries)")
    print(f"     foo.requires_grad = {foo_has_grad}")

    if foo is not None:
        # segment_mask: the critical step
        segment_mask = (foo == 0).float()
        mask_has_grad = segment_mask.requires_grad
        print(f"\n  2. segment_mask = (foo == 0).float()")
        print(f"     segment_mask.requires_grad = {mask_has_grad}")
        if not mask_has_grad:
            print(f"     *** BLOCKED HERE: boolean comparison (== 0) is non-differentiable ***")

        # Even if foo has grad_fn, the == kills it
        if foo_has_grad:
            print(f"\n     Note: foo DOES have gradient, but (foo == 0) discards it.")
            print(f"     foo.grad_fn = {foo.grad_fn}")

    # Also check: does common()'s internal logic preserve gradient?
    print(f"\n  Tracing inside common():")
    boundaries_clone = hard_boundaries.clone()
    segment_ids = boundaries_clone.cumsum(1) - boundaries_clone
    print(f"  3. segment_ids = cumsum(boundaries) - boundaries")
    print(f"     segment_ids.requires_grad = {segment_ids.requires_grad}")

    in_segment = segment_ids.unsqueeze(-1) == torch.arange(3)
    print(f"  4. in_segment = (segment_ids == segment_indices)")
    print(f"     in_segment.requires_grad = {in_segment.requires_grad}")
    print(f"     *** Second block: segment_ids == indices is non-differentiable ***")


# ---------------------------------------------------------------------------
# Test 5: Across many random seeds — is it ever nonzero?
# ---------------------------------------------------------------------------
def test_many_seeds():
    """
    BP2 uses stochastic sampling (RelaxedBernoulli). Maybe some seeds
    produce a configuration where gradient leaks through. Test many seeds.
    """
    print("\n" + "=" * 70)
    print("TEST 5: Check across 50 random seeds")
    print("=" * 70)

    D = 32
    seeds_with_grad = 0
    total_seeds = 50

    for seed in range(total_seeds):
        torch.manual_seed(seed)

        bp = make_bp2(D=D)
        bp.train()

        hidden = torch.randn(2, 20, D, requires_grad=True)
        lengths = torch.tensor([1.0, 0.8])
        target_counts = torch.tensor([4.0, 3.0])

        pooled, bp_loss, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)

        downstream_loss = pooled.pow(2).mean()
        downstream_loss.backward()

        # Check boundary_mlp
        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 1e-12
            for p in bp.boundary_mlp.parameters()
        )
        if has_grad:
            seeds_with_grad += 1

        bp.zero_grad()

    print(f"\n  Seeds where boundary_mlp got downstream gradient: "
          f"{seeds_with_grad}/{total_seeds}")

    if seeds_with_grad == 0:
        print("  CONFIRMED: Downstream loss NEVER reaches boundary params.")
        print("  Boundary placement is trained ONLY by boundary loss.")
    else:
        print(f"  Gradient leaked through in {seeds_with_grad} cases — investigate.")


# ---------------------------------------------------------------------------
# Test 6: What if we replaced the hard segment_mask with a soft version?
# ---------------------------------------------------------------------------
def test_soft_segment_mask_gradient():
    """
    Hypothetical: if _attention_pooling used a SOFT segment mask derived
    from foo (preserving gradient), would the downstream gradient reach
    boundary params?

    This shows the gradient path COULD work if the mask were differentiable.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Hypothetical — soft segment mask enables gradient?")
    print("=" * 70)

    from utils import common

    D = 32
    B, T = 2, 20

    torch.manual_seed(42)
    bp = make_bp2(D=D)
    bp.train()

    hidden = torch.randn(B, T, D)
    lengths = torch.tensor([1.0, 0.8])
    target_counts = torch.tensor([4.0, 3.0])

    # Run forward to get boundaries
    hidden_dropped = bp.dropout(hidden)
    q_input = hidden_dropped[:, :-1]
    k_input = hidden_dropped[:, 1:]

    q_normed = nn.functional.normalize(q_input, dim=-1, eps=1e-8)
    k_normed = nn.functional.normalize(k_input, dim=-1, eps=1e-8)

    q_mlp_out = bp.boundary_mlp(q_normed)
    q_hidden = bp.q_proj_layer(nn.functional.normalize(q_mlp_out + q_normed, dim=-1, eps=1e-8))
    k_mlp_out = bp.boundary_mlp(k_normed)
    k_hidden = bp.k_proj_layer(nn.functional.normalize(k_mlp_out + k_normed, dim=-1, eps=1e-8))

    cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)
    probs = torch.clamp((1.0 - (cos_sim + bp.similarity_bias)) * 0.5, min=0.0, max=1.0)
    probs = nn.functional.pad(probs, (0, 1), value=0.0)

    bernoulli = torch.distributions.RelaxedBernoulli(temperature=0.5, probs=probs)
    soft_boundaries = bernoulli.rsample()
    hard_samples = (soft_boundaries > 0.5).float()
    hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

    # Force at least one boundary
    hard_boundaries[:, -1] = 1.0

    foo = common(hard_boundaries)
    if foo is None:
        print("  No segments — skip")
        return

    # SOFT mask: use sigmoid to approximate (foo == 0) differentiably
    # Large negative scale means sharp transition: sigmoid(-|foo|*scale) ≈ 1 when foo=0
    sharpness = 20.0
    soft_segment_mask = torch.sigmoid(-foo.abs() * sharpness)

    # Use this soft mask in a simplified attention pooling
    max_segments = foo.size(2)
    hidden_normed = bp.pool_layernorm(hidden)
    keys = bp.pool_key(hidden_normed)
    values = bp.pool_value(hidden_normed)
    queries = bp.learned_query.unsqueeze(0).unsqueeze(0).expand(B, max_segments, -1)

    # Simple dot-product attention (single-head for simplicity)
    attn_scores = torch.bmm(queries, keys.transpose(1, 2))  # (B, S, T)

    # Apply SOFT segment mask (this is the key difference)
    attn_scores = attn_scores + (soft_segment_mask.permute(0, 2, 1) - 1) * 1e4

    attn_weights = nn.functional.softmax(attn_scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
    pooled = torch.bmm(attn_weights, values)
    pooled = bp.pool_output(pooled)

    # Downstream loss
    loss = pooled.pow(2).mean()
    loss.backward()

    print("\n  With SOFT segment mask (sigmoid approximation):")
    for name, param in [
        ("boundary_mlp.0.weight", bp.boundary_mlp[0].weight),
        ("similarity_bias", bp.similarity_bias),
        ("q_proj_layer.weight", bp.q_proj_layer.weight),
    ]:
        has_grad = param.grad is not None and param.grad.abs().max().item() > 1e-12
        mag = param.grad.abs().max().item() if has_grad else 0.0
        print(f"    {name:30s}  {'GRADIENT' if has_grad else 'zero/None':10s}  max|grad|={mag:.2e}")

    any_grad = any(
        p.grad is not None and p.grad.abs().max().item() > 1e-12
        for p in bp.boundary_mlp.parameters()
    )
    if any_grad:
        print("\n  CONFIRMED: A soft segment mask WOULD allow gradient to flow.")
        print("  The hard (foo == 0) mask is what blocks it.")


if __name__ == "__main__":
    got_downstream, got_pooling = test_downstream_gradient_reaches_boundary_params()
    got_boundary = test_boundary_loss_gradient()
    test_combined_loss_gradient_attribution()
    test_gradient_blocking_point()
    test_many_seeds()
    test_soft_segment_mask_gradient()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Downstream loss → pooling params:    {'YES' if got_pooling else 'NO'}")
    print(f"  Downstream loss → boundary params:   {'YES' if got_downstream else 'NO'}")
    print(f"  Boundary loss   → boundary params:   {'YES' if got_boundary else 'NO'}")
    print()
    if not got_downstream and got_boundary:
        print("  CONCLUSION: Boundary placement is trained ONLY by boundary loss.")
        print("  The downstream ASR loss (CTC/seq2seq) cannot influence WHERE")
        print("  boundaries are placed — only the attention pooling CONTENT.")
        print()
        print("  Root cause: segment_mask = (foo == 0).float() in _attention_pooling")
        print("  and in_segment = (segment_ids == indices) in common() are both")
        print("  non-differentiable boolean comparisons that block gradient flow.")
