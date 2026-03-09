#!/usr/bin/env python3
"""
Test whether BoundaryPredictor3 boundary-decision parameters receive gradients
from a downstream loss (simulating ASR loss), independent of the binomial loss.

When running pretrain.py with --boundary_predictor_loss_weight=0.0, we see
zero gradients on boundary params. This test isolates the gradient flow to
determine if the issue is in BP3's architecture or elsewhere.
"""

import torch
import torch.nn as nn

# Silence debug prints during test
import flags
flags.PRINT_FLOW = False
flags.PRINT_DATA = False
flags.PRINT_BP_LOSS_CHECKS = False
flags.PRINT_NAN_INF = False

from BoundaryPredictor3 import BoundaryPredictor3


def classify_param(name):
    """Classify a BP3 parameter as 'boundary' or 'pooling'."""
    if any(k in name for k in ['boundary_mlp', 'q_proj', 'k_proj', 'similarity_bias']):
        return 'boundary'
    elif any(k in name for k in ['pool_', 'learned_query']):
        return 'pooling'
    return 'other'


def test_gradient_flow():
    """
    Pass synthetic data through BP3, compute a downstream loss on the pooled
    output (no binomial loss), and check which parameters got gradients.
    """
    torch.manual_seed(42)

    input_dim = 64
    batch_size = 4
    seq_len = 40
    prior = 0.2  # ~5x compression

    bp = BoundaryPredictor3(input_dim=input_dim, prior=prior, temp=0.5)
    bp.train()

    # Synthetic input (requires_grad=False — we only care about BP param grads)
    hidden = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.ones(batch_size)  # full-length sequences
    target_counts = torch.full((batch_size,), seq_len * prior)

    # Forward pass
    (pooled, bp_loss, num_boundaries, total_positions,
     shortened_lengths, boundary_cv, boundary_adjacent_pct,
     bp_probs, bp_num_boundaries_per_sample,
     bp_total_positions_per_sample) = bp(
        hidden=hidden,
        lengths=lengths,
        target_boundary_counts=target_counts,
    )

    print(f"Input shape:  {hidden.shape}")
    print(f"Pooled shape: {pooled.shape}")
    print(f"Boundaries:   {num_boundaries:.0f} / {total_positions:.0f} "
          f"= {num_boundaries/max(total_positions,1):.3f}")
    print(f"BP loss:      {bp_loss.item():.6f}")
    print()

    # =========================================================
    # Test 1: Gradient from downstream loss only (no BP loss)
    # =========================================================
    print("=" * 60)
    print("TEST 1: Downstream loss only (simulates boundary_predictor_loss_weight=0)")
    print("=" * 60)

    bp.zero_grad()
    downstream_loss = pooled.sum()  # simple surrogate for ASR loss
    downstream_loss.backward()

    print(f"\n{'Parameter':<40} {'Type':<10} {'Grad Norm':>12} {'Has Grad':>10}")
    print("-" * 75)

    boundary_grads = []
    pooling_grads = []
    for name, param in bp.named_parameters():
        ptype = classify_param(name)
        if param.grad is not None:
            gnorm = param.grad.norm().item()
            has_grad = gnorm > 0
        else:
            gnorm = 0.0
            has_grad = False
        print(f"{name:<40} {ptype:<10} {gnorm:>12.2e} {'YES' if has_grad else 'NO':>10}")
        if ptype == 'boundary':
            boundary_grads.append((name, gnorm))
        elif ptype == 'pooling':
            pooling_grads.append((name, gnorm))

    bd_total = sum(g for _, g in boundary_grads)
    pl_total = sum(g for _, g in pooling_grads)
    print(f"\nBoundary param total grad norm: {bd_total:.2e}")
    print(f"Pooling param total grad norm:  {pl_total:.2e}")

    if bd_total == 0:
        print("\n*** FAIL: Boundary params got ZERO gradient from downstream loss ***")
        print("    This confirms the gradient-flow bug.")
    else:
        print("\n*** PASS: Boundary params got nonzero gradient from downstream loss ***")

    # =========================================================
    # Test 2: Gradient from BP loss only (for comparison)
    # =========================================================
    print()
    print("=" * 60)
    print("TEST 2: BP (binomial) loss only (fresh forward pass)")
    print("=" * 60)

    bp.zero_grad()
    hidden_t2 = torch.randn(batch_size, seq_len, input_dim)
    (pooled_t2, bp_loss_t2, *_rest) = bp(
        hidden=hidden_t2, lengths=lengths,
        target_boundary_counts=target_counts,
    )
    bp_loss_t2.backward()

    print(f"\n{'Parameter':<40} {'Type':<10} {'Grad Norm':>12}")
    print("-" * 65)

    for name, param in bp.named_parameters():
        ptype = classify_param(name)
        gnorm = param.grad.norm().item() if param.grad is not None else 0.0
        print(f"{name:<40} {ptype:<10} {gnorm:>12.2e}")

    # =========================================================
    # Test 3: Trace gradient flow through intermediate tensors
    # =========================================================
    print()
    print("=" * 60)
    print("TEST 3: Intermediate tensor gradient check")
    print("=" * 60)

    # Re-run forward with intermediate tensors retained for gradient checking
    bp.zero_grad()

    hidden2 = torch.randn(batch_size, seq_len, input_dim)
    hidden2.requires_grad_(True)
    lengths2 = torch.ones(batch_size)
    target_counts2 = torch.full((batch_size,), seq_len * prior)

    # Hook into internals to capture intermediates
    intermediates = {}

    def make_hook(name):
        def hook(grad):
            intermediates[name] = grad.norm().item()
        return hook

    # Manually trace through forward to attach hooks
    hidden_dropped = bp.dropout(hidden2)
    q_input = hidden_dropped[:, :-1]
    k_input = hidden_dropped[:, 1:]

    q_normed = torch.nn.functional.normalize(q_input, dim=-1, eps=1e-8)
    k_normed = torch.nn.functional.normalize(k_input, dim=-1, eps=1e-8)

    q_mlp_out = bp.boundary_mlp(q_normed)
    q_hidden = bp.q_proj_layer(torch.nn.functional.normalize(
        q_mlp_out + q_normed, dim=-1, eps=1e-8))

    k_mlp_out = bp.boundary_mlp(k_normed)
    k_hidden = bp.k_proj_layer(torch.nn.functional.normalize(
        k_mlp_out + k_normed, dim=-1, eps=1e-8))

    cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)
    cos_sim.register_hook(make_hook("cos_sim"))

    probs = torch.clamp((1.0 - (cos_sim + bp.similarity_bias)) * 0.5, min=0.0, max=1.0)
    probs = torch.nn.functional.pad(probs, (0, 1), value=0.0)
    probs.register_hook(make_hook("probs"))

    bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
        temperature=bp.temp, probs=probs)
    soft_boundaries = bernoulli.rsample()
    soft_boundaries.register_hook(make_hook("soft_boundaries"))

    hard_samples = (soft_boundaries > 0.5).float()

    # Mask based on lengths
    boundary_seq_len = soft_boundaries.shape[1]
    actual_lens = (lengths2 * (boundary_seq_len + 1)).long()
    valid_lens = torch.clamp(actual_lens - 1, min=0, max=boundary_seq_len)
    pos_idx = torch.arange(boundary_seq_len).unsqueeze(0)
    valid_mask = (pos_idx < valid_lens.unsqueeze(1)).float()

    soft_boundaries = soft_boundaries * valid_mask
    hard_samples = hard_samples * valid_mask

    needs_boundary_mask = valid_lens < boundary_seq_len
    if needs_boundary_mask.any():
        batch_idx = torch.arange(batch_size)[needs_boundary_mask]
        first_padding_idx = valid_lens[needs_boundary_mask]
        soft_boundaries[batch_idx, first_padding_idx] = 1.0
        hard_samples[batch_idx, first_padding_idx] = 1.0

    # STE
    hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries
    hard_boundaries.register_hook(make_hook("hard_boundaries"))

    # Ensure at least one boundary
    no_bd = (hard_boundaries.sum(dim=1) == 0)
    if no_bd.any():
        indices = no_bd.nonzero(as_tuple=True)[0]
        vl = torch.clamp(actual_lens[indices] - 1, min=0, max=boundary_seq_len)
        bi = torch.clamp(vl, min=0, max=boundary_seq_len - 1)
        hard_boundaries[indices, bi] = 1.0

    # Attention pooling
    pooled2 = bp._attention_pooling(hard_boundaries, hidden2, lengths2)
    pooled2.register_hook(make_hook("pooled"))

    # Downstream loss
    downstream_loss2 = pooled2.sum()
    downstream_loss2.backward()

    print(f"\n{'Tensor':<25} {'Grad Norm':>12}")
    print("-" * 40)
    for name in ["pooled", "hard_boundaries", "soft_boundaries", "probs", "cos_sim"]:
        gnorm = intermediates.get(name, "NOT REGISTERED")
        if isinstance(gnorm, float):
            print(f"{name:<25} {gnorm:>12.2e}")
        else:
            print(f"{name:<25} {gnorm:>12}")

    # Check input hidden grad too
    if hidden2.grad is not None:
        print(f"{'hidden (input)':<25} {hidden2.grad.norm().item():>12.2e}")
    else:
        print(f"{'hidden (input)':<25} {'None':>12}")

    # Boundary param grads from this manual trace
    bd_norm = 0.0
    for name, param in bp.named_parameters():
        if classify_param(name) == 'boundary' and param.grad is not None:
            bd_norm += param.grad.norm().item()

    print(f"\nBoundary param total grad norm (manual trace): {bd_norm:.2e}")
    if bd_norm == 0:
        print("*** FAIL: Still zero. Gradient is blocked somewhere. ***")
    else:
        print("*** PASS: Nonzero gradient through manual trace. ***")

    # =========================================================
    # Test 4: Check the specific soft_segment_mask gradient path
    # =========================================================
    print()
    print("=" * 60)
    print("TEST 4: Soft segment mask gradient path (isolated)")
    print("=" * 60)

    # Create a simple case: boundaries = [1, 0, 1, 0] for 1 batch item
    boundaries = torch.tensor([[1.0, 0.0, 1.0, 0.0]], requires_grad=True)
    from BoundaryPredictor3 import common_cumsum
    foo = common_cumsum(boundaries)

    if foo is not None:
        print(f"foo shape: {foo.shape}")
        print(f"foo values:\n{foo[0]}")

        # Build soft mask same way as BP3
        out_of_segment = (foo.detach() != 0)
        soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
        print(f"\nsoft_mask values:\n{soft_mask[0]}")

        # Compute gradient of soft_mask.sum() w.r.t. boundaries
        loss_mask = soft_mask.sum()
        loss_mask.backward()

        print(f"\nd(soft_mask.sum())/d(boundaries) = {boundaries.grad}")
        if boundaries.grad is not None and boundaries.grad.abs().sum() > 0:
            print("PASS: soft_mask carries gradient w.r.t. boundaries")
        else:
            print("FAIL: soft_mask does NOT carry gradient w.r.t. boundaries")
    else:
        print("FAIL: common_cumsum returned None (no boundaries)")


if __name__ == "__main__":
    test_gradient_flow()
