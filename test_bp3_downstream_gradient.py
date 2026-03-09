#!/usr/bin/env python3
"""
Test that BoundaryPredictor3's boundary-decision parameters receive
gradients from a downstream loss applied to the pooled output.

Compares BP2 (gradient blocked) vs BP3 (gradient flows).
"""

import torch

# Suppress diagnostic prints during test
import flags
flags.PRINT_FLOW = False
flags.PRINT_DATA = False
flags.PRINT_BP_LOSS_CHECKS = False
flags.PRINT_NAN_INF = False

from BoundaryPredictor import BoundaryPredictor2
from BoundaryPredictor3 import BoundaryPredictor3


def make_inputs(B=2, T=20, D=32):
    hidden = torch.randn(B, T, D, requires_grad=True)
    lengths = torch.tensor([1.0, 0.8])
    target_counts = torch.tensor([4.0, 3.0])
    return hidden, lengths, target_counts


def check_boundary_grads(bp, label):
    """Return max|grad| for boundary-decision params and pooling params."""
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

    print(f"\n  {label} — Boundary-decision parameters:")
    any_boundary_grad = False
    for name, param in boundary_params.items():
        has_grad = param.grad is not None and param.grad.abs().max().item() > 1e-12
        mag = param.grad.abs().max().item() if has_grad else 0.0
        status = "GRADIENT" if has_grad else "zero/None"
        print(f"    {name:30s}  {status:10s}  max|grad|={mag:.2e}")
        if has_grad:
            any_boundary_grad = True

    print(f"\n  {label} — Pooling parameters:")
    any_pooling_grad = False
    for name, param in pooling_params.items():
        has_grad = param.grad is not None and param.grad.abs().max().item() > 1e-12
        mag = param.grad.abs().max().item() if has_grad else 0.0
        status = "GRADIENT" if has_grad else "zero/None"
        print(f"    {name:30s}  {status:10s}  max|grad|={mag:.2e}")
        if has_grad:
            any_pooling_grad = True

    return any_boundary_grad, any_pooling_grad


# ---------------------------------------------------------------------------
# Test 1: BP2 vs BP3 — downstream loss gradient to boundary params
# ---------------------------------------------------------------------------
def test_downstream_gradient_comparison():
    print("=" * 70)
    print("TEST 1: Downstream loss → boundary params (BP2 vs BP3)")
    print("=" * 70)

    D = 32
    torch.manual_seed(42)

    results = {}
    for cls, name in [(BoundaryPredictor2, "BP2"), (BoundaryPredictor3, "BP3")]:
        torch.manual_seed(42)
        bp = cls(input_dim=D, prior=0.2, temp=0.5)
        bp.train()
        hidden, lengths, target_counts = make_inputs(D=D)

        pooled, bp_loss, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)
        downstream_loss = pooled.pow(2).mean()
        downstream_loss.backward()

        got_boundary, got_pooling = check_boundary_grads(bp, name)
        results[name] = (got_boundary, got_pooling)

    print("\n" + "-" * 70)
    for name, (gb, gp) in results.items():
        print(f"  {name}: boundary params={'YES' if gb else 'NO':3s}  "
              f"pooling params={'YES' if gp else 'NO':3s}")

    return results


# ---------------------------------------------------------------------------
# Test 2: BP3 boundary loss still works
# ---------------------------------------------------------------------------
def test_bp3_boundary_loss():
    print("\n" + "=" * 70)
    print("TEST 2: BP3 boundary loss → boundary params (sanity check)")
    print("=" * 70)

    D = 32
    torch.manual_seed(42)
    bp = BoundaryPredictor3(input_dim=D, prior=0.2, temp=0.5)
    bp.train()
    hidden, lengths, target_counts = make_inputs(D=D)

    pooled, bp_loss, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)
    bp_loss.backward()

    got_boundary, _ = check_boundary_grads(bp, "BP3 boundary-loss")
    return got_boundary


# ---------------------------------------------------------------------------
# Test 3: Across many seeds — is BP3 gradient consistently nonzero?
# ---------------------------------------------------------------------------
def test_many_seeds():
    print("\n" + "=" * 70)
    print("TEST 3: BP3 downstream gradient across 50 random seeds")
    print("=" * 70)

    D = 32
    seeds_with_grad = 0
    total_seeds = 50

    for seed in range(total_seeds):
        torch.manual_seed(seed)
        bp = BoundaryPredictor3(input_dim=D, prior=0.2, temp=0.5)
        bp.train()

        hidden = torch.randn(2, 20, D, requires_grad=True)
        lengths = torch.tensor([1.0, 0.8])
        target_counts = torch.tensor([4.0, 3.0])

        pooled, bp_loss, *_ = bp(hidden, lengths, target_boundary_counts=target_counts)
        downstream_loss = pooled.pow(2).mean()
        downstream_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().max().item() > 1e-12
            for p in bp.boundary_mlp.parameters()
        )
        if has_grad:
            seeds_with_grad += 1

    print(f"\n  Seeds where boundary_mlp got downstream gradient: "
          f"{seeds_with_grad}/{total_seeds}")
    return seeds_with_grad


# ---------------------------------------------------------------------------
# Test 4: Forward output equivalence — BP2 and BP3 produce same values
# ---------------------------------------------------------------------------
def test_forward_equivalence():
    print("\n" + "=" * 70)
    print("TEST 4: BP2 vs BP3 forward output equivalence")
    print("=" * 70)

    D = 32
    torch.manual_seed(42)

    bp2 = BoundaryPredictor2(input_dim=D, prior=0.2, temp=0.5)
    bp3 = BoundaryPredictor3(input_dim=D, prior=0.2, temp=0.5)

    # Copy BP2 weights to BP3
    bp3.load_state_dict(bp2.state_dict())

    bp2.eval()
    bp3.eval()

    hidden = torch.randn(2, 20, D)
    lengths = torch.tensor([1.0, 0.8])
    target_counts = torch.tensor([4.0, 3.0])

    with torch.no_grad():
        pooled2, _, nb2, tp2, sl2, *_ = bp2(hidden, lengths, target_boundary_counts=target_counts)
        pooled3, _, nb3, tp3, sl3, *_ = bp3(hidden, lengths, target_boundary_counts=target_counts)

    if pooled2.shape == pooled3.shape:
        max_diff = (pooled2 - pooled3).abs().max().item()
        print(f"\n  pooled shape: BP2={pooled2.shape}, BP3={pooled3.shape}")
        print(f"  max |pooled2 - pooled3| = {max_diff:.2e}")
        print(f"  num_boundaries: BP2={nb2}, BP3={nb3}")
        print(f"  shortened_lengths: BP2={sl2.tolist()}, BP3={sl3.tolist()}")
        if max_diff < 1e-5:
            print(f"  MATCH: Forward outputs are equivalent.")
        else:
            print(f"  MISMATCH: Outputs differ (max_diff={max_diff:.2e})")
    else:
        print(f"\n  Shape mismatch: BP2={pooled2.shape}, BP3={pooled3.shape}")
        print(f"  (Expected in eval — same weights should give same boundaries)")

    return pooled2.shape == pooled3.shape


if __name__ == "__main__":
    results = test_downstream_gradient_comparison()
    got_bp3_boundary = test_bp3_boundary_loss()
    seeds_with_grad = test_many_seeds()
    equiv = test_forward_equivalence()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    bp2_boundary, bp2_pooling = results["BP2"]
    bp3_boundary, bp3_pooling = results["BP3"]
    print(f"  BP2: downstream → boundary params: {'YES' if bp2_boundary else 'NO'}")
    print(f"  BP3: downstream → boundary params: {'YES' if bp3_boundary else 'NO'}")
    print(f"  BP3: boundary loss → boundary params: {'YES' if got_bp3_boundary else 'NO'}")
    print(f"  BP3: downstream grad nonzero in {seeds_with_grad}/50 seeds")
    print(f"  Forward equivalence (eval): {'YES' if equiv else 'NO'}")

    if bp3_boundary and not bp2_boundary:
        print("\n  BP3 fixes the gradient blockage.")
        print("  Change: cumsum-based foo + soft_mask multiplication in attention pooling")
        print("  enables loss → pooled → attn*soft_mask → foo → cumsum → boundaries → STE → params")
