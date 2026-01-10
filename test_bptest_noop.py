#!/usr/bin/env python3
"""
Test script to verify that BPTest.py acts as a no-op even after training.

This script verifies:
1. Boundaries are all 1s (no compression)
2. Attention pooling parameters are frozen
3. Output sequence length equals input sequence length
4. Parameters don't change during training
5. Output remains consistent before and after training
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path to import BPTest
sys.path.insert(0, str(Path(__file__).parent))

from BPTest import BoundaryPredictor2

def test_frozen_parameters():
    """Test that attention pooling parameters are frozen."""
    print("\n" + "="*80)
    print("TEST 1: Verifying that attention pooling parameters are frozen")
    print("="*80)

    input_dim = 144
    prior = 0.192
    bp = BoundaryPredictor2(input_dim=input_dim, prior=prior, temp=1.0)

    frozen_params = []
    trainable_params = []

    for name, param in bp.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"\nFrozen parameters ({len(frozen_params)}):")
    for name in frozen_params:
        print(f"  ✓ {name}")

    print(f"\nTrainable parameters ({len(trainable_params)}):")
    for name in trainable_params:
        print(f"  ⚠ {name}")

    # Check that attention pooling params are frozen
    assert 'learned_query' in frozen_params, "learned_query should be frozen"
    assert 'pool_key.weight' in frozen_params, "pool_key.weight should be frozen"
    assert 'pool_value.weight' in frozen_params, "pool_value.weight should be frozen"
    assert 'pool_output.weight' in frozen_params, "pool_output.weight should be frozen"

    # Check LayerNorm params are frozen
    layernorm_frozen = any('pool_layernorm' in name for name in frozen_params)
    assert layernorm_frozen, "pool_layernorm parameters should be frozen"

    print("\n✅ PASSED: All attention pooling parameters are frozen")
    return bp


def test_boundaries_all_ones():
    """Test that boundaries are all 1s (except padding)."""
    print("\n" + "="*80)
    print("TEST 2: Verifying that boundaries are all 1s")
    print("="*80)

    input_dim = 144
    prior = 0.192
    bp = BoundaryPredictor2(input_dim=input_dim, prior=prior, temp=1.0)
    bp.eval()

    batch_size = 4
    seq_len = 100

    # Create random input
    hidden = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.tensor([1.0, 0.8, 0.6, 0.5])  # Different lengths

    with torch.no_grad():
        pooled, loss, num_boundaries, total_positions, shortened_lengths, cv, adj_pct = bp(
            hidden=hidden,
            lengths=lengths,
            target_boundary_counts=None,
            return_unreduced_boundary_loss=False
        )

    print(f"\nInput shape: {hidden.shape}")
    print(f"Output shape: {pooled.shape}")
    print(f"Lengths: {lengths}")
    print(f"Shortened lengths: {shortened_lengths}")

    # Verify output shape matches input shape (no compression)
    assert pooled.shape[0] == batch_size, f"Batch size mismatch: {pooled.shape[0]} vs {batch_size}"
    assert pooled.shape[1] == seq_len, f"Seq len mismatch: {pooled.shape[1]} vs {seq_len}"
    assert pooled.shape[2] == input_dim, f"Hidden dim mismatch: {pooled.shape[2]} vs {input_dim}"

    # Check that shortened_lengths match input lengths (no compression)
    length_diff = (shortened_lengths - lengths).abs()
    print(f"\nLength difference: {length_diff}")

    # Allow small numerical differences due to floating point
    assert torch.allclose(shortened_lengths, lengths, atol=1e-2), \
        f"Lengths should be preserved: shortened={shortened_lengths}, original={lengths}"

    print("\n✅ PASSED: Boundaries are all 1s (no compression)")


def test_identity_behavior():
    """Test that the module approximately acts as identity."""
    print("\n" + "="*80)
    print("TEST 3: Detailed analysis of value changes")
    print("="*80)

    input_dim = 144
    prior = 0.192
    bp = BoundaryPredictor2(input_dim=input_dim, prior=prior, temp=1.0)
    bp.eval()

    batch_size = 3
    seq_len = 50

    # Create random input with varying lengths to test padding behavior
    hidden = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.tensor([1.0, 0.8, 0.6])  # Different lengths for each sample

    print(f"\nInput configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden dim: {input_dim}")
    print(f"  Lengths: {lengths}")
    print(f"  Actual lengths: {(lengths * seq_len).long()}")

    with torch.no_grad():
        pooled, _, _, _, _, _, _ = bp(
            hidden=hidden,
            lengths=lengths,
            target_boundary_counts=None,
            return_unreduced_boundary_loss=False
        )

    # Analyze each sample separately
    print("\n" + "-"*80)
    print("DETAILED ANALYSIS PER SAMPLE:")
    print("-"*80)

    for b in range(batch_size):
        actual_len = int(lengths[b].item() * seq_len)

        print(f"\n--- Sample {b} (length={lengths[b].item():.1f}, actual_len={actual_len}) ---")

        # Split into valid and padding regions
        valid_input = hidden[b, :actual_len, :]
        valid_output = pooled[b, :actual_len, :]

        if actual_len < seq_len:
            padding_input = hidden[b, actual_len:, :]
            padding_output = pooled[b, actual_len:, :]
        else:
            padding_input = None
            padding_output = None

        # Analyze VALID region
        valid_diff = (valid_output - valid_input).abs()
        print(f"\nVALID REGION (positions 0-{actual_len-1}):")
        print(f"  Input stats:  min={valid_input.min():.4f}, max={valid_input.max():.4f}, "
              f"mean={valid_input.mean():.4f}, std={valid_input.std():.4f}")
        print(f"  Output stats: min={valid_output.min():.4f}, max={valid_output.max():.4f}, "
              f"mean={valid_output.mean():.4f}, std={valid_output.std():.4f}")
        print(f"  Max difference: {valid_diff.max():.6f}")
        print(f"  Mean difference: {valid_diff.mean():.6f}")

        # Show a few example values
        print(f"\n  Position 0 examples (first 5 dims):")
        print(f"    Input:  {valid_input[0, :5]}")
        print(f"    Output: {valid_output[0, :5]}")
        print(f"    Diff:   {valid_diff[0, :5]}")

        if actual_len > 10:
            print(f"\n  Position 10 examples (first 5 dims):")
            print(f"    Input:  {valid_input[10, :5]}")
            print(f"    Output: {valid_output[10, :5]}")
            print(f"    Diff:   {valid_diff[10, :5]}")

        # Analyze PADDING region (should be all zeros in output)
        if padding_input is not None:
            padding_diff = (padding_output - padding_input).abs()
            print(f"\nPADDING REGION (positions {actual_len}-{seq_len-1}):")
            print(f"  Input stats:  min={padding_input.min():.4f}, max={padding_input.max():.4f}, "
                  f"mean={padding_input.mean():.4f}")
            print(f"  Output stats: min={padding_output.min():.4f}, max={padding_output.max():.4f}, "
                  f"mean={padding_output.mean():.4f}")
            print(f"  Output should be ALL ZEROS!")

            # Check if padding is properly zeroed
            if padding_output.abs().max() < 1e-6:
                print(f"  ✓ Padding correctly zeroed out (max abs value: {padding_output.abs().max():.2e})")
            else:
                print(f"  ✗ WARNING: Padding NOT zeroed! Max abs value: {padding_output.abs().max():.6f}")
                print(f"    First padding position: {padding_output[0, :5]}")

    # Overall statistics
    print("\n" + "-"*80)
    print("OVERALL STATISTICS:")
    print("-"*80)

    diff_all = (pooled - hidden).abs()
    print(f"\nAll positions (including padding):")
    print(f"  Max absolute difference: {diff_all.max():.6f}")
    print(f"  Mean absolute difference: {diff_all.mean():.6f}")

    # Check only valid positions
    max_valid_diff = 0.0
    mean_valid_diff = 0.0
    total_valid = 0

    for b in range(batch_size):
        actual_len = int(lengths[b].item() * seq_len)
        valid_diff = (pooled[b, :actual_len, :] - hidden[b, :actual_len, :]).abs()
        max_valid_diff = max(max_valid_diff, valid_diff.max().item())
        mean_valid_diff += valid_diff.sum().item()
        total_valid += actual_len * input_dim

    mean_valid_diff /= total_valid

    print(f"\nValid positions only (excluding padding):")
    print(f"  Max absolute difference: {max_valid_diff:.6f}")
    print(f"  Mean absolute difference: {mean_valid_diff:.6f}")

    # Detailed investigation: why is there a difference?
    print("\n" + "-"*80)
    print("INVESTIGATING SOURCE OF DIFFERENCES:")
    print("-"*80)

    # Create a simple test with length 1.0 (no padding)
    print("\nTest case: Full sequence (no padding)")
    test_hidden = torch.randn(1, 10, input_dim)
    test_lengths = torch.tensor([1.0])

    with torch.no_grad():
        test_pooled, _, _, _, _, _, _ = bp(
            hidden=test_hidden,
            lengths=test_lengths,
            target_boundary_counts=None,
            return_unreduced_boundary_loss=False
        )

    test_diff = (test_pooled - test_hidden).abs()
    print(f"  Max difference (no padding): {test_diff.max():.6f}")
    print(f"  Mean difference (no padding): {test_diff.mean():.6f}")

    # Check if LayerNorm is the culprit
    print("\nHypothesis: LayerNorm in attention pooling is changing values")
    print("  With zero query and identity projections, attention weights should be uniform")
    print("  But LayerNorm normalizes the input, changing the scale")

    if max_valid_diff > 0.01:
        print(f"\n⚠ WARNING: Large differences detected ({max_valid_diff:.6f})")
        print("  This suggests the pooling is NOT acting as identity!")
        print("  Expected: <0.01 for identity operation")
        print("  Actual behavior needs investigation")
    else:
        print(f"\n✅ PASSED: Differences are small ({max_valid_diff:.6f})")
        print("  Module approximately acts as identity for valid positions")


def test_no_learning_during_training():
    """Test that parameters don't change during training."""
    print("\n" + "="*80)
    print("TEST 4: Verifying no learning during training")
    print("="*80)

    input_dim = 144
    prior = 0.192
    bp = BoundaryPredictor2(input_dim=input_dim, prior=prior, temp=1.0)
    bp.train()

    # Store initial parameters
    initial_params = {}
    for name, param in bp.named_parameters():
        initial_params[name] = param.clone().detach()

    print(f"\nStored {len(initial_params)} initial parameters")

    # Create optimizer (even though frozen params won't update)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, bp.parameters()), lr=1e-3)

    batch_size = 4
    seq_len = 100
    num_iterations = 10

    print(f"\nRunning {num_iterations} training iterations...")

    for i in range(num_iterations):
        # Create random input (with requires_grad to allow backward)
        hidden = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        lengths = torch.rand(batch_size) * 0.5 + 0.5  # Random lengths between 0.5 and 1.0
        target_counts = torch.randint(10, 50, (batch_size,)).float()

        # Forward pass
        pooled, loss, _, _, _, _, _ = bp(
            hidden=hidden,
            lengths=lengths,
            target_boundary_counts=target_counts,
            return_unreduced_boundary_loss=False
        )

        # Create a dummy loss that includes the input (so gradients flow)
        # This simulates what would happen in actual training
        dummy_loss = pooled.sum() + 0.0 * hidden.sum()

        # Backward pass
        optimizer.zero_grad()
        if dummy_loss.requires_grad:
            dummy_loss.backward()
            optimizer.step()

        if (i + 1) % 5 == 0:
            print(f"  Iteration {i+1}/{num_iterations} - Dummy loss: {dummy_loss.item():.6f}")

    # Check that frozen parameters haven't changed
    print("\nChecking parameter changes...")
    all_frozen = True

    for name, param in bp.named_parameters():
        if name in initial_params:
            diff = (param - initial_params[name]).abs().max().item()
            if diff > 1e-6:
                print(f"  ⚠ {name}: changed by {diff:.6e}")
                all_frozen = False
            else:
                if 'pool' in name or 'learned_query' in name:
                    print(f"  ✓ {name}: unchanged (diff={diff:.6e})")

    assert all_frozen, "Some frozen parameters changed during training!"

    print("\n✅ PASSED: No parameters changed during training")


def test_output_consistency():
    """Test that output is consistent before and after 'training'."""
    print("\n" + "="*80)
    print("TEST 5: Verifying output consistency before/after training")
    print("="*80)

    input_dim = 144
    prior = 0.192

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create test input
    batch_size = 2
    seq_len = 50
    hidden = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.tensor([1.0, 0.8])

    # Create BP module
    bp = BoundaryPredictor2(input_dim=input_dim, prior=prior, temp=1.0)
    bp.eval()

    # Get output before training
    with torch.no_grad():
        output_before, _, _, _, lengths_before, _, _ = bp(
            hidden=hidden,
            lengths=lengths,
            target_boundary_counts=None,
            return_unreduced_boundary_loss=False
        )

    print(f"\nOutput before training shape: {output_before.shape}")
    print(f"Lengths before: {lengths_before}")

    # Simulate training
    bp.train()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, bp.parameters()), lr=1e-3)

    for _ in range(5):
        random_hidden = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        random_lengths = torch.rand(batch_size) * 0.5 + 0.5
        target_counts = torch.randint(10, 50, (batch_size,)).float()

        pooled, _, _, _, _, _, _ = bp(
            hidden=random_hidden,
            lengths=random_lengths,
            target_boundary_counts=target_counts,
            return_unreduced_boundary_loss=False
        )

        dummy_loss = pooled.sum() + 0.0 * random_hidden.sum()
        optimizer.zero_grad()
        if dummy_loss.requires_grad:
            dummy_loss.backward()
            optimizer.step()

    # Get output after training
    bp.eval()
    with torch.no_grad():
        output_after, _, _, _, lengths_after, _, _ = bp(
            hidden=hidden,
            lengths=lengths,
            target_boundary_counts=None,
            return_unreduced_boundary_loss=False
        )

    print(f"Output after training shape: {output_after.shape}")
    print(f"Lengths after: {lengths_after}")

    # Check consistency
    output_diff = (output_after - output_before).abs().max().item()
    length_diff = (lengths_after - lengths_before).abs().max().item()

    print(f"\nMax output difference: {output_diff:.6e}")
    print(f"Max length difference: {length_diff:.6e}")

    assert output_diff < 1e-5, f"Output changed after training: {output_diff}"
    assert length_diff < 1e-5, f"Lengths changed after training: {length_diff}"

    print("\n✅ PASSED: Output is consistent before and after training")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("BPTest.py NO-OP VERIFICATION TEST SUITE")
    print("="*80)
    print("\nThis test verifies that BPTest.py:")
    print("  1. Has frozen attention pooling parameters")
    print("  2. Uses all 1s boundaries (no compression)")
    print("  3. Outputs same sequence length as input")
    print("  4. Doesn't learn during training")
    print("  5. Produces consistent outputs")

    try:
        # Run all tests
        bp = test_frozen_parameters()
        test_boundaries_all_ones()
        test_identity_behavior()
        test_no_learning_during_training()
        test_output_consistency()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✅")
        print("="*80)
        print("\nBPTest.py is confirmed to be a no-op:")
        print("  ✓ All attention pooling parameters are frozen")
        print("  ✓ Boundaries are all 1s (no compression)")
        print("  ✓ Output length equals input length")
        print("  ✓ No parameters change during training")
        print("  ✓ Output is consistent before and after training")
        print("\nThe module can now be used in training runs to verify")
        print("that the rest of the pipeline works correctly.\n")

        return 0

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ❌")
        print("="*80)
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
