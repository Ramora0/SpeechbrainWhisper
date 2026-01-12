"""
BPTest3.py - STAGE 0: COMPLETE IDENTITY (NO-OP)

Complete identity transformation with both boundaries and attention pooling frozen.

CURRENT CONFIGURATION - STAGE 0 (BASELINE):

1. BOUNDARY DETECTION: COMPLETELY BYPASSED
   - No boundary computation at all
   - Forward pass skips all boundary detection logic
   - All boundary detection parameters exist but are completely unused
   - No gradient flow through boundary detection modules

2. ATTENTION POOLING: COMPLETELY BYPASSED
   - No pooling computation at all
   - Input hidden states pass through unchanged
   - All attention pooling parameters exist but are completely unused
   - No gradient flow through attention pooling modules

3. BOUNDARY LOSS: DISABLED
   - Loss always returns 0.0
   - No gradient signal for boundary prediction

4. COMPRESSION: NONE (TRUE IDENTITY)
   - Output = Input (exact copy)
   - Output length = Input length (no compression)
   - shortened_lengths = input lengths (1:1 mapping)
   - Proper padding masking maintained

PURPOSE:
Establish a true no-op baseline where the BoundaryPredictor module has ZERO effect
on the forward pass. This serves as:
- Sanity check that the rest of the pipeline works
- Baseline performance metric (no boundary prediction overhead)
- Starting point for gradual feature enabling

PROGRESSION:
✓ Stage 0: COMPLETE IDENTITY (no boundaries, no pooling) → TRUE NO-OP (current)
- Stage 1: Fixed boundaries=1s, frozen pooling → Test segmentation
- Stage 2: Fixed boundaries=1s, trainable pooling → Test pooling learning
- Stage 3: Learned boundaries, trainable pooling → Full system (see BPTest.py)

Use this to verify the pipeline works correctly before enabling any BoundaryPredictor features!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from loss import binomial_loss_from_target_counts
from utils import common
import flags


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, prior, temp=1):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.compression_schedule = 1.0
        self.target_prior = prior

        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

        self.q_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(input_dim, input_dim, bias=False)

        self.similarity_bias = nn.Parameter(torch.tensor(-1.0))

        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(input_dim))
            self.k_proj_layer.weight.copy_(torch.eye(input_dim))

        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

        self.dropout = nn.Dropout(p=0.1)

        # Multi-head attention configuration
        self.num_heads = 8
        self.head_dim = input_dim // self.num_heads
        assert input_dim % self.num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({self.num_heads})"

        # Learned query vector (shared across all segments)
        # Initialize to small random values to break symmetry
        self.learned_query = nn.Parameter(torch.randn(input_dim) * 0.02)

        # Key and Value projections
        self.pool_key = nn.Linear(input_dim, input_dim, bias=False)
        self.pool_value = nn.Linear(input_dim, input_dim, bias=False)

        # Output projection after pooling (combines heads)
        self.pool_output = nn.Linear(input_dim, input_dim, bias=False)

        # LayerNorm for stabilizing attention inputs
        self.pool_layernorm = nn.LayerNorm(input_dim)

        # Scaling factor for attention scores (per head)
        self.pool_scale = self.head_dim ** -0.5

        # Initialize projections as identity matrices (good starting point for learning)
        with torch.no_grad():
            self.pool_key.weight.copy_(torch.eye(input_dim))
            self.pool_value.weight.copy_(torch.eye(input_dim))
            self.pool_output.weight.copy_(torch.eye(input_dim))

        # Mark as non-reinitializable (so checkpoint loading doesn't reset them)
        self.pool_key.weight._no_reinit = True
        self.pool_value.weight._no_reinit = True
        self.pool_output.weight._no_reinit = True

        # NOTE: All parameters exist for compatibility but are COMPLETELY UNUSED in forward pass
        # - No boundary detection computation
        # - No attention pooling computation
        # - Forward pass is pure identity: output = input

    def set_prior(self, prior):
        self.prior = prior

    def set_temperature(self, temp):
        """Set the temperature for the RelaxedBernoulli distribution."""
        self.temp = temp

    def set_compression_schedule(self, schedule_value):
        """
        Set the compression schedule value (0.0 to 1.0).
        0.0 = no compression (every token is a boundary)
        1.0 = max compression (only target_boundary_counts boundaries)
        """
        self.compression_schedule = float(schedule_value)

    def get_scheduled_prior(self):
        schedule = self.compression_schedule
        target = self.target_prior

        if abs(target - 1.0) < 1e-8:
            return 1.0

        scheduled_prior = target / (target + schedule * (1.0 - target))
        return scheduled_prior

    def _attention_pooling(self, boundaries, hidden, lengths):
        """
        Multi-head attention-based pooling using query matrix applied to boundary positions.

        Args:
            boundaries: (B, L) - binary boundary indicators
            hidden: (B, L, D) - hidden states
            lengths: (B,) - relative lengths (0.0-1.0) for each sequence

        Returns:
            pooled: (B, S, D) - pooled segment representations
        """
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] _attention_pooling INPUT:")
        if flags.PRINT_DATA:
            print(f"  boundaries.shape = {boundaries.shape}")
            print(f"  hidden.shape = {hidden.shape}")
            print(f"  lengths = {lengths}")

        # Step 1: Create segment assignment matrix using existing logic
        foo = common(boundaries)  # B x L x S

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] After common():")
        if flags.PRINT_DATA:
            print(
                f"  foo = {foo if foo is None else f'tensor with shape {foo.shape}'}")

        if foo is None:
            # No boundaries found
            if flags.PRINT_FLOW:
                print(
                    f"[BoundaryPredictor.py] No boundaries found, returning empty tensor")
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)  # S
        if flags.PRINT_DATA:
            print(f"[BoundaryPredictor.py] max_segments = {max_segments}")

        # Step 2: Create binary segment mask (B x L x S)
        segment_mask = (foo == 0).float()  # B x L x S

        # Compute actual lengths for each batch item
        actual_lens = (lengths * seq_len).long()

        # Create length mask for segment_mask (vectorized)
        length_mask = torch.arange(seq_len, device=device).unsqueeze(
            0) < actual_lens.unsqueeze(1)
        length_mask = length_mask.float()

        # Apply length mask to segment_mask
        segment_mask = segment_mask * length_mask.unsqueeze(-1)

        # Step 3: Use learned query vector for all segments
        # Expand learned query to (B, S, D) - same query for all segments in all batches
        queries = self.learned_query.unsqueeze(0).unsqueeze(
            0).expand(batch_size, max_segments, -1)  # (B, S, D)

        # Step 4: Reshape queries for multi-head attention
        queries = queries.view(batch_size, max_segments, self.num_heads,
                               # (B, H, S, head_dim)
                               self.head_dim).transpose(1, 2)

        # Step 5: Apply LayerNorm before projecting to keys and values
        # LayerNorm stabilizes attention by normalizing to mean=0, std=1
        hidden_normed = self.pool_layernorm(hidden)  # (B, L, D)

        # Step 6: Project to keys and values and reshape for multi-head
        keys = self.pool_key(hidden_normed)      # (B, L, D)
        keys = keys.view(batch_size, seq_len, self.num_heads,
                         self.head_dim).transpose(1, 2)  # (B, H, L, head_dim)

        values = self.pool_value(hidden_normed)  # (B, L, D)
        values = values.view(batch_size, seq_len, self.num_heads,
                             # (B, H, L, head_dim)
                             self.head_dim).transpose(1, 2)

        # Step 7: Compute attention scores: queries @ keys
        # queries: (B, H, S, head_dim), keys: (B, H, L, head_dim) -> (B, H, S, L)
        attn_scores = torch.matmul(
            queries, keys.transpose(-2, -1))  # (B, H, S, L)
        attn_scores = attn_scores * self.pool_scale

        # Step 8: Mask out positions not in segment
        # segment_mask is (B, L, S), we need (B, 1, S, L) for broadcasting across heads
        segment_mask_transposed = segment_mask.transpose(
            1, 2).unsqueeze(1)  # (B, 1, S, L)
        attn_scores = attn_scores.masked_fill(
            segment_mask_transposed == 0, float('-inf'))

        # Step 9: Compute attention weights per segment
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, S, L)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Step 10: Apply attention: (B, H, S, L) @ (B, H, L, head_dim) -> (B, H, S, head_dim)
        pooled = torch.matmul(attn_weights, values)  # (B, H, S, head_dim)

        # Step 11: Concatenate heads back together
        pooled = pooled.transpose(1, 2).contiguous()  # (B, S, H, head_dim)
        pooled = pooled.flatten(2)  # (B, S, H*head_dim) -> (B, S, D)

        # Step 12: Output projection to combine information from all heads
        pooled = self.pool_output(pooled)

        # Ensure output maintains same dtype as input
        pooled = pooled.to(dtype=dtype)

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] _attention_pooling OUTPUT:")
        if flags.PRINT_DATA:
            print(f"  pooled.shape = {pooled.shape}")

        return pooled  # B x S x D

    def _simple_average_pooling(self, boundaries, hidden, lengths):
        """
        SIMPLIFIED: Basic average pooling with only masking - no attention mechanism.

        Args:
            boundaries: (B, L) - binary boundary indicators
            hidden: (B, L, D) - hidden states
            lengths: (B,) - relative lengths (0.0-1.0) for each sequence

        Returns:
            pooled: (B, S, D) - pooled segment representations
        """
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] _simple_average_pooling INPUT:")
        if flags.PRINT_DATA:
            print(f"  boundaries.shape = {boundaries.shape}")
            print(f"  hidden.shape = {hidden.shape}")
            print(f"  lengths = {lengths}")

        # Create segment assignment matrix
        foo = common(boundaries)  # B x L x S

        if foo is None:
            if flags.PRINT_FLOW:
                print(
                    f"[BoundaryPredictor.py] No boundaries found, returning empty tensor")
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)  # S

        # Create binary segment mask (B x L x S)
        segment_mask = (foo == 0).float()  # B x L x S

        # Apply length masking (vectorized)
        actual_lens = (lengths * seq_len).long()
        length_mask = torch.arange(seq_len, device=device).unsqueeze(
            0) < actual_lens.unsqueeze(1)
        length_mask = length_mask.float()

        # Apply length mask to segment_mask
        segment_mask = segment_mask * length_mask.unsqueeze(-1)  # B x L x S

        # Simple average pooling: sum over sequence dimension, normalize by count
        # hidden: (B, L, D), segment_mask: (B, L, S)
        # We want to compute weighted average for each segment

        # Expand hidden to (B, L, 1, D) and segment_mask to (B, L, S, 1)
        hidden_expanded = hidden.unsqueeze(2)  # (B, L, 1, D)
        mask_expanded = segment_mask.unsqueeze(-1)  # (B, L, S, 1)

        # Masked hidden states: (B, L, S, D)
        masked_hidden = hidden_expanded * mask_expanded

        # Sum over sequence dimension: (B, S, D)
        pooled = masked_hidden.sum(dim=1)

        # Normalize by segment lengths (count of positions in each segment)
        segment_counts = segment_mask.sum(dim=1, keepdim=True).transpose(
            1, 2)  # (B, 1, S) -> (B, S, 1)
        segment_counts = torch.clamp(
            segment_counts, min=1.0)  # Avoid division by zero

        pooled = pooled / segment_counts  # (B, S, D)

        # Ensure output maintains same dtype as input
        pooled = pooled.to(dtype=dtype)

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] _simple_average_pooling OUTPUT:")
        if flags.PRINT_DATA:
            print(f"  pooled.shape = {pooled.shape}")

        return pooled  # B x S x D

    def forward(
        self,
        hidden,
        lengths,
        target_boundary_counts=None,
        return_unreduced_boundary_loss=False,
    ):
        """
        COMPLETE IDENTITY FORWARD PASS - NO COMPUTATION

        This forward method bypasses ALL boundary detection and attention pooling.
        Output = Input (exact copy with no transformation).

        This serves as a true no-op baseline to verify the rest of the pipeline works.
        """
        batch_size, seq_len, _ = hidden.shape

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] forward() INPUT (IDENTITY MODE):")
        if flags.PRINT_DATA:
            print(f"  hidden.shape = {hidden.shape}")
            print(f"  lengths.shape = {lengths.shape}")
            print(f"  lengths = {lengths}")

        # ========== COMPLETE IDENTITY: NO BOUNDARY DETECTION, NO POOLING ==========
        # Skip all boundary computation and pooling
        # Output is exactly the input hidden states

        pooled = hidden  # Identity: output = input

        # Since output = input, shortened_lengths = input lengths (no compression)
        shortened_lengths = lengths  # Identity: output length = input length

        # No boundary computation, so set dummy values for diagnostic outputs
        num_boundaries = 0.0
        actual_lens = (lengths * seq_len).long()
        total_positions = actual_lens.sum().float().item()

        # No boundary loss since we're not computing boundaries
        loss = torch.tensor(0.0, device=hidden.device)
        if return_unreduced_boundary_loss:
            loss = loss.repeat(batch_size)

        # No diagnostic statistics
        boundary_cv = None
        boundary_adjacent_pct = None

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] IDENTITY FORWARD COMPLETE:")
        if flags.PRINT_DATA:
            print(f"  pooled.shape = {pooled.shape} (same as input)")
            print(f"  shortened_lengths = {shortened_lengths} (same as input)")
            print(f"  num_boundaries = {num_boundaries} (no boundaries computed)")
            print(f"  total_positions = {total_positions}")

        return (
            pooled,
            loss,
            num_boundaries,
            total_positions,
            shortened_lengths,
            boundary_cv,
            boundary_adjacent_pct,
        )

    def calc_loss_target_counts(
        self,
        hard_boundaries,
        soft_boundaries,
        lengths,
        target_boundary_counts,
        reduce=True,
    ):
        if flags.PRINT_BP_LOSS_CHECKS:
            print("[BP calc_loss] Running checks...")

        device = hard_boundaries.device
        per_item_boundaries = hard_boundaries.sum(dim=1)

        seq_len = hard_boundaries.shape[1] + 1  # boundaries are seq_len - 1
        actual_lens = (lengths * seq_len).long()
        per_item_totals = actual_lens.float()

        per_item_totals = per_item_totals.to(dtype=torch.float32)
        target_boundary_counts = target_boundary_counts.to(
            device=device,
            dtype=torch.float32,
        )

        loss_values = binomial_loss_from_target_counts(
            per_item_boundaries.to(dtype=torch.float32),
            per_item_totals,
            target_boundary_counts,
        )

        if reduce:
            final_loss = loss_values.mean()
            return final_loss
        return loss_values

    def calc_ratio_loss(
        self,
        # Indicators (b_t): [batch, seq_len], STE-linked
        hard_boundaries,
        soft_boundaries,        # Probabilities (p_t): [batch, seq_len]
        lengths,                # Effective lengths/masks: [batch]
        target_boundary_counts,  # Target number of boundaries: [batch]
        reduce=True,
    ):
        """
        Implements the Ratio Loss from H-Net (Equation 10).

        This loss guides the model toward a target compression ratio N.
        Crucially, gradients only flow through G (the average probability),
        while F (the actual count) is treated as a non-differentiable constant.
        """
        # N: Target compression ratio (e.g., 6.0 for 1-stage)
        # N = actual_length / target_count
        N = (lengths / target_boundary_counts).unsqueeze(1)  # [batch, 1]

        # F: The actual fraction of vectors selected.
        # We detach hard_boundaries to ensure it doesn't provide gradients,
        # as per the paper's specification that F is non-differentiable.
        sum_b = hard_boundaries.detach().sum(dim=1, keepdim=True)
        F = sum_b / lengths.unsqueeze(1)  # [batch, 1]

        # G: The average boundary probability.
        # This is the differentiable term that allows the router to be trained.
        sum_p = soft_boundaries.sum(dim=1, keepdim=True)
        G = sum_p / lengths.unsqueeze(1)  # [batch, 1]

        # Ratio Loss Formula:
        # L_ratio = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))
        # This formulation is inspired by MoE load balancing.

        term1 = (N - 1) * F * G
        term2 = (1 - F) * (1 - G)
        scaling = N / (N - 1)

        loss_values = scaling * (term1 + term2)

        if reduce:
            # Standard H-Net practice is to average across the batch.
            # This loss is typically scaled by alpha = 0.03.
            return loss_values.mean()

        return loss_values
