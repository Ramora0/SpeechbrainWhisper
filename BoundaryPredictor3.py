import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from loss import binomial_loss_from_target_counts
import flags


def common_cumsum(boundaries):
    """
    Cumsum-based segment distance computation that preserves gradient flow.

    Unlike utils.common() which uses boolean == comparisons (killing gradients),
    this uses only differentiable ops: cumsum, subtraction.

    Returns foo [B, L, S] where foo[b,t,s] = s - segment_id[b,t].
    foo == 0 at positions belonging to segment s.
    """
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if n_segments == 0:
        return None

    segment_indices = torch.arange(
        n_segments, device=boundaries.device, dtype=boundaries.dtype)

    # cumsum - boundaries = segment IDs (differentiable w.r.t. boundaries)
    hh1 = boundaries.cumsum(1) - boundaries

    # foo[b,t,s] = s - segment_id[b,t]; == 0 when position t is in segment s
    foo = segment_indices.view(1, 1, -1) - hh1.unsqueeze(-1)  # B x L x S

    return foo


class BoundaryPredictor3(nn.Module):
    """
    Same as BoundaryPredictor2 but with gradient-preserving attention pooling.

    The key change is in _attention_pooling: uses cumsum-based foo (gradient
    flows through cumsum -> boundaries -> STE -> boundary params) and multiplies
    attention weights by a soft segment mask (1 - foo) so downstream loss
    gradients reach boundary-decision parameters.
    """

    def __init__(self, input_dim, prior, temp=1, similarity_bias=0.0):
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

        # Initialize similarity bias from parameter
        self.similarity_bias = nn.Parameter(torch.tensor(similarity_bias))

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
        self.learned_query = nn.Parameter(torch.randn(input_dim))

        # Key and Value projections
        self.pool_key = nn.Linear(input_dim, input_dim, bias=False)
        self.pool_value = nn.Linear(input_dim, input_dim, bias=False)

        # Output projection after pooling (combines heads)
        self.pool_output = nn.Linear(input_dim, input_dim, bias=False)

        # LayerNorm for stabilizing attention inputs
        self.pool_layernorm = nn.LayerNorm(input_dim)

        # Scaling factor for attention scores (per head)
        self.pool_scale = self.head_dim ** -0.5

        # Initialize projections as identity matrices
        with torch.no_grad():
            self.pool_key.weight.copy_(torch.eye(input_dim))
            self.pool_value.weight.copy_(torch.eye(input_dim))
            self.pool_output.weight.copy_(torch.eye(input_dim))

        self.pool_key.weight._no_reinit = True
        self.pool_value.weight._no_reinit = True
        self.pool_output.weight._no_reinit = True

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
        """
        Get the prior based on compression schedule.

        Scales linearly in compression space (not prior space):
        - schedule=0.0: 2x compression (prior=0.5)
        - schedule=1.0: target compression (prior=target_prior)

        Linear in compression means: 2x -> 4x -> 6x -> 8x
        Not exponential: 2x -> 4x -> 8x (which you'd get from linear prior)
        """
        schedule = self.compression_schedule
        target = self.target_prior

        if abs(target - 1.0) < 1e-8:
            return 1.0

        # Convert target prior to compression ratio
        target_compression = 1.0 / target  # e.g., 8 for prior=0.125
        start_compression = 2.0  # Start at 2x compression (prior=0.5)

        # Linear interpolation in compression space
        current_compression = start_compression + (target_compression - start_compression) * schedule

        # Convert back to prior
        scheduled_prior = 1.0 / current_compression
        return scheduled_prior

    def _attention_pooling(self, boundaries, hidden, lengths):
        """
        Multi-head attention-based pooling with gradient-preserving segment masks.

        Unlike BP2, this uses cumsum-based foo (preserves gradient through
        boundaries via STE) and multiplies attention weights by a soft segment
        mask so downstream loss gradients reach boundary-decision parameters.

        The soft mask values are 1.0 at in-segment positions (no value change),
        but d(1-foo)/d(boundary) is nonzero, enabling gradient flow.
        """
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] _attention_pooling INPUT:")
        if flags.PRINT_DATA:
            print(f"  boundaries.shape = {boundaries.shape}")
            print(f"  hidden.shape = {hidden.shape}")
            print(f"  lengths = {lengths}")

        # Step 1: Compute foo via cumsum (gradient-preserving)
        foo = common_cumsum(boundaries)  # B x L x S

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] After common_cumsum():")
        if flags.PRINT_DATA:
            print(
                f"  foo = {foo if foo is None else f'tensor with shape {foo.shape}'}")

        if foo is None:
            if flags.PRINT_FLOW:
                print(
                    f"[BoundaryPredictor3.py] No boundaries found, returning empty tensor")
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)  # S
        if flags.PRINT_DATA:
            print(f"[BoundaryPredictor3.py] max_segments = {max_segments}")

        # Step 2: Create masks
        # Hard mask (detached) for -inf attention masking
        hard_segment_mask = (foo.detach() == 0).float()  # B x L x S

        # Soft mask (gradient-carrying) for post-attention modulation
        # Values: 1.0 at in-segment (foo=0), 0.0 at out-of-segment
        # Gradient: d(1-foo)/d(boundary) = -d(foo)/d(boundary) is nonzero via cumsum
        out_of_segment = (foo.detach() != 0)
        soft_segment_mask = torch.where(
            out_of_segment, torch.zeros_like(foo), 1.0 - foo)  # B x L x S

        # Compute actual lengths for each batch item
        actual_lens = (lengths * seq_len).long()

        # Create length mask (vectorized)
        pos_indices = torch.arange(
            seq_len, device=device).unsqueeze(0)  # (1, L)
        length_mask = (pos_indices < actual_lens.unsqueeze(1)
                       ).float()  # (B, L)

        # Apply length mask to both masks
        hard_segment_mask = hard_segment_mask * length_mask.unsqueeze(-1)
        soft_segment_mask = soft_segment_mask * length_mask.unsqueeze(-1)

        # Step 3: Use learned query vector for all segments
        queries = self.learned_query.unsqueeze(0).unsqueeze(
            0).expand(batch_size, max_segments, -1)  # (B, S, D)

        # Step 4: Apply LayerNorm and project to keys/values
        hidden_normed = self.pool_layernorm(hidden)  # (B, L, D)

        keys = self.pool_key(hidden_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.pool_value(hidden_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim)

        queries = queries.view(batch_size, max_segments,
                               self.num_heads, self.head_dim)

        # Step 5: Compute attention scores
        attn_scores = torch.einsum(
            'bshd,blhd->bhsl', queries, keys) * self.pool_scale  # (B, H, S, L)

        # Step 6: Hard-mask out-of-segment positions with -inf
        hard_mask_transposed = hard_segment_mask.permute(
            0, 2, 1).unsqueeze(1)  # (B, 1, S, L)
        attn_scores = attn_scores.masked_fill(
            hard_mask_transposed == 0, float('-inf'))

        # Step 7: Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, S, L)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Step 7.5: Multiply by soft mask to enable gradient flow to boundaries
        # Values unchanged: soft_mask=1.0 at in-segment, attn_weights=0 at out-of-segment
        # Gradient flows: loss -> pooled -> attn*soft -> soft_mask -> foo -> cumsum -> boundaries
        soft_mask_transposed = soft_segment_mask.permute(
            0, 2, 1).unsqueeze(1)  # (B, 1, S, L)
        attn_weights = attn_weights * soft_mask_transposed

        # Step 8: Apply attention to values
        pooled = torch.einsum('bhsl,blhd->bshd', attn_weights, values)

        # Step 9: Flatten heads
        pooled = pooled.reshape(batch_size, max_segments, -1)  # (B, S, D)

        # Step 10: Output projection
        pooled = self.pool_output(pooled)

        # Ensure output maintains same dtype as input
        pooled = pooled.to(dtype=dtype)

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] _attention_pooling OUTPUT:")
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
        batch_size, seq_len, _ = hidden.shape

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] forward() INPUT:")
        if flags.PRINT_DATA:
            print(f"  hidden.shape = {hidden.shape}")
            print(f"  lengths.shape = {lengths.shape}")
            print(f"  lengths = {lengths}")

        # Optimized: apply dropout once and normalize once to reduce backward pass cost
        hidden_dropped = self.dropout(hidden)

        # Extract q and k inputs (adjacent frames)
        q_input = hidden_dropped[:, :-1]  # (B, L-1, D)
        k_input = hidden_dropped[:, 1:]   # (B, L-1, D)

        # Normalize once (instead of 4 times)
        q_normed = F.normalize(q_input, dim=-1, eps=1e-8)
        k_normed = F.normalize(k_input, dim=-1, eps=1e-8)

        # Apply MLP with residual and project (fused operations)
        q_mlp_out = self.boundary_mlp(q_normed)
        q_hidden = self.q_proj_layer(F.normalize(
            q_mlp_out + q_normed, dim=-1, eps=1e-8))

        k_mlp_out = self.boundary_mlp(k_normed)
        k_hidden = self.k_proj_layer(F.normalize(
            k_mlp_out + k_normed, dim=-1, eps=1e-8))

        # Compute cosine similarity (already normalized, so this is just dot product)
        cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)

        # Debug: Check for NaN values (only when flag enabled to avoid GPU sync)
        if flags.PRINT_NAN_INF:
            if torch.isnan(cos_sim).any():
                print(f"[BoundaryPredictor3.py] WARNING: NaN detected in cos_sim")
                print(f"  q_hidden has NaN: {torch.isnan(q_hidden).any()}")
                print(f"  k_hidden has NaN: {torch.isnan(k_hidden).any()}")

        # Optimized probability computation (fused operations, safe for gradients)
        probs = torch.clamp(
            (1.0 - (cos_sim + self.similarity_bias)) * 0.5, min=0.0, max=1.0)
        probs = F.pad(probs, (0, 1), value=0.0)

        # Debug: Check for NaN in probs (only when flag enabled to avoid GPU sync)
        if flags.PRINT_NAN_INF:
            if torch.isnan(probs).any():
                print(f"[BoundaryPredictor3.py] ERROR: NaN detected in probs!")
                print(f"  probs shape: {probs.shape}")
                print(f"  Number of NaN values: {torch.isnan(probs).sum()}")
                print(f"  cos_sim min/max: {cos_sim.min()}/{cos_sim.max()}")
                print(f"  similarity_bias: {self.similarity_bias.item()}")
        if self.training:
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
            hard_samples = (soft_boundaries > 0.5).float()
        else:
            # During evaluation, threshold probabilities directly without sampling
            soft_boundaries = probs
            hard_samples = (probs > 0.5).float()

        # Mask boundaries based on lengths (vectorized)
        batch_size, boundary_seq_len = soft_boundaries.shape
        # +1 because boundaries are seq_len-1
        actual_lens = (lengths * (boundary_seq_len + 1)).long()

        # Compute valid lengths and clamp
        valid_lens = torch.clamp(actual_lens - 1, min=0, max=boundary_seq_len)

        # Create position mask
        pos_idx = torch.arange(
            # (1, L)
            boundary_seq_len, device=soft_boundaries.device).unsqueeze(0)
        valid_mask = (pos_idx < valid_lens.unsqueeze(1)).float()  # (B, L)

        # Zero out padding positions
        soft_boundaries = soft_boundaries * valid_mask
        hard_samples = hard_samples * valid_mask
        masked_probs = probs * valid_mask  # (B, L) — for nudge loss

        # Set boundary at first padding position using indexing
        # Only set if valid_len < boundary_seq_len
        needs_boundary_mask = valid_lens < boundary_seq_len
        if needs_boundary_mask.any():
            batch_idx = torch.arange(batch_size, device=soft_boundaries.device)[
                needs_boundary_mask]
            first_padding_idx = valid_lens[needs_boundary_mask]
            soft_boundaries[batch_idx, first_padding_idx] = 1.0
            hard_samples[batch_idx, first_padding_idx] = 1.0

        hard_boundaries = (
            hard_samples - soft_boundaries.detach() + soft_boundaries
        )

        # Ensure each sequence has at least one boundary to prevent empty segments (vectorized)
        boundary_seq_len = hard_boundaries.shape[1]

        # Detect sequences with no boundaries
        sequences_no_boundaries = (hard_boundaries.sum(dim=1) == 0)  # (B,)
        sequences_with_no_boundaries = sequences_no_boundaries.nonzero(as_tuple=True)[
            0].tolist()

        if sequences_no_boundaries.any():
            indices = sequences_no_boundaries.nonzero(as_tuple=True)[0]
            valid_lens_emergency = torch.clamp(
                actual_lens[indices] - 1, min=0, max=boundary_seq_len)
            boundary_idxs = torch.clamp(
                valid_lens_emergency, min=0, max=boundary_seq_len - 1)

            # Set boundaries for sequences that need them
            valid_boundary_mask = boundary_idxs >= 0
            if valid_boundary_mask.any():
                valid_seq_indices = indices[valid_boundary_mask]
                valid_boundary_indices = boundary_idxs[valid_boundary_mask]
                hard_boundaries[valid_seq_indices,
                                valid_boundary_indices] = 1.0

        # Print warning if this happens during training (should be rare)
        if len(sequences_with_no_boundaries) > 0 and self.training:
            print(
                f"[BoundaryPredictor3.py] WARNING: Added emergency boundary for {len(sequences_with_no_boundaries)} sequence(s) with no boundaries during TRAINING")
            print(f"  Affected sequences: {sequences_with_no_boundaries}")

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] BEFORE _attention_pooling:")
        if flags.PRINT_DATA:
            print(f"  hard_boundaries.shape = {hard_boundaries.shape}")
            print(
                f"  hard_boundaries sum per sample = {hard_boundaries.sum(dim=1)}")

        pooled = self._attention_pooling(
            hard_boundaries, hidden, lengths)

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] AFTER _attention_pooling:")
        if flags.PRINT_DATA:
            print(f"  pooled.shape = {pooled.shape}")

        # Compute shortened lengths based on boundary positions
        batch_size = hard_boundaries.shape[0]
        seq_len = hidden.shape[1]
        actual_lens = (lengths * seq_len).long()

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] Computing shortened_lengths:")
        if flags.PRINT_DATA:
            print(f"  batch_size = {batch_size}")
            print(f"  seq_len (hidden) = {seq_len}")
            print(f"  actual_lens = {actual_lens}")

        # Get the actual pooled sequence length from the pooled tensor
        max_segments = pooled.shape[1] if pooled.shape[1] > 0 else 1

        if flags.PRINT_DATA:
            print(f"  max_segments (from pooled) = {max_segments}")

        # Vectorized shortened lengths computation
        boundary_seq_len = hard_boundaries.shape[1]
        pos_mask = torch.arange(boundary_seq_len, device=hidden.device).unsqueeze(
            0) < actual_lens.unsqueeze(1)

        # Count boundaries within valid length per sequence
        num_boundaries_per_sample = (
            hard_boundaries * pos_mask.float()).sum(dim=1)  # (B,)

        # Initialize output
        shortened_lengths = torch.zeros(batch_size, device=hidden.device)

        if max_segments > 0:
            # Case 1: num_boundaries >= max_segments - 1
            case1_mask = num_boundaries_per_sample >= (max_segments - 1)
            shortened_lengths[case1_mask] = 1.0

            # Case 2: 0 < num_boundaries < max_segments - 1
            case2_mask = (num_boundaries_per_sample > 0) & (
                num_boundaries_per_sample < (max_segments - 1))
            shortened_lengths[case2_mask] = (
                num_boundaries_per_sample[case2_mask] + 1) / max_segments

            # Case 3: num_boundaries == 0 -> already 0.0 (default)

        if flags.PRINT_DATA:
            print(
                f"[BoundaryPredictor3.py] shortened_lengths = {shortened_lengths}")

        num_boundaries_tensor = hard_boundaries.sum()
        seq_len = hidden.shape[1]

        actual_lens = (lengths * seq_len).long()
        total_positions_tensor = actual_lens.sum().float()
        total_positions_per_sample = actual_lens.float()  # (B,)

        if self.training:
            per_sample_loss = self.calc_loss_target_counts(
                hard_boundaries,
                soft_boundaries,
                lengths,
                target_boundary_counts,
                reduce=False,
            )

            if return_unreduced_boundary_loss:
                loss = per_sample_loss
            else:
                loss = per_sample_loss.mean()
        else:
            loss = torch.tensor(0.0, device=hidden.device)
            if return_unreduced_boundary_loss:
                loss = loss.repeat(batch_size)

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        boundary_cv = None
        boundary_adjacent_pct = None

        # Only compute expensive boundary statistics during evaluation
        if not self.training:
            with torch.no_grad():
                all_spacings = []
                adjacent_count = 0
                total_boundaries = 0

                seq_len = hard_samples.shape[1]
                valid_lengths = torch.clamp(
                    (lengths * (seq_len + 1)).long() - 1, max=seq_len)
                valid_mask = torch.arange(seq_len, device=hard_samples.device).unsqueeze(
                    0) < valid_lengths.unsqueeze(1)
                masked_boundaries = hard_samples * valid_mask.float()

                for b in range(batch_size):
                    boundary_positions = masked_boundaries[b].nonzero(as_tuple=True)[
                        0]

                    if len(boundary_positions) > 1:
                        spacings = boundary_positions[1:] - \
                            boundary_positions[:-1]
                        all_spacings.extend(spacings.cpu().tolist())
                        adjacent_count += (spacings == 1).sum().item()
                        total_boundaries += len(boundary_positions) - 1

                if len(all_spacings) > 0:
                    spacings_tensor = torch.tensor(
                        all_spacings, dtype=torch.float32)
                    mean_spacing = spacings_tensor.mean()
                    std_spacing = spacings_tensor.std()
                    if mean_spacing > 0:
                        boundary_cv = (std_spacing / mean_spacing).item()
                    else:
                        boundary_cv = 0.0

                if total_boundaries > 0:
                    boundary_adjacent_pct = (
                        adjacent_count / total_boundaries) * 100.0
                else:
                    boundary_adjacent_pct = 0.0

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor3.py] RETURN:")
        if flags.PRINT_DATA:
            print(f"  pooled.shape = {pooled.shape}")
            print(f"  shortened_lengths.shape = {shortened_lengths.shape}")
            print(f"  shortened_lengths = {shortened_lengths}")
            print(f"  num_boundaries = {num_boundaries}")
            print(f"  total_positions = {total_positions}")

        return (
            pooled,
            loss,
            num_boundaries,
            total_positions,
            shortened_lengths,
            boundary_cv,
            boundary_adjacent_pct,
            masked_probs,                # (B, L) — boundary probs, masked
            num_boundaries_per_sample,   # (B,) — boundaries per sample
            total_positions_per_sample,  # (B,) — valid positions per sample
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
            print("[BP3 calc_loss] Running checks...")

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
        hard_boundaries,
        soft_boundaries,
        lengths,
        target_boundary_counts,
        reduce=True,
    ):
        """
        Implements the Ratio Loss from H-Net (Equation 10).
        """
        N = (lengths / target_boundary_counts).unsqueeze(1)

        sum_b = hard_boundaries.detach().sum(dim=1, keepdim=True)
        F_val = sum_b / lengths.unsqueeze(1)

        sum_p = soft_boundaries.sum(dim=1, keepdim=True)
        G = sum_p / lengths.unsqueeze(1)

        term1 = (N - 1) * F_val * G
        term2 = (1 - F_val) * (1 - G)
        scaling = N / (N - 1)

        loss_values = scaling * (term1 + term2)

        if reduce:
            return loss_values.mean()

        return loss_values
