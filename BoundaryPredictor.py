import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from loss import binomial_loss_from_target_counts
from utils import common
import flags


class BoundaryPredictor2(nn.Module):
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
        pos_indices = torch.arange(
            seq_len, device=device).unsqueeze(0)  # (1, L)
        length_mask = (pos_indices < actual_lens.unsqueeze(1)
                       ).float()  # (B, L)

        # Apply length mask to segment_mask
        segment_mask = segment_mask * length_mask.unsqueeze(-1)

        # Step 3: Use learned query vector for all segments
        # Expand learned query to (B, S, D) - same query for all segments in all batches
        queries = self.learned_query.unsqueeze(0).unsqueeze(
            0).expand(batch_size, max_segments, -1)  # (B, S, D)

        # Step 4: Apply LayerNorm and project to keys/values (fused to reduce operations)
        hidden_normed = self.pool_layernorm(hidden)  # (B, L, D)

        # Project keys and values in one go, then reshape
        keys = self.pool_key(hidden_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.pool_value(hidden_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim)

        # Reshape queries
        queries = queries.view(batch_size, max_segments,
                               self.num_heads, self.head_dim)

        # Step 5: Compute attention with fused operations (reduce transposes)
        # queries: (B, S, H, head_dim), keys: (B, L, H, head_dim)
        # Use einsum to avoid multiple transposes
        attn_scores = torch.einsum(
            'bshd,blhd->bhsl', queries, keys) * self.pool_scale  # (B, H, S, L)

        # Step 6: Mask out positions not in segment
        # segment_mask is (B, L, S), we need (B, 1, S, L) for broadcasting across heads
        segment_mask_transposed = segment_mask.permute(
            0, 2, 1).unsqueeze(1)  # (B, 1, S, L)
        attn_scores = attn_scores.masked_fill(
            segment_mask_transposed == 0, float('-inf'))

        # Step 7: Compute attention weights and apply to values
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, S, L)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Apply attention using einsum (avoids transpose)
        # attn_weights: (B, H, S, L), values: (B, L, H, head_dim) -> (B, S, H, head_dim)
        pooled = torch.einsum('bhsl,blhd->bshd', attn_weights, values)

        # Step 8: Flatten heads (no contiguous() needed, einsum output is already contiguous)
        pooled = pooled.reshape(batch_size, max_segments, -1)  # (B, S, D)

        # Step 12: Output projection to combine information from all heads
        pooled = self.pool_output(pooled)

        # Ensure output maintains same dtype as input
        pooled = pooled.to(dtype=dtype)

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] _attention_pooling OUTPUT:")
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
            print(f"[BoundaryPredictor.py] forward() INPUT:")
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

        # Debug: Check for NaN values
        if torch.isnan(cos_sim).any():
            if flags.PRINT_NAN_INF:
                print(f"[BoundaryPredictor.py] WARNING: NaN detected in cos_sim")
                print(f"  q_hidden has NaN: {torch.isnan(q_hidden).any()}")
                print(f"  k_hidden has NaN: {torch.isnan(k_hidden).any()}")
                print(f"  q_residual has NaN: {torch.isnan(q_residual).any()}")
                print(f"  k_residual has NaN: {torch.isnan(k_residual).any()}")

        # Optimized probability computation (fused operations, safe for gradients)
        probs = torch.clamp(
            (1.0 - (cos_sim + self.similarity_bias)) * 0.5, min=0.0, max=1.0)
        probs = F.pad(probs, (0, 1), value=0.0)

        # Debug: Check for NaN in probs
        if torch.isnan(probs).any():
            if flags.PRINT_NAN_INF:
                print(f"[BoundaryPredictor.py] ERROR: NaN detected in probs!")
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

        # Set boundary at first padding position using advanced indexing
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
        # This is a safeguard that should rarely trigger
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
                f"[BoundaryPredictor.py] WARNING: Added emergency boundary for {len(sequences_with_no_boundaries)} sequence(s) with no boundaries during TRAINING")
            print(f"  Affected sequences: {sequences_with_no_boundaries}")

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] BEFORE _attention_pooling:")
        if flags.PRINT_DATA:
            print(f"  hard_boundaries.shape = {hard_boundaries.shape}")
            print(
                f"  hard_boundaries sum per sample = {hard_boundaries.sum(dim=1)}")

        pooled = self._attention_pooling(
            hard_boundaries, hidden, lengths)

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] AFTER _attention_pooling:")
        if flags.PRINT_DATA:
            print(f"  pooled.shape = {pooled.shape}")

        # ERROR CHECK: Detect if entire pooled output is empty
        # if pooled.shape[1] == 0:
        #     mode = "TRAINING" if self.training else "EVALUATION"
        #     print(f"\n{'='*80}")
        #     print(
        #         f"[BoundaryPredictor.py] ERROR: Empty pooled output (no segments) in {mode} mode!")
        #     print(f"{'='*80}")
        #     print(f"DIAGNOSTICS:")
        #     print(f"  Mode: {mode}")
        #     print(f"  pooled.shape = {pooled.shape}")
        #     print(f"  hidden.shape = {hidden.shape}")
        #     print(f"  lengths = {lengths}")
        #     print(f"  hard_boundaries.shape = {hard_boundaries.shape}")
        #     print(
        #         f"  hard_boundaries sum per sample = {hard_boundaries.sum(dim=1)}")
        #     print(f"\n  All samples boundary info:")
        #     for i in range(min(batch_size, 5)):
        #         print(
        #             f"    Sample {i}: sum = {hard_boundaries[i].sum().item()}, first 30 = {hard_boundaries[i, :30]}")
        #     print(f"\n  probs (first 30 values for first 3 samples):")
        #     for i in range(min(batch_size, 3)):
        #         print(f"    Sample {i}: {probs[i, :30]}")
        #     print(f"\n  Boundary predictor state:")
        #     print(f"    self.training = {self.training}")
        #     print(f"    self.temp = {self.temp}")
        #     print(f"    self.prior = {self.prior}")
        #     print(f"    self.similarity_bias = {self.similarity_bias.item()}")
        #     print(f"{'='*80}\n")

        # Compute shortened lengths based on boundary positions
        batch_size = hard_boundaries.shape[0]
        seq_len = hidden.shape[1]
        actual_lens = (lengths * seq_len).long()

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] Computing shortened_lengths:")
        if flags.PRINT_DATA:
            print(f"  batch_size = {batch_size}")
            print(f"  seq_len (hidden) = {seq_len}")
            print(f"  actual_lens = {actual_lens}")

        # Get the actual pooled sequence length from the pooled tensor
        max_segments = pooled.shape[1] if pooled.shape[1] > 0 else 1

        if flags.PRINT_DATA:
            print(f"  max_segments (from pooled) = {max_segments}")

        # Vectorized shortened lengths computation
        # Create valid position mask
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
                f"[BoundaryPredictor.py] shortened_lengths = {shortened_lengths}")

        # ERROR CHECK: Detect if any sequence has zero-length output
        # zero_length_mask = (shortened_lengths == 0.0)
        # if zero_length_mask.any():
        #     mode = "TRAINING" if self.training else "EVALUATION"
        #     num_zero = zero_length_mask.sum().item()
        #     print(f"\n{'='*80}")
        #     print(
        #         f"[BoundaryPredictor.py] ERROR: {num_zero} sequence(s) with zero-length output in {mode} mode!")
        #     print(f"{'='*80}")
        #     print(f"DIAGNOSTICS:")
        #     print(f"  Mode: {mode}")
        #     print(f"  pooled.shape = {pooled.shape}")
        #     print(f"  hidden.shape = {hidden.shape}")
        #     print(f"  lengths = {lengths}")
        #     print(f"  shortened_lengths = {shortened_lengths}")
        #     print(f"  zero_length_mask = {zero_length_mask}")
        #     print(f"\n  hard_boundaries.shape = {hard_boundaries.shape}")
        #     print(
        #         f"  hard_boundaries sum per sample = {hard_boundaries.sum(dim=1)}")
        #     print(f"\n  For samples with zero length:")
        #     for i in range(batch_size):
        #         if zero_length_mask[i]:
        #             valid_len = actual_lens[i].item()
        #             num_boundaries = hard_boundaries[i,
        #                                              :valid_len].sum().item()
        #             print(f"\n    Sample {i}:")
        #             print(f"      input length = {lengths[i].item():.4f}")
        #             print(f"      actual_len = {valid_len}")
        #             print(f"      num_boundaries = {num_boundaries}")
        #             print(f"      max_segments = {max_segments}")
        #             print(f"      probs[:30] = {probs[i, :30]}")
        #             print(
        #                 f"      hard_boundaries[:30] = {hard_boundaries[i, :30]}")
        #     print(f"\n  Boundary predictor state:")
        #     print(f"    self.training = {self.training}")
        #     print(f"    self.temp = {self.temp}")
        #     print(f"    self.prior = {self.prior}")
        #     print(f"    self.similarity_bias = {self.similarity_bias.item()}")
        #     print(f"{'='*80}\n")

        num_boundaries_tensor = hard_boundaries.sum()
        seq_len = hidden.shape[1]

        actual_lens = (lengths * seq_len).long()
        total_positions_tensor = actual_lens.sum().float()

        if self.training:
            per_sample_loss = self.calc_loss_target_counts(
                hard_boundaries,
                soft_boundaries,
                lengths,
                target_boundary_counts,
                reduce=False,
            ) / 10.0

            # per_sample_loss = 0.001 * self.calc_ratio_loss(
            #     hard_boundaries,
            #     soft_boundaries,
            #     lengths,
            #     target_boundary_counts,
            #     reduce=False,
            # )

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

        with torch.no_grad():
            all_spacings = []
            adjacent_count = 0
            total_boundaries = 0

            # Vectorize valid length computation and mask creation
            seq_len = hard_samples.shape[1]
            valid_lengths = torch.clamp(
                (lengths * (seq_len + 1)).long() - 1, max=seq_len)
            valid_mask = torch.arange(seq_len, device=hard_samples.device).unsqueeze(
                0) < valid_lengths.unsqueeze(1)
            masked_boundaries = hard_samples * valid_mask.float()

            # Keep loop for ragged boundary positions (variable length per sample)
            for b in range(batch_size):
                # Find positions where boundaries occur
                boundary_positions = masked_boundaries[b].nonzero(as_tuple=True)[
                    0]

                if len(boundary_positions) > 1:
                    # Calculate spacings between consecutive boundaries
                    spacings = boundary_positions[1:] - boundary_positions[:-1]
                    all_spacings.extend(spacings.cpu().tolist())

                    # Count adjacent boundaries (spacing == 1)
                    adjacent_count += (spacings == 1).sum().item()
                    # Number of gaps between boundaries
                    total_boundaries += len(boundary_positions) - 1

            # Calculate coefficient of variation (CV = std / mean)
            if len(all_spacings) > 0:
                spacings_tensor = torch.tensor(
                    all_spacings, dtype=torch.float32)
                mean_spacing = spacings_tensor.mean()
                std_spacing = spacings_tensor.std()
                if mean_spacing > 0:
                    boundary_cv = (std_spacing / mean_spacing).item()
                else:
                    boundary_cv = 0.0

            # Calculate adjacent percentage
            if total_boundaries > 0:
                boundary_adjacent_pct = (
                    adjacent_count / total_boundaries) * 100.0
            else:
                boundary_adjacent_pct = 0.0

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictor.py] RETURN:")
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
