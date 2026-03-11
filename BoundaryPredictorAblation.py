"""Ablation variants of BoundaryPredictor2.

Drop-in replacement with configurable boundary detection and pooling strategies.
Same forward() return signature (10-tuple) so pretrain.py works unchanged.

Boundary modes:
    learned     - Cosine similarity between adjacent frames (original)
    fixed_width - Boundary every N frames (no learnable boundary params)
    all         - Every position is a boundary (no compression from BP)
    mlp         - Per-frame MLP binary classifier (no adjacent-frame comparison)

Pooling modes:
    attention          - Multi-head attention with learned query (original)
    mean               - Simple mean pooling per segment
    attention_no_value - Attention weights from Q/K, but raw hidden as values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import binomial_loss_from_target_counts
from utils import common
import flags


class BoundaryPredictorAblation(nn.Module):
    def __init__(
        self,
        input_dim,
        prior,
        temp=1,
        similarity_bias=0.0,
        boundary_mode="learned",
        pooling_mode="attention",
        fixed_width=5,
        disable_temp_schedule=False,
    ):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.compression_schedule = 1.0
        self.target_prior = prior
        self.boundary_mode = boundary_mode
        self.pooling_mode = pooling_mode
        self.fixed_width = fixed_width
        self.disable_temp_schedule = disable_temp_schedule

        self.dropout = nn.Dropout(p=0.1)

        # --- Boundary detection modules (only for modes that need them) ---
        if boundary_mode == "learned":
            self.boundary_mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, input_dim),
            )
            self.q_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
            self.k_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
            self.similarity_bias = nn.Parameter(torch.tensor(similarity_bias))

            with torch.no_grad():
                self.q_proj_layer.weight.copy_(torch.eye(input_dim))
                self.k_proj_layer.weight.copy_(torch.eye(input_dim))
            self.q_proj_layer.weight._no_reinit = True
            self.k_proj_layer.weight._no_reinit = True

        elif boundary_mode == "mlp":
            self.boundary_classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.GELU(),
                nn.Linear(input_dim, 1),
            )

        # --- Pooling modules (only for modes that need them) ---
        if pooling_mode in ("attention", "attention_no_value"):
            self.num_heads = 8
            self.head_dim = input_dim // self.num_heads
            assert input_dim % self.num_heads == 0

            self.learned_query = nn.Parameter(torch.randn(input_dim))
            self.pool_key = nn.Linear(input_dim, input_dim, bias=False)
            self.pool_output = nn.Linear(input_dim, input_dim, bias=False)
            self.pool_layernorm = nn.LayerNorm(input_dim)
            self.pool_scale = self.head_dim ** -0.5

            with torch.no_grad():
                self.pool_key.weight.copy_(torch.eye(input_dim))
                self.pool_output.weight.copy_(torch.eye(input_dim))
            self.pool_key.weight._no_reinit = True
            self.pool_output.weight._no_reinit = True

            if pooling_mode == "attention":
                self.pool_value = nn.Linear(input_dim, input_dim, bias=False)
                with torch.no_grad():
                    self.pool_value.weight.copy_(torch.eye(input_dim))
                self.pool_value.weight._no_reinit = True

    # ------------------------------------------------------------------
    # Configuration methods (same API as BoundaryPredictor2)
    # ------------------------------------------------------------------

    def set_prior(self, prior):
        self.prior = prior

    def set_temperature(self, temp):
        if not self.disable_temp_schedule:
            self.temp = temp

    def set_compression_schedule(self, schedule_value):
        self.compression_schedule = float(schedule_value)

    def get_scheduled_prior(self):
        schedule = self.compression_schedule
        target = self.target_prior
        if abs(target - 1.0) < 1e-8:
            return 1.0
        target_compression = 1.0 / target
        start_compression = 2.0
        current_compression = start_compression + (target_compression - start_compression) * schedule
        return 1.0 / current_compression

    # ------------------------------------------------------------------
    # Boundary detection strategies
    # ------------------------------------------------------------------

    def _learned_boundaries(self, hidden, lengths):
        """Cosine similarity between adjacent frames — identical to BoundaryPredictor2."""
        batch_size, seq_len, _ = hidden.shape

        hidden_dropped = self.dropout(hidden)
        q_input = hidden_dropped[:, :-1]
        k_input = hidden_dropped[:, 1:]

        q_normed = F.normalize(q_input, dim=-1, eps=1e-8)
        k_normed = F.normalize(k_input, dim=-1, eps=1e-8)

        q_mlp_out = self.boundary_mlp(q_normed)
        q_hidden = self.q_proj_layer(
            F.normalize(q_mlp_out + q_normed, dim=-1, eps=1e-8)
        )

        k_mlp_out = self.boundary_mlp(k_normed)
        k_hidden = self.k_proj_layer(
            F.normalize(k_mlp_out + k_normed, dim=-1, eps=1e-8)
        )

        cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)

        if flags.PRINT_NAN_INF and torch.isnan(cos_sim).any():
            print("[BoundaryPredictorAblation] WARNING: NaN in cos_sim")

        probs = torch.clamp(
            (1.0 - (cos_sim + self.similarity_bias)) * 0.5, min=0.0, max=1.0
        )
        probs = F.pad(probs, (0, 1), value=0.0)

        if self.training:
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp, probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
            hard_samples = (soft_boundaries > 0.5).float()
        else:
            soft_boundaries = probs
            hard_samples = (probs > 0.5).float()

        # Mask based on lengths
        boundary_seq_len = soft_boundaries.shape[1]
        actual_lens = (lengths * (boundary_seq_len + 1)).long()
        valid_lens = torch.clamp(actual_lens - 1, min=0, max=boundary_seq_len)
        pos_idx = torch.arange(boundary_seq_len, device=hidden.device).unsqueeze(0)
        valid_mask = (pos_idx < valid_lens.unsqueeze(1)).float()

        soft_boundaries = soft_boundaries * valid_mask
        hard_samples = hard_samples * valid_mask
        masked_probs = probs * valid_mask

        # Set boundary at first padding position
        needs_boundary_mask = valid_lens < boundary_seq_len
        if needs_boundary_mask.any():
            batch_idx = torch.arange(batch_size, device=hidden.device)[needs_boundary_mask]
            first_padding_idx = valid_lens[needs_boundary_mask]
            soft_boundaries[batch_idx, first_padding_idx] = 1.0
            hard_samples[batch_idx, first_padding_idx] = 1.0

        hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

        # Emergency: ensure at least one boundary per sequence
        no_boundaries = hard_boundaries.sum(dim=1) == 0
        if no_boundaries.any():
            indices = no_boundaries.nonzero(as_tuple=True)[0]
            emergency_lens = torch.clamp(actual_lens[indices] - 1, min=0, max=boundary_seq_len)
            emergency_pos = torch.clamp(emergency_lens, min=0, max=boundary_seq_len - 1)
            valid_emergency = emergency_pos >= 0
            if valid_emergency.any():
                hard_boundaries[indices[valid_emergency], emergency_pos[valid_emergency]] = 1.0
            if self.training:
                print(
                    f"[BoundaryPredictorAblation] WARNING: emergency boundary for "
                    f"{len(indices)} sequence(s) during TRAINING"
                )

        return hard_boundaries, soft_boundaries, masked_probs

    def _fixed_width_boundaries(self, hidden, lengths):
        """Place a boundary every ``fixed_width`` frames."""
        batch_size, seq_len, _ = hidden.shape
        device = hidden.device
        stride = max(1, self.fixed_width)

        hard_boundaries = torch.zeros(batch_size, seq_len, device=device)
        positions = torch.arange(stride - 1, seq_len, stride, device=device)
        if len(positions) > 0:
            hard_boundaries[:, positions] = 1.0

        # Mask based on lengths
        actual_lens = (lengths * seq_len).long()
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = (pos_idx < actual_lens.unsqueeze(1)).float()
        hard_boundaries = hard_boundaries * valid_mask

        # Ensure boundary at last valid position of each sequence
        last_valid = (actual_lens - 1).clamp(min=0, max=seq_len - 1)
        batch_idx = torch.arange(batch_size, device=device)
        hard_boundaries[batch_idx, last_valid] = 1.0

        return hard_boundaries, hard_boundaries.clone(), hard_boundaries.clone()

    def _all_boundaries(self, hidden, lengths):
        """Every valid position is a boundary — no compression from BP."""
        batch_size, seq_len, _ = hidden.shape
        device = hidden.device

        actual_lens = (lengths * seq_len).long()
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        hard_boundaries = (pos_idx < actual_lens.unsqueeze(1)).float()

        return hard_boundaries, hard_boundaries.clone(), hard_boundaries.clone()

    def _mlp_boundaries(self, hidden, lengths):
        """Per-frame MLP binary classifier."""
        batch_size, seq_len, _ = hidden.shape
        device = hidden.device

        hidden_dropped = self.dropout(hidden)
        logits = self.boundary_classifier(hidden_dropped).squeeze(-1)  # (B, L)
        probs = torch.sigmoid(logits)

        if self.training:
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp, probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
            hard_samples = (soft_boundaries > 0.5).float()
        else:
            soft_boundaries = probs
            hard_samples = (probs > 0.5).float()

        # Mask based on lengths
        actual_lens = (lengths * seq_len).long()
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = (pos_idx < actual_lens.unsqueeze(1)).float()

        soft_boundaries = soft_boundaries * valid_mask
        hard_samples = hard_samples * valid_mask
        masked_probs = probs * valid_mask

        # Ensure boundary at last valid position
        last_valid = (actual_lens - 1).clamp(min=0, max=seq_len - 1)
        batch_idx = torch.arange(batch_size, device=device)
        soft_boundaries[batch_idx, last_valid] = 1.0
        hard_samples[batch_idx, last_valid] = 1.0

        hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

        # Emergency: ensure at least one boundary per sequence
        no_boundaries = hard_boundaries.sum(dim=1) == 0
        if no_boundaries.any():
            indices = no_boundaries.nonzero(as_tuple=True)[0]
            emergency_pos = (actual_lens[indices] - 1).clamp(min=0, max=seq_len - 1)
            hard_boundaries[indices, emergency_pos] = 1.0
            if self.training:
                print(
                    f"[BoundaryPredictorAblation] WARNING: MLP emergency boundary "
                    f"for {len(indices)} sequence(s)"
                )

        return hard_boundaries, soft_boundaries, masked_probs

    # ------------------------------------------------------------------
    # Pooling strategies
    # ------------------------------------------------------------------

    def _attention_pooling(self, boundaries, hidden, lengths):
        """Multi-head attention pooling — identical to BoundaryPredictor2."""
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        foo = common(boundaries)
        if foo is None:
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)
        segment_mask = (foo == 0).float()

        actual_lens = (lengths * seq_len).long()
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        length_mask = (pos_indices < actual_lens.unsqueeze(1)).float()
        segment_mask = segment_mask * length_mask.unsqueeze(-1)

        queries = self.learned_query.unsqueeze(0).unsqueeze(0).expand(
            batch_size, max_segments, -1
        )

        hidden_normed = self.pool_layernorm(hidden)
        keys = self.pool_key(hidden_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        values = self.pool_value(hidden_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        queries = queries.view(
            batch_size, max_segments, self.num_heads, self.head_dim
        )

        attn_scores = (
            torch.einsum("bshd,blhd->bhsl", queries, keys) * self.pool_scale
        )

        segment_mask_t = segment_mask.permute(0, 2, 1).unsqueeze(1)
        attn_scores = attn_scores.masked_fill(segment_mask_t == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        pooled = torch.einsum("bhsl,blhd->bshd", attn_weights, values)
        pooled = pooled.reshape(batch_size, max_segments, -1)
        pooled = self.pool_output(pooled)

        return pooled.to(dtype=dtype)

    def _mean_pooling(self, boundaries, hidden, lengths):
        """Simple mean pooling within each segment."""
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        foo = common(boundaries)
        if foo is None:
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)
        segment_mask = (foo == 0).float()

        actual_lens = (lengths * seq_len).long()
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        length_mask = (pos_indices < actual_lens.unsqueeze(1)).float()
        segment_mask = segment_mask * length_mask.unsqueeze(-1)

        segment_counts = segment_mask.sum(dim=1).clamp(min=1.0)  # (B, S)
        pooled = torch.bmm(segment_mask.permute(0, 2, 1), hidden)  # (B, S, D)
        pooled = pooled / segment_counts.unsqueeze(-1)

        return pooled.to(dtype=dtype)

    def _attention_no_value_pooling(self, boundaries, hidden, lengths):
        """Attention weights from Q/K, but raw hidden states as values."""
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        foo = common(boundaries)
        if foo is None:
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)
        segment_mask = (foo == 0).float()

        actual_lens = (lengths * seq_len).long()
        pos_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        length_mask = (pos_indices < actual_lens.unsqueeze(1)).float()
        segment_mask = segment_mask * length_mask.unsqueeze(-1)

        queries = self.learned_query.unsqueeze(0).unsqueeze(0).expand(
            batch_size, max_segments, -1
        )

        hidden_normed = self.pool_layernorm(hidden)
        keys = self.pool_key(hidden_normed).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        # Use raw hidden (after layernorm) as values — no pool_value projection
        values = hidden_normed.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        queries = queries.view(
            batch_size, max_segments, self.num_heads, self.head_dim
        )

        attn_scores = (
            torch.einsum("bshd,blhd->bhsl", queries, keys) * self.pool_scale
        )

        segment_mask_t = segment_mask.permute(0, 2, 1).unsqueeze(1)
        attn_scores = attn_scores.masked_fill(segment_mask_t == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        pooled = torch.einsum("bhsl,blhd->bshd", attn_weights, values)
        pooled = pooled.reshape(batch_size, max_segments, -1)
        pooled = self.pool_output(pooled)

        return pooled.to(dtype=dtype)

    # ------------------------------------------------------------------
    # Forward — same 10-tuple return as BoundaryPredictor2
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden,
        lengths,
        target_boundary_counts=None,
        return_unreduced_boundary_loss=False,
    ):
        batch_size, seq_len, _ = hidden.shape

        if flags.PRINT_FLOW:
            print(f"[BoundaryPredictorAblation] forward() INPUT:")
        if flags.PRINT_DATA:
            print(f"  hidden.shape = {hidden.shape}")
            print(f"  lengths = {lengths}")
            print(f"  boundary_mode = {self.boundary_mode}")
            print(f"  pooling_mode = {self.pooling_mode}")

        # === Step 1: Compute boundaries ===
        if self.boundary_mode == "learned":
            hard_boundaries, soft_boundaries, masked_probs = (
                self._learned_boundaries(hidden, lengths)
            )
        elif self.boundary_mode == "fixed_width":
            hard_boundaries, soft_boundaries, masked_probs = (
                self._fixed_width_boundaries(hidden, lengths)
            )
        elif self.boundary_mode == "all":
            hard_boundaries, soft_boundaries, masked_probs = (
                self._all_boundaries(hidden, lengths)
            )
        elif self.boundary_mode == "mlp":
            hard_boundaries, soft_boundaries, masked_probs = (
                self._mlp_boundaries(hidden, lengths)
            )
        else:
            raise ValueError(f"Unknown boundary_mode: {self.boundary_mode}")

        if flags.PRINT_DATA:
            print(f"  hard_boundaries sum/sample = {hard_boundaries.sum(dim=1)}")

        # === Step 2: Pool ===
        if self.pooling_mode == "attention":
            pooled = self._attention_pooling(hard_boundaries, hidden, lengths)
        elif self.pooling_mode == "mean":
            pooled = self._mean_pooling(hard_boundaries, hidden, lengths)
        elif self.pooling_mode == "attention_no_value":
            pooled = self._attention_no_value_pooling(hard_boundaries, hidden, lengths)
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")

        if flags.PRINT_DATA:
            print(f"  pooled.shape = {pooled.shape}")

        # === Step 3: Shortened lengths ===
        seq_len = hidden.shape[1]
        actual_lens = (lengths * seq_len).long()
        max_segments = pooled.shape[1] if pooled.shape[1] > 0 else 1

        boundary_seq_len = hard_boundaries.shape[1]
        pos_mask = (
            torch.arange(boundary_seq_len, device=hidden.device).unsqueeze(0)
            < actual_lens.unsqueeze(1)
        )
        num_boundaries_per_sample = (hard_boundaries * pos_mask.float()).sum(dim=1)

        shortened_lengths = torch.zeros(batch_size, device=hidden.device)
        if max_segments > 0:
            case1 = num_boundaries_per_sample >= (max_segments - 1)
            shortened_lengths[case1] = 1.0
            case2 = (num_boundaries_per_sample > 0) & (
                num_boundaries_per_sample < (max_segments - 1)
            )
            shortened_lengths[case2] = (
                num_boundaries_per_sample[case2] + 1
            ) / max_segments

        # === Step 4: Aggregate stats ===
        num_boundaries_tensor = hard_boundaries.sum()
        total_positions_tensor = actual_lens.sum().float()
        total_positions_per_sample = actual_lens.float()

        # === Step 5: Boundary loss ===
        if self.training and self.boundary_mode in ("learned", "mlp"):
            per_sample_loss = self.calc_loss_target_counts(
                hard_boundaries,
                soft_boundaries,
                lengths,
                target_boundary_counts,
                reduce=False,
            )
            loss = per_sample_loss if return_unreduced_boundary_loss else per_sample_loss.mean()
        else:
            loss = torch.tensor(0.0, device=hidden.device)
            if return_unreduced_boundary_loss:
                loss = loss.repeat(batch_size)

        num_boundaries = num_boundaries_tensor.item()
        total_positions = total_positions_tensor.item()

        # === Step 6: Eval statistics ===
        boundary_cv = None
        boundary_adjacent_pct = None

        if not self.training:
            with torch.no_grad():
                all_spacings = []
                adjacent_count = 0
                total_gaps = 0

                hb_len = hard_boundaries.shape[1]
                valid_lengths = torch.clamp(
                    (lengths * (hb_len + 1)).long() - 1, max=hb_len
                )
                valid_mask = (
                    torch.arange(hb_len, device=hidden.device).unsqueeze(0)
                    < valid_lengths.unsqueeze(1)
                )
                masked_hb = hard_boundaries * valid_mask.float()

                for b in range(batch_size):
                    bp = masked_hb[b].nonzero(as_tuple=True)[0]
                    if len(bp) > 1:
                        spacings = bp[1:] - bp[:-1]
                        all_spacings.extend(spacings.cpu().tolist())
                        adjacent_count += (spacings == 1).sum().item()
                        total_gaps += len(bp) - 1

                if all_spacings:
                    st = torch.tensor(all_spacings, dtype=torch.float32)
                    mean_s = st.mean()
                    std_s = st.std()
                    boundary_cv = (std_s / mean_s).item() if mean_s > 0 else 0.0

                boundary_adjacent_pct = (
                    (adjacent_count / total_gaps) * 100.0 if total_gaps > 0 else 0.0
                )

        return (
            pooled,
            loss,
            num_boundaries,
            total_positions,
            shortened_lengths,
            boundary_cv,
            boundary_adjacent_pct,
            masked_probs,
            num_boundaries_per_sample,
            total_positions_per_sample,
        )

    # ------------------------------------------------------------------
    # Loss (same as BoundaryPredictor2)
    # ------------------------------------------------------------------

    def calc_loss_target_counts(
        self,
        hard_boundaries,
        soft_boundaries,
        lengths,
        target_boundary_counts,
        reduce=True,
    ):
        if flags.PRINT_BP_LOSS_CHECKS:
            print("[BoundaryPredictorAblation calc_loss] Running checks...")

        device = hard_boundaries.device
        per_item_boundaries = hard_boundaries.sum(dim=1)

        seq_len = hard_boundaries.shape[1] + 1
        actual_lens = (lengths * seq_len).long()
        per_item_totals = actual_lens.float().to(dtype=torch.float32)

        target_boundary_counts = target_boundary_counts.to(
            device=device, dtype=torch.float32,
        )

        loss_values = binomial_loss_from_target_counts(
            per_item_boundaries.to(dtype=torch.float32),
            per_item_totals,
            target_boundary_counts,
        )

        return loss_values.mean() if reduce else loss_values
