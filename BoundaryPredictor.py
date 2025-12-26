import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from loss import binomial_loss_from_target_counts
from utils import get_sinusoidal_positional_embeddings, common


class BoundaryPredictor2(nn.Module):
    def __init__(self, input_dim, prior, temp=1, threshold=0.5):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.threshold = threshold
        self.compression_schedule = 1.0
        self.target_prior = prior

        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )

        self.q_proj_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj_layer = nn.Linear(input_dim, input_dim, bias=False)

        self.similarity_bias = nn.Parameter(torch.tensor(-0.126))

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

    def _add_positional_embeddings(self, x):
        pos_embeds = get_sinusoidal_positional_embeddings(x)
        return x + pos_embeds

    def _attention_pooling(self, boundaries, hidden, attention_mask=None):
        """
        Multi-head attention-based pooling using query matrix applied to boundary positions.

        Args:
            boundaries: (B, L) - binary boundary indicators
            hidden: (B, L, D) - hidden states
            attention_mask: (B, L) - attention mask

        Returns:
            pooled: (B, S, D) - pooled segment representations
        """
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        # Step 1: Create segment assignment matrix using existing logic
        foo = common(boundaries)  # B x L x S

        if foo is None:
            # No boundaries found
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        max_segments = foo.size(2)  # S

        # Step 2: Create binary segment mask (B x L x S)
        segment_mask = (foo == 0).float()  # B x L x S

        if attention_mask is not None:
            # Apply attention mask: (B, L, 1) * (B, L, S) -> (B, L, S)
            segment_mask = segment_mask * attention_mask.unsqueeze(-1)

        # Step 3: Use learned query vector for all segments
        # Expand learned query to (B, S, D) - same query for all segments in all batches
        queries = self.learned_query.unsqueeze(0).unsqueeze(
            0).expand(batch_size, max_segments, -1)  # (B, S, D)

        # Step 4: Reshape queries for multi-head attention
        queries = queries.view(batch_size, max_segments, self.num_heads,
                               # (B, H, S, head_dim)
                               self.head_dim).transpose(1, 2)

        # Step 5: Apply LayerNorm before projecting to keys and values
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

        return pooled  # B x S x D

    def forward(
        self,
        hidden,
        attention_mask=None,
        target_boundary_counts=None,
        return_unreduced_boundary_loss=False,
    ):
        batch_size = hidden.size(0)

        q_input = F.normalize(self.dropout(hidden[:, :-1]), dim=-1)
        q_mlp_out = self.boundary_mlp(q_input)
        q_residual = q_mlp_out + q_input  # Residual connection
        q_hidden = self.q_proj_layer(F.normalize(q_residual, dim=-1))

        k_input = F.normalize(self.dropout(hidden[:, 1:]), dim=-1)
        k_mlp_out = self.boundary_mlp(k_input)
        k_residual = k_mlp_out + k_input  # Residual connection
        k_hidden = self.k_proj_layer(F.normalize(k_residual, dim=-1))

        cos_sim = torch.einsum("bld,bld->bl", q_hidden, k_hidden)
        probs = torch.clamp(
            (1 - (cos_sim + self.similarity_bias)) * 0.5, min=0.0, max=1.0)
        probs = F.pad(probs, (0, 1), value=0.0)
        if self.training:
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
            hard_samples = (soft_boundaries > self.threshold).float()
        else:
            # During evaluation, threshold probabilities directly without sampling
            soft_boundaries = probs
            hard_samples = (probs > self.threshold).float()

        if attention_mask is not None:
            soft_boundaries = soft_boundaries * attention_mask
            hard_samples = hard_samples * attention_mask

            pad_mask = attention_mask == 0
            if pad_mask.any():
                first_pad_mask = pad_mask & (
                    pad_mask.long().cumsum(dim=1) == 1)
                last_real_mask = torch.roll(first_pad_mask, shifts=-1, dims=1)
                last_real_mask[:, -1] = False
                boundary_mask = last_real_mask.float()
                hard_samples = torch.maximum(hard_samples, boundary_mask)
                soft_boundaries = torch.maximum(soft_boundaries, boundary_mask)

        hard_boundaries = (
            hard_samples - soft_boundaries.detach() + soft_boundaries
        )

        pooled = self._attention_pooling(
            hard_boundaries, hidden, attention_mask)
        pooled = self._add_positional_embeddings(pooled)

        shortened_attention_mask = None
        if attention_mask is not None:
            keep_mask = hard_boundaries == 1
            batch_size = attention_mask.shape[0]
            shortened_masks = []

            for b in range(batch_size):
                keep_indices = keep_mask[b].nonzero(as_tuple=True)[0]
                original_mask = attention_mask[b]
                shortened_mask = original_mask[keep_indices]
                shortened_masks.append(shortened_mask)

            shortened_attention_mask = pad_sequence(
                shortened_masks, batch_first=True, padding_value=0.0
            )

        num_boundaries_tensor = hard_boundaries.sum()
        if attention_mask is not None:
            total_positions_tensor = attention_mask.sum()
        else:
            total_positions_tensor = torch.tensor(
                hard_boundaries.numel(),
                device=hard_boundaries.device,
                dtype=torch.float,
            )

        if self.training:
            per_sample_loss = 10. * self.calc_loss_target_counts(
                hard_boundaries,
                attention_mask,
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

        with torch.no_grad():
            all_spacings = []
            adjacent_count = 0
            total_boundaries = 0

            for b in range(batch_size):
                # Get boundary positions for this sample
                if attention_mask is not None:
                    valid_length = int(attention_mask[b].sum().item())
                    boundaries_b = hard_samples[b, :valid_length]
                else:
                    boundaries_b = hard_samples[b]

                # Find positions where boundaries occur
                boundary_positions = boundaries_b.nonzero(as_tuple=True)[0]

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

        return (
            pooled,
            loss,
            num_boundaries,
            total_positions,
            shortened_attention_mask,
            boundary_cv,
            boundary_adjacent_pct,
        )

    def calc_loss_target_counts(
        self,
        hard_boundaries,
        attention_mask,
        target_boundary_counts,
        reduce=True,
    ):
        device = hard_boundaries.device
        per_item_boundaries = hard_boundaries.sum(dim=1)

        if attention_mask is not None:
            per_item_totals = attention_mask.sum(dim=1)
        else:
            per_item_totals = torch.full(
                (hard_boundaries.size(0),),
                hard_boundaries.size(1),
                device=device,
                dtype=torch.float32,
            )

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
            return loss_values.mean()
        return loss_values
