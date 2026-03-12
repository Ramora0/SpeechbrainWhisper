import torch
import torch.nn as nn


def _common(boundaries):
    """
    Cumsum-based segment assignment. Returns foo [B, L, S] where
    foo[b,t,s] == 0 when position t belongs to segment s.
    """
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if n_segments == 0:
        return None

    segment_indices = torch.arange(n_segments, device=boundaries.device)
    hh1 = boundaries.cumsum(1) - boundaries
    foo = segment_indices.view(1, 1, -1) - hh1.unsqueeze(-1)  # B x L x S
    return foo


class BoundaryPredictor4(nn.Module):
    """
    Simplified BoundaryPredictor with per-frame MLP boundary prediction
    and mean pooling.
    """

    def __init__(self, input_dim, prior, temp=1):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.compression_schedule = 1.0
        self.target_prior = prior

        # Per-frame MLP: frame -> scalar boundary logit
        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )

        self.dropout = nn.Dropout(p=0.1)

    def set_prior(self, prior):
        self.prior = prior

    def set_temperature(self, temp):
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

    def _mean_pooling(self, boundaries, hidden):
        """Mean pooling using cumsum-based segment assignment (a la dynamic-pooling)."""
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        foo = _common(boundaries)  # B x L x S
        if foo is None:
            return torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)

        # Build normalized segment weights: 1-foo where in-segment, 0 elsewhere
        bar = 1 - foo
        bar[foo != 0] = 0
        bar = bar / (bar.sum(dim=1, keepdim=True) + 1e-9)

        pooled = torch.einsum('bls,bld->bsd', bar, hidden)
        return pooled.to(dtype=dtype)

    def forward(
        self,
        hidden,
        lengths,
        target_boundary_counts=None,  # unused, kept for interface compat
        return_unreduced_boundary_loss=False,
    ):
        batch_size, seq_len, _ = hidden.shape

        hidden_dropped = self.dropout(hidden)

        # Per-frame boundary logits
        logits = self.boundary_mlp(hidden_dropped).squeeze(-1)  # (B, L)
        probs = torch.sigmoid(logits)

        if self.training:
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
        else:
            soft_boundaries = probs
        hard_samples = (soft_boundaries > 0.5).float()

        # Mask boundaries based on lengths
        boundary_seq_len = soft_boundaries.shape[1]
        actual_lens = (lengths * boundary_seq_len).long()
        pos_idx = torch.arange(boundary_seq_len, device=hidden.device).unsqueeze(0)
        valid_mask = (pos_idx < actual_lens.unsqueeze(1)).float()

        soft_boundaries = soft_boundaries * valid_mask
        hard_samples = hard_samples * valid_mask
        masked_probs = probs * valid_mask

        # Set boundary at last valid position to close the final segment
        last_valid_idx = torch.clamp(actual_lens - 1, min=0, max=boundary_seq_len - 1)
        batch_idx = torch.arange(batch_size, device=hidden.device)
        soft_boundaries[batch_idx, last_valid_idx] = 1.0
        hard_samples[batch_idx, last_valid_idx] = 1.0

        hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

        pooled = self._mean_pooling(hard_boundaries, hidden)

        # Compute shortened lengths
        max_segments = pooled.shape[1] if pooled.shape[1] > 0 else 1
        num_boundaries_per_sample = hard_boundaries.sum(dim=1)
        shortened_lengths = num_boundaries_per_sample / max_segments

        # Loss
        actual_lens = (lengths * seq_len).long()
        num_boundaries = hard_boundaries.sum().item()
        total_positions = actual_lens.sum().float().item()
        total_positions_per_sample = actual_lens.float()

        if self.training:
            per_sample_loss = self._binomial_loss(hard_boundaries, lengths)
            loss = per_sample_loss if return_unreduced_boundary_loss else per_sample_loss.mean()
        else:
            loss = torch.tensor(0.0, device=hidden.device)
            if return_unreduced_boundary_loss:
                loss = loss.repeat(batch_size)

        # --- Eval-only boundary statistics (CV, adjacency) ---
        boundary_cv = None
        boundary_adjacent_pct = None
        if not self.training:
            with torch.no_grad():
                all_spacings = []
                adjacent_count = 0
                total_boundary_pairs = 0
                for b in range(batch_size):
                    bp = hard_samples[b].nonzero(as_tuple=True)[0]
                    if len(bp) > 1:
                        spacings = bp[1:] - bp[:-1]
                        all_spacings.extend(spacings.cpu().tolist())
                        adjacent_count += (spacings == 1).sum().item()
                        total_boundary_pairs += len(bp) - 1
                if all_spacings:
                    st = torch.tensor(all_spacings, dtype=torch.float32)
                    m = st.mean()
                    boundary_cv = (st.std() / m).item() if m > 0 else 0.0
                boundary_adjacent_pct = (adjacent_count / total_boundary_pairs * 100.0) if total_boundary_pairs > 0 else 0.0
        # --- End eval-only statistics ---

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

    def _binomial_loss(self, hard_boundaries, lengths):
        """Fixed-prior binomial loss (a la dynamic-pooling)."""
        seq_len = hard_boundaries.shape[1]
        actual_lens = (lengths * seq_len).long()
        binomial = torch.distributions.binomial.Binomial(
            total_count=actual_lens.float(),
            probs=torch.tensor([self.prior], device=hard_boundaries.device),
        )
        num_boundaries = hard_boundaries.sum(dim=1)
        return -binomial.log_prob(num_boundaries) / actual_lens.float().clamp(min=1)
