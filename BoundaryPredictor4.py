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

    boundary_mode controls how boundaries are determined:
      - "learned": MLP predicts per-frame boundary logits (default)
      - "all": every position is a boundary (no compression, sanity check)
      - "alternating": boundary every other position (~2x compression)
    """

    def __init__(self, input_dim, prior, temp=1, boundary_mode="learned"):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.boundary_mode = boundary_mode
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

    # -------------------------------------------------------------------------
    # Pooling
    # -------------------------------------------------------------------------

    def _mean_pooling(self, boundaries, hidden):
        """Mean pooling using cumsum-based segment assignment."""
        batch_size, _, hidden_dim = hidden.shape
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

    # -------------------------------------------------------------------------
    # Boundary computation
    # -------------------------------------------------------------------------

    def _apply_length_mask(self, boundaries, lengths, seq_len, batch_size):
        """Mask out positions beyond actual lengths and force last-valid boundary."""
        device = boundaries.device
        actual_lens = (lengths * seq_len).long()
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = (pos_idx < actual_lens.unsqueeze(1)).float()

        boundaries = boundaries * valid_mask

        # Force boundary at last valid position to close the final segment
        last_valid_idx = torch.clamp(actual_lens - 1, min=0, max=seq_len - 1)
        batch_idx = torch.arange(batch_size, device=device)
        boundaries[batch_idx, last_valid_idx] = 1.0

        return boundaries, valid_mask

    def _compute_learned_boundaries(self, hidden, lengths):
        """Compute boundaries using learned MLP + RelaxedBernoulli sampling."""
        batch_size, seq_len, _ = hidden.shape

        hidden_dropped = self.dropout(hidden)
        logits = self.boundary_mlp(hidden_dropped).squeeze(-1)  # (B, L)
        probs = torch.sigmoid(logits)

        if self.training:
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp, probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
        else:
            soft_boundaries = probs

        hard_samples = (soft_boundaries > 0.5).float()

        # Mask by lengths
        soft_boundaries, valid_mask = self._apply_length_mask(
            soft_boundaries, lengths, seq_len, batch_size)
        hard_samples, _ = self._apply_length_mask(
            hard_samples, lengths, seq_len, batch_size)
        masked_probs = probs * valid_mask

        # STE
        hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

        return hard_boundaries, hard_samples, masked_probs

    def _compute_forced_boundaries(self, hidden, lengths, every_n=1):
        """Compute deterministic boundaries (no learning).

        every_n=1: boundary at every position (no compression)
        every_n=2: boundary every other position (~2x compression)
        """
        batch_size, seq_len, _ = hidden.shape
        device = hidden.device

        if every_n == 1:
            hard_boundaries = torch.ones(batch_size, seq_len, device=device)
        else:
            hard_boundaries = torch.zeros(batch_size, seq_len, device=device)
            # [0, 1, 0, 1, ...] pattern — boundary at positions 1, 3, 5, ...
            hard_boundaries[:, every_n - 1::every_n] = 1.0

        hard_boundaries, _ = self._apply_length_mask(
            hard_boundaries, lengths, seq_len, batch_size)

        # No learned probs for forced modes
        probs = hard_boundaries.clone()

        return hard_boundaries, hard_boundaries, probs

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def _binomial_loss(self, hard_boundaries, lengths):
        """Fixed-prior binomial loss."""
        seq_len = hard_boundaries.shape[1]
        actual_lens = (lengths * seq_len).long()
        binomial = torch.distributions.binomial.Binomial(
            total_count=actual_lens.float(),
            probs=torch.tensor([self.prior], device=hard_boundaries.device),
        )
        num_boundaries = hard_boundaries.sum(dim=1)
        return -binomial.log_prob(num_boundaries) / actual_lens.float().clamp(min=1)

    # -------------------------------------------------------------------------
    # Eval-only statistics
    # -------------------------------------------------------------------------

    def _compute_eval_stats(self, hard_samples, batch_size):
        """Compute boundary spacing CV and adjacency % (eval only)."""
        boundary_cv = None
        boundary_adjacent_pct = None

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
            boundary_adjacent_pct = (
                adjacent_count / total_boundary_pairs * 100.0
            ) if total_boundary_pairs > 0 else 0.0

        return boundary_cv, boundary_adjacent_pct

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        hidden,
        lengths,
        target_boundary_counts=None,  # noqa: ARG002 - unused, kept for interface compat
        return_unreduced_boundary_loss=False,
    ):
        batch_size, seq_len, _ = hidden.shape

        # --- Compute boundaries based on mode ---
        if self.boundary_mode == "all":
            hard_boundaries, hard_samples, masked_probs = (
                self._compute_forced_boundaries(hidden, lengths, every_n=1))
        elif self.boundary_mode == "alternating":
            hard_boundaries, hard_samples, masked_probs = (
                self._compute_forced_boundaries(hidden, lengths, every_n=2))
        else:  # "learned"
            hard_boundaries, hard_samples, masked_probs = (
                self._compute_learned_boundaries(hidden, lengths))

        # --- Pooling ---
        if self.boundary_mode == "all":
            # All boundaries = identity (each frame is its own segment).
            # Skip _mean_pooling to avoid O(B*L*L) memory.
            pooled = hidden
        else:
            pooled = self._mean_pooling(hard_boundaries, hidden)

        # --- Lengths ---
        max_segments = pooled.shape[1] if pooled.shape[1] > 0 else 1
        num_boundaries_per_sample = hard_boundaries.sum(dim=1)
        shortened_lengths = num_boundaries_per_sample / max_segments

        # --- Stats ---
        actual_lens = (lengths * seq_len).long()
        num_boundaries = hard_boundaries.sum().item()
        total_positions = actual_lens.sum().float().item()
        total_positions_per_sample = actual_lens.float()

        # --- Loss (only for learned mode during training) ---
        if self.training and self.boundary_mode == "learned":
            per_sample_loss = self._binomial_loss(hard_boundaries, lengths)
            loss = per_sample_loss if return_unreduced_boundary_loss else per_sample_loss.mean()
        else:
            loss = torch.tensor(0.0, device=hidden.device)
            if return_unreduced_boundary_loss:
                loss = loss.repeat(batch_size)

        # --- Eval-only stats ---
        boundary_cv = None
        boundary_adjacent_pct = None
        if not self.training:
            boundary_cv, boundary_adjacent_pct = self._compute_eval_stats(
                hard_samples, batch_size)

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
