"""Forced-all BoundaryPredictor: every valid position is a boundary (no BP compression).

Drop-in replacement for BoundaryPredictor2 — same forward() 10-tuple return.
Useful as an ablation baseline: only CNN compression applies.
"""

import torch
import torch.nn as nn

from utils import common


class BoundaryPredictorAll(nn.Module):
    def __init__(self, input_dim, prior=1.0, temp=1, **kwargs):
        super().__init__()
        self.prior = prior
        self.temp = temp

    def set_prior(self, prior):
        self.prior = prior

    def set_temperature(self, temp):
        self.temp = temp

    def set_compression_schedule(self, schedule_value):
        pass

    def get_scheduled_prior(self):
        return 1.0

    def forward(self, hidden, lengths, target_boundary_counts=None,
                return_unreduced_boundary_loss=False):
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        actual_lens = (lengths * seq_len).long()
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = (pos_idx < actual_lens.unsqueeze(1)).float()

        # Every valid position is a boundary
        hard_boundaries = valid_mask

        # --- Mean pooling ---
        foo = common(hard_boundaries)
        if foo is None:
            pooled = torch.empty(batch_size, 0, hidden_dim, device=device, dtype=dtype)
        else:
            segment_mask = (foo == 0).float() * valid_mask.unsqueeze(-1)
            segment_counts = segment_mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.bmm(segment_mask.permute(0, 2, 1), hidden)
            pooled = (pooled / segment_counts.unsqueeze(-1)).to(dtype=dtype)

        # --- Shortened lengths ---
        max_segments = pooled.shape[1] if pooled.shape[1] > 0 else 1
        num_boundaries_per_sample = (hard_boundaries * valid_mask).sum(dim=1)

        shortened_lengths = torch.zeros(batch_size, device=device)
        if max_segments > 0:
            full = num_boundaries_per_sample >= (max_segments - 1)
            shortened_lengths[full] = 1.0
            partial = (num_boundaries_per_sample > 0) & (num_boundaries_per_sample < (max_segments - 1))
            shortened_lengths[partial] = (num_boundaries_per_sample[partial] + 1) / max_segments

        num_boundaries = hard_boundaries.sum()
        total_positions = actual_lens.sum().float()

        # No learned boundaries — zero loss
        loss = torch.tensor(0.0, device=device)
        if return_unreduced_boundary_loss:
            loss = loss.repeat(batch_size)

        return (
            pooled,
            loss,
            num_boundaries.item(),
            total_positions.item(),
            shortened_lengths,
            None,   # boundary_cv
            None,   # boundary_adjacent_pct
            valid_mask,  # masked_probs (all 1s for valid positions)
            num_boundaries_per_sample,
            actual_lens.float(),
        )
