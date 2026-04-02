"""Forced-all BoundaryPredictor with direct frame selection (no mean pooling).

Every valid position is a boundary. Instead of going through segment assignment
and mean pooling, frames are selected directly — bypasses the common() machinery.
"""

import torch
import torch.nn as nn


class BoundaryPredictorSelect(nn.Module):
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

        actual_lens = (lengths * seq_len).long()
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = (pos_idx < actual_lens.unsqueeze(1)).float()

        hard_boundaries = valid_mask

        # Direct selection: each frame is its own segment, just pass through
        pooled = hidden

        # Shortened lengths: just the original valid lengths
        max_segments = seq_len
        num_boundaries_per_sample = actual_lens.float()
        shortened_lengths = actual_lens.float() / max_segments

        num_boundaries = hard_boundaries.sum()
        total_positions = actual_lens.sum().float()

        loss = torch.tensor(0.0, device=device)
        if return_unreduced_boundary_loss:
            loss = loss.repeat(batch_size)

        return (
            pooled,
            loss,
            num_boundaries.item(),
            total_positions.item(),
            shortened_lengths,
            None,
            None,
            valid_mask,
            num_boundaries_per_sample,
            actual_lens.float(),
        )
