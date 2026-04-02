"""Identity BoundaryPredictor: complete pass-through, no boundary logic at all.

Returns hidden states unchanged. Drop-in replacement with same 10-tuple return.
"""

import torch
import torch.nn as nn


class BoundaryPredictorIdentity(nn.Module):
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
        batch_size, seq_len, _ = hidden.shape
        device = hidden.device

        actual_lens = (lengths * seq_len).long()
        shortened_lengths = lengths

        num_boundaries = actual_lens.sum().float()
        total_positions = num_boundaries

        loss = torch.tensor(0.0, device=device)
        if return_unreduced_boundary_loss:
            loss = loss.repeat(batch_size)

        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        valid_mask = (pos_idx < actual_lens.unsqueeze(1)).float()

        return (
            hidden,
            loss,
            num_boundaries.item(),
            total_positions.item(),
            shortened_lengths,
            None,
            None,
            valid_mask,
            actual_lens.float(),
            actual_lens.float(),
        )
