"""Minimal BoundaryPredictor: MLP boundary detection + mean pooling.

Drop-in replacement for BoundaryPredictor2 — same forward() 10-tuple return.
"""

import torch
import torch.nn as nn

from loss import binomial_loss_from_target_counts
from utils import common


class BoundaryPredictorMLP(nn.Module):
    def __init__(self, input_dim, prior, temp=1, disable_temp_schedule=False):
        super().__init__()
        self.temp = temp
        self.prior = prior
        self.compression_schedule = 1.0
        self.target_prior = prior
        self.disable_temp_schedule = disable_temp_schedule

        self.dropout = nn.Dropout(p=0.1)
        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )

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

    def _valid_mask(self, lengths, seq_len, device):
        actual_lens = (lengths * seq_len).long()
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        return actual_lens, (pos_idx < actual_lens.unsqueeze(1)).float()

    def forward(self, hidden, lengths, target_boundary_counts=None,
                return_unreduced_boundary_loss=False):
        batch_size, seq_len, hidden_dim = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        actual_lens, valid_mask = self._valid_mask(lengths, seq_len, device)

        # --- Boundary detection: per-frame MLP ---
        logits = self.boundary_mlp(self.dropout(hidden)).squeeze(-1)
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

        soft_boundaries = soft_boundaries * valid_mask
        hard_samples = hard_samples * valid_mask
        masked_probs = probs * valid_mask

        # Ensure boundary at last valid position
        last_valid = (actual_lens - 1).clamp(min=0, max=seq_len - 1)
        batch_idx = torch.arange(batch_size, device=device)
        soft_boundaries[batch_idx, last_valid] = 1.0
        hard_samples[batch_idx, last_valid] = 1.0

        hard_boundaries = hard_samples - soft_boundaries.detach() + soft_boundaries

        # Emergency: ensure at least one boundary
        no_boundaries = hard_boundaries.sum(dim=1) == 0
        if no_boundaries.any():
            indices = no_boundaries.nonzero(as_tuple=True)[0]
            emergency_pos = (actual_lens[indices] - 1).clamp(min=0, max=seq_len - 1)
            hard_boundaries[indices, emergency_pos] = 1.0
            if self.training:
                print(f"[BoundaryPredictorMLP] WARNING: emergency boundary for {len(indices)} sequence(s)")

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

        # --- Loss ---
        num_boundaries = hard_boundaries.sum()
        total_positions = actual_lens.sum().float()

        if self.training:
            per_item_boundaries = hard_boundaries.sum(dim=1).to(dtype=torch.float32)
            loss_totals = (lengths * (seq_len + 1)).long().float().to(dtype=torch.float32)
            loss_values = binomial_loss_from_target_counts(
                per_item_boundaries,
                loss_totals,
                target_boundary_counts.to(device=device, dtype=torch.float32),
            )
            loss = loss_values if return_unreduced_boundary_loss else loss_values.mean()
        else:
            loss = torch.tensor(0.0, device=device)
            if return_unreduced_boundary_loss:
                loss = loss.repeat(batch_size)

        # --- Eval statistics ---
        boundary_cv = None
        boundary_adjacent_pct = None
        if not self.training:
            with torch.no_grad():
                all_spacings = []
                adjacent_count = 0
                total_gaps = 0
                hb_len = hard_boundaries.shape[1]
                valid_lengths = torch.clamp((lengths * (hb_len + 1)).long() - 1, max=hb_len)
                eval_mask = (torch.arange(hb_len, device=device).unsqueeze(0) < valid_lengths.unsqueeze(1))
                masked_hb = hard_boundaries * eval_mask.float()
                for b in range(batch_size):
                    bp = masked_hb[b].nonzero(as_tuple=True)[0]
                    if len(bp) > 1:
                        spacings = bp[1:] - bp[:-1]
                        all_spacings.extend(spacings.cpu().tolist())
                        adjacent_count += (spacings == 1).sum().item()
                        total_gaps += len(bp) - 1
                if all_spacings:
                    st = torch.tensor(all_spacings, dtype=torch.float32)
                    boundary_cv = (st.std() / st.mean()).item() if st.mean() > 0 else 0.0
                boundary_adjacent_pct = (adjacent_count / total_gaps * 100.0) if total_gaps > 0 else 0.0

        return (
            pooled,
            loss,
            num_boundaries.item(),
            total_positions.item(),
            shortened_lengths,
            boundary_cv,
            boundary_adjacent_pct,
            masked_probs,
            num_boundaries_per_sample,
            actual_lens.float(),
        )
