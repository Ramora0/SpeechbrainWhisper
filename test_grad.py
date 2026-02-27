#!/usr/bin/env python3
"""
Prove that arithmetic common() + soft mask preserves gradients to boundary MLP,
while boolean common() + hard mask does not.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Two versions of common() ──────────────────────────────────────

def common_arithmetic(boundaries):
    """Wave2Vec2ASR version: pure arithmetic, preserves gradients."""
    boundaries = boundaries.clone()
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if n_segments == 0:
        return None
    tmp = torch.zeros_like(boundaries).unsqueeze(2) + torch.arange(
        start=0, end=n_segments, device=boundaries.device)
    hh1 = boundaries.cumsum(1)
    hh1 -= boundaries
    foo = tmp - hh1.unsqueeze(-1)
    return foo


def common_boolean(boundaries):
    """SpeechbrainWhisper version: uses == comparison, kills gradients."""
    boundaries = boundaries.clone()
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if n_segments == 0:
        return None
    batch_size, seq_len = boundaries.shape
    device = boundaries.device
    segment_ids = boundaries.cumsum(1) - boundaries
    positions = torch.arange(seq_len, device=device, dtype=torch.float)
    segment_indices = torch.arange(n_segments, device=device)
    in_segment = segment_ids.unsqueeze(-1) == segment_indices.unsqueeze(0).unsqueeze(0)
    pos_for_min = torch.where(
        in_segment,
        positions.view(1, -1, 1).expand(batch_size, -1, n_segments),
        torch.tensor(float('inf'), device=device))
    pos_for_max = torch.where(
        in_segment,
        positions.view(1, -1, 1).expand(batch_size, -1, n_segments),
        torch.tensor(float('-inf'), device=device))
    seg_starts = pos_for_min.min(dim=1).values
    seg_ends = pos_for_max.max(dim=1).values
    pos_expanded = positions.view(1, -1, 1)
    foo = torch.where(
        in_segment,
        torch.zeros_like(pos_expanded),
        torch.where(
            pos_expanded < seg_starts.unsqueeze(1),
            pos_expanded - seg_starts.unsqueeze(1),
            pos_expanded - seg_ends.unsqueeze(1)))
    return foo


# ── Two versions of mean pooling ──────────────────────────────────

def mean_pool_hard(foo, hidden):
    """Current SpeechbrainWhisper: hard binary mask. Kills gradient from foo."""
    segment_mask = (foo == 0).float()          # [B, L, S]
    counts = segment_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(segment_mask.transpose(1, 2), hidden)
    return seg_sum / counts.unsqueeze(-1)


def mean_pool_soft(foo, hidden):
    """Wave2Vec2ASR final()-style: soft weights that preserve gradient from foo."""
    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
    # Normalize per segment (same as final())
    soft_mask = soft_mask / (soft_mask.sum(dim=1, keepdim=True) + 1e-9)
    return torch.bmm(soft_mask.transpose(1, 2), hidden)


# ── Minimal boundary predictor ────────────────────────────────────

class MinimalBP(nn.Module):
    def __init__(self, dim, temp=0.5):
        super().__init__()
        self.temp = temp
        self.boundary_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        with torch.no_grad():
            self.boundary_mlp[-1].bias.fill_(0.0)

    def forward(self, hidden):
        logits = self.boundary_mlp(hidden).squeeze(-1)
        probs = torch.sigmoid(logits)

        dist = torch.distributions.RelaxedBernoulli(
            temperature=self.temp, probs=probs.clamp(1e-6, 1 - 1e-6))
        soft = dist.rsample()
        hard = (soft > 0.5).float()
        boundaries = hard - soft.detach() + soft  # STE

        # Ensure at least one boundary per sequence
        if (boundaries.sum(dim=1) == 0).any():
            idx = (boundaries.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            boundaries[idx, -1] = 1.0

        return boundaries, probs


# ── Test harness ──────────────────────────────────────────────────

def test(name, common_fn, pool_fn):
    torch.manual_seed(42)
    B, T, D = 2, 20, 16
    hidden = torch.randn(B, T, D, requires_grad=True)

    bp = MinimalBP(D)
    bp.train()

    boundaries, probs = bp(hidden)
    foo = common_fn(boundaries)
    if foo is None:
        print(f"  {name}: no boundaries, skip")
        return
    pooled = pool_fn(foo, hidden)

    loss = pooled.pow(2).mean()
    loss.backward()

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  {'Parameter':<35s} {'Grad norm':>12s}  Has grad?")
    print(f"  {'-' * 55}")
    for pname, p in bp.named_parameters():
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        tag = "YES" if gn > 1e-12 else "ZERO"
        print(f"  {pname:<35s} {gn:>12.2e}  {tag}")


# ── Test real BoundaryPredictor2 ──────────────────────────────────

def test_real_bp(name, module_name):
    torch.manual_seed(42)
    B, T, D = 2, 20, 16
    hidden = torch.randn(B, T, D)
    lengths = torch.ones(B)
    targets = torch.tensor([5.0, 5.0])

    if module_name == "BoundaryPredictor":
        from BoundaryPredictor import BoundaryPredictor2
    else:
        from BoundaryPredictor2 import BoundaryPredictor2

    bp = BoundaryPredictor2(input_dim=D, prior=0.2, temp=0.5)
    bp.train()

    pooled, bp_loss, *rest = bp(hidden, lengths, target_boundary_counts=targets)

    # Downstream loss ONLY — ignore boundary loss
    downstream_loss = pooled.pow(2).mean()
    downstream_loss.backward()

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  {'Parameter':<35s} {'Grad norm':>12s}  Has grad?")
    print(f"  {'-' * 55}")
    for pname, p in bp.named_parameters():
        gn = p.grad.norm().item() if p.grad is not None else 0.0
        tag = "YES" if gn > 1e-12 else "ZERO"
        print(f"  {pname:<35s} {gn:>12.2e}  {tag}")


# ── Run all tests ─────────────────────────────────────────────────

print("\n\nPart 1: Minimal mean-pool tests (four combinations)")
print("=" * 60)

test("BOOLEAN common + HARD pool (current SpeechbrainWhisper)",
     common_boolean, mean_pool_hard)

test("BOOLEAN common + SOFT pool",
     common_boolean, mean_pool_soft)

test("ARITHMETIC common + HARD pool",
     common_arithmetic, mean_pool_hard)

test("ARITHMETIC common + SOFT pool (Wave2Vec2ASR style)",
     common_arithmetic, mean_pool_soft)

print("\n\nPart 2: Real BoundaryPredictor2 with attention pooling")
print("=" * 60)

test_real_bp("ORIGINAL BoundaryPredictor.py (broken)", "BoundaryPredictor")
test_real_bp("FIXED BoundaryPredictor2.py", "BoundaryPredictor2")
