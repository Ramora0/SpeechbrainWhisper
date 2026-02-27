#!/usr/bin/env python3
"""
Can the boundary predictor learn per-sample adaptive compression?

Setup: two input classes per batch —
  SIMPLE  = piecewise constant (few boundaries needed)
  COMPLEX = random noise (many boundaries needed)

The boundary detector is a standard per-token MLP (hidden → logit → sigmoid).
No cosine similarity, no adjacent-frame comparison — just a pointwise
predictor. This removes any inductive bias so that any separation between
classes must come from the training signal.

Experiments:
  1. GRADIENT DIRECTION — When adding a boundary would help the downstream
     loss, does the gradient through cumsum+soft_mask agree? 50% = no signal.

  2. DOWNSTREAM ONLY — Train with reconstruction loss (upstream frozen).
     The only way to reduce loss is by changing boundary count/placement.

  3. BINOMIAL LOSS — Train with per-sample binomial loss (different target
     counts for simple vs complex).

  4. BOTH LOSSES — Binomial + downstream together.

Uses gradient-preserving cumsum + soft_mask pooling from BoundaryPredictor2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


# ──────────────────────────────────────────────────────────────────────
# common() and pooling — from BoundaryPredictor2.py
# ──────────────────────────────────────────────────────────────────────

def common(boundaries):
    """Arithmetic segment-distance matrix (gradient-preserving)."""
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


def mean_pool(boundaries, hidden):
    """Gradient-preserving mean pooling via soft_mask = (1 - foo)."""
    B, T, D = hidden.shape
    foo = common(boundaries)
    if foo is None:
        return torch.zeros(B, 1, D, device=hidden.device)
    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
    counts = soft_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
    return seg_sum / counts.unsqueeze(-1)


def binomial_loss(num_boundaries, total_positions, target_counts, eps=1e-6):
    """Per-sample binomial loss (from loss.py)."""
    clamped_totals = total_positions.clamp(min=1.0)
    clamped_targets = torch.minimum(target_counts, clamped_totals)
    target_probs = (clamped_targets / clamped_totals).clamp(min=eps, max=1 - eps)
    dist = torch.distributions.Binomial(
        total_count=clamped_totals, probs=target_probs)
    return -dist.log_prob(num_boundaries) / clamped_totals


# ──────────────────────────────────────────────────────────────────────
# Boundary predictor: standard per-token MLP
# ──────────────────────────────────────────────────────────────────────

class MinimalBP(nn.Module):
    def __init__(self, input_dim, temp=0.5, init_bias=0.0):
        super().__init__()
        self.temp = temp

        # Upstream linear (simulates CNN)
        self.upstream = nn.Linear(input_dim, input_dim, bias=False)
        with torch.no_grad():
            self.upstream.weight.copy_(torch.eye(input_dim))

        # Per-token MLP: hidden → logit → sigmoid
        self.boundary_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )
        with torch.no_grad():
            self.boundary_mlp[-1].bias.fill_(init_bias)

    def _compute_probs(self, hidden):
        logits = self.boundary_mlp(hidden).squeeze(-1)  # [B, T]
        return torch.sigmoid(logits)

    def _sample(self, probs):
        dist = torch.distributions.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs.clamp(1e-6, 1 - 1e-6),
        )
        soft = dist.rsample()
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft, soft

    def forward(self, x):
        hidden = self.upstream(x)
        probs = self._compute_probs(hidden)
        boundaries, soft = self._sample(probs)
        if (boundaries.sum(dim=1) == 0).any():
            idx = (boundaries.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            boundaries[idx, -1] = 1.0
        pooled = mean_pool(boundaries, hidden)
        return pooled, boundaries, probs, soft


def eval_boundary_rate(bp, x):
    """Deterministic boundary count and mean prob, per sample."""
    bp.eval()
    with torch.no_grad():
        hidden = bp.upstream(x)
        probs = bp._compute_probs(hidden)
        count = (probs > 0.5).float().sum(dim=1)
        mp = probs.mean(dim=1)
    bp.train()
    return count, mp


# ──────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────

T = 40
D = 32
B = 8  # 4 simple + 4 complex
SIMPLE_TARGET = 5.0
COMPLEX_TARGET = 15.0


def make_batch():
    """First B//2 = SIMPLE (piecewise constant), last B//2 = COMPLEX (random)."""
    simple = torch.zeros(B // 2, T, D)
    for i in range(B // 2):
        n_segs = 5
        seg_len = T // n_segs
        for s in range(n_segs):
            val = torch.randn(D) * 2.0
            start = s * seg_len
            end = (s + 1) * seg_len if s < n_segs - 1 else T
            simple[i, start:end] = val
    complex_ = torch.randn(B // 2, T, D)
    x = torch.cat([simple, complex_], dim=0)
    targets = torch.tensor(
        [SIMPLE_TARGET] * (B // 2) + [COMPLEX_TARGET] * (B // 2))
    return x, targets


def print_rates(step, loss, bp, x):
    counts, mps = eval_boundary_rate(bp, x)
    sr = mps[:B // 2].mean().item()
    cr = mps[B // 2:].mean().item()
    sc = counts[:B // 2].mean().item()
    cc = counts[B // 2:].mean().item()
    print(f"  Step {step:4d}: loss={loss:.6f}  "
          f"simple={sr:.4f} ({sc:.0f}b)  complex={cr:.4f} ({cc:.0f}b)")
    return sr, cr


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Gradient direction accuracy
# ──────────────────────────────────────────────────────────────────────

def experiment_1():
    print("=" * 70)
    print("EXPERIMENT 1: Gradient direction accuracy")
    print("=" * 70)
    print()
    print("  Does the continuous gradient through pooling correctly predict")
    print("  whether adding a boundary reduces reconstruction loss?")
    print("  50% = random chance = no useful signal.")
    print()

    N_TRIALS = 1000
    aligned = 0
    misaligned = 0
    zero_grad = 0

    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        hidden = torch.randn(1, T, D)

        n_existing = (trial % 6) + 1
        spacing = T // (n_existing + 1)
        existing = [(i + 1) * spacing for i in range(n_existing)]

        all_pts = [0] + sorted(existing) + [T - 1]
        gaps = [(all_pts[i + 1] - all_pts[i], (all_pts[i] + all_pts[i + 1]) // 2)
                for i in range(len(all_pts) - 1)]
        new_pos = max(gaps)[1]

        prob = torch.tensor(0.3, requires_grad=True)
        boundaries = torch.zeros(1, T)
        for ep in existing:
            boundaries[0, ep] = 1.0
        hard = (prob > 0.5).float()
        boundaries[0, new_pos] = hard - prob.detach() + prob
        boundaries[0, -1] = 1.0

        pooled = mean_pool(boundaries, hidden)
        if pooled.shape[1] == 0:
            continue
        seg_ids = (boundaries.cumsum(1) - boundaries).long().clamp(
            0, pooled.shape[1] - 1)
        recon = torch.gather(
            pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
        loss = (recon - hidden).pow(2).mean()
        loss.backward()
        grad = prob.grad.item()

        b_with = torch.zeros(1, T)
        for ep in existing:
            b_with[0, ep] = 1.0
        b_with[0, new_pos] = 1.0
        b_with[0, -1] = 1.0
        p_w = mean_pool(b_with, hidden)
        si_w = (b_with.cumsum(1) - b_with).long().clamp(
            0, max(p_w.shape[1] - 1, 0))
        loss_with = (torch.gather(
            p_w, 1, si_w.unsqueeze(-1).expand(-1, -1, D)
        ) - hidden).pow(2).mean().item()

        b_without = torch.zeros(1, T)
        for ep in existing:
            b_without[0, ep] = 1.0
        b_without[0, -1] = 1.0
        p_wo = mean_pool(b_without, hidden)
        si_wo = (b_without.cumsum(1) - b_without).long().clamp(
            0, max(p_wo.shape[1] - 1, 0))
        loss_without = (torch.gather(
            p_wo, 1, si_wo.unsqueeze(-1).expand(-1, -1, D)
        ) - hidden).pow(2).mean().item()

        adding_helps = loss_with < loss_without

        if abs(grad) < 1e-12:
            zero_grad += 1
        elif (grad < 0) == adding_helps:
            aligned += 1
        else:
            misaligned += 1

    total_nonzero = aligned + misaligned
    accuracy = 100 * aligned / total_nonzero if total_nonzero > 0 else 0

    print(f"  {N_TRIALS} trials:")
    print(f"    Agrees:    {aligned}")
    print(f"    Disagrees: {misaligned}")
    print(f"    Zero:      {zero_grad}")
    print(f"    Accuracy:  {accuracy:.1f}%")

    return accuracy


# ──────────────────────────────────────────────────────────────────────
# Experiment 2: Downstream loss only (frozen upstream)
# ──────────────────────────────────────────────────────────────────────

def experiment_2(steps=2000):
    """
    Reconstruction loss with FROZEN upstream weights.
    Only the boundary MLP can change.
    """
    print()
    print("=" * 70)
    print("EXPERIMENT 2: Downstream loss only (upstream FROZEN)")
    print("=" * 70)
    print()
    print("  Reconstruction loss. Upstream frozen — only boundary MLP learns.")
    print(f"  Optimal: ~{SIMPLE_TARGET:.0f}b for simple,"
          f" ~{COMPLEX_TARGET:.0f}b for complex.")
    print()

    torch.manual_seed(42)
    bp = MinimalBP(input_dim=D, temp=0.5, init_bias=0.0)

    # Freeze upstream
    bp.upstream.weight.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in bp.parameters() if p.requires_grad], lr=0.003)

    for step in range(steps):
        x, _ = make_batch()
        optimizer.zero_grad()
        pooled, boundaries, probs, soft = bp(x)
        n_seg = pooled.shape[1]
        if n_seg == 0:
            continue
        seg_ids = (boundaries.cumsum(1) - boundaries).long().clamp(
            0, n_seg - 1)
        hidden = bp.upstream(x)
        recon = torch.gather(
            pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
        loss = (recon - hidden.detach()).pow(2).mean()
        loss.backward()
        optimizer.step()
        if step % 400 == 0:
            print_rates(step, loss.item(), bp, x)

    x, _ = make_batch()
    counts, mps = eval_boundary_rate(bp, x)
    sr = mps[:B // 2].mean().item()
    cr = mps[B // 2:].mean().item()
    sc = counts[:B // 2].mean().item()
    cc = counts[B // 2:].mean().item()
    diff = cr - sr

    print()
    print(f"  Final: simple={sr:.4f} ({sc:.0f}b)  "
          f"complex={cr:.4f} ({cc:.0f}b)  diff={diff:+.4f}")
    return diff


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Binomial loss with per-sample targets
# ──────────────────────────────────────────────────────────────────────

def experiment_3(steps=2000):
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Binomial loss with per-sample targets")
    print("=" * 70)
    print()
    print(f"  SIMPLE target: {SIMPLE_TARGET:.0f} boundaries")
    print(f"  COMPLEX target: {COMPLEX_TARGET:.0f} boundaries")
    print()

    torch.manual_seed(42)
    bp = MinimalBP(input_dim=D, temp=0.5, init_bias=0.0)
    optimizer = torch.optim.Adam(bp.parameters(), lr=0.003)

    for step in range(steps):
        x, targets = make_batch()
        optimizer.zero_grad()
        pooled, boundaries, probs, soft = bp(x)
        per_sample_count = boundaries.sum(dim=1)
        total_positions = torch.tensor(float(T)).expand(B)
        loss = binomial_loss(
            per_sample_count, total_positions, targets).mean()
        loss.backward()
        optimizer.step()
        if step % 400 == 0:
            print_rates(step, loss.item(), bp, x)

    x, targets = make_batch()
    counts, mps = eval_boundary_rate(bp, x)
    sr = mps[:B // 2].mean().item()
    cr = mps[B // 2:].mean().item()
    sc = counts[:B // 2].mean().item()
    cc = counts[B // 2:].mean().item()
    diff = cr - sr

    print()
    print(f"  Final: simple={sr:.4f} ({sc:.0f}b, target={SIMPLE_TARGET:.0f})  "
          f"complex={cr:.4f} ({cc:.0f}b, target={COMPLEX_TARGET:.0f})  "
          f"diff={diff:+.4f}")
    return diff


# ──────────────────────────────────────────────────────────────────────
# Experiment 4: Both losses
# ──────────────────────────────────────────────────────────────────────

def experiment_4(steps=2000):
    print()
    print("=" * 70)
    print("EXPERIMENT 4: Binomial + downstream loss")
    print("=" * 70)
    print()
    print(f"  SIMPLE target: {SIMPLE_TARGET:.0f} boundaries")
    print(f"  COMPLEX target: {COMPLEX_TARGET:.0f} boundaries")
    print(f"  Loss = reconstruction + binomial")
    print()

    torch.manual_seed(42)
    bp = MinimalBP(input_dim=D, temp=0.5, init_bias=0.0)
    optimizer = torch.optim.Adam(bp.parameters(), lr=0.003)

    for step in range(steps):
        x, targets = make_batch()
        optimizer.zero_grad()
        pooled, boundaries, probs, soft = bp(x)
        n_seg = pooled.shape[1]
        if n_seg == 0:
            continue
        seg_ids = (boundaries.cumsum(1) - boundaries).long().clamp(
            0, n_seg - 1)
        hidden = bp.upstream(x)
        recon = torch.gather(
            pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
        recon_loss = (recon - hidden.detach()).pow(2).mean()
        per_sample_count = boundaries.sum(dim=1)
        total_positions = torch.tensor(float(T)).expand(B)
        bp_loss = binomial_loss(
            per_sample_count, total_positions, targets).mean()
        loss = recon_loss + bp_loss
        loss.backward()
        optimizer.step()
        if step % 400 == 0:
            print_rates(step, loss.item(), bp, x)

    x, targets = make_batch()
    counts, mps = eval_boundary_rate(bp, x)
    sr = mps[:B // 2].mean().item()
    cr = mps[B // 2:].mean().item()
    sc = counts[:B // 2].mean().item()
    cc = counts[B // 2:].mean().item()
    diff = cr - sr

    print()
    print(f"  Final: simple={sr:.4f} ({sc:.0f}b, target={SIMPLE_TARGET:.0f})  "
          f"complex={cr:.4f} ({cc:.0f}b, target={COMPLEX_TARGET:.0f})  "
          f"diff={diff:+.4f}")
    return diff


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print()
    print("Can the boundary predictor learn per-sample adaptive compression?")
    print(f"SIMPLE = piecewise constant ({SIMPLE_TARGET:.0f} segments)")
    print(f"COMPLEX = random noise (needs {COMPLEX_TARGET:.0f}+ boundaries)")
    print(f"Boundary detector: per-token MLP (no inductive bias)")
    print()

    accuracy = experiment_1()
    diff_downstream = experiment_2()
    diff_binomial = experiment_3()
    diff_both = experiment_4()

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
  Exp 1 (Gradient direction):
    Accuracy: {accuracy:.1f}%  {"← random chance" if accuracy < 55 else ""}

  Exp 2 (Downstream only, upstream frozen):
    Rate diff: {diff_downstream:+.4f}  {"← no adaptation" if diff_downstream < 0.03 else "← adapted!"}

  Exp 3 (Binomial, per-sample targets):
    Rate diff: {diff_binomial:+.4f}  {"← no adaptation" if abs(diff_binomial) < 0.03 else "← adapted!"}

  Exp 4 (Binomial + downstream):
    Rate diff: {diff_both:+.4f}  {"← no adaptation" if abs(diff_both) < 0.03 else "← adapted!"}""")

    downstream_fails = diff_downstream < 0.03  # must be positive (complex > simple)
    binomial_works = diff_binomial > 0.03

    print()
    if downstream_fails and binomial_works:
        print("  CONCLUSION:")
        print("  Downstream loss cannot teach adaptive compression — gradient")
        print("  through cumsum + soft_mask + pooling is random (exp 1).")
        print("  Binomial loss with per-sample targets CAN.")
    elif downstream_fails and not binomial_works:
        print("  CONCLUSION:")
        print("  Neither loss achieves adaptive compression with a pointwise MLP.")
        print("  The MLP processes each token independently and cannot distinguish")
        print("  which sample a token belongs to. Per-sample targets require an")
        print("  architecture that sees inter-token structure (e.g. adjacent-frame")
        print("  comparison) to differentiate simple from complex inputs.")
    elif not downstream_fails:
        print("  CONCLUSION:")
        print("  Downstream loss CAN drive adaptive compression.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
