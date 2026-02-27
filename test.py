#!/usr/bin/env python3
"""
Empirical test: Can downstream gradients change boundary COUNTS
when flowing through sampled boundaries + mean pooling?

Minimal setup:
  - MLP boundary detector (hidden -> sigmoid -> per-position probability)
  - RelaxedBernoulli sampling + STE
  - Mean pooling using common() from utils.py (same segment assignment
    as the real BoundaryPredictor2)
  - Downstream loss on the pooled output

Experiments:
  1. GRADIENT INSPECTION — Check which params get gradient from downstream loss.

  2a. Loss REWARDS FEWER boundaries — same init as 2b. Count should not drop.
  2b. Loss REWARDS MORE boundaries  — same init as 2a. Count should not rise.

  3. DIRECT CONTROL — Loss on boundary probs. Count converges. Proves MLP works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from utils import common


# ──────────────────────────────────────────────────────────────────────
# Minimal boundary predictor using common() for segment assignment
# ──────────────────────────────────────────────────────────────────────

class MinimalBoundaryPredictor(nn.Module):
    def __init__(self, input_dim, temp=0.5, init_bias=0.0):
        super().__init__()
        self.temp = temp

        # Upstream linear (simulates CNN)
        self.upstream = nn.Linear(input_dim, input_dim, bias=False)
        with torch.no_grad():
            self.upstream.weight.copy_(torch.eye(input_dim))

        # Boundary detector: MLP -> scalar logit per position
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

    def _sample_boundaries(self, probs):
        dist = torch.distributions.RelaxedBernoulli(
            temperature=self.temp,
            probs=probs.clamp(1e-6, 1 - 1e-6),
        )
        soft = dist.rsample()
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft, soft

    def _mean_pool(self, boundaries, hidden):
        """Mean-pool using common() from utils.py — same as real BP."""
        B, T, D = hidden.shape

        foo = common(boundaries)  # [B, L, S] distance matrix

        if foo is None:
            return torch.zeros(B, 1, D, device=hidden.device)

        max_segs = foo.size(2)

        # Segment membership: same as BoundaryPredictor2 line 158
        segment_mask = (foo == 0).float()  # [B, L, S]

        # Mean pool: sum / count using the membership mask
        counts = segment_mask.sum(dim=1).clamp(min=1e-8)  # [B, S]
        seg_sum = torch.bmm(segment_mask.transpose(1, 2), hidden)  # [B, S, D]
        pooled = seg_sum / counts.unsqueeze(-1)

        return pooled

    def forward(self, raw_input):
        hidden = self.upstream(raw_input)
        probs = self._compute_probs(hidden)
        boundaries, soft = self._sample_boundaries(probs)

        if (boundaries.sum(dim=1) == 0).any():
            idx = (boundaries.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            boundaries[idx, -1] = 1.0

        pooled = self._mean_pool(boundaries, hidden)
        return pooled, boundaries.sum().item(), probs


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

BD_KEYS = ["boundary_mlp"]
UP_KEYS = ["upstream"]


def eval_stats(bp, x):
    bp.eval()
    with torch.no_grad():
        hidden = bp.upstream(x)
        probs = bp._compute_probs(hidden)
        count = (probs > 0.5).float().sum().item()
        mp = probs.mean().item()
    bp.train()
    return count, mp


def snapshot(model, keys):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if any(k in n for k in keys)}


def max_change(before, after):
    return max((after[n] - before[n]).abs().max().item() for n in before) if before else 0.0


# ──────────────────────────────────────────────────────────────────────
# Experiment 1: Gradient Inspection
# ──────────────────────────────────────────────────────────────────────

def experiment_1(x):
    print("=" * 70)
    print("EXPERIMENT 1: Gradient Inspection")
    print("=" * 70)
    print("Forward downstream loss through mean-pooled output (using common()).")
    print("Check which parameters receive nonzero gradient.\n")

    D = x.shape[2]
    bp = MinimalBoundaryPredictor(input_dim=D, temp=0.5)
    bp.train()

    pooled, num_b, probs = bp(x)
    loss = pooled.pow(2).mean()
    loss.backward()

    print(f"  {'Parameter':<40s} {'Grad norm':>10s}  Category")
    print(f"  {'-'*40} {'-'*10}  {'-'*15}")

    all_bd_zero = True
    up_nonzero = False
    for name, param in bp.named_parameters():
        gn = param.grad.norm().item() if param.grad is not None else 0.0
        is_bd = any(k in name for k in BD_KEYS)
        is_up = any(k in name for k in UP_KEYS)
        cat = "BOUNDARY-MLP" if is_bd else ("UPSTREAM/CNN" if is_up else "???")
        tag = "ZERO" if gn < 1e-10 else "NONZERO"
        print(f"  {name:<40s} {gn:>10.2e}  [{cat}] {tag}")
        if is_bd and gn > 1e-10:
            all_bd_zero = False
        if is_up and gn > 1e-10:
            up_nonzero = True

    print()
    if all_bd_zero and up_nonzero:
        print("  Boundary MLP gets ZERO gradient from downstream loss.")
        print("  Upstream (CNN) gets NONZERO gradient.")
    elif all_bd_zero and not up_nonzero:
        print("  Both boundary MLP and upstream get zero gradient.")
    else:
        print("  Boundary MLP got NONZERO gradient from downstream loss!")
    return all_bd_zero, not all_bd_zero


# ──────────────────────────────────────────────────────────────────────
# Experiment 2a: Fewer boundaries would help
# ──────────────────────────────────────────────────────────────────────

def experiment_2a(x, steps=500):
    print()
    print("=" * 70)
    print("EXPERIMENT 2a: Downstream loss REWARDS FEWER boundaries")
    print("=" * 70)
    print("Same init as 2b (init_bias=0.0, ~50% boundary rate).")
    print("Loss = pooled.pow(2).mean() — fewer segments = less to minimize.")
    print(f"If gradient worked, boundary count would DROP over {steps} steps.\n")

    D = x.shape[2]
    bp = MinimalBoundaryPredictor(input_dim=D, temp=0.5, init_bias=0.0)

    init_count, init_mp = eval_stats(bp, x)
    bd_before = snapshot(bp, BD_KEYS)

    optimizer = torch.optim.Adam(bp.parameters(), lr=0.01)
    bp.train()

    first_loss = None
    for step in range(steps):
        optimizer.zero_grad()
        pooled, num_b, probs = bp(x)
        loss = pooled.pow(2).mean()
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()
        if step % 100 == 0:
            ec, emp = eval_stats(bp, x)
            print(f"  Step {step:4d}: boundaries={ec:.0f}  "
                  f"mean_prob={emp:.4f}  loss={loss.item():.6f}")

    final_count, final_mp = eval_stats(bp, x)
    bd_change = max_change(bd_before, snapshot(bp, BD_KEYS))
    loss_decreased = loss.item() < first_loss * 0.8

    print()
    print(f"  Boundary prob:       {init_mp:.4f} -> {final_mp:.4f}  "
          f"(delta={final_mp - init_mp:+.6f})")
    print(f"  Boundary count:      {init_count:.0f} -> {final_count:.0f}")
    print(f"  BD param max change: {bd_change:.2e}")
    print(f"  Loss:                {first_loss:.6f} -> {loss.item():.6f}  "
          f"({'decreased' if loss_decreased else 'did NOT decrease'})")

    prob_delta = abs(final_mp - init_mp)
    count_dropped = final_count < init_count * 0.7
    return prob_delta, bd_change, count_dropped, "fewer"


# ──────────────────────────────────────────────────────────────────────
# Experiment 2b: More boundaries would help
# ──────────────────────────────────────────────────────────────────────

def experiment_2b(x, steps=500):
    print()
    print("=" * 70)
    print("EXPERIMENT 2b: Downstream loss REWARDS MORE boundaries")
    print("=" * 70)
    print("Same init as 2a (init_bias=0.0, ~50% boundary rate).")
    print("Loss = reconstruction error (pooled repeated to frames vs original).")
    print("More segments = better reconstruction = lower loss.")
    print(f"If gradient worked, boundary count would RISE over {steps} steps.\n")

    D = x.shape[2]
    B = x.shape[0]
    T = x.shape[1]
    bp = MinimalBoundaryPredictor(input_dim=D, temp=0.5, init_bias=0.0)

    init_count, init_mp = eval_stats(bp, x)
    bd_before = snapshot(bp, BD_KEYS)

    optimizer = torch.optim.Adam(bp.parameters(), lr=0.01)
    bp.train()

    first_loss = None
    for step in range(steps):
        optimizer.zero_grad()
        pooled, num_b, probs = bp(x)

        # Reconstruct: get segment IDs from boundaries, gather pooled per frame
        hidden = bp.upstream(x)
        boundaries, _ = bp._sample_boundaries(bp._compute_probs(hidden))
        if (boundaries.sum(dim=1) == 0).any():
            idx = (boundaries.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            boundaries[idx, -1] = 1.0
        seg_ids = (boundaries.cumsum(dim=1) - boundaries).long()
        seg_ids_clamped = seg_ids.clamp(0, pooled.shape[1] - 1)
        reconstructed = torch.gather(
            pooled, 1,
            seg_ids_clamped.unsqueeze(-1).expand(-1, -1, D)
        )
        loss = (reconstructed - hidden.detach()).pow(2).mean()
        loss.backward()
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()
        if step % 100 == 0:
            ec, emp = eval_stats(bp, x)
            print(f"  Step {step:4d}: boundaries={ec:.0f}  "
                  f"mean_prob={emp:.4f}  loss={loss.item():.6f}")

    final_count, final_mp = eval_stats(bp, x)
    bd_change = max_change(bd_before, snapshot(bp, BD_KEYS))

    print()
    print(f"  Boundary prob:       {init_mp:.4f} -> {final_mp:.4f}  "
          f"(delta={final_mp - init_mp:+.6f})")
    print(f"  Boundary count:      {init_count:.0f} -> {final_count:.0f}")
    print(f"  BD param max change: {bd_change:.2e}")
    print(f"  Loss:                {first_loss:.6f} -> {loss.item():.6f}")

    prob_delta = abs(final_mp - init_mp)
    count_rose = final_count > init_count * 1.3
    return prob_delta, bd_change, count_rose, "more"


# ──────────────────────────────────────────────────────────────────────
# Experiment 3: Direct loss on probs (control)
# ──────────────────────────────────────────────────────────────────────

def experiment_3(x, steps=500):
    print()
    print("=" * 70)
    print("EXPERIMENT 3: Direct loss on boundary probs (control)")
    print("=" * 70)
    print(f"Training for {steps} steps with loss = (mean_prob - 0.05)^2.")
    print("Direct gradient to boundary MLP. Count SHOULD converge.\n")

    D = x.shape[2]
    T = x.shape[1]
    bp = MinimalBoundaryPredictor(input_dim=D, temp=0.5, init_bias=0.0)

    init_count, init_mp = eval_stats(bp, x)

    optimizer = torch.optim.Adam(bp.parameters(), lr=0.01)
    bp.train()

    target_rate = 0.05

    for step in range(steps):
        optimizer.zero_grad()
        hidden = bp.upstream(x)
        probs = bp._compute_probs(hidden)
        loss = (probs.mean() - target_rate).pow(2)
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            ec, emp = eval_stats(bp, x)
            print(f"  Step {step:4d}: boundaries={ec:.0f}  "
                  f"mean_prob={emp:.4f}  loss={loss.item():.6f}")

    final_count, final_mp = eval_stats(bp, x)

    print()
    print(f"  Boundary prob:  {init_mp:.4f} -> {final_mp:.4f}  "
          f"(target={target_rate:.4f})")
    print(f"  Boundary count: {init_count:.0f} -> {final_count:.0f}  "
          f"(target ~{target_rate * T:.0f} per sample)")

    prob_delta = abs(final_mp - init_mp)
    moved = abs(final_mp - target_rate) < abs(init_mp - target_rate)
    ok = prob_delta > 0.05 and moved
    print()
    if ok:
        print(f"  PASS: Direct gradient changed probs by {prob_delta:.4f}.")
        print(f"        The MLP CAN learn boundary counts with direct signal.")
    else:
        print(f"  FAIL: Probs didn't move enough ({prob_delta:.4f})")
    return ok


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    D = 64
    T = 40
    B = 4

    x = torch.randn(B, T, D)

    print()
    print("Can downstream gradients change boundary COUNTS")
    print("through sampling + mean pooling (using common() from utils.py)?")
    print(f"Setup: B={B}, T={T}, D={D}, temp=0.5")
    print()

    bd_zero, bd_nonzero = experiment_1(x.clone())

    r2a_prob, r2a_bd, r2a_moved, _ = experiment_2a(x.clone())
    r2b_prob, r2b_bd, r2b_moved, _ = experiment_2b(x.clone())
    r3 = experiment_3(x.clone())

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  Exp 1  (Gradient inspection):")
    if bd_zero:
        print(f"    Boundary MLP grad = ZERO from downstream loss")
    else:
        print(f"    Boundary MLP grad = NONZERO from downstream loss!")

    print(f"\n  Exp 2a (Wants fewer boundaries):")
    print(f"    Prob delta: {r2a_prob:.6f}, BD param change: {r2a_bd:.2e}, "
          f"Count dropped: {r2a_moved}")

    print(f"\n  Exp 2b (Wants more boundaries):")
    print(f"    Prob delta: {r2b_prob:.6f}, BD param change: {r2b_bd:.2e}, "
          f"Count rose: {r2b_moved}")

    print(f"\n  Exp 3  (Direct prob loss):  {'PASS' if r3 else 'FAIL'}")

    # Determine overall result
    can_change_count = r2a_moved or r2b_moved or bd_nonzero

    print()
    if not can_change_count and r3:
        print("CONCLUSION: Downstream loss CANNOT change boundary counts.")
        print()
        print("  Neither direction works:")
        print("  - Loss wanting fewer boundaries: count unchanged (2a)")
        print("  - Loss wanting more boundaries:  count unchanged (2b)")
        print("  - Not a collapse — opposite losses, same result: no movement.")
        print()
        print("  Only a direct loss on probs (exp 3) can change count.")
    elif can_change_count:
        print("CONCLUSION: Downstream loss CAN influence boundary counts!")
        print()
        if bd_nonzero:
            print("  Gradient inspection shows nonzero gradient to boundary MLP.")
        if r2a_moved:
            print("  Count dropped when loss wanted fewer boundaries.")
        if r2b_moved:
            print("  Count rose when loss wanted more boundaries.")
    else:
        print("CONCLUSION: Mixed results — needs further investigation.")

    return 0 if (not can_change_count and r3) else 1


if __name__ == "__main__":
    sys.exit(main())
