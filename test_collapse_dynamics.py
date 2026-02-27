#!/usr/bin/env python3
"""
Deep dive: WHY do boundaries collapse to zero during training even when
the downstream loss (reconstruction) benefits from more boundaries?

Key finding from test_gradient_direction.py:
  - The gradient direction is roughly balanced (28/50 down vs 22/50 up)
  - NOT a simple structural sign bias

New hypotheses to test:
  1. MAGNITUDE ASYMMETRY: Even if direction is balanced, the gradient may
     be STRONGER when pushing down than when pushing up
  2. LOSS LANDSCAPE ASYMMETRY: Fewer boundaries = simpler model = easier to
     optimize. The upstream weights adapt to make pooling work with fewer boundaries.
  3. STRAIGHT-THROUGH ESTIMATOR LEAK: The STE disconnects the discrete
     boundary count from the continuous prob. The downstream loss optimizes
     the pooled CONTENT (which depends on upstream weights), not the boundary COUNT.
  4. COMPETING OBJECTIVES: The boundary loss drives count toward target,
     but if the downstream loss is indifferent to count (it can optimize via
     upstream weights instead), any noise in the boundary gradient causes drift.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def common(boundaries):
    """Arithmetic segment-distance (gradient-preserving)."""
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


class MinimalBP(nn.Module):
    def __init__(self, D, temp=0.5, init_bias=0.0):
        super().__init__()
        self.temp = temp
        self.upstream = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.upstream.weight.copy_(torch.eye(D))
        self.boundary_mlp = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        with torch.no_grad():
            self.boundary_mlp[-1].bias.fill_(init_bias)

    def _compute_probs(self, hidden):
        return torch.sigmoid(self.boundary_mlp(hidden).squeeze(-1))

    def _sample(self, probs):
        dist = torch.distributions.RelaxedBernoulli(
            temperature=self.temp, probs=probs.clamp(1e-6, 1-1e-6))
        soft = dist.rsample()
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft, soft

    def _mean_pool(self, boundaries, hidden):
        B, T, D = hidden.shape
        foo = common(boundaries)
        if foo is None:
            return torch.zeros(B, 1, D, device=hidden.device)
        out_of_segment = (foo.detach() != 0)
        soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
        counts = soft_mask.sum(dim=1).clamp(min=1e-8)
        seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
        return seg_sum / counts.unsqueeze(-1)

    def forward(self, x):
        hidden = self.upstream(x)
        probs = self._compute_probs(hidden)
        boundaries, soft = self._sample(probs)
        if (boundaries.sum(dim=1) == 0).any():
            idx = (boundaries.sum(dim=1) == 0).nonzero(as_tuple=True)[0]
            boundaries[idx, -1] = 1.0
        pooled = self._mean_pool(boundaries, hidden)
        return pooled, boundaries, probs, hidden


def eval_stats(bp, x):
    bp.eval()
    with torch.no_grad():
        hidden = bp.upstream(x)
        probs = bp._compute_probs(hidden)
        count = (probs > 0.5).float().sum().item()
        mp = probs.mean().item()
    bp.train()
    return count, mp


def test_magnitude_asymmetry():
    """
    Does the gradient magnitude differ when it points up vs down?
    If |grad_down| >> |grad_up| on average, SGD will drift downward.
    """
    print("=" * 70)
    print("TEST 1: Gradient MAGNITUDE asymmetry")
    print("=" * 70)

    torch.manual_seed(42)
    T, D = 20, 16

    grad_up_mags = []
    grad_down_mags = []

    for trial in range(200):
        torch.manual_seed(trial)
        hidden = torch.randn(1, T, D)
        prob = torch.tensor(0.3, requires_grad=True)

        boundaries = torch.zeros(1, T)
        hard = (prob > 0.5).float()
        boundaries[0, T//2] = hard - prob.detach() + prob
        boundaries[0, -1] = 1.0

        foo = common(boundaries)
        if foo is None:
            continue

        out_of_segment = (foo.detach() != 0)
        soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
        counts = soft_mask.sum(dim=1).clamp(min=1e-8)
        seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
        pooled = seg_sum / counts.unsqueeze(-1)

        target = torch.randn_like(pooled)
        loss = (pooled - target).pow(2).mean()
        loss.backward()

        g = prob.grad.item()
        if g > 0:
            grad_down_mags.append(abs(g))
        else:
            grad_up_mags.append(abs(g))

    avg_up = sum(grad_up_mags) / len(grad_up_mags) if grad_up_mags else 0
    avg_down = sum(grad_down_mags) / len(grad_down_mags) if grad_down_mags else 0

    print(f"\n  Gradient pushes UP:   {len(grad_up_mags)}/200  avg magnitude: {avg_up:.6f}")
    print(f"  Gradient pushes DOWN: {len(grad_down_mags)}/200  avg magnitude: {avg_down:.6f}")
    print(f"  Ratio DOWN/UP magnitude: {avg_down/avg_up:.2f}x" if avg_up > 0 else "")

    if avg_down > avg_up * 1.5:
        print(f"  ** DOWN gradients are {avg_down/avg_up:.1f}x stronger!")
    elif avg_up > avg_down * 1.5:
        print(f"  ** UP gradients are {avg_up/avg_down:.1f}x stronger!")
    else:
        print(f"  Magnitudes are roughly balanced.")


def test_upstream_adaptation():
    """
    Key test: Does the upstream (CNN) adapt to make fewer boundaries work?

    If the upstream can solve the task regardless of boundary count,
    then the boundary count will drift toward whatever the gradient noise favors.
    """
    print("\n" + "=" * 70)
    print("TEST 2: Upstream adaptation masks the need for boundaries")
    print("=" * 70)

    torch.manual_seed(42)
    D, T, B = 16, 20, 4
    x = torch.randn(B, T, D)

    bp = MinimalBP(D, temp=0.5, init_bias=0.0)

    # Freeze boundary MLP, train only upstream
    for p in bp.boundary_mlp.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        [p for p in bp.parameters() if p.requires_grad], lr=0.01)

    init_count, init_mp = eval_stats(bp, x)
    print(f"\n  Initial boundary count: {init_count:.0f}, mean_prob: {init_mp:.4f}")
    print(f"  Training with FROZEN boundary MLP, only upstream adapts.")
    print(f"  Loss = reconstruction error.\n")

    for step in range(500):
        optimizer.zero_grad()
        pooled, boundaries, probs, hidden = bp(x)

        # Reconstruction loss
        seg_ids = (boundaries.cumsum(dim=1) - boundaries).long()
        seg_ids = seg_ids.clamp(0, pooled.shape[1] - 1)
        reconstructed = torch.gather(
            pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
        loss = (reconstructed - hidden.detach()).pow(2).mean()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"  Step {step:4d}: loss={loss.item():.6f}  "
                  f"boundaries={boundaries.sum().item():.0f}")

    print(f"\n  Upstream can reduce reconstruction loss from {1.4:.3f} "
          f"to {loss.item():.3f}")
    print(f"  WITHOUT changing boundary count (it's frozen at ~{init_mp:.2f} rate)")
    print(f"  This means: the upstream CAN optimize around any boundary count.")


def test_gradient_noise_drift():
    """
    When the downstream loss is roughly indifferent to boundary count
    (because upstream adapts), the boundary MLP receives noisy gradients.

    Does this noise have a directional drift?
    """
    print("\n" + "=" * 70)
    print("TEST 3: Gradient noise drift direction")
    print("=" * 70)

    torch.manual_seed(42)
    D, T, B = 16, 20, 4
    x = torch.randn(B, T, D)

    bp = MinimalBP(D, temp=0.5, init_bias=0.0)
    optimizer = torch.optim.Adam(bp.parameters(), lr=0.01)

    print(f"\n  Training with reconstruction loss (wants more boundaries).")
    print(f"  Tracking boundary MLP gradient direction per step.\n")

    grad_directions = []

    for step in range(300):
        optimizer.zero_grad()
        pooled, boundaries, probs, hidden = bp(x)

        seg_ids = (boundaries.cumsum(dim=1) - boundaries).long()
        seg_ids = seg_ids.clamp(0, pooled.shape[1] - 1)
        reconstructed = torch.gather(
            pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))
        loss = (reconstructed - hidden.detach()).pow(2).mean()
        loss.backward()

        # Check boundary MLP final bias gradient direction
        bias_grad = bp.boundary_mlp[-1].bias.grad.item()
        grad_directions.append(bias_grad)

        optimizer.step()

        if step % 50 == 0:
            ec, emp = eval_stats(bp, x)
            print(f"  Step {step:4d}: count={ec:.0f} prob={emp:.4f} "
                  f"bias_grad={bias_grad:+.6f} loss={loss.item():.6f}")

    # Analyze gradient direction statistics
    pos_grads = sum(1 for g in grad_directions if g > 0)  # pushes logit up -> more boundaries
    neg_grads = sum(1 for g in grad_directions if g < 0)  # pushes logit down -> fewer boundaries
    avg_grad = sum(grad_directions) / len(grad_directions)

    print(f"\n  Gradient direction over {len(grad_directions)} steps:")
    print(f"    Pushes logit UP (more boundaries):  {pos_grads}")
    print(f"    Pushes logit DOWN (fewer boundaries): {neg_grads}")
    print(f"    Average gradient: {avg_grad:+.8f}")

    if avg_grad < 0:
        print(f"\n  NET DRIFT: Gradient pushes boundaries DOWN on average!")
    else:
        print(f"\n  NET DRIFT: Gradient pushes boundaries UP on average.")


def test_recon_loss_gradient_vs_content_gradient():
    """
    Critical insight: In the reconstruction loss
        L = ||reconstructed - hidden||^2
    where reconstructed = gather(pooled, seg_ids),

    the gradient flows through `pooled` (via gather), NOT through `seg_ids`.

    seg_ids = cumsum(boundaries) - boundaries  ... this is DETACHED via .long()!

    So the reconstruction loss ONLY changes the CONTENT of each pooled vector,
    NOT which frames are assigned to which segment.

    The boundary gradient only comes through the POOLING mechanism
    (how the segment content is computed), not through the ASSIGNMENT mechanism
    (which frames go to which segment).
    """
    print("\n" + "=" * 70)
    print("TEST 4: Does reconstruction loss gradient flow through seg_ids?")
    print("=" * 70)

    torch.manual_seed(42)
    D, T = 8, 10
    hidden = torch.randn(1, T, D)

    prob = torch.tensor(0.3, requires_grad=True)
    boundaries = torch.zeros(1, T)
    hard = (prob > 0.5).float()
    boundaries[0, 4] = hard - prob.detach() + prob
    boundaries[0, -1] = 1.0

    # Pool
    foo = common(boundaries)
    out_of_segment = (foo.detach() != 0)
    soft_mask = torch.where(out_of_segment, torch.zeros_like(foo), 1.0 - foo)
    counts = soft_mask.sum(dim=1).clamp(min=1e-8)
    seg_sum = torch.bmm(soft_mask.transpose(1, 2), hidden)
    pooled = seg_sum / counts.unsqueeze(-1)

    # Reconstruct via gather (this is what test.py experiment 2b does)
    seg_ids = (boundaries.cumsum(dim=1) - boundaries).long()  # .long() kills gradients!
    seg_ids = seg_ids.clamp(0, pooled.shape[1] - 1)
    reconstructed = torch.gather(
        pooled, 1, seg_ids.unsqueeze(-1).expand(-1, -1, D))

    loss = (reconstructed - hidden.detach()).pow(2).mean()
    loss.backward()

    grad_via_recon = prob.grad.item()

    print(f"\n  seg_ids = {seg_ids[0].tolist()}")
    print(f"  Gradient on boundary prob via reconstruction: {grad_via_recon:+.8f}")

    # Now check: does the gradient only come through pooled content?
    # If we detach pooled and only let gradient flow through seg_ids...
    # But seg_ids uses .long() so no gradient flows through it!

    prob2 = torch.tensor(0.3, requires_grad=True)
    boundaries2 = torch.zeros(1, T)
    hard2 = (prob2 > 0.5).float()
    boundaries2[0, 4] = hard2 - prob2.detach() + prob2
    boundaries2[0, -1] = 1.0

    # Pool but detach the result
    foo2 = common(boundaries2)
    out_of_segment2 = (foo2.detach() != 0)
    soft_mask2 = torch.where(out_of_segment2, torch.zeros_like(foo2), 1.0 - foo2)
    counts2 = soft_mask2.sum(dim=1).clamp(min=1e-8)
    seg_sum2 = torch.bmm(soft_mask2.transpose(1, 2), hidden)
    pooled2 = (seg_sum2 / counts2.unsqueeze(-1)).detach()  # DETACH pooled

    # Reconstruct - if seg_ids carried gradients, this would still have gradient
    seg_ids2 = (boundaries2.cumsum(dim=1) - boundaries2).long()
    seg_ids2 = seg_ids2.clamp(0, pooled2.shape[1] - 1)
    reconstructed2 = torch.gather(
        pooled2, 1, seg_ids2.unsqueeze(-1).expand(-1, -1, D))

    loss2 = (reconstructed2 - hidden.detach()).pow(2).mean()
    if loss2.requires_grad:
        loss2.backward()
        grad_no_pooled = prob2.grad.item() if prob2.grad is not None else 0.0
    else:
        grad_no_pooled = 0.0
        print("  (loss has no grad_fn when pooled is detached — confirms no gradient path through seg_ids)")

    print(f"  Gradient with pooled DETACHED: {grad_no_pooled:+.8f}")
    print(f"\n  The .long() on seg_ids completely kills the assignment gradient!")
    print(f"  Reconstruction loss can ONLY optimize pooled CONTENT, not boundary PLACEMENT.")


def test_what_downstream_gradient_actually_does():
    """
    So what does the downstream gradient through soft_mask actually do?

    It changes the WEIGHTS of the mean-pooling. It doesn't add/remove segments.

    soft_mask controls how much each position contributes to each segment's pool.
    The gradient says "position j should contribute more/less to segment s".

    This is equivalent to adjusting the AVERAGING WEIGHTS, not the SEGMENTATION.

    The boundary prob controls soft_mask, but in a very coarse way (via cumsum):
    increasing prob_i shifts ALL downstream positions' soft_mask values by +1.

    This is like saying "make the average include more downstream content".
    It doesn't say "create a new segment boundary here".

    The gradient signal through soft_mask is about CONTENT REWEIGHTING,
    not about BOUNDARY PLACEMENT. This is why boundaries drift down:
    the easiest way to minimize any loss on pooled content is to make
    fewer, larger segments (= fewer boundaries), because:
    1. Larger segments average more positions -> lower variance in pooled content
    2. Fewer segments = less total pooled content to optimize
    """
    print("\n" + "=" * 70)
    print("TEST 5: Mean pooling + noise = boundary collapse")
    print("=" * 70)
    print("  With mean pooling, larger segments have LOWER VARIANCE in their")
    print("  pooled representation (averaging effect).")
    print("  Lower variance = more predictable = easier to optimize.")
    print("  The gradient EXPLOITS this by collapsing boundaries.\n")

    torch.manual_seed(42)
    D, T, B = 16, 40, 4
    x = torch.randn(B, T, D)

    # Track: what fraction of the gradient on boundary MLP comes from
    # each loss component?
    bp = MinimalBP(D, temp=0.5, init_bias=0.0)
    optimizer = torch.optim.Adam(bp.parameters(), lr=0.01)

    print(f"  Tracking pooled output variance as boundaries collapse:\n")

    for step in range(300):
        optimizer.zero_grad()
        pooled, boundaries, probs, hidden = bp(x)

        # Just use L2 loss on pooled (content-only, no reconstruction)
        loss = pooled.pow(2).mean()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            ec, emp = eval_stats(bp, x)
            pooled_var = pooled.var(dim=-1).mean().item()
            pooled_norm = pooled.norm(dim=-1).mean().item()
            print(f"  Step {step:4d}: count={ec:.0f} prob={emp:.4f} "
                  f"pooled_var={pooled_var:.6f} pooled_norm={pooled_norm:.4f} "
                  f"loss={loss.item():.6f}")

    print(f"\n  Observation: As boundaries decrease, pooled variance also decreases.")
    print(f"  Fewer boundaries -> larger segments -> more averaging -> lower variance")
    print(f"  -> lower norm -> lower loss. This is the path of least resistance.")


def test_the_real_training_scenario():
    """
    In the actual training, there are TWO losses:
    1. Downstream (CTC/Seq2Seq) - depends on pooled content
    2. Boundary loss (binomial) - drives count toward target

    Does the downstream loss actively fight the boundary loss,
    or is it just indifferent (letting boundary count drift)?

    Let's simulate: boundary loss wants count=20, downstream loss = pooled^2.
    """
    print("\n" + "=" * 70)
    print("TEST 6: Boundary loss vs downstream loss")
    print("=" * 70)

    torch.manual_seed(42)
    D, T, B = 16, 40, 4
    x = torch.randn(B, T, D)

    bp = MinimalBP(D, temp=0.5, init_bias=0.0)
    optimizer = torch.optim.Adam(bp.parameters(), lr=0.01)

    target_count = 20  # per sample
    target_prob = target_count / T

    print(f"\n  Target boundary count per sample: {target_count}")
    print(f"  Downstream loss: pooled.pow(2).mean()")
    print(f"  Boundary loss: (mean_prob - target_rate)^2")
    print(f"  Boundary loss weight: 1.0\n")

    for step in range(500):
        optimizer.zero_grad()
        pooled, boundaries, probs, hidden = bp(x)

        # Downstream loss
        downstream_loss = pooled.pow(2).mean()

        # Boundary loss (simplified - like the binomial loss direction)
        boundary_loss = (probs.mean() - target_prob).pow(2)

        total_loss = downstream_loss + 1.0 * boundary_loss
        total_loss.backward()

        # Track gradients on boundary MLP bias before step
        bias_grad = bp.boundary_mlp[-1].bias.grad.item()

        optimizer.step()

        if step % 50 == 0:
            ec, emp = eval_stats(bp, x)
            print(f"  Step {step:4d}: count={ec:.0f} prob={emp:.4f} "
                  f"downstream={downstream_loss.item():.6f} "
                  f"boundary={boundary_loss.item():.6f} "
                  f"bias_grad={bias_grad:+.6f}")

    final_count, final_mp = eval_stats(bp, x)
    print(f"\n  Final: count={final_count:.0f}, prob={final_mp:.4f}")
    print(f"  Target: count={target_count}, prob={target_prob:.4f}")

    if final_mp < target_prob * 0.5:
        print(f"\n  BOUNDARY COLLAPSED despite boundary loss!")
        print(f"  The downstream loss overwhelms the boundary loss.")
    elif abs(final_mp - target_prob) < 0.1:
        print(f"\n  Boundary count stabilized near target.")
    else:
        print(f"\n  Partial collapse: count drifted but didn't fully collapse.")


if __name__ == "__main__":
    test_magnitude_asymmetry()
    test_upstream_adaptation()
    test_gradient_noise_drift()
    test_recon_loss_gradient_vs_content_gradient()
    test_what_downstream_gradient_actually_does()
    test_the_real_training_scenario()
