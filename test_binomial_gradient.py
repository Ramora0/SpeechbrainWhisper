#!/usr/bin/env python3
"""
Focused test: Does the BINOMIAL LOSS itself have a gradient bias that
pushes boundaries down?

The binomial loss operates on:
  - num_boundaries = hard_boundaries.sum(dim=1)  (STE-linked to probs)
  - target_counts = phoneme counts from g2p
  - total_positions = sequence length

The loss is: -log P(X = num_boundaries | n=total, p=target/total) / total

The gradient flows: loss -> num_boundaries -> STE -> soft_boundaries -> probs -> MLP

KEY QUESTION: When num_boundaries matches target, is the gradient zero?
Or does it have a bias?

Also: The STE is hard_boundaries = hard_samples - soft.detach() + soft
So gradient of hard_boundaries.sum() = gradient of soft_boundaries.sum()
= sum of all soft boundary values

But soft_boundaries come from RelaxedBernoulli sampling, not directly from probs.
The gradient through RelaxedBernoulli may itself have a bias.
"""

import torch
import torch.nn as nn


def binomial_loss_from_target_counts(num_boundaries, total_positions, target_counts, eps=1e-6):
    """Copy of loss.py"""
    device = num_boundaries.device
    num_boundaries = num_boundaries.to(dtype=torch.float32)
    total_positions = total_positions.to(device=device, dtype=torch.float32) if not isinstance(total_positions, torch.Tensor) else total_positions.to(dtype=torch.float32)
    target_counts = target_counts.to(device=device, dtype=torch.float32) if not isinstance(target_counts, torch.Tensor) else target_counts.to(dtype=torch.float32)

    clamped_totals = total_positions.clamp(min=1.0)
    clamped_targets = torch.minimum(target_counts, clamped_totals)
    target_probs = (clamped_targets / clamped_totals).clamp(min=eps, max=1-eps)

    binomial = torch.distributions.binomial.Binomial(
        total_count=clamped_totals, probs=target_probs)
    log_prob = binomial.log_prob(num_boundaries)

    loss = -log_prob
    return loss / clamped_totals


def test_binomial_loss_gradient_at_target():
    """
    When num_boundaries == target, what is the gradient on each boundary?

    The binomial log_prob: log C(n,k) + k*log(p) + (n-k)*log(1-p)
    d/dk = log(p) - log(1-p) = log(p/(1-p)) = logit(p)

    When k = n*p (at target): logit(p) is NOT zero unless p=0.5.
    For p < 0.5 (target < total/2), logit(p) < 0 -> gradient pushes k UP.
    For p > 0.5, logit(p) > 0 -> gradient pushes k DOWN.

    So the binomial loss gradient at the target is zero ONLY for p=0.5.
    For our case, p = target_count / total_positions ~ 0.192, so p < 0.5.
    logit(0.192) = log(0.192/0.808) = log(0.238) = -1.44

    This means: at the target count, the binomial loss gradient actually
    pushes boundaries UP (logit < 0 -> -loss has positive gradient on k).

    Wait, let me be more careful. The loss is -log_prob / total.
    d(loss)/d(k) = -d(log_prob)/d(k) / total = -(log(p) - log(1-p)) / total
    = -logit(p) / total

    For p=0.192: -logit(0.192) / total = -(-1.44) / total = +1.44/total > 0
    Positive gradient on k means gradient descent DECREASES k!

    Hmm, but k is the actual count, not directly a learnable parameter.
    Through STE: dk/d(boundary_i) = 1 (since k = sum of boundaries).

    So d(loss)/d(boundary_i) = d(loss)/d(k) = +1.44/total > 0
    Gradient descent: boundary_i -= lr * (+1.44/total)
    -> boundary values DECREASE -> probs DECREASE -> count DROPS

    BUT WAIT: this is the gradient at exactly k = target.
    When k < target, the gradient should push k UP.
    When k > target, the gradient should push k DOWN.

    The binomial loss IS a proper loss - it's minimized when k = target.
    The issue is that d(loss)/d(k) at k=target is NOT zero for discrete k!

    Actually for continuous k, d(loss)/dk = -(log(p/(1-p)) + psi(n-k+1) - psi(k+1))
    where psi is the digamma function. At k = n*p, this IS approximately zero.

    Let me just compute it numerically.
    """
    print("=" * 70)
    print("TEST: Binomial loss gradient at and around target count")
    print("=" * 70)

    total = torch.tensor(100.0)
    target = torch.tensor(20.0)  # target_prob = 0.2

    print(f"\n  total_positions = {total.item():.0f}")
    print(f"  target_count = {target.item():.0f}")
    print(f"  target_prob = {target.item()/total.item():.3f}")

    print(f"\n  {'count':>8}  {'loss':>10}  {'d(loss)/d(count)':>18}  {'direction':>12}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*18}  {'-'*12}")

    for count_val in [5, 10, 15, 18, 19, 20, 21, 22, 25, 30, 40, 50]:
        count = torch.tensor(float(count_val), requires_grad=True)
        loss = binomial_loss_from_target_counts(count, total, target)
        loss.backward()
        grad = count.grad.item()
        direction = "DECREASE" if grad > 0 else "INCREASE"
        at_target = " <-- TARGET" if count_val == 20 else ""
        print(f"  {count_val:>8}  {loss.item():>10.6f}  {grad:>+18.8f}  {direction:>12}{at_target}")


def test_binomial_through_STE():
    """
    More realistic: gradient through RelaxedBernoulli + STE.

    probs -> RelaxedBernoulli -> soft -> hard (STE) -> sum -> binomial_loss

    Does the STE + sampling introduce additional bias?
    """
    print("\n" + "=" * 70)
    print("TEST: Gradient through RelaxedBernoulli + STE + binomial loss")
    print("=" * 70)

    T = 100
    target = torch.tensor(20.0)
    total = torch.tensor(float(T))

    for temp in [1.0, 0.5, 0.1, 0.01]:
        print(f"\n  Temperature = {temp}")

        # Average gradient over many samples
        up_grads = []
        down_grads = []

        for trial in range(100):
            torch.manual_seed(trial)

            # Create probs that should give ~20 boundaries (p=0.2)
            logit = torch.tensor(0.0, requires_grad=True)  # sigmoid(0) = 0.5
            # Adjust to get p ≈ 0.2: logit = log(0.2/0.8) = -1.386
            target_logit = torch.tensor(-1.386, requires_grad=True)
            probs = torch.sigmoid(target_logit).expand(T)

            # Sample
            dist = torch.distributions.RelaxedBernoulli(
                temperature=temp, probs=probs.clamp(1e-6, 1-1e-6))
            soft = dist.rsample()
            hard = (soft > 0.5).float()
            boundaries = hard - soft.detach() + soft  # STE

            count = boundaries.sum()
            loss = binomial_loss_from_target_counts(
                count.unsqueeze(0), total.unsqueeze(0), target.unsqueeze(0))
            loss.backward()

            grad = target_logit.grad.item()
            actual_count = hard.sum().item()

            if grad > 0:
                down_grads.append((grad, actual_count))
            else:
                up_grads.append((grad, actual_count))

        avg_up = sum(g for g, _ in up_grads) / len(up_grads) if up_grads else 0
        avg_down = sum(g for g, _ in down_grads) / len(down_grads) if down_grads else 0
        avg_count_up = sum(c for _, c in up_grads) / len(up_grads) if up_grads else 0
        avg_count_down = sum(c for _, c in down_grads) / len(down_grads) if down_grads else 0
        overall_avg = sum(g for g, _ in up_grads + down_grads) / (len(up_grads) + len(down_grads))

        print(f"    UP grads:   {len(up_grads):3d}/100  avg_grad={avg_up:+.6f}  avg_count={avg_count_up:.1f}")
        print(f"    DOWN grads: {len(down_grads):3d}/100  avg_grad={avg_down:+.6f}  avg_count={avg_count_down:.1f}")
        print(f"    Overall avg gradient: {overall_avg:+.8f}")
        print(f"    Net direction: {'DOWN (fewer boundaries)' if overall_avg > 0 else 'UP (more boundaries)'}")


def test_binomial_loss_vs_mse_loss():
    """
    Compare the binomial loss gradient with a simple MSE loss on count.

    MSE: (count - target)^2 / total
    Binomial: -log P(X=count | n=total, p=target/total) / total

    For the binomial, the gradient at count=target is NOT exactly zero
    because the distribution is discrete (the mode may not equal n*p exactly).
    """
    print("\n" + "=" * 70)
    print("TEST: Binomial loss gradient vs MSE at various counts")
    print("=" * 70)

    total = torch.tensor(100.0)
    target = torch.tensor(20.0)

    print(f"\n  {'count':>8}  {'binomial_grad':>14}  {'mse_grad':>14}  {'binom_dir':>10}  {'mse_dir':>10}")
    print(f"  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*10}  {'-'*10}")

    for count_val in [5, 10, 15, 18, 19, 20, 21, 22, 25, 30, 40]:
        # Binomial gradient
        c1 = torch.tensor(float(count_val), requires_grad=True)
        loss1 = binomial_loss_from_target_counts(c1, total, target)
        loss1.backward()
        bgrad = c1.grad.item()

        # MSE gradient
        c2 = torch.tensor(float(count_val), requires_grad=True)
        loss2 = (c2 - target).pow(2) / total
        loss2.backward()
        mgrad = c2.grad.item()

        bdir = "DOWN" if bgrad > 0 else "UP"
        mdir = "DOWN" if mgrad > 0 else "UP"

        print(f"  {count_val:>8}  {bgrad:>+14.8f}  {mgrad:>+14.8f}  {bdir:>10}  {mdir:>10}")


def test_gradient_at_low_counts():
    """
    Critical scenario: what happens when count is already BELOW target?

    The binomial loss should push it back UP, but does it?
    And how does this interact with the STE + sampling?
    """
    print("\n" + "=" * 70)
    print("TEST: Can binomial loss recover from low boundary counts?")
    print("=" * 70)

    T = 100
    target = torch.tensor(20.0)
    total = torch.tensor(float(T))

    for start_logit in [-4.0, -3.0, -2.0, -1.0]:
        start_prob = torch.sigmoid(torch.tensor(start_logit)).item()
        expected_count = start_prob * T

        print(f"\n  Starting logit={start_logit:.1f}, prob={start_prob:.4f}, "
              f"expected_count={expected_count:.1f}")

        logit = torch.tensor(start_logit, requires_grad=True)
        optimizer = torch.optim.SGD([logit], lr=0.1)

        for step in range(20):
            optimizer.zero_grad()
            probs = torch.sigmoid(logit).expand(T)

            dist = torch.distributions.RelaxedBernoulli(
                temperature=0.5, probs=probs.clamp(1e-6, 1-1e-6))
            soft = dist.rsample()
            hard = (soft > 0.5).float()
            boundaries = hard - soft.detach() + soft

            count = boundaries.sum()
            loss = binomial_loss_from_target_counts(
                count.unsqueeze(0), total.unsqueeze(0), target.unsqueeze(0))
            loss.backward()
            optimizer.step()

            if step % 5 == 0:
                current_prob = torch.sigmoid(logit).item()
                print(f"    Step {step:3d}: logit={logit.item():+.4f} "
                      f"prob={current_prob:.4f} count={hard.sum().item():.0f} "
                      f"grad={logit.grad.item():+.6f}")

        final_prob = torch.sigmoid(logit).item()
        target_prob = target.item() / T
        print(f"    Final: prob={final_prob:.4f} (target={target_prob:.4f})")
        if final_prob > start_prob:
            print(f"    Recovered: prob went UP from {start_prob:.4f} to {final_prob:.4f}")
        else:
            print(f"    FAILED to recover: prob went DOWN from {start_prob:.4f} to {final_prob:.4f}")


if __name__ == "__main__":
    test_binomial_loss_gradient_at_target()
    test_binomial_through_STE()
    test_binomial_loss_vs_mse_loss()
    test_gradient_at_low_counts()
