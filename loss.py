import torch
import flags


def _to_device_tensor(value, device):
    """Convert a value to a tensor on the specified device."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(value, device=device, dtype=torch.float32)


def binomial_loss_from_target_counts(num_boundaries, total_positions, target_counts, eps=1e-6):
    print("=================BOOOOO====================")
    """Binomial loss where the expected count matches target boundary counts."""
    if not isinstance(num_boundaries, torch.Tensor):
        raise ValueError(
            "num_boundaries must be a tensor for per-example loss computation")

    if flags.PRINT_BP_LOSS_CHECKS:
        print("[binomial_loss] Running checks...")

    device = num_boundaries.device
    num_boundaries = num_boundaries.to(dtype=torch.float32)
    total_positions = _to_device_tensor(total_positions, device)
    target_counts = _to_device_tensor(target_counts, device)

    if flags.PRINT_DATA:
        print(f"[binomial_loss] num_boundaries: {num_boundaries}")
        print(f"[binomial_loss] total_positions: {total_positions}")
        print(f"[binomial_loss] target_counts: {target_counts}")

    clamped_totals = total_positions.clamp(min=1.0)
    clamped_targets = torch.minimum(target_counts, clamped_totals)
    target_probs = (clamped_targets /
                    clamped_totals).clamp(min=eps, max=1 - eps)

    if flags.PRINT_DATA:
        print(f"[binomial_loss] clamped_totals: {clamped_totals}")
        print(f"[binomial_loss] clamped_targets: {clamped_targets}")
        print(f"[binomial_loss] target_probs: {target_probs}")

    # Check if num_boundaries is valid for the binomial distribution
    if (num_boundaries > clamped_totals).any():
        if flags.PRINT_BP_LOSS_CHECKS:
            print(f"[binomial_loss] ERROR: num_boundaries > clamped_totals!")
            invalid_mask = num_boundaries > clamped_totals
            print(f"  Invalid boundaries: {num_boundaries[invalid_mask]}")
            print(f"  Corresponding totals: {clamped_totals[invalid_mask]}")

    if (num_boundaries < 0).any():
        if flags.PRINT_BP_LOSS_CHECKS:
            print(f"[binomial_loss] ERROR: num_boundaries < 0!")
            print(
                f"  Negative boundaries: {num_boundaries[num_boundaries < 0]}")

    binomial = torch.distributions.binomial.Binomial(
        total_count=clamped_totals,
        probs=target_probs
    )
    log_prob = binomial.log_prob(num_boundaries)

    if flags.PRINT_DATA:
        print(f"[binomial_loss] log_prob: {log_prob}")
    if flags.PRINT_NAN_INF:
        print(
            f"[binomial_loss] log_prob has NaN: {torch.isnan(log_prob).any()}")
        print(
            f"[binomial_loss] log_prob has Inf: {torch.isinf(log_prob).any()}")

    loss = -log_prob
    final_loss = loss / clamped_totals

    if flags.PRINT_DATA:
        print(f"[binomial_loss] loss before division: {loss}")
        print(f"[binomial_loss] final_loss: {final_loss}")
    if flags.PRINT_NAN_INF:
        print(
            f"[binomial_loss] final_loss has NaN: {torch.isnan(final_loss).any()}")
        print(
            f"[binomial_loss] final_loss has Inf: {torch.isinf(final_loss).any()}")

    return final_loss
