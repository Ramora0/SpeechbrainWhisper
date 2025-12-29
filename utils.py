import torch


def common(boundaries):
    """
    Create position-wise distance matrix for each segment.

    For each (position, segment) pair:
    - Distance = 0 if position is in the segment
    - Distance = negative if position is before segment (position - segment_start)
    - Distance = positive if position is after segment (position - segment_end)

    Args:
        boundaries: [B, L] tensor with 1s marking segment starts

    Returns:
        foo: [B, L, S] distance tensor where S is max number of segments
    """
    boundaries = boundaries.clone()
    n_segments = int(boundaries.sum(dim=-1).max().item())

    if n_segments == 0:
        return None

    batch_size, seq_len = boundaries.shape
    device = boundaries.device

    # Get segment index for each position [B, L]
    segment_ids = boundaries.cumsum(1) - boundaries

    # Position indices [L]
    positions = torch.arange(seq_len, device=device, dtype=torch.float)

    # Create segment index tensor [S]
    segment_indices = torch.arange(n_segments, device=device)

    # Create mask [B, L, S] - True where position belongs to segment
    in_segment = segment_ids.unsqueeze(
        -1) == segment_indices.unsqueeze(0).unsqueeze(0)

    # For each segment, find start position (min) and end position (max)
    # Use extreme values for positions not in segment
    pos_for_min = torch.where(
        in_segment,
        positions.view(1, -1, 1).expand(batch_size, -1, n_segments),
        torch.tensor(float('inf'), device=device)
    )
    pos_for_max = torch.where(
        in_segment,
        positions.view(1, -1, 1).expand(batch_size, -1, n_segments),
        torch.tensor(float('-inf'), device=device)
    )

    # Get min and max positions for each segment [B, S]
    seg_starts = pos_for_min.min(dim=1).values  # [B, S]
    seg_ends = pos_for_max.max(dim=1).values    # [B, S]

    # Expand positions for broadcasting [1, L, 1]
    pos_expanded = positions.view(1, -1, 1)

    # Compute position-wise distances [B, L, S]
    # If in segment: 0
    # If before segment: position - seg_start (negative)
    # If after segment: position - seg_end (positive)
    foo = torch.where(
        in_segment,
        torch.zeros_like(pos_expanded),
        torch.where(
            pos_expanded < seg_starts.unsqueeze(1),  # [B, 1, S]
            pos_expanded - seg_starts.unsqueeze(1),  # Before segment
            pos_expanded - seg_ends.unsqueeze(1)     # After segment
        )
    )

    return foo
