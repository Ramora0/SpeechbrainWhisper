import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormer(nn.Module):
    """Fixed-rate compression via chunked mean-pool queries + cross-attention.

    A drop-in replacement for BoundaryPredictor2 that uses deterministic
    fixed-rate compression instead of learned boundary detection.

    Args:
        input_dim: Dimension of input features.
        compression_rate: Integer k — compress by k× (e.g. 4 means 4× compression).
        num_heads: Number of attention heads for cross-attention.
        dropout: Dropout rate for attention weights.
    """

    def __init__(self, input_dim, compression_rate, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.compression_rate = compression_rate
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        assert input_dim % num_heads == 0, (
            f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, input_dim)

        self.norm_q = nn.LayerNorm(input_dim)
        self.norm_kv = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden,
        lengths,
        target_boundary_counts=None,
        return_unreduced_boundary_loss=False,
    ):
        """Forward pass — chunk, mean-pool, cross-attend.

        Args:
            hidden: (B, T, D) input tensor.
            lengths: (B,) relative lengths in [0, 1].
            target_boundary_counts: Ignored (kept for API compatibility).
            return_unreduced_boundary_loss: Ignored.

        Returns:
            Tuple of (compressed, loss, num_boundaries, total_positions,
                      new_lengths, cv, adjacent_pct) matching BoundaryPredictor2.
        """
        B, T, D = hidden.shape
        k = self.compression_rate
        num_chunks = math.ceil(T / k)

        # --- Build per-position mask from relative lengths ---
        abs_lengths = (lengths * T).long().clamp(min=1)  # (B,)
        # (B, T) — True for valid positions
        pos_mask = torch.arange(T, device=hidden.device).unsqueeze(0) < abs_lengths.unsqueeze(1)

        # --- Chunk & mean-pool to create queries ---
        # Pad hidden and mask to be evenly divisible by k
        pad_len = num_chunks * k - T
        if pad_len > 0:
            hidden_padded = F.pad(hidden, (0, 0, 0, pad_len))  # (B, num_chunks*k, D)
            mask_padded = F.pad(pos_mask, (0, pad_len), value=False)  # (B, num_chunks*k)
        else:
            hidden_padded = hidden
            mask_padded = pos_mask

        # Reshape into chunks: (B, num_chunks, k, D)
        hidden_chunks = hidden_padded.reshape(B, num_chunks, k, D)
        mask_chunks = mask_padded.reshape(B, num_chunks, k)  # (B, num_chunks, k)

        # Masked mean-pool each chunk
        mask_chunks_f = mask_chunks.float().unsqueeze(-1)  # (B, num_chunks, k, 1)
        chunk_sums = (hidden_chunks * mask_chunks_f).sum(dim=2)  # (B, num_chunks, D)
        chunk_counts = mask_chunks_f.sum(dim=2).clamp(min=1)  # (B, num_chunks, 1)
        queries = chunk_sums / chunk_counts  # (B, num_chunks, D)

        # --- Cross-attention: queries attend to all input tokens ---
        q = self.norm_q(queries)
        kv_input = self.norm_kv(hidden)

        q = self.q_proj(q)  # (B, num_chunks, D)
        k_att = self.k_proj(kv_input)  # (B, T, D)
        v = self.v_proj(kv_input)  # (B, T, D)

        # Reshape for multi-head attention
        q = q.reshape(B, num_chunks, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, num_chunks, head_dim)
        k_att = k_att.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, head_dim)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k_att.transpose(-2, -1)) / scale  # (B, H, num_chunks, T)

        # Mask out padded key positions
        key_mask = ~pos_mask  # (B, T) — True for padded positions
        attn = attn.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, num_chunks, head_dim)
        out = out.transpose(1, 2).reshape(B, num_chunks, D)  # (B, num_chunks, D)
        compressed = self.out_proj(out)  # (B, num_chunks, D)

        # --- Compute new relative lengths ---
        # Number of non-empty chunks per sequence
        abs_new_lengths = torch.ceil(abs_lengths.float() / k).long().clamp(min=1)
        new_lengths = abs_new_lengths.float() / num_chunks  # relative lengths
        new_lengths = new_lengths.clamp(max=1.0)

        # --- Dummy outputs for API compatibility ---
        loss = torch.tensor(0.0, device=hidden.device)
        num_boundaries = abs_new_lengths.sum().item()
        total_positions = abs_lengths.sum().item()
        cv = None
        adjacent_pct = None

        return compressed, loss, num_boundaries, total_positions, new_lengths, cv, adjacent_pct
