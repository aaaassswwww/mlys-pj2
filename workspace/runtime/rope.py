"""RoPE utilities."""

from __future__ import annotations

import torch


def apply_rotary_pos_emb(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_ids: torch.Tensor,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = query_states.size(-1)
    if head_dim % 2 != 0:
        raise ValueError("RoPE head_dim must be even")

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=query_states.device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("bt,d->btd", position_ids.to(torch.float32), inv_freq)
    cos = freqs.cos().unsqueeze(1).to(query_states.dtype)
    sin = freqs.sin().unsqueeze(1).to(query_states.dtype)
    return rotate_interleaved(query_states, cos, sin), rotate_interleaved(key_states, cos, sin)


def rotate_interleaved(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    rotated = torch.stack(
        (
            even * cos - odd * sin,
            even * sin + odd * cos,
        ),
        dim=-1,
    )
    return rotated.flatten(-2)
