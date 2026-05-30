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
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().unsqueeze(1).to(query_states.dtype)
    sin = emb.sin().unsqueeze(1).to(query_states.dtype)

    query_states = (query_states * cos) + (rotate_half(query_states) * sin)
    key_states = (key_states * cos) + (rotate_half(key_states) * sin)
    return query_states, key_states


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first_half = hidden_states[..., : hidden_states.size(-1) // 2]
    second_half = hidden_states[..., hidden_states.size(-1) // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)

