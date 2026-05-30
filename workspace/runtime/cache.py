"""KV cache abstractions for incremental decode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class LayerKVCache:
    key: torch.Tensor
    value: torch.Tensor

    @property
    def seq_len(self) -> int:
        return int(self.key.size(-2))


@dataclass
class RequestKVCache:
    layers: list[LayerKVCache]

    @property
    def seq_len(self) -> int:
        if not self.layers:
            return 0
        return self.layers[0].seq_len

    def num_layers(self) -> int:
        return len(self.layers)


@dataclass
class CacheHandle:
    slot: int
    seq_len: int


class SlotKVCacheManager:
    def __init__(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.device = device
        self.dtype = dtype
        self.capacity_slots = 0
        self.capacity_seq_len = 0
        self.key_caches: list[torch.Tensor] = []
        self.value_caches: list[torch.Tensor] = []

    def ensure_capacity(self, required_slot_count: int, required_seq_len: int) -> None:
        with torch.inference_mode():
            if required_seq_len > self.max_position_embeddings:
                raise ValueError(
                    f"Requested sequence length {required_seq_len} exceeds max_position_embeddings={self.max_position_embeddings}"
                )

            required_slot_count = max(1, int(required_slot_count))
            required_seq_len = max(1, int(required_seq_len))
            if required_slot_count <= self.capacity_slots and required_seq_len <= self.capacity_seq_len:
                return

            new_slot_capacity = max(1, self.capacity_slots)
            while new_slot_capacity < required_slot_count:
                new_slot_capacity *= 2

            new_seq_capacity = max(16, self.capacity_seq_len)
            while new_seq_capacity < required_seq_len:
                new_seq_capacity *= 2
            new_seq_capacity = min(new_seq_capacity, self.max_position_embeddings)

            expanded_keys = []
            expanded_values = []
            if self.capacity_slots == 0:
                layer_pairs = [(None, None)] * self.num_layers
            else:
                layer_pairs = zip(self.key_caches, self.value_caches)

            for key_cache, value_cache in layer_pairs:
                new_key_cache = torch.empty(
                    (new_slot_capacity, self.num_kv_heads, new_seq_capacity, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                )
                new_value_cache = torch.empty(
                    (new_slot_capacity, self.num_kv_heads, new_seq_capacity, self.head_dim),
                    device=self.device,
                    dtype=self.dtype,
                )
                if key_cache is not None and value_cache is not None:
                    new_key_cache[: self.capacity_slots, :, : self.capacity_seq_len, :].copy_(
                        key_cache[: self.capacity_slots, :, : self.capacity_seq_len, :]
                    )
                    new_value_cache[: self.capacity_slots, :, : self.capacity_seq_len, :].copy_(
                        value_cache[: self.capacity_slots, :, : self.capacity_seq_len, :]
                    )
                expanded_keys.append(new_key_cache)
                expanded_values.append(new_value_cache)

            self.key_caches = expanded_keys
            self.value_caches = expanded_values
            self.capacity_slots = new_slot_capacity
            self.capacity_seq_len = new_seq_capacity

    def store_request_cache(self, slot: int, request_cache: RequestKVCache) -> CacheHandle:
        with torch.inference_mode():
            self.ensure_capacity(slot + 1, request_cache.seq_len)
            seq_len = request_cache.seq_len
            for layer_index, layer_cache in enumerate(request_cache.layers):
                self.key_caches[layer_index][slot, :, :seq_len, :].copy_(layer_cache.key.squeeze(0))
                self.value_caches[layer_index][slot, :, :seq_len, :].copy_(layer_cache.value.squeeze(0))
            return CacheHandle(slot=slot, seq_len=seq_len)

    def append_layer_tokens(
        self,
        layer_index: int,
        slot_ids: torch.Tensor,
        positions: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        with torch.inference_mode():
            self.ensure_capacity(int(slot_ids.max().item()) + 1, int(positions.max().item()) + 1)
            batch_indices = torch.arange(slot_ids.numel(), device=slot_ids.device)
            self.key_caches[layer_index][slot_ids, :, positions, :] = key_states[batch_indices, :, 0, :]
            self.value_caches[layer_index][slot_ids, :, positions, :] = value_states[batch_indices, :, 0, :]

    def get_layer_cache(self, layer_index: int, slot_ids: torch.Tensor, total_length: int) -> LayerKVCache:
        return LayerKVCache(
            key=self.key_caches[layer_index][slot_ids, :, :total_length, :],
            value=self.value_caches[layer_index][slot_ids, :, :total_length, :],
        )


def stack_request_caches(caches: Iterable[RequestKVCache]) -> RequestKVCache:
    caches = list(caches)
    if not caches:
        raise ValueError("Cannot stack an empty cache list")
    seq_lens = {cache.seq_len for cache in caches}
    if len(seq_lens) != 1:
        raise ValueError("All caches must have the same sequence length to stack")

    num_layers = caches[0].num_layers()
    stacked_layers: list[LayerKVCache] = []
    for layer_index in range(num_layers):
        keys = [cache.layers[layer_index].key for cache in caches]
        values = [cache.layers[layer_index].value for cache in caches]
        stacked_layers.append(
            LayerKVCache(
                key=torch.cat(keys, dim=0),
                value=torch.cat(values, dim=0),
            )
        )
    return RequestKVCache(stacked_layers)


def split_request_cache(cache: RequestKVCache) -> list[RequestKVCache]:
    if not cache.layers:
        return []
    batch_size = cache.layers[0].key.size(0)
    per_request_layers: list[list[LayerKVCache]] = [[] for _ in range(batch_size)]
    for layer in cache.layers:
        key_chunks = layer.key.split(1, dim=0)
        value_chunks = layer.value.split(1, dim=0)
        for request_layers, key_chunk, value_chunk in zip(per_request_layers, key_chunks, value_chunks):
            request_layers.append(
                LayerKVCache(
                    key=key_chunk,
                    value=value_chunk,
                )
            )
    return [RequestKVCache(layers) for layers in per_request_layers]
