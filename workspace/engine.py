"""Evaluator-facing engine entrypoint.

This module intentionally stays small and stable.
All runtime internals should live under workspace/runtime/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch

try:
    from .runtime.cache import CacheHandle, SlotKVCacheManager, split_request_cache, stack_request_caches
    from .runtime.loader import build_config, load_model
    from .runtime.request_state import RequestStateTable
    from .runtime.scheduler import group_pairs_by_sequence_length, group_request_ids_by_sequence_length
except ImportError:  # pragma: no cover - supports direct file import by evaluator
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    from runtime.cache import CacheHandle, SlotKVCacheManager, split_request_cache, stack_request_caches
    from runtime.loader import build_config, load_model
    from runtime.request_state import RequestStateTable
    from runtime.scheduler import group_pairs_by_sequence_length, group_request_ids_by_sequence_length


def create_engine(model_config: dict, weight_dir: str, device: str = "cuda") -> "Engine":
    """Create the evaluator-facing runtime instance.

    Phase 1 provides a correctness-first baseline that recomputes full
    request sequences on each decode step.
    """
    return Engine(model_config=model_config, weight_dir=weight_dir, device=device)


class Engine:
    """Stable runtime facade for the evaluator contract."""

    def __init__(self, model_config: Dict, weight_dir: str, device: str = "cuda") -> None:
        self.model_config = model_config
        self.weight_dir = weight_dir
        self.device = _resolve_device(device)
        self.runtime_config = build_config(model_config)
        self.model = load_model(model_config, weight_dir, device=self.device)
        self.requests = RequestStateTable(device=torch.device(self.device))
        model_dtype = self.model.lm_head.weight.dtype
        self.cache_manager = SlotKVCacheManager(
            num_layers=self.runtime_config.num_hidden_layers,
            num_kv_heads=self.runtime_config.num_key_value_heads,
            head_dim=self.runtime_config.head_dim,
            max_position_embeddings=self.runtime_config.max_position_embeddings,
            device=torch.device(self.device),
            dtype=model_dtype,
        )

    def prefill(self, request_ids: Iterable[int], input_ids: List[object]):
        request_ids = [int(request_id) for request_id in request_ids]
        if len(request_ids) != len(input_ids):
            raise ValueError("request_ids and input_ids must have the same length")

        sequences_by_request: dict[int, torch.Tensor] = {}
        sequence_lengths: list[int] = []
        for request_id, tokens in zip(request_ids, input_ids):
            sequence = _normalize_sequence(tokens, device=self.device)
            if sequence.numel() == 0:
                raise ValueError("prefill sequences must be non-empty")
            sequences_by_request[request_id] = sequence
            sequence_lengths.append(sequence.numel())

        logits_by_request: dict[int, torch.Tensor] = {}
        grouped_requests = group_request_ids_by_sequence_length(request_ids, sequence_lengths)
        for _, grouped_request_ids in grouped_requests.items():
            batch_input = torch.stack([sequences_by_request[request_id] for request_id in grouped_request_ids], dim=0)
            if len(grouped_request_ids) == 1:
                logits, kv_cache = self.model.logits_and_cache_for_prefill(batch_input)
                request_id = grouped_request_ids[0]
                state = self.requests.upsert_prompt(request_id, sequences_by_request[request_id])
                cache_handle = self.cache_manager.store_request_cache(state.cache_slot, kv_cache)
                self.requests.update_kv_cache(request_id, cache_handle)
                logits_by_request[request_id] = logits.squeeze(0)
                continue

            logits, batch_cache = self.model.logits_and_cache_for_prefill_batch(batch_input)
            per_request_caches = split_request_cache(batch_cache)
            for request_id, row_logits, request_cache in zip(grouped_request_ids, logits, per_request_caches):
                state = self.requests.upsert_prompt(request_id, sequences_by_request[request_id])
                cache_handle = self.cache_manager.store_request_cache(state.cache_slot, request_cache)
                self.requests.update_kv_cache(request_id, cache_handle)
                logits_by_request[request_id] = row_logits

        return torch.stack([logits_by_request[request_id] for request_id in request_ids], dim=0)

    def decode(self, request_ids: Iterable[int], token_ids: object):
        request_ids = [int(request_id) for request_id in request_ids]
        token_ids = _normalize_decode_tokens(token_ids, expected=len(request_ids), device=self.device)
        token_values = _tensor_to_int_list(token_ids)

        state_by_request_id = {}
        cached_request_ids: list[int] = []
        cached_sequence_lengths: list[int] = []
        cached_indices: list[int] = []
        fallback_request_ids: list[int] = []

        for index, request_id in enumerate(request_ids):
            token_value = token_values[index]
            state = self.requests.append_token(request_id, token_value)
            state_by_request_id[request_id] = state
            if state.kv_cache is None:
                fallback_request_ids.append(request_id)
            else:
                cached_request_ids.append(request_id)
                cached_sequence_lengths.append(state.seq_len - 1)
                cached_indices.append(index)

        logits_by_request: dict[int, torch.Tensor] = {}
        for request_id in fallback_request_ids:
            state = state_by_request_id[request_id]
            if state.tokens is None:
                raise RuntimeError("Missing full token history for non-cached request")
            logits, kv_cache = self.model.logits_and_cache_for_prefill(state.tokens.view(1, -1))
            cache_handle = self.cache_manager.store_request_cache(state.cache_slot, kv_cache)
            self.requests.update_kv_cache(state.request_id, cache_handle)
            logits_by_request[state.request_id] = logits.squeeze(0)

        grouped_state_pairs = group_pairs_by_sequence_length(
            cached_request_ids,
            cached_indices,
            cached_sequence_lengths,
        )

        for cache_len, request_index_pairs in grouped_state_pairs.items():
            request_group = [request_id for request_id, _ in request_index_pairs]
            group_indices = [token_index for _, token_index in request_index_pairs]
            items = [state_by_request_id[request_id] for request_id in request_group]

            batch_tokens = token_ids[group_indices].unsqueeze(1)
            position_ids = torch.full(
                (len(request_group), 1),
                fill_value=cache_len,
                device=self.device,
                dtype=torch.long,
            )
            slot_ids = torch.tensor([state.cache_slot for state in items], device=self.device, dtype=torch.long)
            logits = self.model.logits_for_decode_batch_with_manager(
                batch_tokens,
                cache_manager=self.cache_manager,
                cache_slot_ids=slot_ids,
                position_ids=position_ids,
            )
            for state, row_logits in zip(items, logits):
                self.requests.update_kv_cache(
                    state.request_id,
                    CacheHandle(slot=state.cache_slot, seq_len=state.seq_len),
                )
                logits_by_request[state.request_id] = row_logits

        return torch.stack([logits_by_request[request_id] for request_id in request_ids], dim=0)

    def remove(self, request_ids: Iterable[int]):
        self.requests.remove(request_ids)


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _normalize_sequence(tokens: object, device: str) -> torch.Tensor:
    if not isinstance(tokens, torch.Tensor):
        tokens = torch.as_tensor(tokens, dtype=torch.long)
    return tokens.to(device=device, dtype=torch.long).view(-1)


def _normalize_decode_tokens(token_ids: object, expected: int, device: str) -> torch.Tensor:
    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    token_ids = token_ids.to(device=device, dtype=torch.long).view(-1)
    if token_ids.numel() != expected:
        raise ValueError(f"Expected {expected} decode tokens, got {token_ids.numel()}")
    return token_ids


def _tensor_to_int_list(token_ids: torch.Tensor) -> list[int]:
    if token_ids.device.type == "cpu":
        return token_ids.tolist()
    return token_ids.detach().to(device="cpu").tolist()
