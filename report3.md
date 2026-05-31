# Report 3

## 1. Task Summary

This project implements a decoder-only LLM inference runtime for the Phase 3 evaluator contract. The required public API is exposed through `workspace/engine.py` and includes:

- `create_engine(model_config, weight_dir, device="cuda")`
- `prefill(request_ids, input_ids)`
- `decode(request_ids, token_ids)`
- `remove(request_ids)`

The final implementation is correctness-first, then optimized for grouped batching and incremental decode with KV cache reuse.

## 2. High-Level Design

The runtime is organized around three layers:

- `engine.py`
  - Evaluator-facing orchestration layer.
  - Owns request lifecycle, batching decisions, and optional compile warmup.
- `runtime/`
  - Core model execution, KV cache data structures, weight loading, and scheduling helpers.
- `workspace/`
  - Compatibility layer kept for the evaluator contract.

The evaluator imports `workspace/engine.py`, which forwards to the root implementation. This keeps the submission layout compatible while allowing the actual runtime code to live at the repository root for easier deployment and debugging.

## 3. Model and Execution Path

The runtime builds a lightweight LLaMA-like decoder-only model dynamically from `model_config`. The model includes:

- token embedding
- repeated decoder layers
- RMSNorm
- grouped-query self-attention
- RoPE position encoding
- gated MLP
- LM head

Weights are loaded dynamically from `weight_dir`, and the loader normalizes a few common checkpoint naming schemes so the model code stays independent from a specific file format.

## 4. Request State and KV Cache

Each active request is tracked in `RequestStateTable`. A request state stores:

- `request_id`
- `seq_len`
- optional full token history before cache handoff
- per-request KV cache
- reusable cache slot id

The runtime uses a per-request cache ownership model:

- `prefill` builds the initial KV cache from the prompt
- `decode` appends one token at a time using the cached prefix
- `remove` deletes the request state and releases its cache slot

For batched decode, requests are grouped by current cache length so they can share a decode batch without padding or attention-mask complexity. Per-request caches are temporarily stacked into a batch cache and then split back after the decode step.

## 5. Prefill and Decode Strategy

### Prefill

`prefill` normalizes each prompt, groups requests by prompt length, and runs same-length prompts as a batch. For each request it stores:

- prompt length
- KV cache produced by prefill
- final-token logits returned to the evaluator

### Decode

`decode` first advances request state, then separates requests into:

- cached requests, which can use incremental decode
- fallback requests, which must still prefill from full token history

Cached requests are grouped by cache length. Each group runs one batched incremental decode step using:

- the new token ids
- the stacked batch KV cache
- a shared position id equal to the current cache length

This avoids recomputing the whole sequence on every decode step.

## 6. Optimization Decisions

The optimization path followed a staged approach:

1. Build a fully correct baseline.
2. Introduce per-request KV cache for incremental decode.
3. Batch same-length requests in prefill and decode.
4. Reduce Python overhead and unnecessary cache split/clone work.
5. Prefer `scaled_dot_product_attention` when available.
6. Cache RoPE tables and reduce repeated dtype/cache preparation.
7. Treat `torch.compile` as an optional acceleration path with fallback.

A more aggressive slot-backed cache experiment was tested, but it increased complexity and regressed public throughput, so the final implementation kept the simpler and more stable per-request cache approach.

## 7. Robustness and Evaluation Fit

The project was shaped around the evaluator contract rather than a single benchmark script:

- `run.sh` handles preparation and local self-check.
- `workspace/results.log` is maintained for debugging.
- `workspace/engine.py` remains the stable evaluator entrypoint.
- hidden test variation is handled by dynamic config loading instead of hardcoded dimensions.

The implementation also supports both CPU-only development and CUDA execution. If CUDA-specific compile optimization fails, the model falls back to eager execution instead of risking correctness or startup failure.

## 8. Final Outcome

The final runtime provides:

- dynamic model construction
- dynamic weight loading
- grouped batched prefill
- grouped incremental decode
- request insertion, continuation, and removal
- optional compile optimization with safe fallback

The resulting system is a serving-style runtime rather than a single forward-only model wrapper, which matches the intent of the Phase 3 evaluation.
