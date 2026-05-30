# Decode Optimization Plan

## Goal

- Keep public evaluator correctness green.
- Prioritize `decode tokens/s`.
- Track `mixed tokens/s` alongside decode so we do not win a micro-benchmark and lose serving behavior.

## Baseline

Initial public evaluator result:

- `prefill tokens/s = 58569.42`
- `decode tokens/s = 900.55`
- `mixed tokens/s = 7211.01`

## Optimization Rules

- Only change one optimization theme at a time.
- Every round must pass local correctness tests before GPU validation.
- Every GPU round must rerun public correctness and throughput.
- If correctness regresses, stop and revert or repair before the next optimization.

## Completed Stable Rounds

1. Reduced decode-side clone and Python overhead.
2. Unified batched decode handling.
3. Enabled `scaled_dot_product_attention` on the attention path.
4. Cached rotary embeddings.
5. Reduced `split_request_cache` overhead.
6. Streamlined rotary rotation.

## Current High-Risk Round

### Theme

- Replace per-step decode cache `stack/split` assembly with slot-backed KV storage.

### Intended Effect

- Remove repeated per-request cache packing on the decode hot path.
- Keep request lifecycle semantics unchanged.
- Preserve evaluator-facing `prefill / decode / remove` behavior.

### Implementation Notes

- Request state stores `CacheHandle(slot, seq_len)` after cache materialization.
- Prefill still materializes full request KV once, then writes it into slot storage.
- Batched decode writes new K/V directly into slot-backed caches and reads them back by slot id.
- Slot-backed storage now expands on demand across both slot count and sequence length instead of allocating full `max_position_embeddings` up front.

## Validation Checklist

### Local

```bash
python -m unittest \
  workspace.tests.test_correctness_local \
  workspace.tests.test_request_lifecycle \
  workspace.tests.test_tools_local \
  workspace.tests.test_submission_entrypoint
```

### GPU Public Correctness

```bash
python3 evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

### GPU Public Throughput

```bash
python3 evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

### GPU Decode Profile

```bash
python3 workspace/tools/profile_decode.py --device auto --batch-size 8 --prompt-len 24 --decode-steps 32 --top 20
```

## Metrics To Record Each Round

- `prefill tokens/s`
- `decode tokens/s`
- `mixed tokens/s`
- `peak_memory_mb`
- top decode profile entries

## Current Status

- Local regression: passed on `2026-05-30`
- Current phase: `Step 3 in progress`
- Next action: rerun public GPU evaluator after the dynamic-capacity slot cache fix
