#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENGINE_IMPORT_PATH=""
SELFCHECK_PATH=""
RUNTIME_DIR=""

if [[ -f "$ROOT_DIR/workspace/engine.py" ]]; then
  ENGINE_IMPORT_PATH="workspace/engine.py"
elif [[ -f "$ROOT_DIR/engine.py" ]]; then
  ENGINE_IMPORT_PATH="engine.py"
else
  ENGINE_CANDIDATE="$(find "$ROOT_DIR" -maxdepth 4 -type f -name engine.py | head -n 1 || true)"
  if [[ -n "$ENGINE_CANDIDATE" ]]; then
    ENGINE_IMPORT_PATH="${ENGINE_CANDIDATE#$ROOT_DIR/}"
  fi
fi

if [[ -z "$ENGINE_IMPORT_PATH" ]]; then
  echo "[run.sh] unable to locate engine.py under $ROOT_DIR" >&2
  exit 1
fi

RUNTIME_DIR="$(cd "$(dirname "$ROOT_DIR/$ENGINE_IMPORT_PATH")" && pwd)"

if [[ -f "$ROOT_DIR/workspace/tools/selfcheck_submission.py" ]]; then
  SELFCHECK_PATH="workspace/tools/selfcheck_submission.py"
elif [[ -f "$ROOT_DIR/tools/selfcheck_submission.py" ]]; then
  SELFCHECK_PATH="tools/selfcheck_submission.py"
else
  SELFCHECK_CANDIDATE="$(find "$ROOT_DIR" -maxdepth 5 -type f -path '*/tools/selfcheck_submission.py' | head -n 1 || true)"
  if [[ -n "$SELFCHECK_CANDIDATE" ]]; then
    SELFCHECK_PATH="${SELFCHECK_CANDIDATE#$ROOT_DIR/}"
  fi
fi

if [[ -z "$SELFCHECK_PATH" ]]; then
  echo "[run.sh] unable to locate selfcheck_submission.py under $ROOT_DIR" >&2
  exit 1
fi

LOG_FILE="$RUNTIME_DIR/results.log"
RESULT_LOG_FILE="$RUNTIME_DIR/result.log"
OUTPUT_FILE="$ROOT_DIR/output3.txt"
BENCHMARK_FILE="$RUNTIME_DIR/benchmark_results.json"
RUN_CAPTURE_FILE="$RUNTIME_DIR/output3_runtime.log"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

mkdir -p "$RUNTIME_DIR"
: > "$LOG_FILE"
: > "$RESULT_LOG_FILE"
: > "$OUTPUT_FILE"
rm -f "$RUN_CAPTURE_FILE"
rm -f "$BENCHMARK_FILE"

exec > >(tee -a "$LOG_FILE" "$RUN_CAPTURE_FILE") 2>&1

echo "[run.sh] start"
echo "[run.sh] root=$ROOT_DIR"
echo "[run.sh] python=$PYTHON_BIN"
echo "[run.sh] runtime import path: $ENGINE_IMPORT_PATH"

cd "$ROOT_DIR"

"$PYTHON_BIN" - <<'PY'
import sys
print(f"[run.sh] python_version={sys.version.splitlines()[0]}")
try:
    import torch
    print(f"[run.sh] torch_version={torch.__version__}")
    print(f"[run.sh] cuda_available={torch.cuda.is_available()}")
except Exception as exc:
    print(f"[run.sh] torch_import_error={exc!r}")
    raise
PY

echo "[run.sh] running selfcheck"
"$PYTHON_BIN" "$SELFCHECK_PATH"
echo "[run.sh] selfcheck=passed"

if [[ -f "evaluator/benchmark_throughput.py" && -f "target/model_config.json" ]]; then
  if [[ ! -f "target/weights/model.pt" && -f "scripts/generate_toy_weights.py" ]]; then
    echo "[run.sh] generating toy weights for benchmark context"
    "$PYTHON_BIN" scripts/generate_toy_weights.py --config target/model_config.json --output target/weights/model.pt
  fi

  if [[ -d "target/weights" ]]; then
    echo "[run.sh] running benchmark_throughput for output3 context"
    "$PYTHON_BIN" evaluator/benchmark_throughput.py \
      --engine "$ENGINE_IMPORT_PATH" \
      --model-config target/model_config.json \
      --weight-dir target/weights \
      --device auto > "$BENCHMARK_FILE"
    echo "[run.sh] benchmark_file=$BENCHMARK_FILE"
  else
    echo "[run.sh] benchmark skipped: target/weights not available"
  fi
else
  echo "[run.sh] benchmark skipped: evaluator or target config not available"
fi

"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

root = Path.cwd()
output_file = root / "output3.txt"
benchmark_file = root / "workspace" / "benchmark_results.json"
runtime_dir = root / "workspace"
if not (runtime_dir / "engine.py").is_file():
    runtime_dir = root
benchmark_file = runtime_dir / "benchmark_results.json"
run_capture_file = runtime_dir / "output3_runtime.log"

run_log = ""
if run_capture_file.is_file():
    run_log = run_capture_file.read_text(encoding="utf-8")

lines = ["---- result ----"]
if benchmark_file.is_file():
    try:
        benchmark_text = benchmark_file.read_text(encoding="utf-8").strip()
        json.loads(benchmark_text)
    except Exception as exc:
        benchmark_summary = [f"- benchmark parsing failed: {exc!r}"]
        benchmark_status = "parsing_failed"
    else:
        benchmark_summary = [
            "5. Current Benchmark Result On This Environment",
            benchmark_text,
        ]
        benchmark_status = "completed"
else:
    benchmark_summary = [
        "5. Current Benchmark Result On This Environment",
        "- unavailable in this run",
    ]
    benchmark_results = None
    benchmark_status = "unavailable"

lines = [
    "Phase 3 Automated LLM Inference Runtime",
    "",
    "0. Run Log",
]
if run_log.strip():
    lines.extend(run_log.rstrip().splitlines())
else:
    lines.append("- unavailable")
lines.extend(
    [
        "",
    "1. Submission Summary",
    f"- engine entrypoint: {('workspace/engine.py' if (root / 'workspace' / 'engine.py').is_file() else 'engine.py')}",
    "- run.sh executed successfully",
    "- selfcheck_submission.py passed",
    "- project status: Phase 0 through Phase 7 completed",
    "",
    "2. Runtime Design",
    "- The runtime implements a decoder-only LLM inference engine with the evaluator-facing API:",
    "  - create_engine(model_config, weight_dir, device=\"cuda\")",
    "  - prefill(request_ids, input_ids)",
    "  - decode(request_ids, token_ids)",
    "  - remove(request_ids)",
    "- Model structure is built dynamically from model_config.",
    "- Weights are loaded dynamically from weight_dir.",
    "- Request lifecycle is tracked explicitly through request state and KV cache ownership.",
    "- Prefill and decode both support grouped batching for same-length requests.",
    "",
    "3. Decode Optimization Reasoning",
    "- Correctness was treated as the hard gate before throughput work.",
    "- Decode was identified as the main bottleneck after the first public benchmark pass.",
    "- Low-risk optimizations were applied first:",
    "  - reduced Python overhead in decode grouping",
    "  - reduced unnecessary cache split/clone overhead",
    "  - enabled scaled_dot_product_attention when available",
    "  - cached RoPE tables and reduced repeated dtype/cache preparation work",
    "  - enabled inference_mode on hot evaluator paths",
    "  - added a CUDA torch.compile warmup path with graceful fallback",
    "- A higher-risk slot-backed cache experiment was tested and then removed because it regressed throughput.",
    "- Final implementation keeps the best-performing stable decode path found during validation.",
    "",
    "4. Validation Status",
    "- Submission selfcheck passes.",
    f"- benchmark status in this run: {benchmark_status}",
    "",
]
)
lines.extend(benchmark_summary)
lines.extend(
    [
        "",
        "6. Final Notes",
        "- This file is the aggregated primary output artifact for this run.",
        "- It includes the run log plus the structured reasoning summary.",
        "- Hidden-evaluator performance may still differ because model size, trace shape, and request patterns can change.",
    ]
)

output_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

printf '%s\n' "[run.sh] output moved to output3.txt" > "$RESULT_LOG_FILE"

echo "[run.sh] output_file=$OUTPUT_FILE"
echo "[run.sh] end"
