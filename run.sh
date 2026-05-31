#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$ROOT_DIR/workspace"
ENGINE_IMPORT_PATH="workspace/engine.py"
SELFCHECK_PATH="workspace/tools/selfcheck_submission.py"

if [[ ! -f "$ROOT_DIR/$ENGINE_IMPORT_PATH" ]]; then
  echo "[run.sh] missing required engine entrypoint: $ROOT_DIR/$ENGINE_IMPORT_PATH" >&2
  exit 1
fi

if [[ ! -f "$ROOT_DIR/$SELFCHECK_PATH" ]]; then
  echo "[run.sh] missing selfcheck helper: $ROOT_DIR/$SELFCHECK_PATH" >&2
  exit 1
fi

LOG_FILE="$WORKSPACE_DIR/results.log"
OUTPUT_FILE="$ROOT_DIR/output3.txt"
BENCHMARK_FILE="$WORKSPACE_DIR/benchmark_results.json"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

export MLSYS_DEBUG_RESULT_LOG="${MLSYS_DEBUG_RESULT_LOG:-0}"

mkdir -p "$WORKSPACE_DIR"
: > "$LOG_FILE"
: > "$OUTPUT_FILE"
rm -f "$BENCHMARK_FILE"

exec > >(tee -a "$LOG_FILE" "$OUTPUT_FILE") 2>&1

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
    MLSYS_DISABLE_COMPILE=1 "$PYTHON_BIN" evaluator/benchmark_throughput.py \
      --engine "$ENGINE_IMPORT_PATH" \
      --model-config target/model_config.json \
      --weight-dir target/weights \
      --device auto | tee "$BENCHMARK_FILE"
    echo "[run.sh] benchmark_file=$BENCHMARK_FILE"
  else
    echo "[run.sh] benchmark skipped: target/weights not available"
  fi
else
  echo "[run.sh] benchmark skipped: evaluator or target config not available"
fi

echo "[run.sh] output_file=$OUTPUT_FILE"
echo "[run.sh] end"
