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

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

export MLSYS_DEBUG_RESULT_LOG="${MLSYS_DEBUG_RESULT_LOG:-1}"

mkdir -p "$RUNTIME_DIR"
: > "$LOG_FILE"
: > "$RESULT_LOG_FILE"
: > "$OUTPUT_FILE"
rm -f "$BENCHMARK_FILE"

exec > >(tee -a "$LOG_FILE" "$RESULT_LOG_FILE" "$OUTPUT_FILE") 2>&1

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
