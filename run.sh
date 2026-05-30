#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$ROOT_DIR/workspace"
LOG_FILE="$WORKSPACE_DIR/results.log"
RESULT_LOG_FILE="$WORKSPACE_DIR/result.log"
OUTPUT_FILE="$ROOT_DIR/output3.txt"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

mkdir -p "$WORKSPACE_DIR"
: > "$LOG_FILE"
: > "$RESULT_LOG_FILE"
: > "$OUTPUT_FILE"

exec > >(tee -a "$LOG_FILE" "$RESULT_LOG_FILE") 2>&1

echo "[run.sh] start"
echo "[run.sh] root=$ROOT_DIR"
echo "[run.sh] python=$PYTHON_BIN"
echo "[run.sh] runtime import path: workspace/engine.py"

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
"$PYTHON_BIN" workspace/tools/selfcheck_submission.py
echo "[run.sh] selfcheck=passed"

cat > "$OUTPUT_FILE" <<'EOF'
Phase 3 runtime submission selfcheck summary

- engine entrypoint: workspace/engine.py
- run.sh executed successfully
- selfcheck_submission.py passed
- project status: Phase 0 through Phase 7 completed
- note: final throughput depends on target Linux + GPU environment
EOF

echo "[run.sh] output_file=$OUTPUT_FILE"
echo "[run.sh] end"
