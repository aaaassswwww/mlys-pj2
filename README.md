# Automated LLM Inference Runtime

This repository is organized around the Phase 3 runtime contract.

Key entrypoints:
- `run.sh`
- `workspace/engine.py`
- `docs/PROJECT_PLAN.md`
- `docs/ARCHITECTURE.md`

Development rule:
- implement strictly according to the staged plan in `docs/PROJECT_PLAN.md`
- keep `workspace/engine.py` as the stable evaluator-facing entrypoint
- place runtime internals under `workspace/runtime/`

Useful local commands:
- `python -m unittest workspace.tests.test_interface workspace.tests.test_correctness_local workspace.tests.test_request_lifecycle workspace.tests.test_tools_local`
- `python workspace/tools/benchmark_local.py --batch-size 2 --prompt-len 8 --decode-steps 4 --repeat 1 --warmup 0`
- `python workspace/tools/profile_decode.py --batch-size 2 --prompt-len 8 --decode-steps 4 --top 5`
- `bash run.sh`
