# 项目交接文档

## 1. 当前项目状态

当前仓库已经完成从 `Phase 0` 到 `Phase 7` 的全部规划内工作，状态以 [docs/PROJECT_PLAN.md](/D:/VS%20Code/mlys-pj2/docs/PROJECT_PLAN.md) 为准。

当前整体状态：
- `Phase 0 completed`
- `Phase 1 completed`
- `Phase 2 completed`
- `Phase 3 completed`
- `Phase 4 completed`
- `Phase 5 completed`
- `Phase 6 completed`
- `Phase 7 completed`

当前 git 状态：
- 已初始化 git 仓库
- 当前分支：`main`
- 已有初始提交：`1241d24 chore: initial project scaffold and runtime baseline`

## 2. 这个项目现在具备什么

当前项目已经具备：
- 动态读取 `model_config`
- 动态加载权重目录
- evaluator-facing 入口 `workspace/engine.py`
- `prefill(request_ids, input_ids)`
- `decode(request_ids, token_ids)`
- `remove(request_ids)`
- per-layer KV cache
- 单请求增量 decode
- 多请求 batched incremental decode
- mixed serving runtime 语义
- 逻辑 cache slot 分配与回收
- batched prefill
- 本地 benchmark / profiling 工具链
- 提交入口 `run.sh`
- 提交前自检脚本 `workspace/tools/selfcheck_submission.py`

## 3. 关键文件

优先阅读这些文件：
- [docs/PROJECT_PLAN.md](/D:/VS%20Code/mlys-pj2/docs/PROJECT_PLAN.md)
- [docs/ARCHITECTURE.md](/D:/VS%20Code/mlys-pj2/docs/ARCHITECTURE.md)
- [workspace/engine.py](/D:/VS%20Code/mlys-pj2/workspace/engine.py)
- [workspace/runtime/model.py](/D:/VS%20Code/mlys-pj2/workspace/runtime/model.py)
- [workspace/runtime/layers.py](/D:/VS%20Code/mlys-pj2/workspace/runtime/layers.py)
- [workspace/runtime/request_state.py](/D:/VS%20Code/mlys-pj2/workspace/runtime/request_state.py)
- [workspace/runtime/cache.py](/D:/VS%20Code/mlys-pj2/workspace/runtime/cache.py)

工具和验证入口：
- [workspace/tools/selfcheck_submission.py](/D:/VS%20Code/mlys-pj2/workspace/tools/selfcheck_submission.py)
- [workspace/tools/benchmark_local.py](/D:/VS%20Code/mlys-pj2/workspace/tools/benchmark_local.py)
- [workspace/tools/profile_decode.py](/D:/VS%20Code/mlys-pj2/workspace/tools/profile_decode.py)
- [workspace/tests/test_correctness_local.py](/D:/VS%20Code/mlys-pj2/workspace/tests/test_correctness_local.py)
- [workspace/tests/test_request_lifecycle.py](/D:/VS%20Code/mlys-pj2/workspace/tests/test_request_lifecycle.py)
- [workspace/tests/test_submission_entrypoint.py](/D:/VS%20Code/mlys-pj2/workspace/tests/test_submission_entrypoint.py)

## 4. 当前开发环境说明

这次开发是在下面这个环境完成的：
- 操作系统：Windows
- 本地 PyTorch：CPU-only
- 无 GPU
- 本机系统 `bash.exe` 不可用

这意味着：
- correctness 和 runtime 语义验证已经做了
- benchmark / profile 工具链已经能跑
- 当前性能数字只代表 CPU-only 本地环境
- `run.sh` 已按 Linux evaluator 风格编写，但本机不能完整做 `bash run.sh` 集成验证

## 5. 切换到新设备后建议先做什么

如果新设备是 Linux + GPU 服务器，建议按这个顺序继续：

1. 拉取仓库并进入项目目录
2. 确认 Python 和 PyTorch 可用
3. 确认 `torch.cuda.is_available()` 为 `True`
4. 先跑提交自检
5. 再跑完整回归
6. 再跑 benchmark
7. 再跑 decode profile
8. 最后再做 evaluator 风格验证

## 6. 新设备上的建议执行命令

### 6.1 基础检查

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

### 6.2 提交前自检

```bash
python workspace/tools/selfcheck_submission.py
```

预期至少看到：

```text
[selfcheck] import=ok
[selfcheck] prefill_decode_remove=ok
```

### 6.3 完整本地回归

```bash
python -m unittest \
  workspace.tests.test_interface \
  workspace.tests.test_correctness_local \
  workspace.tests.test_request_lifecycle \
  workspace.tests.test_tools_local \
  workspace.tests.test_submission_entrypoint
```

### 6.4 本地 benchmark

```bash
python workspace/tools/benchmark_local.py --device auto
```

### 6.5 decode profile

```bash
python workspace/tools/profile_decode.py --device auto
```

### 6.6 evaluator 风格入口检查

```bash
bash run.sh
cat workspace/results.log
```

## 7. 当前已知限制

这些不是 bug，但在新设备上需要有预期：
- 当前 batching 仍按“相同长度分组”进行
- prefill 还没有做跨不同 prompt 长度的 padded batch
- decode 还没有做跨不同 cache 长度的统一 padded batch
- cache slot 回收当前是逻辑层语义，不是统一大块 GPU allocator
- 没有 Triton / CUDA extension
- 没有做 GPU 专项数值与吞吐验证

## 8. 如果新设备上出现问题，优先检查什么

优先排查顺序：
- `workspace/engine.py` 是否能被直接 import
- 权重文件名是否符合 `workspace/runtime/loader.py` 的加载规则
- `model_config` 字段名是否覆盖当前隐藏配置
- CUDA 设备是否真的启用
- `run.sh` 是否能在目标 shell 中执行
- logits 是否先过 correctness，再谈吞吐

## 9. 后续最有价值的工作

如果换到 Linux + GPU 环境，最值得继续做的是：
- 先完成一轮真实 CUDA 回归验证
- 跑 benchmark，建立 GPU 基线
- 跑 `profile_decode.py` 找 GPU 热点
- 再决定是否需要 Triton / CUDA 优化
- 如果 evaluator 环境允许，再做更贴近 hidden trace 的压测

## 10. 文档维护约定

后续如果继续开发，必须同步更新：
- [docs/PROJECT_PLAN.md](/D:/VS%20Code/mlys-pj2/docs/PROJECT_PLAN.md)
- [docs/ARCHITECTURE.md](/D:/VS%20Code/mlys-pj2/docs/ARCHITECTURE.md)
- 本文档 `docs/HANDOFF.md`

特别是当发生以下情况时必须更新：
- 环境变更
- 新增性能结论
- 新增已知限制
- 进入新的优化阶段
- 准备正式提交
