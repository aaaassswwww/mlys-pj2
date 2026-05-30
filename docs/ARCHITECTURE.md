# 架构说明

## 1. 目录职责

### `workspace/engine.py`

评测入口层。

职责：
- 提供 `create_engine(...)`
- 暴露评测接口 `prefill / decode / remove`
- 保持外部 API 稳定

不负责：
- 复杂算子实现
- 低层 cache 管理细节

### `workspace/runtime/loader.py`

职责：
- 解析 `model_config`
- 加载权重
- 构建内部模型对象

### `workspace/runtime/model.py`

职责：
- 组织 decoder-only 模型 forward
- 提供 prefill / decode 所需执行路径

### `workspace/runtime/layers.py`

职责：
- RMSNorm
- attention block
- MLP block
- lm head 相关组装

### `workspace/runtime/rope.py`

职责：
- RoPE 位置编码逻辑

### `workspace/runtime/cache.py`

职责：
- per-layer KV cache 抽象
- request 级 KV cache 组织
- 后续 cache slot 分配与释放

### `workspace/runtime/request_state.py`

职责：
- request 生命周期状态
- request 到 cache slot 的映射
- active request 管理

### `workspace/runtime/scheduler.py`

职责：
- 后续 mixed trace / batched 执行调度辅助

## 2. 稳定抽象

以下抽象在后续开发中默认冻结：

### Engine 接口

- `prefill(request_ids, input_ids)`
- `decode(request_ids, token_ids)`
- `remove(request_ids)`

### RequestState 最小字段

- `request_id`
- `seq_len`
- `cache_slot`
- `active`

### KV Cache 目标能力

- 按 layer 存储 K/V
- 支持 prefill 写入
- 支持 decode 追加
- 支持 remove 后 slot 回收

## 3. 开发边界

为了避免后续反复返工，默认遵守：
- 不把所有逻辑塞进 `engine.py`
- 不为 public toy case 写死结构参数
- 不在 correctness baseline 完成前做复杂 kernel 优化
- 不在缺少 mixed trace 语义验证时提前锁死 cache layout

## 4. 当前实现状态

当前已完成 Phase 2 baseline+incremental decode：
- `workspace/engine.py` 已提供稳定入口
- 已实现动态构模与权重加载
- 已实现基于 request state 的 `prefill / decode / remove`
- `prefill` 会建立 per-layer KV cache
- `decode` 已切换为基于历史 cache 的单 token 增量计算
- 多请求 `decode` 已支持按 cache 长度分组的 batched incremental execution
- request state 已支持逻辑 cache slot 分配、remove 后回收、以及 mixed trace 下的持续服务
- `prefill` 已支持按 prompt 长度分组的 batched cache 构建
- 已提供本地 benchmark 与 decode profiling 工具，支持后续性能决策
- 已提供 submission selfcheck 与 `run.sh` 日志化入口

当前环境限制：
- 目前开发验证基于 CPU-only PyTorch
- correctness 与运行时语义结论可信
- GPU 吞吐与 CUDA 专项优化仍待后续环境验证
- 当前 Windows 开发机上的系统 `bash.exe` 不可用，因此 `run.sh` 的集成执行需在兼容的 shell 环境或 evaluator 环境中验证

下一阶段目标：
- 当前阶段已完成，可在后续获得 GPU / Linux evaluator 环境后继续做最终提交前验证
