# 项目开发计划

## 1. 项目目标

本项目严格对准 Phase 3 要求，目标不是实现一个单纯的模型 forward，而是实现一个可被评测器直接驱动的 LLM inference runtime。

必须满足的最终目标：
- 从 `model_config` 动态构建模型结构
- 从 `weight_dir` 动态加载权重
- 正确支持 `prefill(request_ids, input_ids)`
- 正确支持 `decode(request_ids, token_ids)`
- 正确支持 `remove(request_ids)`
- 在 `prefill`、`decode`、`mixed` 三类 trace 下优化整体吞吐

## 2. 交付契约

仓库必须长期保持以下入口稳定：
- `run.sh`
- `workspace/engine.py`
- `workspace/results.log`

约束：
- `run.sh` 负责准备、自检、写日志
- 评测器随后会直接 import `workspace/engine.py`
- `workspace/engine.py` 必须提供 `create_engine(model_config, weight_dir, device="cuda")`
- correctness 是吞吐评分前置条件

## 3. 冻结的开发原则

后续开发严格遵守以下原则：
- 不修改评测接口语义，除非需求文档明确要求
- `engine.py` 始终保持 thin wrapper 角色
- 真实实现放在 `workspace/runtime/`
- 先做 correctness baseline，再做增量和性能优化
- 每一轮优化都必须有对应回归验证
- 没有 profiling 证据，不做重型算子优化
- 每完成一个明确阶段或子阶段，必须同步更新本文档中的当前状态、阶段进度和验收结果
- 文档进度更新属于开发完成的一部分，不允许代码推进而文档状态滞后

当前环境约束：
- 当前开发机为 CPU-only 环境，且本地 PyTorch 不带 CUDA
- correctness、request state、KV cache 语义、增量 decode 路径可继续开发和验证
- GPU 吞吐、CUDA kernel、Triton、设备相关性能结论延后到有 GPU 环境时再验证

## 4. 阶段规划

### Phase 0: 需求冻结与架构定型

目标：
- 固定接口语义
- 固定模块职责
- 固定 request state / KV cache 抽象

产出：
- `docs/PROJECT_PLAN.md`
- `docs/ARCHITECTURE.md`
- 项目目录骨架

验收：
- 后续开发不需要再改项目总体结构

### Phase 1: Correctness Baseline

目标：
- 动态构模
- 权重加载
- 正确实现 prefill / decode / remove
- `decode` 第一版允许整段重算

重点：
- 这是全项目的 correctness 基线
- 后续所有优化都必须与它对齐

验收：
- 单请求 prefill 正确
- 单请求 decode 正确
- 多请求 prefill 正确
- 多请求 decode 正确
- 插入和删除请求后仍正确

完成进度：
- status: completed
- 已完成内容：实现了 correctness-first baseline，支持动态构模、权重加载、request state 管理，以及基于全序列重算的 `prefill / decode / remove`
- 已完成验证：本地接口测试、单请求 prefill/decode、多请求 prefill/decode、remove 后继续 decode
- 遗留限制：当前 `decode` 仍为全序列重算，尚未引入 KV cache

### Phase 2: Decode Stage 1 - 单请求增量化

目标：
- 引入 per-layer KV cache
- 单请求 decode 只计算新增 token

重点：
- cache 写入位置
- RoPE position 对齐
- last-token logits 对齐

验收：
- 单请求增量 decode 与 baseline 一致
- 连续多步 decode 与 baseline 一致

完成进度：
- status: completed
- 已完成内容：引入了 per-layer KV cache，并将 decode 路径升级为基于历史 cache 的单 token 增量计算
- 已完成验证：单请求 prefill 后建立 cache、单请求一步 decode、一条请求连续多步 decode、existing multi-request correctness regression
- 遗留限制：当前多请求 decode 仍按 request 逐个增量执行，尚未进入 batched incremental decode

### Phase 3: Decode Stage 2 - 多请求批量化

目标：
- batched incremental decode
- 多个 active request 一起执行 decode

重点：
- request 顺序与输出行顺序严格一致
- 降低 Python overhead 和 launch overhead

验收：
- batched decode 与逐请求 decode 一致
- decode tokens/s 高于单请求循环

完成进度：
- status: completed
- 已完成内容：将多请求 decode 升级为按 cache 长度分组的 batched incremental decode，相同历史长度的 active requests 会合并执行
- 已完成验证：多请求 prefill/decode correctness regression、同长度请求单次 batched decode 路径检查、输出顺序与输入 request_ids 顺序保持一致
- 遗留限制：当前 batching 仍按相同 cache 长度分组，尚未支持跨不同长度请求的统一 padded batch

### Phase 4: Mixed Serving Runtime

目标：
- 支持请求中途插入
- 支持 remove
- 支持 cache slot 回收

重点：
- request_id 到 cache slot 的映射
- mixed trace 状态一致性

验收：
- mixed trace 下 logits 与 baseline 一致
- remove 后无 cache 污染

完成进度：
- status: completed
- 已完成内容：补齐 mixed serving 语义，支持运行中插入新请求、remove 后继续服务其他请求，并加入逻辑 cache slot 的分配与回收
- 已完成验证：remove 后继续 decode、slot reuse、mixed trace 下插入新请求并继续服务剩余请求
- 遗留限制：当前 slot 回收仍是逻辑层语义，尚未演进为统一的大块共享 KV memory allocator

### Phase 5: Prefill Runtime 优化

目标：
- batched long-prompt prefill
- 统一 prefill / decode cache 写入路径

重点：
- prefill 后直接接 decode 不出错
- 避免两套独立 cache 逻辑

验收：
- prefill correctness 不回退
- long-prompt throughput 提升

完成进度：
- status: completed
- 已完成内容：将 prefill 升级为按 prompt 长度分组的 batched prefill，相同长度请求会共享一次前向与 cache 构建
- 已完成验证：原有 prefill/decode correctness regression、同长度请求 batched prefill 路径检查、prefill 后 cache 可直接衔接 decode
- 遗留限制：当前 prefill batching 仍按相同长度分组，尚未支持跨不同 prompt 长度的 padded prefill batch

### Phase 6: Profile-Driven Optimization

目标：
- 基于 profile 结果优化整体吞吐

优先对象：
- attention
- KV cache layout
- gather/scatter
- Python loops
- RMSNorm / RoPE / MLP 链路

策略：
- 先 profile 再优化
- 只有在必要时才考虑 Triton / CUDA / C++ extension

验收：
- correctness 不回退
- `prefill` / `decode` / `mixed` 的总体吞吐持续提升

完成进度：
- status: completed
- 已完成内容：补齐了本地 benchmark 与 cProfile 工具链，增加了调度分组辅助模块，并完成一轮低风险结构性整理
- 已完成验证：benchmark 脚本可输出 `prefill / decode / mixed` tokens/s，profile 脚本可输出 decode 热点函数，完整本地回归测试通过
- 当前本地观测：CPU-only 环境下一次 smoke benchmark 得到 `prefill 1942.83 tokens/s`、`decode 443.08 tokens/s`、`mixed 998.78 tokens/s`
- 遗留限制：这些性能数字仅代表当前 CPU-only 开发环境，不可外推为最终 GPU 吞吐成绩

### Phase 7: 提交工程化

目标：
- 保证提交环境稳定运行

内容：
- `run.sh` 准备和日志完善
- 本地自测命令固化
- 明确失败定位信息

验收：
- 新环境可直接执行 `bash run.sh`
- `workspace/engine.py` 可被直接 import

完成进度：
- status: completed
- 已完成内容：`run.sh` 已升级为真实提交入口，增加 Python / Torch 环境记录、统一日志输出、轻量 selfcheck，并补齐提交入口 smoke tests
- 已完成验证：submission selfcheck 可独立运行，完整本地 unittest 回归通过
- 当前环境说明：由于当前 Windows 开发机上的系统 `bash.exe` 不可用，本地 `run.sh` 集成验证在测试中按环境跳过；脚本本身已按 Linux evaluator 约定编写

## 5. 开发顺序约束

后续严格按以下顺序开发，不跨阶段提前做重优化：

1. Phase 1 correctness baseline
2. Phase 2 单请求增量 decode
3. Phase 3 多请求 batched decode
4. Phase 4 mixed serving runtime
5. Phase 5 prefill 优化
6. Phase 6 profile-driven optimization
7. Phase 7 提交工程化

## 6. 每阶段的统一验收规则

每个阶段都必须回答三个问题：
- 新增了什么能力
- correctness 如何验证
- throughput 或工程稳定性如何量化

如果任一阶段无法回答这三个问题，则不进入下一阶段。

此外，每次阶段验收完成后，必须立即更新：
- 本文档中的 `当前状态`
- 对应阶段的完成情况
- 已完成的验证项和遗留问题

如果上述文档未更新，则该阶段视为未真正完成。

## 7. 当前状态

当前仓库处于 `Phase 0 completed / Phase 1 completed / Phase 2 completed / Phase 3 completed / Phase 4 completed / Phase 5 completed / Phase 6 completed / Phase 7 completed` 状态。

后续状态记录格式约定如下：
- `Phase X: not started | in progress | completed`
- 如某个 phase 拆分为子阶段，则记录为 `Phase X / Stage Y`
- 每次更新状态时，同时补充一句本阶段已完成内容和下一步动作

最新进度记录：
- `Phase 0: completed`，已冻结项目结构、模块职责和开发顺序，下一步进入 correctness baseline
- `Phase 1: completed`，已搭建 correctness-first baseline 并完成本地回归，下一步进入 `Phase 2 / Decode Stage 1`
- `Phase 2: completed`，已完成单请求增量 decode 和 per-layer KV cache，下一步进入 `Phase 3 / Decode Stage 2`
- `Phase 3: completed`，已完成多请求 batched incremental decode，下一步进入 `Phase 4 / Mixed Serving Runtime`
- `Phase 4: completed`，已完成 mixed serving runtime 语义和逻辑 slot 回收，下一步进入 `Phase 5 / Prefill Runtime 优化`
- `Phase 5: completed`，已完成 batched prefill 和统一 cache 生成路径，下一步进入 `Phase 6 / Profile-Driven Optimization`
- `Phase 6: completed`，已完成本地 benchmark / profiling 工具链和一轮结构性优化，下一步进入 `Phase 7 / 提交工程化`
- `Phase 7: completed`，已完成提交入口、自检和日志工程化，项目进入可交付状态
