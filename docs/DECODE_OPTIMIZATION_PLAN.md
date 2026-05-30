# Decode 优化计划

## 1. 当前背景

根据公共 evaluator 的一次基准结果，当前项目状态为：
- correctness 已通过公共样例
- prefill 表现较强
- mixed 表现可接受
- decode 是当前最明显瓶颈

一次公共样例结果如下：
- `prefill tokens/s = 58569.42`
- `decode tokens/s = 900.55`
- `mixed tokens/s = 7211.01`

当前优化主目标：
- 优先提升 `decode tokens/s`
- 同时观察 `mixed tokens/s` 是否同步改善
- 不回退 correctness

## 2. 优化原则

后续 decode 优化严格遵守：
- 先做低风险结构性优化
- 每轮只做一小步，避免混合多个变量
- 每轮优化后必须重新跑公共 throughput benchmark
- 如无 correctness 保障，不进入下一轮优化
- 如无 benchmark 结果，不判断优化是否有效

## 3. 当前推测瓶颈

结合当前实现，decode 路径可能的主要瓶颈是：
- `engine.decode` 中的 Python 分组与调度
- `stack_request_caches / split_request_cache`
- 小 tensor 的频繁构造
- `tolist()` 和 Python 标量往返
- batched decode 后将 cache 拆回 per-request 的额外开销

## 4. 分阶段优化路线

### Step 1: 低风险优化

目标：
- 先压缩 decode 路径中的额外 Python / clone / 小对象开销

动作：
- 尽量减少 `tolist()` 和 Python 标量往返
- 尽量减少 decode 中的小 tensor 重建
- 去掉 batched cache split 时不必要的 `clone()`
- 收紧 decode 过程中的中间数据结构

预期：
- `decode tokens/s` 有第一轮可见提升
- `mixed tokens/s` 至少不下降

### Step 2: 中风险优化

目标：
- 进一步减少 per-request cache 与 batch cache 来回转换的成本

动作：
- 收紧 grouped decode 的批组织路径
- 尽量减少 `stack/split` 次数
- 评估是否能保留更长生命周期的 batched cache 视图

预期：
- decode 路径进一步变紧凑

### Step 3: 高风险优化

目标：
- 如果前两轮收益不够，再考虑重构 cache layout

动作：
- 更统一的 batched KV cache 组织
- 更接近 serving runtime 的共享内存布局
- 必要时再考虑 CUDA / Triton 方向

预期：
- 这是最后手段，不作为第一优先级

## 5. 每轮优化后的测试命令

### 公共 correctness

```bash
python3 evaluator/test_correctness.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

### 公共 throughput

```bash
python3 evaluator/benchmark_throughput.py \
  --engine workspace/engine.py \
  --model-config target/model_config.json \
  --weight-dir target/weights \
  --device auto
```

### 本地 decode profile

```bash
python3 workspace/tools/profile_decode.py --device auto --batch-size 8 --prompt-len 24 --decode-steps 32 --top 20
```

## 6. 每轮需要记录的数据

每次回传至少记录：
- `prefill tokens/s`
- `decode tokens/s`
- `mixed tokens/s`
- `peak_memory_mb`

如果有 profile，再额外记录：
- decode 热点前 10-20 行

## 7. 当前轮次状态

当前处于：
- `Step 2 in progress`

当前轮次目标：
- 在保持 correctness 的前提下压 attention 主链耗时
- 优先尝试 PyTorch `scaled_dot_product_attention` 路径
- 继续观察 decode 与 mixed 是否同步提升

最新尝试方向：
- 为 evaluator 热路径的 `engine.prefill / decode / remove` 启用 `torch.inference_mode()`
- 继续压缩 RoPE 路径中的 dtype 转换与重复 cache 准备开销
