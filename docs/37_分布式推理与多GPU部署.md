# 37. 分布式推理与多GPU部署（多卡/多实例/负载均衡/队列限流/流式输出）

> 重要性：面试官问"你们只用了单卡？如果并发再增大怎么办？"——这个问题几乎必问。本章覆盖推理阶段的多GPU扩展路径、负载均衡策略、队列限流设计、流式输出的原理和实践。
>
> 关联文件：`34_vLLM推理部署与性能优化.md`（单卡优化、压测、性能指标）、`13_推理部署与vLLM.md`（基础版）、`15_安全对齐与医疗场景.md`（安全审核与分流）。

---

## 1. 训练并行 vs 推理并行：根本区别

### 1.1 相同点：都涉及将计算分布到多张 GPU 上

训练并行和推理并行在"多卡协同计算"这个层面有技术交集——都用到了 Tensor Parallel、Pipeline Parallel 等基本范式——但需求、瓶颈和最优策略完全不同。

### 1.2 核心区别一览

| 维度 | 训练并行 | 推理并行 |
|------|---------|---------|
| **主要瓶颈** | 显存（模型权重 + 优化器状态 + 激活值 + 梯度） | 显存（KV Cache）+ 延迟 |
| **通信频率** | 每个 micro-batch 都需要通信（每步 all-reduce 梯度） | Prefill 阶段每层通信，Decode 阶段通信量小 |
| **通信量** | 极大（梯度、优化器状态、参数 all-gather） | 相对小（主要是每层的 all-reduce 激活值） |
| **最优并行策略** | ZeRO-3/FSDP（数据并行 + 参数分片） | Tensor Parallel（模型分片）或 Replica（多副本） |
| **延迟要求** | 不敏感（batch 训练，分钟级） | 高度敏感（在线推理，毫秒级） |
| **batch 特性** | 大 batch（利用 GPU 算力） | 小 batch 但持续（delay-sensitive） |
| **弹性扩缩** | 不太需要（训练任务固定） | 需要（流量波动，auto-scaling） |
| **KV Cache** | 无需关心 | 核心瓶颈 |

### 1.3 为什么推理并行不等于"用训练并行的方法做推理"

训练时每张 GPU 既要存模型又要存优化器状态（Adam 的 m 和 v 各占 4 bytes/参数），一张 32GB 的卡可能连 8B 模型的 FP16 版本都装不下，必须用 ZeRO 分片或 tensor parallel。

推理时**不需要优化器状态**，显存压力来自 KV Cache。如果模型权重（8B BF16 ≈ 16GB）能放进单卡，最有效的方式不是分片模型，而是**多个完整副本各自服务**——这就是 Replica Parallel 的逻辑。

> **面试金句**："训练并行解决的是'单卡装不下模型+优化器'的问题，推理并行解决的是'单卡服务不够多用户'的问题。训练的瓶颈是优化器状态和激活值，推理的瓶颈是 KV Cache 显存和并发排队。因此最优策略完全不同——训练用 FSDP/ZeRO-3 做参数分片，推理能用单卡放模型时优先多副本横向扩展。"

### 1.4 一个具体的量化对比

```
场景：8B 模型，BF16（16GB），RTX 5090 32GB

训练 (QLoRA SFT):
  Base Model (NF4):           4 GB
  LoRA Weights:               0.05 GB
  Optimizer States (Adam):    0.10 GB
  Activations (bs=2):         ~12 GB
  Gradients:                  0.05 GB
  Total:                      ~18 GB  → 单卡刚好够
  如果用全参训练:              需要 ~64 GB+ → 必须多卡 FSDP/ZeRO

推理 (vLLM):
  模型权重 (BF16):            16 GB
  KV Cache (16 seq × 1500):  ~3 GB
  CUDA overhead:              ~2 GB
  Total:                      ~21 GB → 单卡够用
  "瓶颈"不在模型大小，在 KV Cache 容量限制了并发数
```

---

## 2. 推理阶段的三个核心瓶颈

### 2.1 瓶颈一：Prefill 计算瓶颈

- **本质**：Compute-bound，prompt 的 attention 计算是 O(N²)。
- **现象**：长 prompt（RAG 场景）的 prefill 耗时显著增加，TTFT 恶化。
- **优化方向**：减少 prompt 长度、启用 prefix caching、使用 chunked prefill。
- **是否可通过多 GPU 缓解**：可以——Tensor Parallel 可以加速单个 prefill（因为 attention 矩阵被分片计算），但对实际 TTFT 的改善受通信开销限制。

### 2.2 瓶颈二：Decode 显存带宽瓶颈

- **本质**：Memory-bound，每个 decode step 计算量很小，但需要从显存读取整个 KV Cache。
- **现象**：TPOT 主要由显存带宽而非 GPU 算力决定。
- **优化方向**：KV Cache 量化（FP8）、减少 KV Cache 大小（减小 max_model_len）。
- **是否可通过多 GPU 缓解**：不行。Tensor Parallel 在 decode 阶段的通信开销反而可能增加 TPOT（每层都需要 all-reduce，延迟敏感）。

### 2.3 瓶颈三：KV Cache 显存瓶颈

- **本质**：KV Cache 占用大量显存（每 token ~128KB），限制了单 GPU 能服务的并发请求数。
- **现象**：并发数受限于 KV Cache pool 大小，超过限制就会 OOM。
- **优化方向**：减小 max_model_len、限制 max_new_tokens、量化 KV Cache、PagedAttention。
- **是否可通过多 GPU 缓解**：可以——多 GPU 多副本（每个副本有独立的 KV Cache pool），线性扩展并发能力。但 Tensor Parallel 不增加 KV Cache pool（同一个 pool 分布在多卡上）。

```
误区：Tensor Parallel 多卡 → 总 KV Cache 容量 = 0.9 × (32 × 2) - 16 ≈ 41.6 GB？
事实：Tensor Parallel 下 KV Cache 各层分布在多卡上，每卡的 KV Cache pool 存储的是不同层，
     总容量 = 各卡 pool 之和，但单请求的 KV Cache 占用也分布在多卡上。
     所以 Tensor Parallel 的 KV Cache 容量 ≈ 各卡容量之和（减去模型权重分片）。
     
     但如果每卡都存完整模型（Replica），则总 KV Cache 容量 = N × 单卡 KV Cache pool，
     这才是真正线性扩展并发能力。
```

---

## 3. 部署架构演进：从单卡到多副本

### 3.1 级别一：单 GPU 单实例

```
┌─────────────────────────────────┐
│         RTX 5090 (32GB)         │
│  ┌───────────────────────────┐  │
│  │     vLLM Instance         │  │
│  │  - Model: 16GB            │  │
│  │  - KV Cache Pool: 12.8GB  │  │
│  │  - max_num_seqs: 16       │  │
│  │  - Port: 8000             │  │
│  └───────────────────────────┘  │
│          API: :8000              │
└─────────────────────────────────┘

适用：模型能放进单卡 + 并发需求低（<16 QPS）
本项目当前架构
```

### 3.2 级别二：单 GPU 多实例（不推荐）

```
┌─────────────────────────────────┐
│         RTX 5090 (32GB)         │
│  ┌────────────┐ ┌────────────┐  │
│  │ Instance 1 │ │ Instance 2 │  │
│  │ :8000      │ │ :8001      │  │
│  │ model: 16G │ │ model: 16G │  │  ← 两份模型权重 = 32GB！
│  └────────────┘ └────────────┘  │      直接 OOM
└─────────────────────────────────┘

问题：单卡显存不够放两份模型权重（32GB + KV Cache > 32GB）
除非模型很小（< 4GB）或者量化到 INT4（< 4GB）
本项目不适用（8B BF16 已占 16GB，放不下两份）
```

### 3.3 级别三：多 GPU 单模型（Tensor Parallel）

```
┌──────────────┐   ┌──────────────┐
│ GPU 0 (32GB) │   │ GPU 1 (32GB) │
│ Model 层0-15 │◄──┤ Model 层16-31│  NVLink/PCIe
│ KV Cache 层0 │   │ KV Cache 层1 │
│ :8000        │   │ :8000 (同一服务) │
└──────────────┘   └──────────────┘
     TP=2，一个 vLLM 实例横跨 2 张卡
     
适用：模型放不进单卡（如 70B+ 模型）+ 并发需求不高
不适用本项目：8B 单卡能放下，加 TP 反而引入通信开销
```

### 3.4 级别四：多 GPU 多副本（Replica Parallel，推荐扩展方案）

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ GPU 0 (32GB) │   │ GPU 1 (32GB) │   │ GPU 2 (32GB) │   │ GPU 3 (32GB) │
│vLLM Instance │   │vLLM Instance │   │vLLM Instance │   │vLLM Instance │
│ :8000        │   │ :8001        │   │ :8002        │   │ :8003        │
│ 独立模型副本  │   │ 独立模型副本  │   │ 独立模型副本  │   │ 独立模型副本  │
│ 独立KV Cache │   │ 独立KV Cache │   │ 独立KV Cache │   │ 独立KV Cache │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │                  │
       └──────────────────┴──────────────────┴──────────────────┘
                           │
                    ┌──────┴──────┐
                    │   Nginx LB  │
                    │   :80       │
                    └─────────────┘
                    负载均衡对外暴露

适用：模型能放进单卡 + 并发需求高
每加一张 GPU，总吞吐 ≈ 线性扩展（减去负载均衡开销）
本项目如果扩展到多卡，优先选这个方案
```

> **面试金句**："多 GPU 推理的扩展策略决策树：模型能放进单卡吗？能→多副本横向扩展（线性扩展，无通信开销）。不能→Tensor Parallel 纵向切分（单请求延迟可能增加但能跑大模型）。大并发时→在多副本基础上继续加副本。混部→部分卡 TP 跑大模型，部分卡多副本跑小模型。"

### 3.5 级别五：多节点多副本（集群级部署）

```
机房节点1:  4 × RTX 5090 → 4 个 vLLM 实例 :8000-8003
机房节点2:  4 × RTX 5090 → 4 个 vLLM 实例 :8000-8003
                                ↓
                    全局负载均衡器 (API Gateway)
                                ↓
                           客户端请求

开销：跨节点网络延迟（>1ms） + 负载均衡开销
适用：超大规模并发（>100 QPS）+ 需要高可用 + 需要异地多活
本项目暂时不需要，但架构上已预留扩展空间
```

---

## 4. 四种推理并行策略深度解析

### 4.1 Tensor Parallel（张量并行）

**原理**：将单层 Transformer 的权重矩阵**按列或按行切分**到多张 GPU 上，每张 GPU 只计算一部分，然后通过 all-reduce / all-gather 通信聚合结果。

```
Attention 层的 TP (TP=2):
┌──────────────────────────────────────────────────────┐
│  Input: [batch, seq_len, hidden_dim]                 │
│                         │                            │
│         ┌───────────────┴───────────────┐            │
│    GPU0: Q/K/V/O 的前半部分              GPU1: 后半   │
│    Q0 = X × W_Q[: , :half]              Q1 = ...    │
│    K0 = X × W_K[: , :half]              K1 = ...    │
│    V0 = X × W_V[: , :half]              V1 = ...    │
│    Attn0 = softmax(Q0×K0^T/√d)×V0      Attn1 = ...  │
│    O0 = Attn0 × W_O[half:, :]           O1 = ...     │
│         └───────────────┬───────────────┘            │
│                    all-reduce(O0, O1)                │
│                         │                            │
│  Output = O0 + O1      (等价于完整结果)               │
└──────────────────────────────────────────────────────┘
```

**优点**：
- 降低单卡显存需求（权重分片，每卡存 1/TP 的权重）。
- 对 prefill 阶段有加速效果（attention 矩阵被分片并行计算，但通信开销抵消部分收益）。
- 对 batch size 较大的请求利用率高。

**缺点**：
- 每层都需要 GPU 间通信（all-reduce），对网络带宽要求高。消费级显卡用 PCIe 通信延迟大，企业级用 NVLink 好很多。
- Decode 阶段每次只处理 1 个 token，计算量极小，通信开销占比高，**TP 反而可能增加 decode 延迟**。
- 总 KV Cache 容量 = 各卡容量之和（减去模型分片），不是简单 N 倍。

**何时用 TP**：
- 模型太大，单卡放不下（如 70B 模型 FP16 需要 140GB，需要 4-8 张卡 TP）。
- **如果模型能放进单卡，加 TP 是负优化。**

```bash
# vLLM TP 启动
vllm serve ./model --tensor-parallel-size 2 --dtype bfloat16
```

### 4.2 Pipeline Parallel（流水线并行）

**原理**：将模型按层切分成多个 stage，每个 GPU 负责一部分层。请求像流水线一样依次经过各 stage。

```
PP=2:
GPU0: Embedding → Layer 0-15 → 中间激活
                                     ↓ (通过 PCIe/NVLink)
GPU1:                               Layer 16-31 → LM Head → output

Micro-batch 调度:
时间 →
GPU0: [B1_L0-15] [B2_L0-15] [B3_L0-15] ...
GPU1:    (idle)   [B1_L16-31] [B2_L16-31] [B3_L16-31] ...
                  ↑ 存在 pipeline bubble (空闲时间)
```

**优点**：
- 减少单卡显存需求（只存部分层）。
- 通信只在 stage 边界发生（比 TP 的每层通信少）。

**缺点**：
- Pipeline bubble：第一个 stage 在处理 batch 时，最后一个 stage 空闲等待，反之亦然。
- 不适合在线推理：单请求延迟被各 stage 串行叠加。
- 适合离线批量推理（大 batch 可掩盖 pipeline bubble）。

**vLLM 中的 PP 支持**：
```bash
vllm serve ./model --pipeline-parallel-size 2 --dtype bfloat16
```
注意：vLLM 的 PP 支持不如 TP 成熟，对于大模型部署通常首选 TP。

### 4.3 Data Parallel / Replica Parallel（数据并行 / 副本并行）

**原理**：每张 GPU 持有一份**完整的模型副本**，各自独立服务请求。负载均衡器将请求分发到不同副本。

```
GPU0: 完整模型副本 + 独立 KV Cache → 服务请求 1,3,5,7...
GPU1: 完整模型副本 + 独立 KV Cache → 服务请求 2,4,6,8...
GPU2: 完整模型副本 + 独立 KV Cache → 服务请求 ...
GPU3: 完整模型副本 + 独立 KV Cache → 服务请求 ...

没有 GPU 间通信！每个副本完全独立！
```

**优点**：
- **零通信开销**：每个副本独立运行，不需要 GPU 间同步。
- **线性扩展**：N 个副本 → 总吞吐 ≈ N × 单副本吞吐（减去负载均衡器开销）。
- **高可用**：一个副本宕机，其他副本继续服务。
- **延迟不增加**：单请求只在一个副本上处理，延迟与单卡一致。

**缺点**：
- 每张卡都需要能装下完整模型（8B BF16 需要 16GB，RTX 5090 32GB 可以）。
- 各副本之间 KV Cache 不共享（无法共享 prefix cache 跨越副本）。

**何时用 Replica**：
- 模型能放进单卡（这是前提）。
- 需要高吞吐（并发请求多）。
- 需要高可用（容错）。
- **本项目扩展首选方案。**

### 4.4 混合策略：TP + Replica（工业界大模型部署常用）

```
┌──────────────────────────────────────────────────┐
│  4 × RTX 5090 集群                               │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐                │
│  │ GPU0 + GPU1 │  │ GPU2 + GPU3 │                │
│  │   TP=2      │  │   TP=2      │                │
│  │  Replica 1  │  │  Replica 2  │                │
│  │  :8000      │  │  :8001      │                │
│  └──────┬──────┘  └──────┬──────┘                │
│         └────────┬───────┘                        │
│             Nginx LB                             │
└──────────────────────────────────────────────────┘

适用场景：大模型（如 70B）+ 高并发
  TP=2：让 70B 模型能放进 2 × 32GB = 64GB
  Replica=2：横向扩展并发能力
  总计：4 张卡跑 2 个 TP-2 副本
```

---

## 5. 负载均衡策略详解

### 5.1 基础策略

| 策略 | 原理 | 优点 | 缺点 | 本项目适用性 |
|------|------|------|------|------------|
| **Round-Robin** | 依次轮询分发请求 | 最简单，无状态 | 不考虑后端负载，可能不均衡 | 多副本均质部署适用 |
| **Least Connections** | 发给当前活跃连接数最少的后端 | 比 RR 更均衡 | 不考虑请求复杂度差异 | **推荐** |
| **Random** | 随机选择后端 | 简单 | 长尾可能不均衡 | 不推荐 |
| **IP Hash** | 按客户端 IP 哈希选后端 | 同一用户始终打到同一后端 | 后端宕机导致该用户全部失败 | session 需要时用 |
| **Weighted** | 按权重分配（高性能 GPU 设更高权重） | 异构硬件友好 | 需要手动维护权重 | 异构 GPU 集群专用 |

### 5.2 高级策略

**Latency-Aware Routing（延迟感知路由）**：
```
监控每个后端的 p50/p95 延迟 → 优先发给延迟低的后端
实现：定期 health check 统计延迟 → 更新权重
```

**Queue-Aware Routing（队列感知路由）**：
```
监控每个后端的排队请求数 → 发给排队最少的后端
vLLM 可通过 /health 或 metrics endpoint 暴露排队深度
```

**GPU Memory-Aware Routing（显存感知路由）**：
```
监控每个后端的 KV Cache 使用率 → 发给显存最充裕的后端
避免发给 KV Cache 快满的后端（可能导致 OOM）
```

**Token-Length-Aware Routing（长度感知路由）**：
```
检查 prompt 长度 → 短 prompt 发给高负载实例，长 prompt 发给低负载实例
避免长 prompt 集中打到一个实例导致该实例的 TTFT 恶化
```

### 5.3 医学场景的请求分流策略

医学 LLM Teacher 场景下，不同请求对延迟/安全的要求差异很大，适合做**差异化路由**：

```
                      ┌──────────────────┐
                      │   请求分类器      │
                      │ (基于 prompt/用户) │
                      └────────┬─────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
   ┌────────────┐    ┌────────────┐    ┌────────────┐
   │ 普通问答   │    │ RAG 检索   │    │ 高风险医疗  │
   │ 快速链路   │    │ 标准链路   │    │ 安全链路    │
   │            │    │            │    │            │
   │ 短 prompt  │    │ 长 prompt  │    │ Safety-RAG │
   │ 不检索     │    │ 需检索     │    │ 需审核     │
   │ 低延迟优先 │    │ 准确性优先 │    │ 安全优先   │
   └─────┬──────┘    └─────┬──────┘    └─────┬──────┘
         │                 │                  │
         ▼                 ▼                  ▼
   Instance A,B      Instance C,D       Instance E
   (max_seqs=16)    (max_seqs=8)      (max_seqs=4,
   无 RAG           RAG 优化配参       安全审核链路)
```

**分流的好处**：
1. 普通简单问答走快速链路，不受 RAG 请求的 prefill 拖累。
2. RAG 长 prompt 请求独立队列，不会因为 prefill 慢而阻塞快速请求。
3. 高风险医疗（含处方/诊断）走安全链路，带完整的安全审核流程。
4. 各链路的 vLLM 参数可独立优化：快速链路设更大的 max-num-seqs，安全链路设更小以留安全余量。

> **面试金句**："医学场景不应该所有请求走同一链路。简单概念问答（'什么是高血压'）和复杂临床问题（'这个病人应该用什么药'）的延迟要求和安全要求完全不同。分类路由让快速请求不被慢请求阻塞，高风险请求不跳过安全检查。"

---

## 6. 队列与限流设计

### 6.1 为什么要排队和限流

vLLM 本身有内部调度队列，但如果并发超限（超过 max_num_seqs 能承载的量），新请求会被阻塞在 HTTP 层等待。不做外部队列/限流会导致：

1. **TCP 连接堆积**：太多请求等待 → 连接超时 → 用户重试 → 更多请求 → 雪崩。
2. **长尾爆炸**：队列中的请求延迟无保障，p99 可能从 500ms 飙到几十秒。
3. **恶意/异常请求**：超长 prompt 会占用大量 prefill 时间和 KV Cache，影响其他用户。

### 6.2 请求队列设计

```python
"""
request_queue.py
请求队列伪代码：令牌桶 + 优先级队列 + 超时控制
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable


class Priority(Enum):
    HIGH = 0    # 高风险医疗（安全审核优先）
    NORMAL = 1  # 普通问答 + RAG
    LOW = 2     # 离线批处理 / 系统维护请求


@dataclass(order=True)
class QueuedRequest:
    """优先级队列中的请求"""
    priority: int              # Priority enum 的值
    enqueue_time: float = field(compare=False)  # 入队时间
    timeout_s: float = field(compare=False)     # 超时时间（秒）
    request_data: dict = field(compare=False)   # 原始请求数据
    callback: Callable = field(compare=False)   # 实际执行的协程


class RequestQueue:
    """基于优先级的异步请求队列 + 限流"""

    def __init__(
        self,
        max_concurrent: int = 16,       # 最大并发（对应 vLLM max_num_seqs）
        max_queue_size: int = 100,      # 最大排队长度
        default_timeout_s: float = 30.0,  # 默认超时
        high_priority_timeout_s: float = 60.0,  # 高风险请求更长超时
    ):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=max_queue_size
        )
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._default_timeout = default_timeout_s
        self._high_priority_timeout = high_priority_timeout_s

        # 统计指标
        self._total_requests = 0
        self._rejected_requests = 0
        self._timeout_requests = 0

    async def enqueue(
        self,
        request_data: dict,
        callback: Callable[[dict], Awaitable[dict]],
        priority: Priority = Priority.NORMAL,
        timeout_s: Optional[float] = None,
    ) -> dict:
        """
        将请求加入队列并等待执行

        Args:
            request_data: 原始请求数据
            callback: 实际执行推理的 async 函数
            priority: 优先级
            timeout_s: 超时时间，None 则使用默认

        Returns:
            推理结果

        Raises:
            asyncio.TimeoutError: 请求超时
            QueueFull: 队列已满
        """
        if timeout_s is None:
            timeout_s = (
                self._high_priority_timeout
                if priority == Priority.HIGH
                else self._default_timeout
            )

        request = QueuedRequest(
            priority=priority.value,
            enqueue_time=time.time(),
            timeout_s=timeout_s,
            request_data=request_data,
            callback=callback,
        )

        # 尝试入队（如果队列满了会抛 QueueFull）
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            self._rejected_requests += 1
            raise QueueFull("Request queue is full, try again later")

        self._total_requests += 1

        # 等待获取执行槽位（受并发数限制）
        try:
            async with asyncio.timeout(timeout_s):
                async with self._semaphore:
                    # 从队列取出并执行
                    queued = await self._queue.get()

                    # 检查是否已超时（排队时间已超过 timeout）
                    wait_time = time.time() - queued.enqueue_time
                    if wait_time > queued.timeout_s:
                        self._timeout_requests += 1
                        raise asyncio.TimeoutError(
                            f"Request timed out after waiting {wait_time:.1f}s"
                        )

                    # 执行推理
                    result = await queued.callback(queued.request_data)
                    return result

        except asyncio.TimeoutError:
            self._timeout_requests += 1
            raise

    @property
    def queue_depth(self) -> int:
        """当前排队请求数"""
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "queue_depth": self.queue_depth,
            "active_requests": self._max_concurrent - self._semaphore._value,
            "total_requests": self._total_requests,
            "rejected_requests": self._rejected_requests,
            "timeout_requests": self._timeout_requests,
        }


class QueueFull(Exception):
    pass
```

### 6.3 限流设计

```python
"""
rate_limiter.py
多层级限流：用户级、IP级、全局token budget
"""
import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """限流配置"""
    requests_per_second: int = 20         # 全局 QPS 上限
    requests_per_user_per_minute: int = 10  # 每用户每分钟上限
    requests_per_ip_per_second: int = 5     # 每 IP 每秒上限
    max_prompt_tokens: int = 3000           # 单次请求 prompt 上限
    max_total_tokens_per_request: int = 4096  # 单次请求总 token 上限


class RateLimiter:
    """
    滑动窗口 + Token Bucket 混合限流器
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config

        # Token Bucket（全局）
        self._global_tokens = config.requests_per_second
        self._global_max_tokens = config.requests_per_second
        self._last_refill = time.monotonic()

        # 用户级滑动窗口计数
        self._user_window: dict[str, list[float]] = defaultdict(list)

        # IP 级滑动窗口计数
        self._ip_window: dict[str, list[float]] = defaultdict(list)

        self._lock = asyncio.Lock()

    async def _refill_global(self):
        """令牌桶补充"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        refill = elapsed * self.config.requests_per_second
        self._global_tokens = min(
            self._global_max_tokens, self._global_tokens + refill
        )
        self._last_refill = now

    def _clean_sliding_window(self, window: list[float], window_seconds: float):
        """清理滑动窗口中的过期记录"""
        cutoff = time.time() - window_seconds
        while window and window[0] < cutoff:
            window.pop(0)

    async def check_and_acquire(
        self,
        user_id: str = "anonymous",
        ip_address: str = "unknown",
        prompt_tokens: int = 0,
    ) -> tuple[bool, str]:
        """
        检查并获取请求许可

        Returns:
            (allowed: bool, reason: str)
        """
        async with self._lock:
            now = time.time()

            # === 检查1: Prompt 长度限制 ===
            if prompt_tokens > self.config.max_prompt_tokens:
                return False, (
                    f"Prompt too long: {prompt_tokens} tokens, "
                    f"max {self.config.max_prompt_tokens}"
                )

            # === 检查2: 用户级限流（每分钟） ===
            user_window = self._user_window[user_id]
            self._clean_sliding_window(user_window, 60.0)
            if len(user_window) >= self.config.requests_per_user_per_minute:
                return False, "User rate limit exceeded (requests/min)"

            # === 检查3: IP 级限流（每秒） ===
            ip_window = self._ip_window[ip_address]
            self._clean_sliding_window(ip_window, 1.0)
            if len(ip_window) >= self.config.requests_per_ip_per_second:
                return False, "IP rate limit exceeded (requests/s)"

            # === 检查4: 全局令牌桶 ===
            await self._refill_global()
            if self._global_tokens < 1.0:
                return False, "Global rate limit exceeded"
            self._global_tokens -= 1.0

            # === 全部通过，记录 ===
            user_window.append(now)
            ip_window.append(now)
            return True, "OK"

    def get_stats(self) -> dict:
        """获取限流统计"""
        return {
            "global_tokens_available": self._global_tokens,
            "active_users": len(self._user_window),
            "active_ips": len(self._ip_window),
        }
```

### 6.4 超时与重试设计

```python
"""
timeout_retry.py
超时控制 + 指数退避重试 + Circuit Breaker
"""
import asyncio
import random
import time
from enum import Enum
from typing import Optional


class CircuitState(Enum):
    CLOSED = "closed"          # 正常
    OPEN = "open"              # 熔断
    HALF_OPEN = "half_open"    # 半开（试探恢复）


class CircuitBreaker:
    """
    熔断器：当连续失败超过阈值时自动切断请求，
    避免雪崩效应，一段时间后半开试探恢复。
    """

    def __init__(
        self,
        failure_threshold: int = 5,       # 连续失败 N 次后熔断
        recovery_timeout: float = 30.0,   # 熔断后 30s 试探恢复
        half_open_max_requests: int = 2,  # 半开状态允许的试探请求数
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_requests = 0

    async def call(
        self,
        coro,
        fallback=None,
    ):
        """
        通过熔断器执行请求
        """
        # 熔断状态：直接拒绝
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                # 进入半开状态
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
            else:
                if fallback:
                    return await fallback() if callable(fallback) else fallback
                raise ServiceUnavailableError("Circuit breaker is OPEN")

        # 半开状态：限制试探请求数
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_requests >= self.half_open_max_requests:
                if fallback:
                    return await fallback() if callable(fallback) else fallback
                raise ServiceUnavailableError("Circuit breaker is HALF_OPEN")

        # 执行请求
        try:
            result = await coro
            # 成功：重置
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_requests += 1
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        if self.state == CircuitState.HALF_OPEN:
            # 半开状态下连续成功 → 恢复
            self.state = CircuitState.CLOSED
        self.failure_count = 0

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class ServiceUnavailableError(Exception):
    pass


async def call_with_retry(
    coro_factory,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (asyncio.TimeoutError, ConnectionError),
):
    """
    指数退避重试（带 jitter）

    Args:
        coro_factory: 返回 coroutine 的可调用对象（每次重试创建新的）
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        jitter: 是否加随机抖动（避免惊群效应）
        retryable_exceptions: 可重试的异常类型
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except retryable_exceptions as e:
            last_exception = e
            if attempt == max_retries:
                break

            # 指数退避
            delay = min(base_delay * (2 ** attempt), max_delay)
            if jitter:
                delay = delay * (0.5 + random.random())  # 50%-150% 抖动

            print(f"Retry {attempt + 1}/{max_retries} after {delay:.1f}s: {e}")
            await asyncio.sleep(delay)

    raise last_exception


# ==================== 使用示例 ====================
async def example_usage():
    """超时 + 重试 + 熔断 组合使用"""
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)

    async def make_request():
        return await call_with_retry(
            lambda: vllm_client.generate({"prompt": "..."}),
            max_retries=3,
            base_delay=0.5,
        )

    try:
        # 通过熔断器发送请求
        result = await breaker.call(
            make_request(),
            fallback=lambda: {"response": "Service temporarily unavailable, please retry."}
        )
        return result
    except ServiceUnavailableError:
        return {"error": "Service is currently overloaded."}
```

---

## 7. 流式输出（Streaming）

### 7.1 流式输出是什么

流式输出（Server-Sent Events, SSE）是 vLLM 在生成每个 token 时立即推送给客户端，而不是等全部生成完成再一次性返回。

```
非流式:
Client ──Request──> Server ──[生成全部 256 tokens]──> Response
        等待全部完成 ↑ (TTFT = 总等待时间)

流式:
Client ──Request──> Server ──tok1──> Client (TTFT ~42ms)
                             ──tok2──> Client
                             ──tok3──> Client
                             ...
                             ──tok256──> Client [DONE]
                             ↑ 每个 token 生成后立即发送
```

### 7.2 流式输出的核心价值：降低感知延迟

```
非流式输出 (total_time=2s):
  用户等待 2s → 看到完整回答
  感知延迟 = 2s

流式输出 (total_time=2s, TTFT=50ms):
  用户等待 50ms → 看到第一个 token → 后续 token 流式到达
  感知延迟 ≈ 50ms（远低于 300ms 人类感知阈值）
  
总延迟不变（都是 2s），但用户感知的等待从 2s 变成了 50ms！
```

**注意**：流式输出**不降低总延迟**，有时甚至因 SSE framing 和 HTTP chunking 略微增加总时间（1-2%）。它的价值纯粹在于**感知优化**。

### 7.3 SSE (Server-Sent Events) 协议

vLLM 的流式输出使用 SSE 协议（HTTP chunked transfer + `text/event-stream` content type）：

```
HTTP Response Headers:
  Content-Type: text/event-stream
  Cache-Control: no-cache
  Connection: keep-alive
  Transfer-Encoding: chunked

Body (SSE events):
  data: {"id":"cmpl-xxx","choices":[{"delta":{"content":"高"},"index":0}]}

  data: {"id":"cmpl-xxx","choices":[{"delta":{"content":"血"},"index":0}]}

  data: {"id":"cmpl-xxx","choices":[{"delta":{"content":"压"},"index":0}]}
  ...

  data: [DONE]
```

### 7.4 医疗问答是否适合流式输出

| 场景 | 适合流式？ | 理由 |
|------|----------|------|
| 医学概念解释 | **适合** | 用户想尽快看到内容，流式输出感知快 |
| 诊断建议 | **需谨慎** | 诊断结果不应"逐字流出"——用户可能在看到结论前就基于不完整信息做判断 |
| 药物信息查询 | **适合** | 但需要在完整回答前做安全过滤 |
| 高风险医疗回答 | **不适合** | 应该**先完整生成 → 安全审核 → 再一次性返回** |
| 急救指导 | **不适合** | 生命攸关，宁可等 2s 也要保证信息的完整和准确 |

### 7.5 安全审核在流式输出前还是后

这是一个关键的设计决策：

**方案一：先完整生成 → 审核 → 再流式输出（安全优先）**
```
用户请求 → [完整生成全部 tokens] → [安全审核] → [流式输出给用户]
                     ↑                         ↑
              内部完整生成                审核通过后才逐 token 推送
```
- 优点：安全审核在用户看到内容之前完成，保证输出安全。
- 缺点：用户的实际 TTFT = 完整生成时间 + 审核时间（失去了流式的感知优势）。
- 适用：高风险医疗场景。

**方案二：先流式生成 → 边生成边审核（延迟优先）**
```
用户请求 → [逐 token 生成] → [实时安全审核]
                 │                      │
                 └── 同时推送 ──────────┘
```
- 优点：用户 TTFT 不变，感知延迟低。
- 缺点：如果中间 token 触发安全规则——前面已经推送的 token 无法撤回（覆水难收）。
- 适用：低风险场景（概念解释、学习问答）。

**方案三：分类决策（本项目推荐）**
```
请求分类 → 低风险医学概念 → 方案二（边生成边审核边推送）
        → 高风险医疗     → 方案一（先完整生成审核再输出）
        → 包含处方/剂量  → 方案一 + 额外人工审核标记
```

> **面试金句**："流式输出在医学场景是双刃剑。低风险的医学概念解释适合流式（感知延迟低，用户满意），但涉及诊断建议或药物剂量时，必须完整生成并通过安全审核后再输出——不能为了快而牺牲安全。我们采用按风险等级分类路由的策略：低风险走流式快速通道，高风险走完整审核通道。"

### 7.6 流式输出 API 示例

```python
"""
streaming_api.py
vLLM 流式输出 + 安全审核的完整示例
"""
from openai import OpenAI
import asyncio

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)


# ==================== 低风险：流式输出（边生成边推送） ====================
def stream_low_risk(prompt: str):
    """低风险医学概念问答：直接流式输出"""
    stream = client.chat.completions.create(
        model="medical-llm-teacher",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ==================== 高风险：先完整生成 → 审核 → 再输出 ====================
def generate_and_audit(prompt: str):
    """
    高风险医疗：先生成完整回答 → 安全审核 → 流式输出
    用户感知的 TTFT = 完整生成时间 + 审核时间
    """
    # 1. 先生成完整回答（非流式）
    response = client.chat.completions.create(
        model="medical-llm-teacher",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=512,
        stream=False,
    )
    full_answer = response.choices[0].message.content

    # 2. 安全审核（可以用另一个模型或规则引擎）
    safety_result = safety_audit(full_answer)

    # 3. 审核通过 → 流式输出（模拟流式用户体验）
    if safety_result["passed"]:
        # 逐 token 模拟流式输出（实际是完整文本按 token 拆分模拟）
        words = full_answer  # 简化：直接返回完整文本
        return {"status": "ok", "answer": words}
    else:
        return {
            "status": "blocked",
            "reason": safety_result["reason"],
            "fallback": "抱歉，该问题涉及高风险医疗建议，请咨询专业医生。"
        }


def safety_audit(text: str) -> dict:
    """
    安全审核（简化版）
    实际可用：
    - 正则匹配高风险模式（处方、剂量、手术建议）
    - 调用 Safety-RAG Verifier
    - 调用 LLM-as-Judge 安全评分
    """
    # 危险关键词检测
    dangerous_patterns = [
        "每天服用", "一次吃", "推荐剂量", "建议使用.*mg",
        "不需要去医院", "自己在家", "不用看医生",
    ]
    import re
    for pattern in dangerous_patterns:
        if re.search(pattern, text):
            return {"passed": False, "reason": f"Matched dangerous pattern: {pattern}"}

    return {"passed": True, "reason": ""}


# ==================== 分类路由 ====================
def classify_risk(prompt: str) -> str:
    """根据 prompt 内容判断风险等级"""
    high_risk_keywords = [
        "吃药", "处方", "剂量", "手术", "急救", "诊断",
        "我该用什么药", "怎么治疗", "严重吗", "会不会死",
    ]
    prompt_lower = prompt.lower()
    for kw in high_risk_keywords:
        if kw in prompt_lower:
            return "high"
    return "low"


def smart_generate(prompt: str):
    """智能路由：按风险等级选择策略"""
    risk = classify_risk(prompt)
    if risk == "high":
        return generate_and_audit(prompt)
    else:
        return stream_low_risk(prompt)
```

### 7.7 WebSocket vs SSE 对比

| 维度 | SSE (Server-Sent Events) | WebSocket |
|------|-------------------------|-----------|
| 通信方向 | 单向（Server → Client） | 双向（Server ↔ Client） |
| 协议 | HTTP (chunked transfer) | 独立协议 (ws://) |
| 实现复杂度 | 简单（标准 HTTP） | 较复杂（需要升级握手） |
| 浏览器支持 | 原生 EventSource API | 原生 WebSocket API |
| 适用场景 | LLM 流式输出（客户端只需接收） | 需要双向通信（如实时对话+中断） |
| vLLM 支持 | 原生支持（SSE） | 需额外封装 |

**vLLM 默认使用 SSE**，因为 LLM 推理场景中客户端只需接收流式数据，不需要双向通信。

---

## 8. 代码合集

### 8.1 vLLM 单卡启动命令

```bash
#!/bin/bash
# single_gpu_start.sh
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve ./merged_qwen3_8b_medical \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name medical-llm-teacher \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --chat-template ./chat_template.jinja \
  --trust-remote-code \
  --disable-log-requests
```

### 8.2 vLLM Tensor Parallel 多卡启动命令

```bash
#!/bin/bash
# tp2_start.sh
# 2 张 GPU Tensor Parallel（适用于 70B+ 模型放不进单卡时）

export CUDA_VISIBLE_DEVICES=0,1
export VLLM_USE_FLASHINFER_SAMPLER=0

vllm serve ./Llama-3.1-70B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name llama-70b \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \          # 跨 2 张 GPU 切分模型
  --max-model-len 4096 \
  --max-num-seqs 8 \                  # 大模型更吃显存，降低并发
  --gpu-memory-utilization 0.85 \     # TP 留更多余量
  --trust-remote-code \
  --disable-log-requests
```

### 8.3 多实例启动脚本（Replica Parallel）

```bash
#!/bin/bash
# multi_instance_start.sh
# 4 × RTX 5090 多实例部署（每张 GPU 一个独立 vLLM 实例）

# FlashInfer 兼容性
export VLLM_USE_FLASHINFER_SAMPLER=0

# 模型公共参数
MODEL_PATH="./merged_qwen3_8b_medical"
CHAT_TEMPLATE="./chat_template.jinja"

# 定义实例配置：GPU编号, 端口
INSTANCES=(
    "0 8000"
    "1 8001"
    "2 8002"
    "3 8003"
)

# 启动所有实例
echo "Starting vLLM multi-instance deployment..."
for instance in "${INSTANCES[@]}"; do
    read -r GPU PORT <<< "$instance"

    CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL_PATH" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --served-model-name medical-llm-teacher \
      --dtype bfloat16 \
      --max-model-len 4096 \
      --max-num-seqs 16 \
      --gpu-memory-utilization 0.90 \
      --enable-prefix-caching \
      --chat-template "$CHAT_TEMPLATE" \
      --trust-remote-code \
      --disable-log-requests \
      > "vllm_gpu${GPU}_port${PORT}.log" 2>&1 &

    echo "  GPU ${GPU}: Port ${PORT} (PID $!)"
done

echo "All instances started."
echo ""
echo "Waiting for all instances to be healthy..."

# 等待所有实例就绪
HEALTHY=false
MAX_WAIT=120
ELAPSED=0
while [ "$HEALTHY" = false ] && [ $ELAPSED -lt $MAX_WAIT ]; do
    ALL_READY=true
    for instance in "${INSTANCES[@]}"; do
        read -r GPU PORT <<< "$instance"
        if ! curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            ALL_READY=false
            break
        fi
    done
    if [ "$ALL_READY" = true ]; then
        HEALTHY=true
        echo "All instances are healthy!"
    else
        sleep 3
        ELAPSED=$((ELAPSED + 3))
        echo "  Waiting... (${ELAPSED}s/${MAX_WAIT}s)"
    fi
done

if [ "$HEALTHY" = false ]; then
    echo "ERROR: Not all instances became healthy within ${MAX_WAIT}s"
    exit 1
fi

echo "Multi-instance deployment ready!"
echo "Endpoints: http://localhost:8000, :8001, :8002, :8003"
```

### 8.4 Nginx 负载均衡配置

```nginx
# nginx_vllm_lb.conf
# Nginx 负载均衡配置：将请求分发到 4 个 vLLM 实例

upstream vllm_backend {
    # 最少连接数策略（推荐）
    least_conn;

    # 后端实例列表
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;

    # 健康检查（需要 nginx plus 或 openresty lua 模块）
    # 开源 nginx 使用 max_fails + fail_timeout 做被动健康检查

    # 长连接保持（减少 TCP 握手开销）
    keepalive 64;
    keepalive_requests 1000;
}

# 高风险医学请求专用 upstream（安全审核链路）
upstream vllm_safety_backend {
    least_conn;
    server 127.0.0.1:8010 max_fails=2 fail_timeout=15s;
    server 127.0.0.1:8011 max_fails=2 fail_timeout=15s;
    keepalive 32;
}

server {
    listen 80;
    server_name medical-llm.local;

    # 客户端请求体大小限制（防止超大 prompt）
    client_max_body_size 1m;

    # 超时设置
    proxy_read_timeout 120s;     # LLM 推理可能较慢
    proxy_send_timeout 30s;
    proxy_connect_timeout 10s;

    # 通用路由：普通医学问答 + RAG
    location /v1/chat/completions {
        # 通过请求头判断是否为高风险（由 API Gateway 预处理后设置）
        set $backend "vllm_backend";
        if ($http_x_risk_level = "high") {
            set $backend "vllm_safety_backend";
        }

        proxy_pass http://$backend;
        proxy_http_version 1.1;

        # 关键：流式输出必须禁用缓冲！
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;

        # 转发必要头
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";  # 保持长连接

        # 超时重试
        proxy_next_upstream error timeout http_502 http_503;
        proxy_next_upstream_tries 2;
    }

    # 健康检查 endpoint
    location /health {
        return 200 '{"status":"ok"}';
        add_header Content-Type application/json;
    }

    # vLLM 其他 API
    location /v1/ {
        proxy_pass http://vllm_backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header Connection "";
    }

    # 速率限制（使用 limit_req 模块）
    # 定义限流区域：每秒 20 个请求
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=20r/s;
    limit_req zone=api_limit burst=10 nodelay;
    limit_req_status 429;
}
```

### 8.5 FastAPI 转发 vLLM 示例

```python
"""
fastapi_gateway.py
FastAPI API Gateway：请求预处理 + 分类路由 + 多 vLLM 后端转发
"""
import asyncio
import random
from typing import Optional
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Medical LLM API Gateway")

# ==================== 后端实例配置 ====================
VLLM_BACKENDS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
]
VLLM_SAFETY_BACKENDS = [
    "http://127.0.0.1:8010",
    "http://127.0.0.1:8011",
]

# 后端状态跟踪
backend_states = {
    url: {"active_connections": 0, "healthy": True}
    for url in VLLM_BACKENDS + VLLM_SAFETY_BACKENDS
}

# ==================== 请求模型 ====================
class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.3
    max_tokens: int = 512
    stream: bool = False
    user_id: Optional[str] = None


# ==================== 路由选择策略 ====================
def classify_risk(messages: list[dict]) -> str:
    """根据消息内容判断风险等级"""
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user")
    high_risk_keywords = [
        "吃药", "处方", "剂量", "手术", "急救", "诊断是什么",
        "我该用什么", "怎么治", "严重吗",
    ]
    for kw in high_risk_keywords:
        if kw in user_text:
            return "high"
    return "low"


def select_backend(risk_level: str) -> str:
    """
    Least-Connections 策略选择后端

    生产环境可用更复杂策略：
    - 查询每个后端的 /metrics 获取 queue depth
    - 结合延迟统计做 latency-aware routing
    """
    if risk_level == "high":
        pool = VLLM_SAFETY_BACKENDS
    else:
        pool = VLLM_BACKENDS

    # 按活跃连接数排序，选最小的
    healthy_backends = [
        url for url in pool
        if backend_states[url]["healthy"]
    ]
    if not healthy_backends:
        raise HTTPException(503, "No healthy backends available")

    return min(
        healthy_backends,
        key=lambda url: backend_states[url]["active_connections"],
    )


# ==================== API 端点 ====================
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """转发请求到合适的 vLLM 后端"""
    risk_level = classify_risk(req.messages)
    backend = select_backend(risk_level)

    payload = req.model_dump(exclude_none=True)

    try:
        backend_states[backend]["active_connections"] += 1

        if req.stream:
            # 流式：直接透传 SSE
            async def stream_forward():
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream(
                        "POST",
                        f"{backend}/v1/chat/completions",
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    ) as resp:
                        resp.raise_for_status()
                        async for chunk in resp.aiter_bytes():
                            yield chunk

            return StreamingResponse(
                stream_forward(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Risk-Level": risk_level,
                    "X-Backend": backend,
                },
            )
        else:
            # 非流式：正常 POST
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    f"{backend}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                data = resp.json()
                return JSONResponse(
                    content=data,
                    headers={
                        "X-Risk-Level": risk_level,
                        "X-Backend": backend,
                    },
                )

    except httpx.HTTPError as e:
        backend_states[backend]["healthy"] = False
        raise HTTPException(502, f"Backend error: {e}")
    finally:
        backend_states[backend]["active_connections"] -= 1


@app.get("/health")
async def health():
    """健康检查：检查所有后端状态"""
    return {
        "status": "ok",
        "backends": {
            url: state
            for url, state in backend_states.items()
        },
    }


@app.get("/metrics")
async def metrics():
    """暴露指标（供负载均衡器或监控系统使用）"""
    return {
        "backends": [
            {
                "url": url,
                "active": state["active_connections"],
                "healthy": state["healthy"],
            }
            for url, state in backend_states.items()
        ],
    }


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### 8.6 批量压测多实例脚本

```python
"""
benchmark_multi_instance.py
压测多实例部署：测试负载均衡下多个后端的聚合吞吐
"""
import asyncio
import time
import httpx

BACKENDS = [
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8003",
]
MODEL = "medical-llm-teacher"


async def send_to_backend(backend_url: str, prompt: str, max_tokens: int = 256):
    """向指定后端发送请求并统计延迟"""
    t_start = time.time()
    first_token = None
    token_count = 0

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{backend_url}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": True,
            },
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and "[DONE]" not in line:
                    if first_token is None:
                        first_token = time.time() - t_start
                    token_count += 1

    elapsed = time.time() - t_start
    return {
        "backend": backend_url,
        "ttft": first_token or elapsed,
        "total_time": elapsed,
        "tokens": token_count,
    }


async def benchmark_multi_instance(
    total_requests: int = 60,
    max_tokens: int = 256,
):
    """
    多实例均匀负载压测
    每个实例分配 total_requests / len(BACKENDS) 个请求
    所有实例并发运行
    """
    per_instance = total_requests // len(BACKENDS)
    prompt = "请详细解释高血压的发病机制和分类标准。"

    all_tasks = []
    for backend in BACKENDS:
        for _ in range(per_instance):
            all_tasks.append(send_to_backend(backend, prompt, max_tokens))

    print(f"Starting multi-instance benchmark:")
    print(f"  Instances: {len(BACKENDS)}")
    print(f"  Requests per instance: {per_instance}")
    print(f"  Total requests: {len(all_tasks)}")
    print(f"  Max tokens/request: {max_tokens}")
    print()

    t_start = time.time()
    results = await asyncio.gather(*all_tasks)
    total_elapsed = time.time() - t_start

    # 按后端汇总
    from collections import defaultdict
    by_backend = defaultdict(list)
    for r in results:
        by_backend[r["backend"]].append(r)

    total_tokens = sum(r["tokens"] for r in results)
    aggregate_throughput = total_tokens / total_elapsed

    print("=" * 80)
    print(f"{'Backend':<25} {'Count':>6} {'AvgTTFT':>10} {'Tokens':>8} "
          f"{'Thru(t/s)':>10}")
    print("-" * 80)

    for backend in BACKENDS:
        br = by_backend[backend]
        if br:
            avg_ttft = sum(r["ttft"] for r in br) / len(br) * 1000
            tokens = sum(r["tokens"] for r in br)
            thru = tokens / total_elapsed * len(BACKENDS)  # 近似各实例同时结束
            print(f"{backend:<25} {len(br):>6} {avg_ttft:>9.0f}ms {tokens:>8} "
                  f"{thru:>10.0f}")

    print("-" * 80)
    print(f"{'AGGREGATE':<25} {len(results):>6} {'-':>10} {total_tokens:>8} "
          f"{aggregate_throughput:>10.0f}")
    print("=" * 80)
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print(f"Aggregate throughput: {aggregate_throughput:.0f} tok/s")
    print(f"Expected (4x single): {aggregate_throughput:.0f} tok/s")
    print()

    # 效率分析
    single_throughput = 877  # 单实例参考吞吐
    expected = single_throughput * len(BACKENDS)
    efficiency = aggregate_throughput / expected * 100
    print(f"Scaling efficiency: {efficiency:.1f}%")
    print(f"(Expected {expected:.0f} tok/s based on {single_throughput} tok/s × {len(BACKENDS)})")


if __name__ == "__main__":
    asyncio.run(benchmark_multi_instance(total_requests=60, max_tokens=256))
```

---

## 面试1分钟回答

"多 GPU 推理部署的核心决策取决于两点：模型能不能放进单卡，以及并发需求多大。

我们的 Qwen3-8B 在 32GB RTX 5090 上单卡够用，所以首选多副本横向扩展。4 张卡各跑一个独立 vLLM 实例，Nginx least-connections 负载均衡，总吞吐接近线性扩展到 ~3400 tok/s，且延迟不增加（每个请求只在一个副本上处理）。

如果模型放不进单卡（如 70B），才考虑 Tensor Parallel 纵向切分。TP 的缺点是每层都需要 GPU 间通信，对 decode 阶段延迟可能有负面影响。

医学场景的关键设计是按风险分流：普通问答走流式快速通道，高风险医疗请求走安全审核链路（先完整生成审核再输出），RAG 长 prompt 请求单独队列避免阻塞快速请求。

负载均衡方面，Least Connections 比 Round-Robin 更均衡，配合健康检查和自动重启保证高可用。前端加请求队列 + 令牌桶限流防止过载雪崩，熔断器在连续失败时自动断路保护。"

---

## 面试3分钟回答

"多 GPU 推理的设计思路和训练并行完全不同。训练用 FSDP/ZeRO 分片是因为单卡放不下模型+优化器+激活值，推理的瓶颈在 KV Cache 给的并发上限，而非模型大小。

**架构演进**：我们是从单 GPU 单实例出发的。8B 模型 BF16 占 16GB，12.8GB 给 KV Cache pool，能支撑 16 并发。如果需要更大并发，我们首选多副本（Replica Parallel）方案——每张 GPU 各跑一个完整模型副本，Nginx least-connections 分发请求。不需要 GPU 间通信，线性扩展，高可用（一个实例挂掉不影响整体）。

**Tensor Parallel 只在必要时用**：当模型放不进单卡时才需要 TP。比如 70B 模型 FP16 需要 ~140GB，至少 4-5 张 32GB 卡用 TP=4 或 TP=8。TP 每层都需要 all-reduce 通信，在 decode 阶段（每次只处理 1 个 token）通信开销占比很高，对 RTX 5090 这种 PCIe 连接的消费卡尤其明显。这也是为什么我们不建议"能跑单卡的模型硬上 TP"。

**医学场景的三链路分类路由**是我认为最有工程价值的点：普通概念问答走流式快速通道（TTFT ~50ms）、RAG 检索问答走标准链路（长 prompt 独立队列）、高风险医疗请求走安全链路（先完整审核再输出）。这样快请求不被慢请求阻塞，高风险请求不会绕过安全检查。

**队列和限流方面**：我们用优先级队列 + Semaphore 控制实际并发，令牌桶做全局限流，用户级和 IP 级做多级限流。熔断器在连续 5 次失败后自动断路 30s，避免雪崩。指数退避重试（带 jitter）处理瞬时故障。

**流式输出的权衡**：流式输出能大幅降低感知延迟（TTFT 代替端到端延迟成为用户唯一感知的等待），但总延迟不变。对于低风险医学概念解释，我们直接用 SSE 流式输出。对于涉及诊断/处方的回答，必须完整生成并审核后再推送——不会为了快而牺牲安全。

**当前未用到但预留的设计**：如果未来需要跑 70B 模型 + 高并发，会用 TP+Replica 混部（比如 TP=2 跨两张卡跑一份模型，然后做 2 个副本，共 4 张卡）。API Gateway 层支持按 prompt 长度做 token-length-aware routing，长 prompt 单独队列防止 prefill 阻塞。"

---

## 背诵版总结

### 核心概念速记

| 概念 | 一句话 |
|------|--------|
| 训练并行 vs 推理并行 | 训练瓶颈在优化器+激活值（用FSDP/ZeRO），推理瓶颈在 KV Cache+延迟（用Replica/TP）|
| Replica Parallel | 每卡一个完整模型副本，零通信，线性扩展，模型能放进单卡首选 |
| Tensor Parallel | 权重切分到多卡，每层通信，模型放不进单卡时才用 |
| Pipeline Parallel | 按层切分，有 pipeline bubble，适合离线批量 |
| Continuous Batching | vLLM 动态混合 prefill/decode，无需多卡即可提升吞吐 |
| Least Connections LB | 发给活跃连接最少的后端，比 Round-Robin 更均衡 |
| 三链路分流 | 快速链路(无RAG) + 标准链路(RAG) + 安全链路(高风险) |
| Circuit Breaker | 连续失败 > 阈值 → 断路保护，定时半开试探恢复 |
| Token Bucket | 固定速率补充令牌，控制全局 QPS |
| 流式输出 SSE | 逐 token 推送，降低感知延迟，不降低总延迟 |

### 扩展决策树

```
需要扩展推理能力？
├─ 模型能放进单卡？
│   ├─ 是 → 多副本横向扩展（Replica Parallel）
│   │       每卡加一个实例，吞吐线性增长
│   │       延迟不变，高可用
│   └─ 否 → Tensor Parallel 纵向切分
│            先用 TP=2, 不够再加
│            注意：TP 增加可能增加 decode 延迟
│
├─ 并发需求进一步增大？
│   └─ 在现有架构上加更多副本
│       (TP=2 × Replica=2 → TP=2 × Replica=3 ...)
│
└─ 需要异地多活？
    └─ 多节点 + 全局 DNS/API Gateway 负载均衡
```

### 关键数字速记

| 参数 | 值 |
|------|-----|
| 单卡 RTX 5090 vLLM 吞吐 | ~877 tok/s |
| 4 卡 Replica 预期吞吐 | ~3400 tok/s (~97% 扩展效率) |
| 单卡 max_num_seqs | 16 |
| 4 卡 Replica 等效并发 | ~64 |
| Nginx least_conn overhead | < 1ms per request |
| Circuit breaker 默认阈值 | 5 次连续失败 |
| Retry 默认策略 | 3 次，指数退避 base=0.5s, max=10s |
| Token bucket 全局 QPS | 20 r/s（可配） |

### 高频面试问答

**Q: 多卡推理和训练并行有什么不同？**
A: 训练并行（FSDP/ZeRO）因为优化器状态和激活值导致单卡放不下模型，必须分片。推理不需要优化器，瓶颈在 KV Cache 显存限制了并发数。模型能放单卡时优先多副本横向扩展（无通信开销），放不下时才用 Tensor Parallel。

**Q: 为什么不直接用 Tensor Parallel 多用几张卡？**
A: TP 每层都需要 all-reduce 通信。Decode 阶段每次只处理 1 个 token，计算量极小，通信开销占比高。对消费级 PCIe 显卡，TP 可能使 decode 延迟恶化。能放单卡的模型硬上 TP 是负优化。

**Q: 医学场景怎么做请求分流？**
A: 按风险等级三链路：普通问答流式快速通道、RAG 问答独立队列、高风险医疗走安全审核链路。让快请求不被慢请求阻塞，高风险请求不跳过安全检查。

**Q: 流式输出安全吗？**
A: 低风险场景流式没问题（感知延迟低）。高风险医疗（诊断/处方）必须先完整生成并安全审核再输出——不能因为快而牺牲安全。这是设计选择而非技术限制。

**Q: 怎么防止系统过载？**
A: 多级限流（用户级 + IP 级 + 全局令牌桶）+ 请求队列（带超时）+ 熔断器（连续失败断路）+ 指数退避重试。每一层独立工作，层层保护。
