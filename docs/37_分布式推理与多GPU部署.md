# 37. 分布式推理与多GPU部署

> 重要性：面试官问"你们只用了单卡？如果并发再增大怎么办？"——这个问题几乎必问。本章覆盖推理阶段的多GPU扩展路径、负载均衡策略、队列限流设计、流式输出的原理和实践。

---

## 1. 训练并行 vs 推理并行：根本区别

### 核心区别一览

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

> **面试金句**："训练并行解决的是'单卡装不下模型+优化器'的问题，推理并行解决的是'单卡服务不够多用户'的问题。训练的瓶颈是优化器状态和激活值，推理的瓶颈是 KV Cache 显存和并发排队。因此最优策略完全不同——训练用 FSDP/ZeRO-3 做参数分片，推理能用单卡放模型时优先多副本横向扩展。"

### 量化对比

```
场景：8B 模型，BF16（16GB），RTX 5090 32GB

训练 (QLoRA SFT):
  Base Model (NF4):           4 GB
  LoRA Weights:               0.05 GB
  Optimizer States:           0.10 GB
  Activations (bs=2):         ~12 GB
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

### 瓶颈一：Prefill 计算瓶颈

Compute-bound，prompt 的 attention 计算是 O(N²)。长 prompt（RAG 场景）的 prefill 耗时显著增加，TTFT 恶化。Tensor Parallel 可以加速单个 prefill。

### 瓶颈二：Decode 显存带宽瓶颈

Memory-bound，每个 decode step 计算量很小，但需要从显存读取整个 KV Cache。TPOT 主要由显存带宽决定。Tensor Parallel 在 decode 阶段的通信开销反而可能增加 TPOT。

### 瓶颈三：KV Cache 显存瓶颈

KV Cache 占用大量显存（每 token ~128KB），限制了单 GPU 能服务的并发请求数。多 GPU 多副本可以线性扩展并发能力，但 Tensor Parallel 不增加 KV Cache pool。

---

## 3. 部署架构演进：从单卡到多副本

### 级别一：单 GPU 单实例（本项目当前架构）

```
┌─────────────────────────────────┐
│         RTX 5090 (32GB)         │
│     vLLM Instance :8000         │
│  - Model: 16GB, KV Cache: 12.8GB│
│  - max_num_seqs: 16             │
└─────────────────────────────────┘
适用：模型能放进单卡 + 并发需求低（<16 QPS）
```

### 级别四：多 GPU 多副本（Replica Parallel，推荐扩展方案）

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ GPU 0 (32GB) │   │ GPU 1 (32GB) │   │ GPU 2 (32GB) │
│vLLM :8000    │   │vLLM :8001    │   │vLLM :8002    │
│ 独立模型副本  │   │ 独立模型副本  │   │ 独立模型副本  │
│ 独立KV Cache │   │ 独立KV Cache │   │ 独立KV Cache │
└──────────────┘   └──────────────┘   └──────────────┘
                        │
                   Nginx LB :80
                        │
                   客户端请求

适用：模型能放进单卡 + 并发需求高
每加一张 GPU，总吞吐 ≈ 线性扩展
```

> **面试金句**："多 GPU 推理的扩展策略决策树：模型能放进单卡吗？能→多副本横向扩展（线性扩展，无通信开销）。不能→Tensor Parallel 纵向切分。大并发时→在多副本基础上继续加副本。"

---

## 4. 四种推理并行策略深度解析

### Tensor Parallel（张量并行）

将单层 Transformer 的权重矩阵按列或行切分到多 GPU。每层都需要 GPU 间 all-reduce 通信。

- **优点**：降低单卡显存需求；对 prefill 阶段有加速效果
- **缺点**：每层都需通信，对网络带宽要求高；Decode 阶段通信开销占比高，可能增加延迟
- **何时用**：模型太大放不进单卡（如 70B 模型需要 4-8 卡 TP）。**如果模型能放进单卡，加 TP 是负优化。**

### Pipeline Parallel（流水线并行）

将模型按层切分成多个 stage，每个 GPU 负责一部分层。存在 pipeline bubble。适合离线批量推理，不适合在线推理。

### Replica Parallel（副本并行，本项目首选）

每张 GPU 持有一份完整模型副本，各自独立服务请求。负载均衡器分发请求。

- **优点**：零通信开销、线性扩展、高可用、延迟不增加
- **缺点**：每张卡都需要能装下完整模型；KV Cache 不跨副本共享
- **何时用**：模型能放进单卡 + 需要高吞吐/高可用

### 混合策略：TP + Replica

大模型（70B+）通常用 TP + DP 混合：TP=4 切分到 4 卡（解决显存），DP=2 起两组副本（提高并发）。总 GPU = TP × DP。

---

## 5. 负载均衡策略详解

| 策略 | 原理 | 优缺点 |
|------|------|------|
| **Round-Robin** | 依次轮询分发 | 最简单，无状态，不考虑后端负载 |
| **Least Connections** | 发给当前活跃连接数最少的后端 | **推荐**，比 RR 更均衡 |
| **Random** | 随机选择 | 长尾可能不均衡 |
| **IP Hash** | 按客户端 IP 哈希 | session 需要时用 |
| **Weighted** | 按权重分配 | 异构 GPU 集群专用 |

### 医学场景的三链路分流

```
请求分类器 → 普通问答（快速链路，无RAG，低延迟优先）
           → RAG 检索（标准链路，长prompt独立队列）
           → 高风险医疗（安全链路，Safety-RAG，安全优先）
```

> **面试金句**："医学场景不应该所有请求走同一链路。简单概念问答和复杂临床问题的延迟要求和安全要求完全不同。分类路由让快速请求不被慢请求阻塞，高风险请求不跳过安全检查。"

---

## 6. 队列与限流设计

### 请求队列关键设计

- **优先级队列**：高优先级（高风险医疗安全审核）→ 普通 → 低优先级（离线批处理）
- **Semaphore 并发控制**：限制同时执行数对应 vLLM max_num_seqs
- **超时控制**：排队超时返回 503 而非无限等待
- **最大队列长度**：超出则返回 429（保护系统不雪崩）

### 限流设计

多级限流：用户级（滑动窗口）+ IP 级 + 全局令牌桶。超限返回 429 + Retry-After。

### 超时与重试设计

熔断器（Circuit Breaker）：Closed→Open→Half-Open 三态。连续失败 N 次熔断，定时半开探测。指数退避重试（base=0.5s, max=10s, jitter）。

---

## 7. 流式输出（Streaming）

### 流式输出的核心价值：降低感知延迟

总延迟不变（都是 2s），但用户感知的等待从 2s 变成了 TTFT（~50ms）。

### 医疗场景的流式输出考量

| 场景 | 适合流式？ | 理由 |
|------|----------|------|
| 医学概念解释 | **适合** | 感知快，用户体验好 |
| 诊断建议 | **需谨慎** | 用户可能在看到完整结论前做错误判断 |
| 药物信息查询 | **适合** | 但需要安全过滤 |
| 高风险医疗回答 | **不适合** | 应先完整生成→安全审核→再一次性返回 |
| 急救指导 | **不适合** | 生命攸关，保证信息完整准确 |

### 安全审核在流式输出前还是后？

- **方案一**（安全优先）：先完整生成 → 审核 → 再流式输出（适用高风险医疗）
- **方案二**（延迟优先）：边生成边审核边推送（适用低风险场景）
- **方案三**（分类决策，本项目推荐）：请求分类 → 低风险走方案二 → 高风险走方案一

> **面试金句**："流式输出在医学场景是双刃剑。低风险的医学概念解释适合流式（感知延迟低，用户满意），但涉及诊断建议或药物剂量时，必须完整生成并通过安全审核后再输出——不能为了快而牺牲安全。"

### vLLM SSE 流式输出 API 示例

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# 低风险：流式输出
stream = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3, max_tokens=512, stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        yield chunk.choices[0].delta.content
```

---

## 8. 代码合集

### vLLM 单卡启动

```bash
export CUDA_VISIBLE_DEVICES=0
vllm serve ./merged_qwen3_8b_medical \
  --host 0.0.0.0 --port 8000 \
  --served-model-name medical-llm-teacher \
  --dtype bfloat16 --max-model-len 4096 \
  --max-num-seqs 16 --gpu-memory-utilization 0.90 \
  --enable-prefix-caching --trust-remote-code
```

### vLLM Tensor Parallel 多卡启动

```bash
export CUDA_VISIBLE_DEVICES=0,1
vllm serve ./Llama-3.1-70B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 4096 --max-num-seqs 8 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

### Nginx 负载均衡配置（关键部分）

```nginx
upstream vllm_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8002 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8003 max_fails=3 fail_timeout=30s;
    keepalive 64;
}

server {
    listen 80;
    location /v1/chat/completions {
        proxy_pass http://vllm_backend;
        proxy_buffering off;  # 流式输出必须禁用缓冲！
        proxy_read_timeout 120s;
    }
}
```

---

## 面试1分钟回答

"多 GPU 推理部署的核心决策取决于两点：模型能不能放进单卡，以及并发需求多大。

我们的 Qwen3-8B 在 32GB RTX 5090 上单卡够用，所以首选多副本横向扩展。4 张卡各跑一个独立 vLLM 实例，Nginx least-connections 负载均衡，总吞吐接近线性扩展到 ~3400 tok/s，且延迟不增加。

如果模型放不进单卡（如 70B），才考虑 Tensor Parallel 纵向切分。TP 的缺点是每层都需要 GPU 间通信，对 decode 阶段延迟可能有负面影响。

医学场景的关键设计是按风险分流：普通问答走流式快速通道，高风险医疗请求走安全审核链路，RAG 长 prompt 请求单独队列避免阻塞快速请求。"

---

## 面试3分钟回答

"多 GPU 推理的设计思路和训练并行完全不同。训练用 FSDP/ZeRO 分片是因为单卡放不下模型+优化器+激活值，推理的瓶颈在 KV Cache 给的并发上限，而非模型大小。

**架构演进**：从单 GPU 单实例出发。8B 模型 BF16 占 16GB，KV Cache pool 能支撑 16 并发。如果需要更大并发，首选多副本（Replica Parallel）方案——每张 GPU 各跑一个完整模型副本，Nginx least-connections 分发请求。不需要 GPU 间通信，线性扩展，高可用。

**Tensor Parallel 只在必要时用**：当模型放不进单卡时才需要 TP。TP 每层都需要 all-reduce 通信，在 decode 阶段（每次只处理 1 个 token）通信开销占比很高，对 RTX 5090 这种 PCIe 连接的消费卡尤其明显。

**医学场景的三链路分类路由**：普通概念问答走流式快速通道（TTFT ~50ms）、RAG 检索问答走标准链路（长 prompt 独立队列）、高风险医疗请求走安全链路（先完整审核再输出）。快请求不被慢请求阻塞，高风险请求不会绕过安全检查。

**队列和限流方面**：优先级队列 + Semaphore 控制实际并发，令牌桶做全局限流，多级限流。熔断器在连续 5 次失败后自动断路 30s，避免雪崩。

**流式输出的权衡**：流式输出能大幅降低感知延迟，但总延迟不变。对于低风险概念解释，直接用 SSE 流式输出。对于涉及诊断/处方的回答，必须完整生成并审核后再推送——不会为了快而牺牲安全。"

---

## 背诵版总结

### 核心概念速记

| 概念 | 一句话 |
|------|--------|
| 训练并行 vs 推理并行 | 训练瓶颈在优化器+激活值（用FSDP/ZeRO），推理瓶颈在 KV Cache+延迟（用Replica/TP）|
| Replica Parallel | 每卡一个完整模型副本，零通信，线性扩展，模型能放进单卡首选 |
| Tensor Parallel | 权重切分到多卡，每层通信，模型放不进单卡时才用 |
| Pipeline Parallel | 按层切分，有 pipeline bubble，适合离线批量 |
| Least Connections LB | 发给活跃连接最少的后端，比 Round-Robin 更均衡 |
| 三链路分流 | 快速链路(无RAG) + 标准链路(RAG) + 安全链路(高风险) |
| Circuit Breaker | 连续失败 > 阈值 → 断路保护，定时半开试探恢复 |
| 流式输出 SSE | 逐 token 推送，降低感知延迟，不降低总延迟 |

### 扩展决策树

```
需要扩展推理能力？
├─ 模型能放进单卡？
│   ├─ 是 → 多副本横向扩展（Replica Parallel）
│   │       每卡加一个实例，吞吐线性增长，延迟不变
│   └─ 否 → Tensor Parallel 纵向切分
│            注意：TP 可能增加 decode 延迟
├─ 并发需求进一步增大？
│   └─ 在现有架构上加更多副本
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
| Circuit breaker 默认阈值 | 5 次连续失败 |
| Retry 默认策略 | 3 次，指数退避 base=0.5s, max=10s |

### 高频面试问答

**Q: 多卡推理和训练并行有什么不同？**
A: 训练并行（FSDP/ZeRO）因为优化器状态和激活值导致单卡放不下模型，必须分片。推理不需要优化器，瓶颈在 KV Cache 显存限制了并发数。模型能放单卡时优先多副本横向扩展。

**Q: 为什么不直接用 Tensor Parallel 多用几张卡？**
A: TP 每层都需要 all-reduce 通信。Decode 阶段每次只处理 1 个 token，计算量极小，通信开销占比高。对消费级 PCIe 显卡，TP 可能使 decode 延迟恶化。能放单卡的模型硬上 TP 是负优化。

**Q: 医学场景怎么做请求分流？**
A: 按风险等级三链路：普通问答流式快速通道、RAG 问答独立队列、高风险医疗走安全审核链路。

**Q: 流式输出安全吗？**
A: 低风险场景流式没问题（感知延迟低）。高风险医疗（诊断/处方）必须先完整生成并安全审核再输出。

**Q: 怎么防止系统过载？**
A: 多级限流（用户级 + IP 级 + 全局令牌桶）+ 请求队列（带超时）+ 熔断器 + 指数退避重试。
