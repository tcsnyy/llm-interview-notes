# 13. 推理部署与 vLLM

---

## 目录

1. [PagedAttention vs 传统 KV Cache](#q-pagedattention-是什么对比传统-kv-cache)
2. [Continuous Batching、Prefill vs Decode](#q-continuous-batchingprefilldecode-是什么)
3. [核心性能指标](#q-核心性能指标-ttfttpot等怎么理解)
4. [关键配置参数](#q-vllm-的关键配置参数怎么设)
5. [Tensor Parallel](#q-tensor-parallel-是什么本项目为什么不用)
6. [LoRA Merge 部署流程](#q-lora-merge-之后怎么部署)
7. [OpenAI-Compatible API](#q-vllm-的-openai-兼容-api-怎么用)
8. [Chat Template 一致性与 think 泄漏](#q-chat-template-一致性问题和-think-泄漏怎么解决)
9. [推理超参数详解](#q-推理超参数-temperaturetop-ptop-k等怎么选)
10. [推理精度对比](#q-bf16-vs-fp16等推理精度怎么选)
11. [显存估算](#q-vllm-的显存怎么估算)
12. [vLLM vs SGLang vs TensorRT-LLM](#q-vllm-vs-sglang-vs-tensorrt-llm-怎么选)
13. [医学项目部署](#q-医学项目部署怎么讲)
14. [压测结果解释](#q-压测结果怎么解释)
15. [FlashInfer 兼容性](#q-rtx-5090-上-flashinfer-不兼容怎么解决)
16. [TTFT/TPOT 深入](#q-ttfttpot-深入解释)
17. [p95/p99 Latency](#q-p95p99-latency-是什么意思)
18. [Streaming Output / SSE](#q-streaming-输出和-sse-是什么)
19. [Prefix Cache 和 Prompt Cache](#q-prefix-cache-和-prompt-cache-是什么)
20. [多实例部署和负载均衡](#q-多实例部署和负载均衡怎么做)
21. [单卡 vs 多卡速查](#q-单卡部署-vs-多卡部署怎么选)
22. [背诵版总结](#背诵版总结)

---

### Q: PagedAttention 是什么？对比传统 KV Cache

自回归生成时，每个 token 的 attention 计算依赖之前所有 token 的 Key/Value。为了避免重复计算，会把 K/V 缓存下来——这就是 KV Cache。

**传统 KV Cache 的问题：** 传统框架（如 HuggingFace Transformers）为每个序列预分配一块**连续的、最大长度**的 KV Cache 显存。
- **显存碎片化**：每个请求预分配全部空间，即使实际只生成 50 个 token
- **利用率低**：预分配但未使用的空间被浪费，限制并发数
- **无法共享**：不同请求的 prompt 前缀相同（如 system prompt），各自独立分配，无法复用

**PagedAttention 核心思想：** vLLM 把 KV Cache 切成固定大小的 **block**（类似操作系统中的内存页），每个 block 包含固定数量 token 的 K/V。KV Cache 不再要求物理连续，而是通过 **block table** 做逻辑到物理的映射。

```
传统 KV Cache:
[=====请求1预分配 max_len=2048=====][=====请求2预分配=====]...
          ↑ 碎片浪费

PagedAttention:
[Block0][Block1][Block2][Block3][Block4][Block5]...
  请求1 ──→ Block0 → Block3 → Block1     (按需分配，无需连续)
  请求2 ──→ Block2 → Block5               (按需分配)
```

**优势：**
1. **近乎零浪费**：按需分配 block，不存在预分配浪费
2. **显存共享**：相同 prefix 的 block 可被多个请求共享（如 system prompt 只需存一份）
3. **更高吞吐**：同样显存可服务更多并发请求
4. **copy-on-write**：共享 block 在需要修改时才会复制

> **面试金句**："PagedAttention 把 KV Cache 管理类比为操作系统虚拟内存分页，通过 block table 实现逻辑连续、物理离散的 KV Cache 分配，解决了传统预分配方案的内存碎片和利用率低的问题。"

---

### Q: Continuous Batching、Prefill/Decode 是什么？

**Prefill 阶段：** 输入 prompt 的所有 token 一次性送入模型做**并行前向传播**。计算每个 token 的 K/V 并存入 KV Cache，生成第一个输出 token 的概率分布。**计算密集型**：对 GPU 算力（FLOPS）要求极高。Prefill 耗时近似与 prompt 长度的平方成正比（attention 的 O(n^2)）。

**Decode 阶段：** 逐 token 自回归生成。每个 step 只处理 1 个新 token，读取 KV Cache 中的历史 K/V 做 attention。**显存带宽密集型**：大部分时间花在从显存读取 KV Cache 上。Decode 单个 step 耗时近似与已有的 KV Cache 长度成正比（O(n)）。

**Continuous Batching（连续批处理）：** 传统静态 batching 等一个 batch 的所有请求都生成完成才能接收新请求——资源利用率低。Continuous Batching 的核心：在任何时刻，既有请求在做 prefill，也有请求在做 decode，动态混合调度。

```
时间轴:
t=0:  [Req1 prefill] → [Req1 decode] → [Req1 decode] → 完成
t=1:       [Req2 prefill] → [Req2 decode] → 完成
t=2:            [Req3 prefill] → [Req3 decode] → 完成
                              ↑ Req1/Req2/Req3 在同一个 decode step 中被批处理
```

**实现要点：** vLLM scheduler 在每个 step 决定哪些请求加入/退出 batch；新请求到达时立即调度 prefill，已 prefill 完的请求继续 decode；最大化 GPU 利用率，消灭"等待空洞"。

> **面试金句**："Continuous Batching 让 GPU 在任意时刻同时处理 prefill 和 decode 请求，immediate scheduling 消灭了传统静态 batching 的等待时间，显著提升吞吐。"

---

### Q: 核心性能指标 TTFT/TPOT 等怎么理解？

| 指标 | 英文 | 含义 | 用户体感 |
|------|------|------|---------|
| **TTFT** | Time To First Token | 从发送请求到收到第一个 token 的时间 | 响应快不快 |
| **TPOT** | Time Per Output Token | 每个输出 token 的平均生成时间（不含首个） | 生成流不流畅 |
| **ITL** | Inter-Token Latency | 相邻两个 token 之间的延迟 | 打字机效果 |
| **Throughput** | — | 单位时间处理的请求数或生成的 token 数 | 系统容量 |
| **Latency** | — | 单个请求从发起到完成的端到端时间 | 用户体验 |
| **QPS** | Queries Per Second | 每秒处理的请求数 | 服务能力 |

- **TTFT** = prefill 时间 + 调度排队时间
- **TPOT** = decode 每步时间（主要由显存带宽决定）
- 总延迟 = TTFT + 输出 token 数 x TPOT
- 长 prompt 场景：TTFT 是瓶颈（prefill 算力密集）
- 长输出场景：TPOT 是瓶颈（decode 显存带宽密集）

**延迟与吞吐的 Trade-off：** 低并发 TTFT 极低但 GPU 利用率低；高并发 throughput 高但排队时间增长。

> "在医学项目中，我们设置 max_num_seqs=16 作为饱和并发，此时吞吐 ~877 tok/s，TTFT ~100ms。这个点是我们认为延迟和吞吐的最佳平衡——再提高并发虽然吞吐略增，但延迟会显著恶化。"

---

### Q: vLLM 的关键配置参数怎么设？

**max_num_seqs（最大并发序列数）：** vLLM 不设固定的 batch size，而是用 max_num_seqs 动态控制。

```bash
--max-num-seqs 16  # 同一时刻最多 16 条序列在跑
```

**max_model_len：** 模型能处理的最大 token 数（prompt + output）。

```bash
--max-model-len 4096  # 超过此长度的请求会被截断或拒绝
```

该项目设为 4096，因为 Qwen3-8B 原生支持 32K，但在医学 QA 场景中 4096 已足够，设置更小可**显著减少 KV Cache 显存占用**。

**gpu_memory_utilization：** vLLM 允许使用的 GPU 显存比例。

```bash
--gpu-memory-utilization 0.90  # 使用 90% 显存给 KV Cache
```

这个值不宜设满 1.0，因为 KV Cache 动态分配时需要缓冲空间、CUDA context/cuBLAS workspace 也需要显存，设 0.90 可避免 OOM。

---

### Q: Tensor Parallel 是什么？本项目为什么不用？

将模型的权重矩阵**切分到多张 GPU 上**并行计算。常见切分方式：
- **按行切分（Row Parallel）**：注意力头的 Q/K/V/O 投影矩阵按行分片
- **按列切分（Column Parallel）**：FFN 的 up/gate 投影矩阵按列分片

每张 GPU 计算一部分，然后通过 **all-reduce** 或 **all-gather** 通信聚合结果。

**本项目不需要**：Qwen3-8B (16GB) 单卡 RTX 5090 (32GB) 完全够用。如果面试官追问："为什么不用 TP？"——答："单卡能装下模型 + KV Cache，TP 引入 GPU 间通信开销反而降低吞吐。"

---

### Q: LoRA Merge 之后怎么部署？

**为什么要 merge：** QLoRA 训练完成后，base model 是 4-bit 量化权重 + 16-bit LoRA adapter。部署时两种选择：
1. **不 merge**：base weight + LoRA adapter 分开加载，推理时 W = W0 + BA，多一步加法
2. **Merge**：把 LoRA 权重 merge 回 base model，得到一个完整的 16-bit 模型（本项目选择的方式）

**Merge 步骤：**

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

# 1. 加载 4-bit base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B", load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
)
# 2. 加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, "./qlora_checkpoint/checkpoint-1000")
# 3. Merge
model = model.merge_and_unload()  # LoRA 权重融入 base weight，转为 FP16/BF16
# 4. 保存合并后的模型
model.save_pretrained("./merged_qwen3_8b_medical", safe_serialization=True)
```

**vLLM 启动命令：**

```bash
vllm serve ./merged_qwen3_8b_medical \
  --host 0.0.0.0 --port 8000 --dtype float16 \
  --max-model-len 4096 --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 --enable-prefix-caching \
  --chat-template ./chat_template.jinja --served-model-name medical-llm-teacher
```

---

### Q: vLLM 的 OpenAI 兼容 API 怎么用？

vLLM 默认启动后提供与 OpenAI API 兼容的接口：

**Chat Completions：**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[
        {"role": "system", "content": "你是一个专业的医学教学助手。"},
        {"role": "user", "content": "请解释高血压的病理机制。"},
    ],
    temperature=0.7, max_tokens=512, top_p=0.9,
    stop=["</s>", "<|im_end|>"],
    extra_body={"repetition_penalty": 1.1, "top_k": 50},
)
print(response.choices[0].message.content)
```

**Streaming：**

```python
stream = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[{"role": "user", "content": "解释糖尿病的分型"}],
    stream=True, max_tokens=512,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

**兼容的 endpoint 列表：** Chat Completions (`/v1/chat/completions`)、Completions (`/v1/completions`)、Models (`/v1/models`)、Tokenize (`/tokenize`)、Detokenize (`/detokenize`)。

---

### Q: Chat Template 一致性问题和 think 泄漏怎么解决？

**问题场景：** Qwen3 系列模型原生支持 `enable_thinking` 参数。训练时使用 `enable_thinking=False`，但 vLLM 默认模板不会传递这个参数——导致推理时可能生成 `<think>...</think>` 标签，破坏输出格式。

**解决方案：自定义 chat_template.jinja：**

```jinja
{# Qwen3 Medical Teacher Chat Template #}
{%- if tools %}
    {{- '<|im_start|>system\n' }}
{%- else %}
    {{- '<|im_start|>system\n' }}
    {{- messages[0]['content'] }}
    {{- '\n\n请以专业医学教师的身份回答。回答要求：\n' }}
    {{- '1. 不要使用<think>标签\n' }}
    {{- '2. 不要展示思考过程\n' }}
    {{- '3. 直接给出专业回答\n' }}
    {{- '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages[1:] %}
    {%- if message['role'] == 'user' %}
        {{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|im_start|>assistant\n' }}
        {# 强制 enable_thinking=False #}
        {{- '<|im_start|>think\n\n<|im_end|>\n' }}
        {{- message['content'] + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {{- '<|im_start|>think\n\n<|im_end|>\n' }}
{%- endif %}
```

**关键点**：模板中显式插入空的 `<|im_start|>think\n\n<|im_end|>` ——这样 Qwen3 模型推理时就不会再生成 think 标签内容。

> **面试金句**："训练和推理的 chat template 必须一致。如果训练时 disable_thinking=False 而推理时模板没有对应的 think 标签占位，模型行为会漂移。我们在模板中显式插入空 think 块，保证推理时不会泄漏 `<think>` 标签。"

---

### Q: 推理超参数 Temperature/Top-p/Top-k 等怎么选？

**Temperature（温度）**：控制输出的**随机性**。对 logits 除以 temperature 再做 softmax。

```python
temperature=0.0   # 贪婪解码，完全确定（医学场景常用）
temperature=0.7   # 中等随机性（创意生成）
temperature=1.0   # 无调整（使用原始概率分布）
temperature=1.5+  # 输出趋向随机/混乱
```

医学场景推荐 `temperature=0.1~0.3`，保证输出稳定性，减少幻觉。

**Top-p（Nucleus Sampling）**：只从累积概率超过 p 的最小 token 集合中采样。

```python
top_p=0.9  # 只考虑累积概率 >= 90% 的 tokens
```

**Top-k**：只从概率最高的 k 个 token 中采样。

```python
top_k=50   # 只考虑 Top-50 tokens
```

**Repetition Penalty**：惩罚已出现过的 token，减少重复生成。推荐 1.1。

**Stop Tokens**：遇到指定字符串时立即停止生成。常见：EOS token、对话轮次分隔符、用户角色标识。

---

### Q: BF16 vs FP16 等推理精度怎么选？

| 精度 | 每参数字节 | 8B 模型大小 | 推理速度 | 精度损失 | 适用场景 |
|------|-----------|------------|---------|---------|---------|
| **FP32** | 4B | 32GB | 最慢 | 无 | 训练（不用于推理） |
| **FP16** | 2B | 16GB | 较快 | 可忽略 | 通用推理 |
| **BF16** | 2B | 16GB | 同 FP16 | 可忽略 | 更稳定（本项目使用） |
| **INT8** | 1B | 8GB | 快 | 轻微 | 显存受限 |
| **INT4** | 0.5B | 4GB | 最快 | 有损 | 边缘部署 |
| **NF4** | 0.5B | 4GB | 需反量化 | 有损 | QLoRA 训练 |

**BF16 vs FP16：**
- BF16（Brain Float 16）：指数位 8bit（同 FP32），尾数位 7bit
- FP16：指数位 5bit，尾数位 10bit
- BF16 动态范围更大（同 FP32），**不易溢出**，训练/推理更稳定
- **本项目选择 BF16** 因为 Qwen3 原始训练精度就是 BF16，保持一致性

---

### Q: vLLM 的显存怎么估算？

**模型权重：** 8B 参数 x 2 bytes/BF16 = 16 GB

**KV Cache（PagedAttention 下）：**

```
KV Cache ≈ 2 × num_layers × num_kv_heads × head_dim × (prompt_len + output_len) × dtype_size

示例 (Qwen3-8B, 假设 GQA num_kv_heads=8, head_dim=128):
单 token KV Cache = 2 × 32 layers × 8 heads × 128 × 2 bytes = 128 KB
4096 token 上下文 = 128 KB × 4096 = 512 MB（单序列）
16 条并发       = 512 MB × 16 = 8 GB
```

**本项目显存预算：**

| 项目 | 估算大小 |
|------|---------|
| 模型权重 (BF16) | 16 GB |
| KV Cache (16 seq × 4096) | ~8 GB |
| CUDA 开销 + 激活 | ~2 GB |
| **总计** | **~26 GB** |
| RTX 5090 可用 (32GB × 0.90) | 28.8 GB |
| 余量 | ~2.8 GB |

压测验证：长输出压测（max_tokens=1536）时显存峰值 28.86 GB，与估算吻合。

---

### Q: vLLM vs SGLang vs TensorRT-LLM 怎么选？

| 特性 | vLLM | SGLang | TensorRT-LLM |
|------|------|--------|--------------|
| **KV Cache 管理** | PagedAttention | RadixAttention (前缀树共享) | 预分配 + reuse |
| **调度** | Continuous Batching | Continuous Batching + 结构化生成引导 | In-flight Batching |
| **部署复杂度** | 低（pip install 即用） | 低 | 高（需编译引擎） |
| **FP8/量化** | 支持 FP8/W8A8 | FP8 | 深度优化（FP8/INT8/INT4） |
| **前缀缓存** | automatic prefix caching | RadixAttention（天然共享前缀树） | 手动配置 |
| **生态** | OpenAI-compatible API | 自己的一套 DSL (SGLang) | Triton Server |

**为什么本项目选 vLLM：**
1. **部署最快**：不需要编译、不需要复杂配置
2. **生态完善**：中文社区资料多，PagedAttention 论文热度高，面试加分
3. **OpenAI 兼容**：前端直接对接，无需改代码
4. **RTX 5090 消费卡**：TensorRT-LLM 某些优化需要企业卡（A100/H100），消费卡收益有限
5. **快速迭代**：SFT/DPO 训练完 -> merge -> vLLM 启动，流程只需几分钟

> **面试金句**："我们选 vLLM 是因为它部署成本最低、生态最完善。对于 8B 级别的模型在单张 RTX 5090 上，PagedAttention 的内存节省和 continuous batching 的调度优化已经能带来非常好的吞吐表现。"

---

### Q: 医学项目部署怎么讲？

> **面试官**："你们医学 LLM Teacher 项目是怎么部署的？"

**回答框架：**

"我们的部署流程分为四步：

**第一步：QLoRA 训练产出 checkpoint。** 在 RTX 5090 上完成 QLoRA SFT 和 DPO 训练，得到一个 4-bit base model 加 LoRA adapter 的 checkpoint。

**第二步：LoRA Merge。** 使用 PEFT 的 `merge_and_unload()` 将 LoRA 权重融合回 base model，得到一个完整的 BF16 模型，大小约 16GB。

**第三步：vLLM 启动服务。** 配置上我们设置了 `max-model-len=4096` 控制 KV Cache 大小，`max-num-seqs=16` 控制最大并发，`gpu-memory-utilization=0.90` 使用 90% 显存。

**第四步：前端对接。** vLLM 提供 OpenAI-compatible API，医疗教学平台前端直接通过 `/v1/chat/completions` 接口调用。

部署中遇到的一个关键问题是 chat template 一致性和 FlashInfer 兼容性——我们在 RTX 5090 的 Blackwell SM 12.x 架构上遇到了 FlashInfer 采样器崩溃，通过 `VLLM_USE_FLASHINFER_SAMPLER=0` 解决。另外我们修改了 chat template，确保推理时不会泄漏 `<think>` 标签。"

---

### Q: 压测结果怎么解释？

```
| 并发数 | TTFT(ms) | TPOT(ms) | Throughput(tok/s) | 显存峰值(GB) |
|--------|----------|----------|-------------------|-------------|
| 1      | 42       | 18       | 95                | 18.2        |
| 2      | 48       | 18       | 188               | 18.9        |
| 4      | 55       | 19       | 362               | 20.5        |
| 8      | 71       | 21       | 643               | 23.8        |
| 16     | 100      | 24       | 877               | 27.1        |
| 32     | 280      | 35       | 892               | 28.5 (OOM风险)|
```

**讲解要点：**

1. **饱和并发 16**："并发从 16 提升到 32 时吞吐几乎不涨（877→892），但 TTFT 从 100ms 暴涨到 280ms。这说明 GPU 在 16 并发时已饱和。"

2. **TTFT 增长原因**："并发增加 → 更多请求排队等 prefill → TTFT 上升。但 TPOT 变化不大说明 decode 阶段主要受显存带宽限制。"

3. **极限吞吐 877 tok/s**："8B 模型在消费级显卡 RTX 5090 上达到近 900 tok/s 的吞吐，医学教学场景下一个医学生提问大概只需 200-500 个 token 的回答，响应时间不到 1 秒。"

4. **长输出压测 (max_tokens=1536)**："显存峰值达到 28.86GB，接近 RTX 5090 的 32GB 上限。我们设置 max-model-len=4096 就是为了在长输出场景下留足安全余量。"

> **面试金句**："吞吐和延迟本质上是 trade-off。我们的目标是找到延迟可接受前提下的最大吞吐点。对于医学教学场景，100ms 的 TTFT 已经远低于人类感知阈值（~300ms），所以 16 并发是最优配置。"

---

### Q: RTX 5090 上 FlashInfer 不兼容怎么解决？

**问题现象：** RTX 5090 基于 Blackwell 架构（SM 12.x），vLLM 默认启用的 FlashInfer 采样器在 Blackwell 上 CUDA kernel 不兼容，导致 `RuntimeError: FlashInfer sampler kernel launch failed`。

**根本原因：** FlashInfer 的 CUDA kernel 预编译时针对 Ada Lovelace (SM 8.9) 和 Hopper (SM 9.0) 架构，Blackwell (SM 12.x) 的 PTX 指令集变化导致某些 warp-level 操作行为不同。

**解决方案：**

```bash
# 方案1: 环境变量禁用（本项目采用）
export VLLM_USE_FLASHINFER_SAMPLER=0

# 方案2: 启动时指定
vllm serve ./model --disable-flashinfer-sampler
```

对性能的影响：禁用 FlashInfer 采样器后，采样回退到 PyTorch 原生实现。单步采样从 ~0.1ms 增加到 ~0.3ms，但在整个 decode 延迟（~20ms/step）中占比极小，**吞吐影响 < 2%**。

> **面试金句**："遇到 Blackwell 兼容性问题时，我们的决策是优先稳定性——禁用 FlashInfer 采样器性能损失不到 2%，但避免了随机崩溃。等社区正式支持 Blackwell 后再启用。"

---

### Q: TTFT/TPOT 深入解释

**TTFT (Time To First Token)：** 从客户端发送请求到收到第一个 token 的时间。

```
TTFT = 网络传输 + 排队等待 + Prefill 计算 + 首次 Decode
```

TTFT 增长的主因是排队时间而非 prefill 本身。从 1 并发 42ms 到 16 并发 100ms，prefill 时间基本不变（同样的 prompt），增加的都是排队时间。

**TPOT (Time Per Output Token)：** 相邻两个输出 token 之间的平均延迟。

```
TPOT = Decode 单步耗时 = attention(读 KV Cache) + FFN + sampling
```

TPOT 不随并发数显著变化（因为每步 decode 的工作量固定）。16 并发 TPOT ~24ms，1 并发 ~18ms，差距来自 GPU 内部资源竞争。

> "TTFT 反映系统响应速度，受排队和 prefill 长度影响大。TPOT 反映生成流畅度，主要由显存带宽决定。我们项目的 42ms TTFT 和 18ms TPOT 意味着用户感知是'秒回+流畅输出'。"

---

### Q: p95/p99 Latency 是什么意思？

**为什么不能只看平均值：**

```
假设 100 个请求的 TTFT（单位 ms）：
95 个: 40-60ms, 4 个: 200-300ms, 1 个: 2000ms

平均 TTFT = 77.5ms  ← 看起来还行
p95 TTFT = 250ms   ← 5% 的用户等了超过 250ms
p99 TTFT = 2000ms  ← 1% 的用户等了 2 秒
```

**p95 latency**：95% 的请求延迟低于此值，反映"大多数用户"的体验。
**p99 latency**：99% 的请求延迟低于此值，反映"长尾用户体验"，是 SLA 的常见指标。

> "在 16 并发下，平均 TTFT ~100ms，p95 ~180ms，p99 ~350ms。p95 在 200ms 以内说明绝大多数用户体验良好，p99 的上升主要来自极端排队和偶发的 CUDA kernel 编译。在医学教学场景，即使 p99 的 350ms 也在可接受范围内。"

---

### Q: Streaming 输出和 SSE 是什么？

**SSE (Server-Sent Events)：** 一种服务器向客户端推送事件的 HTTP 协议。

```
HTTP Response:
Content-Type: text/event-stream

data: {"token": "高"}
data: {"token": "血"}
data: {"token": "压"}
data: [DONE]
```

**Streaming vs Non-streaming：**

```
Non-streaming: 用户发送请求 → 等待(t=0到t=2s) → 一次性收到完整回答
用户感知：等了2秒什么也没看到

Streaming: 用户发送请求 → 等42ms(TTFT) → 开始看到token逐字流出
用户感知：立刻开始响应，内容在"打字机效果"中逐步显示
```

**感知延迟 = TTFT**（而非总时间），这让 42ms TTFT 的用户体验极佳。

> "Streaming 输出的核心价值是降低**感知延迟**。虽然总生成时间不变，但用户看到第一个词的延迟从'全部生成完'变成了 TTFT（42ms）。在医学教学场景，我们使用 SSE 协议的 streaming 接口，让学生获得'AI 在实时思考回答'的自然交互体验。"

---

### Q: Prefix Cache 和 Prompt Cache 是什么？

**Prefix Cache（vLLM 的 Automatic Prefix Caching）：** 多个请求共享相同的 prompt 前缀时，prefix 的 KV Cache 只计算一次。

```
请求1: [System Prompt + User Query1]
请求2: [System Prompt + User Query2]
                    ↑ 相同的 System Prompt 部分

无 Prefix Cache: System Prompt 的 KV Cache 各算一遍 → 2× prefill
有 Prefix Cache: System Prompt 的 KV Cache 算一遍 → 复用给请求2
```

```bash
vllm serve ./model --enable-prefix-caching
```

适用场景：长 system prompt 不变、多轮对话、RAG 场景（相同的检索结果前缀可复用）。

**Prompt Cache（更广义的概念）：**
- 层级1: KV Cache 复用（vLLM 的 prefix caching）
- 层级2: Embedding Cache（相同请求文本 → 直接用缓存的 embedding）
- 层级3: 回答 Cache（相同语义 → 直接返回缓存回答）

> "Prefix caching 是 vLLM 的一个重要优化。在医学教学场景，长 system prompt 在多轮对话中复用，prefix caching 能节省 30-50% 的 prefill 时间。"

---

### Q: 多实例部署和负载均衡怎么做？

**为什么需要多实例：** 单卡 RTX 5090 + Qwen3-8B 的极限吞吐约 877 tok/s。如果日活用户 1000+、峰值 QPS 50+，单卡不够。

**多实例架构：**

```
                    ┌──────────┐
                    │  Nginx /  │  ← 反向代理 + 负载均衡
                    │  Envoy   │
                    └────┬─────┘
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ vLLM #1 │  │ vLLM #2 │  │ vLLM #3 │  ← 3 张 RTX 5090
      └─────────┘  └─────────┘  └─────────┘
```

**负载均衡策略：** 轮询（Round Robin）、最少连接（Least Connections）、一致性哈希（相同用户路由到同一实例，利用 prefix cache）。

**多实例 vs 张量并行：**

| 维度 | 多实例（多副本） | 张量并行（TP） |
|------|---------------|--------------|
| **适用条件** | 单卡能放下模型 | 单卡放不下模型 |
| **吞吐** | 近线性扩展（N副本 ≈ N×吞吐） | 次线性扩展（通信开销） |
| **延迟** | 不变 | 可能略有增加 |
| **硬件** | 多张独立 GPU | 多张 GPU 需高速互联（NVLink） |

> "对于 8B 级别的模型，多实例部署比张量并行更高效。单卡就能装下模型，多实例实现近线性扩展，没有张量并行的通信开销。"

---

### Q: 单卡部署 vs 多卡部署怎么选？

| 场景 | 方案 | 配置 | 预期吞吐 | 延迟 |
|------|------|------|---------|------|
| < 10B 模型，低并发 | 单卡 vLLM | max-num-seqs=16 | ~900 tok/s | TTFT ~100ms |
| < 10B 模型，高并发 | 多实例 (N 卡) | N个vLLM + LB | ~900×N tok/s | TTFT ~100ms |
| 10-30B 模型 | TP=2 | 2卡张量并行 | ~600 tok/s | TTFT ~150ms |
| 30-70B 模型 | TP=4 或 TP=8 | 4-8卡张量并行 | ~300 tok/s | TTFT ~200ms+ |
| 70B+ 模型 | TP + PP | 跨节点流水线并行 | — | — |

**面试速记**："小于 13B 多副本，大于 13B 用 TP，大于 70B 考虑 PP。"

---

## 背诵版总结

### 核心概念一句话

- **PagedAttention**：把 KV Cache 切块按需分配，类比 OS 虚拟内存，解决碎片问题。
- **Continuous Batching**：动态混合 prefill/decode，消灭静态 batching 等待空洞。
- **TTFT**：首 token 延迟，用户体验的"响应感"。
- **TPOT**：每 token 生成延迟，影响"流畅感"。
- **LoRA Merge**：把 LoRA adapter 权重融回 base model 再部署。
- **Chat Template 一致性**：训练推理模板必须一致，否则模型行为漂移。
- **max-model-len**：控制 KV Cache 上限，越小越省显存。
- **gpu-memory-utilization**：vLLM 的显存使用比例，设 0.90 留余量防 OOM。
- **Speculative Decoding**：用小模型（draft）快速生成候选 token，大模型一次验证多个。无损加速 2-3x——输出分布完全相同。draft 命中率是关键。

### 代码速查

```bash
# vLLM 启动
export VLLM_USE_FLASHINFER_SAMPLER=0
vllm serve ./merged_model --dtype float16 --max-model-len 4096 \
  --max-num-seqs 16 --gpu-memory-utilization 0.90 \
  --chat-template ./chat_template.jinja

# OpenAI API 调用
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="medical-teacher", messages=[{"role":"user","content":"..."}],
    temperature=0.1, max_tokens=512)
```

### 项目数字速记

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-8B, BF16, 16GB |
| GPU | RTX 5090 32GB |
| max-model-len | 4096 |
| max-num-seqs | 16 |
| 极限吞吐 | ~877 tok/s |
| 1并发TTFT | ~42ms |
| 16并发TTFT | ~100ms |
| 显存峰值 | 28.86 GB |
| FlashInfer | Blackwell SM12.x 不兼容，已禁用 |

### 高频面试问答

**Q: PagedAttention 和传统 KV Cache 区别？**
A: 传统预分配连续显存，碎片严重；PagedAttention 按 block 动态分配，支持共享和零碎片。

**Q: 为什么不设更大的并发？**
A: 16 并发已饱和，再增加吞吐不涨但延迟恶化。100ms TTFT 对医学教学场景已足够。

**Q: 训练用什么精度，部署用什么精度？**
A: 训练用 NF4 (QLoRA)，推理用 BF16。Merge 后回到 BF16 保证精度无损。

**Q: 如果开源模型不支持你的 chat template 怎么办？**
A: 用 --chat-template 参数指定自定义 Jinja2 模板，确保训练推理一致。
