# 34. vLLM 推理部署与性能优化

![vLLM 推理架构图](images/vllm%E6%8E%A8%E7%90%86%E6%9E%B6%E6%9E%84%E5%9B%BE.png)

![PagedAttention 虚拟内存类比图](images/PagedAttention%E8%99%9A%E6%8B%9F%E5%86%85%E5%AD%98%E7%B1%BB%E6%AF%94%E5%9B%BE.png)

> 重要性：本章是"项目落地能力"的核心考察点。面试官会从"你们怎么部署的"切入，深挖你对推理性能的理解。本章覆盖 vLLM 原理、性能指标体系、参数调优、RAG 慢查询排查、压测脚本全链路。

---

## 1. vLLM 是什么？为什么大模型部署都选它？

### Q: vLLM 是什么？为什么比 HuggingFace generate 快？⭐⭐⭐⭐⭐

vLLM 是 UC Berkeley 开源的**大模型推理服务引擎**，核心创新是 **PagedAttention**——把 KV Cache 像操作系统虚拟内存一样分页管理，解决传统框架的显存碎片和利用率低的问题。

**传统推理框架的三大痛点**：
1. **显存预分配浪费**：传统 HF Transformers 为每个请求预分配"最大长度"的连续 KV Cache。比如 `max_length=4096`，即使只生成 50 token 也预留 4096 的空间（内部碎片）。
2. **显存碎片化**：不同请求序列长度参差不齐，预分配造成大量"用不了又释放不掉"的碎片。
3. **相同前缀无法共享**：每个请求的 system prompt 各自分配一份 KV Cache。

**PagedAttention 的解决方案**：把 KV Cache 切成固定大小的 **block**（默认 16 token/block），通过 **block table** 映射逻辑位置到物理位置。不要求物理连续，按需分配。

```
传统 KV Cache:
请求1: [████████████████░░░░░░░░░░░░░░░░]  预分配 4096，浪费大量空间

PagedAttention:
显存池: [B0][B1][B2][B3][B4][B5][B6][B7]...  每个 block=16 tokens
请求1:  B0 → B3 → B5 → B9 → ...        (用多少拿多少)
请求2:  B1 → B7                           (同上)
共享 system prompt: B0 被多请求共享（只存一份）
```

**四大优势**：零碎片（按需分配，显存利用率接近 100%）、Prefix Sharing（相同前缀 block 物理共享）、Copy-on-Write（共享 block 修改时才复制）、更高并发（同等显存下并发数提升 2-4x）。

> **面试金句**："PagedAttention 的灵感来自操作系统的虚拟内存分页。它把 KV Cache 从'连续物理分配'变成'逻辑映射+按需分配'，让显存利用率从传统框架的 20-40% 提升到接近 100%。这是 vLLM 比 HuggingFace 推理快 10-20x 的根本原因。"

**vLLM 的三层加速**：
- **第一层：内存管理**——PagedAttention，显存利用率从 20-40% 提升到接近 100%，同样显存下并发数 2-4x
- **第二层：调度优化**——Continuous Batching，每 step 动态决定 prefill/decode 组成，消灭等待空洞，吞吐再提升 2-3x
- **第三层：工程优化**——CUDA graph 加速、自定义 CUDA kernel、prefix caching、FP8 KV Cache，再提升 30-50%

综合起来，vLLM 的吞吐比 HuggingFace generate 高 10-20 倍。在我们项目中，RTX 5090 单卡跑 Qwen3-8B，vLLM 达到 877 tok/s，而 HuggingFace generate 只有约 30-40 tok/s。

---

## 2. vLLM vs 其他推理框架

### Q: vLLM / TGI / SGLang / TensorRT-LLM 怎么选？⭐⭐⭐⭐

| 维度 | HuggingFace | vLLM | TGI | SGLang | TensorRT-LLM |
|------|-----------|------|-----|--------|--------------|
| **KV Cache** | 连续预分配 | PagedAttention | 类似PagedAttention | RadixAttention | 预分配+reuse |
| **调度策略** | 静态batching | Continuous Batching | Continuous Batching | Continuous Batching | In-flight Batching |
| **前缀缓存** | 无 | Automatic Prefix Caching | 支持 | RadixAttention(前缀树) | 手动配置 |
| **部署方式** | Python脚本 | OpenAI-compatible server | REST+gRPC | 原生HTTP+DSL | Triton Server |
| **安装复杂度** | pip install | pip install | Docker为主 | pip install | 需编译引擎 |
| **性能(8B吞吐)** | ~20-30 tok/s | ~800-900 tok/s | ~600-800 tok/s | ~800-1000 tok/s | ~900-1100 tok/s |
| **适用场景** | 调试/研究 | 通用推理部署 | 生产级API服务 | 复杂prompt编排 | 极致性能优化 |

**为什么本项目选 vLLM**：部署效率优先（改模型路径重启服务，不到2分钟）、硬件约束（RTX 5090消费级显卡，TensorRT-LLM的极致优化需要A100/H100）、面试生态（PagedAttention是必考论文）。

> **面试金句**："框架选择没有绝对的优劣，要看场景。8B 模型单卡部署，vLLM 的易用性和性能已经足够。如果在 H100 集群上部署 70B+ 模型，追求极致吞吐，TensorRT-LLM 的 FP8 和 in-flight batching 会更合适。如果多轮对话场景很多，SGLang 的 RadixAttention 前缀树共享优势明显。"

---

## 3. OpenAI-Compatible API、Serve 模式、Offline Inference 模式

### Q: vLLM 有哪几种使用模式？各适合什么场景？⭐⭐⭐⭐

| 模式 | 命令 | 用途 | 接口 |
|------|------|------|------|
| **Online Serving (serve)** | `vllm serve ./model` | 启动 HTTP 服务器 | OpenAI-compatible REST API |
| **Offline Inference** | `from vllm import LLM` | 本地批量推理 | Python API |

**Serve 模式**：启动 HTTP 服务器，提供与 OpenAI API 完全兼容的接口。提供 `/v1/chat/completions`（对话生成）、`/v1/completions`（文本续写）、`/v1/models`（模型列表）、`/health`（健康检查）等端点。任何用 `openai` 库写的代码都可以**只改 `base_url`** 就无缝切换到本地 vLLM。

**Offline 模式**：直接在 Python 进程中调用，适合批量处理数据——Rejected Generation（DPO训练前批量生成rejected response）、Knowledge Distillation（Teacher批量生成训练数据）、评测批量推理、Embedding离线计算。

**Offline vs Serve 对比**：

| 维度 | Offline Inference | Serve |
|------|------------------|-------|
| 延迟 | 更高（批处理延迟） | 更低（即时响应） |
| 吞吐 | 更高（最大化batching） | 受并发和排队影响 |
| 使用方式 | Python脚本 | HTTP API（任意语言） |
| 本项目场景 | Rejected generation | 医学教学平台API |

---

## 4. Merged Model 部署 vs LoRA Serving

### Q: LoRA Merge 后部署和 LoRA Serving 有什么区别？怎么选？⭐⭐⭐⭐

**策略一：Merged Model（本项目采用）**：把 LoRA adapter 的权重通过 `W_merged = W_base + B × A / α` 融回 base model，得到一个完整的 BF16 模型文件后直接部署。

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model_4bit, "./qlora_checkpoint")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model", safe_serialization=True)
```

**策略二：LoRA Serving**：不 merge，base model 和 LoRA adapter 分开加载，vLLM 通过 `--enable-lora` 参数在推理时动态应用 LoRA adapter。

| 维度 | Merged Model | LoRA Serving |
|------|-------------|-------------|
| **计算开销** | 无需额外计算 | 每个token需 W = W0 + BA 的加法（微小） |
| **显存占用** | 一份完整模型(16GB BF16) | base(16GB) + LoRA adapter(50MB × N) |
| **多LoRA支持** | 不行 | 一个base可同时服务多个LoRA |
| **部署复杂度** | 简单 | 需要配置--enable-lora等参数 |

**为什么选 Merge**：我们的场景下只有一个 LoRA adapter（SFT+DPO 训练产物），没有多 LoRA 服务的需求。Merge 成一个模型更简单，不需要管理 base-adapter 版本对应关系。Merge 后推理路径和原生模型完全一致。

---

## 5. KV Cache、Prefill、Decode、PagedAttention 深度解析

### Q: Prefill 和 Decode 阶段分别做什么？各有什么计算特征？⭐⭐⭐⭐⭐

**自回归生成的本质**：LLM 按 token 逐个生成——每次预测下一个 token，拼到输入后面，再预测下一个。每次预测都要对整个序列做 self-attention。

**Prefill 阶段（预填充）**：处理输入 prompt 的所有 token，一次性送入模型做并行前向传播。并行计算每个 token 的 K/V 存入 KV Cache，拿到最后一个 token 的 hidden state 得到第一个输出 token 的概率分布。**计算密集型（Compute-bound）**：prompt N 个 token 同时做矩阵乘法，耗时与 O(N²) 近似。

**Decode 阶段（逐 token 生成）**：从第一个输出 token 开始，逐 token 自回归生成直到 EOS 或 max_tokens。每步将上一步生成的 1 个 token 送入模型，计算新 token 的 Q/K/V（K/V 追加到 KV Cache），用 Q attend 缓存中的所有历史 K/V。**显存带宽密集型（Memory-bound）**：每步只做 1 个 token 的矩阵乘法，但需要从显存读取整个 KV Cache（数十 GB 级别）。

| 维度 | Prefill | Decode |
|------|---------|--------|
| **输入token数** | N（prompt） | 1（当前生成的token） |
| **计算量** | O(N²) — 大 | O(N) — 小 |
| **瓶颈类型** | Compute-bound | Memory-bound |
| **GPU SM占用** | 高 | 低 |
| **优化方向** | 减少prompt长度、提高FLOPS | 提高显存带宽、减少KV Cache大小 |

> **面试金句**："KV Cache 的本质是用空间换时间——多花显存存储已计算的 K/V，避免每个 decode step 都重新计算整个序列的 attention。没有 KV Cache，生成第 N 个 token 要做 O(N²) 的 attention 计算；有了 KV Cache，只需 O(N)。"

---

## 6. KV Cache 显存估算与关键参数

### Q: 怎么估算 KV Cache 的显存占用？各参数对显存的影响？⭐⭐⭐⭐⭐

**KV Cache 精确估算公式**：
```
KV Cache (per token) = 2 × num_layers × num_kv_heads × head_dim × dtype_size
```

**Qwen3-8B 的 KV Cache 估算（GQA num_kv_heads=8, head_dim=128, num_layers=32）**：
```
单 token KV Cache = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB
```

**本项目完整显存预算**：

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型权重 (BF16) | 16 GB | 8B × 2 bytes |
| KV Cache (16 seq × avg 1500 tokens) | ~3 GB | 16 × 1500 × 128KB |
| CUDA Context + cuBLAS workspace | ~1.5 GB | PyTorch/CUDA 运行时开销 |
| 激活值 (推理) | ~0.5 GB | 远小于训练（无反向传播） |
| 其他 | ~0.5 GB | — |
| **总计** | **~21.5 GB** | 均值 |
| **峰值（长输出）** | **~28.86 GB** | 压测验证值 |

RTX 5090 32GB × 0.90 = 28.8 GB 可用。峰值 28.86 GB 非常接近上限，这就是为什么 `gpu_memory_utilization=0.90` 和 `max_model_len=4096` 是经过仔细考量的。

**各参数对显存的影响**：
- `max_model_len=4096`：单请求最大 block 数 = 4096/16 = 256。设越大→同样显存可服务的并发数越少
- `max_num_seqs=16`：最多同时 16 条 sequence 在内存中，间接控制 KV Cache 总量
- `gpu_memory_utilization=0.90`：32GB × 0.90 - 16GB ≈ 12.8 GB 用于 KV Cache block pool（约 100,000 token 容量）
- RAG 场景冲击：prompt 从 200 膨胀到 2000 token，KV Cache 从 ~25MB/请求 膨胀到 ~250MB/请求，并发能力显著下降

> **面试金句**："vLLM 参数调优的核心思想是'显存预算制'——先算模型权重占多少，再算 KV Cache pool 剩多少，然后根据业务场景分配 max_model_len（单请求上限）和 max_num_seqs（并发数）。这两者是 trade-off。"

---

## 7. RAG Context 太长为什么会拖慢 Prefill

### Q: RAG 接入后为什么推理变慢了？怎么解决？⭐⭐⭐⭐⭐

**原因拆解**：

第一，**prefill 的 O(N²) attention 复杂度**是罪魁祸首。无 RAG 时 prompt 约 200 token，attention 计算量约 40K 元素。接入 RAG 后 prompt 膨胀到 2000 token（system prompt + query + 3 篇检索文档），attention 计算量约 4M 元素——prompt 长度 10x，attention 计算量 100x。TTFT 从 42ms 飙到 300ms+。

第二，**KV Cache 显存压力**。2000 token prompt 单请求 KV Cache 约 250MB（vs 原来 25MB）。12.8GB 的 KV Cache pool 原来能服务数百条短请求，现在只能服务约 50 条 RAG 请求。

第三，**prefix cache 命中率降低**。RAG 检索到的文档各不相同，文档部分无法缓存。

**量化分析**：

| prompt 长度 | prefill 估算耗时 | TTFT (1并发) | TTFT (16并发) |
|------------|-----------------|-------------|---------------|
| 200 (无RAG) | ~20 ms | ~42 ms | ~100 ms |
| 2000 (中RAG) | ~250 ms | ~300 ms | ~900 ms |
| 3000 (重RAG) | ~500 ms | ~560 ms | ~1800 ms |

**解决方案**：
1. 限制检索文档数量和长度：从 Top-10 降到 Top-3，每篇截断到 500 字符
2. 使用文档摘要替代原文：摘要比原文短 5-8x
3. 启用 prefix caching：system prompt 和 instruction 模板部分跨请求共享
4. 分层缓存：RAG 检索层加 query 缓存
5. Prompt 压缩工具：如 LLMLingua 把 prompt 压缩到 50%

优化后 TTFT 从 300ms 降到约 150ms。虽然仍比无 RAG 的 42ms 慢，但在医学教学场景中 150ms 的响应完全可接受，而 RAG 带来的事实准确性提升远超这个延迟代价。

---

## 8. 推理性能指标体系

### Q: TTFT、TPOT、Latency 分别是什么意思？有什么关系？⭐⭐⭐⭐⭐

**延迟类指标**：

| 指标 | 定义 | 用户体感 |
|------|------|---------|
| **TTFT** (Time To First Token) | 从发送请求到收到第一个token的时间 | "响应快不快"——最关键 |
| **TPOT** (Time Per Output Token) | 除首token外每个输出token的平均时间 | "生成流畅不流畅" |
| **Latency** (端到端) | 从发送请求到收到完整响应的总时间 | "等了多久拿到完整回答" |
| **p50/p95/p99 Latency** | 50%/95%/99% 请求延迟低于此值 | 稳定性——p99代表最差1%用户 |

**吞吐类指标**：Throughput（tokens/s）、QPS（每秒请求数）、Prompt tokens/s、Generation tokens/s。

**指标之间的关系**：
```
End-to-End Latency = Queueing + TTFT + (num_output_tokens - 1) × TPOT
TTFT ≈ Queueing + Prefill Latency
TPOT ≈ Decode Per-Step Latency（主要由显存带宽决定）
Prefill Latency ∝ prompt_len²（近似）
Decode Latency ∝ (prompt_len + generated_len)（线性）
```

**本项目数据验证**：

| 并发 | TTFT | Throughput | GPU利用率 |
|------|------|-----------|----------|
| 1 | 42ms | 95 tok/s | ~30% |
| 4 | 55ms | 362 tok/s | ~60% |
| 8 | 71ms | 643 tok/s | ~80% |
| 16 | 100ms | 877 tok/s | ~90% |
| 32 | 280ms | 892 tok/s | ~92% (排队严重) |

16 并发是"甜蜜点"——吞吐达到峰值的 98%，TTFT 在可接受范围。32 并发时吞吐几乎不增但 TTFT 翻了近 3 倍。

**医学场景中 p95/p99 的重要性**：p99=2s 的系统意味着每 100 个医学生中就有 1 个要等 2 秒。目标是 p95 < 300ms, p99 < 500ms。

---

## 9. vLLM 关键参数详解

### Q: vLLM 启动时各关键参数怎么选？⭐⭐⭐⭐

**本项目完整启动命令**：
```bash
export VLLM_USE_FLASHINFER_SAMPLER=0  # Blackwell SM12.x 兼容性修复

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

**核心参数解释**：

| 参数 | 本项目值 | 说明 |
|------|---------|------|
| `--dtype` | bfloat16 | 与BF16训练一致，精度无损 |
| `--max-model-len` | 4096 | 单请求最大token数，控制KV Cache上限 |
| `--max-num-seqs` | 16 | 最大并发序列数（饱和并发点） |
| `--gpu-memory-utilization` | 0.90 | 使用90%显存，留10%缓冲 |
| `--enable-prefix-caching` | 开启 | system prompt的KV Cache复用 |
| `--tensor-parallel-size` | 1（默认） | 8B模型单卡够用，TP反而降速 |

**如何根据显存调整参数**：
```
32GB显卡 + 8B模型(16GB): max_model_len=4096, max_num_seqs=16 → KV Cache pool≈12.8GB
24GB显卡 + 8B模型:       max_model_len=2048, max_num_seqs=8  → 需减半
48GB显卡 + 8B模型:       max_model_len=8192, max_num_seqs=32 → 发挥大显存优势
32GB + INT4量化8B模型:   max_model_len=8192, max_num_seqs=32 → 量化释放的显存提升并发
```

---

## 10. 性能优化策略（全栈清单）

### Q: 推理性能优化的完整手段有哪些？⭐⭐⭐⭐⭐

| 优先级 | 优化手段 | 收益 | 代价 | 本项目是否采用 |
|--------|---------|------|------|--------------|
| P0 | Continuous Batching | 吞吐 +2-4x | 无 | 是（vLLM默认） |
| P0 | 减少 max-model-len | 显存省20-50% | 长文本截断 | 是（4096） |
| P0 | 限制 max_new_tokens | 防OOM | 输出被截断 | 是（API层限制） |
| P1 | Prefix Caching | TTFT -20-30% | 少量显存 | 是 |
| P1 | 流式输出 | 感知延迟 -80% | 总延迟不变 | 是 |
| P1 | 控制RAG context长度 | TTFT -50-70% | RAG召回略降 | 是 |
| P2 | Embedding缓存 | Prefill前延迟-50% | 存储空间 | 可选 |
| P2 | Prompt压缩 | Prefill延迟 -30-50% | 信息损失 | 可选 |
| P3 | 量化部署(INT4) | 显存省75% | 精度略降 | 未采用 |

**降低医疗问答 TTFT 的具体策略**：
- Prompt层：精简system prompt（300→100 token）、控制检索文档量、使用prompt压缩工具
- 调度层：启用prefix caching、降低最大并发数（牺牲吞吐换延迟）
- 部署层：流式输出、warmup预热、多实例+负载均衡、RAG和非RAG请求分流

---

## 11. 代码示例（核心片段）

### vLLM OpenAI-Compatible API 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[
        {"role": "system", "content": "你是一个专业的医学教学助手..."},
        {"role": "user", "content": "请解释II型糖尿病的发病机制和主要治疗策略。"},
    ],
    temperature=0.3,
    max_tokens=512,
)
print(response.choices[0].message.content)
```

### 流式输出 + 指标统计

```python
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

def stream_generate(prompt: str, max_tokens: int = 512):
    t_start = time.time()
    first_token_time = None
    token_count = 0

    stream = client.chat.completions.create(
        model="medical-llm-teacher",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=max_tokens, stream=True,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            if first_token_time is None:
                first_token_time = time.time() - t_start
                print(f"\n[TTFT: {first_token_time*1000:.0f}ms]\n")
            token_count += 1
            print(chunk.choices[0].delta.content, end="", flush=True)

    total_time = time.time() - t_start
    tpot = (total_time - first_token_time) / max(token_count - 1, 1) if token_count > 1 else 0
    print(f"\nTTFT: {first_token_time*1000:.0f}ms, TPOT: {tpot*1000:.0f}ms, Tokens/s: {token_count/total_time:.1f}")
```

---

## 面试1分钟回答

"vLLM 是我们医学 LLM Teacher 项目的推理部署方案。核心优势是 PagedAttention——把 KV Cache 像操作系统虚拟内存一样分页管理，按需分配、物理离散、逻辑连续，显存利用率从传统框架的 20-40% 提升到接近 100%。加上 Continuous Batching 动态混合 prefill/decode 调度，比 HuggingFace generate 快 10-20 倍。我们部署 Qwen3-8B BF16 合并模型在 RTX 5090 单卡上，max-model-len=4096, max-num-seqs=16, gpu_memory_utilization=0.90。极限吞吐 ~877 tok/s，1 并发 TTFT~42ms，16 并发 TTFT~100ms。"

---

## 面试3分钟回答

"我们用 vLLM 部署医学 LLM Teacher 项目，涉及几个关键决策和优化。

**为什么选 vLLM**：PagedAttention 解决了传统 KV Cache 预分配的碎片问题，Continuous Batching 消灭了静态 batching 的等待空洞。

**部署配置**：LoRA Merge 后得到完整 BF16 模型（16GB）。关键参数：max-model-len=4096、max-num-seqs=16、gpu_memory_utilization=0.90。显存预算：权重 16GB + KV Cache pool 12.8GB（约 100K token 容量）。

**性能指标**：极限吞吐 877 tok/s，饱和并发 16。TTFT 从 1 并发的 42ms 涨到 16 并发的 100ms。32 并发时吞吐几乎不涨但 TTFT 暴涨到 280ms——这就是饱和并发。

**RAG 场景优化**：RAG context 从 200 膨胀到 2000 token，prefill 的 O(N²) 复杂度导致 TTFT 从 42ms 飙到 ~300ms。优化方案：限制检索 Top-3、文档截断、启用 prefix caching、query 缓存。优化后 TTFT 回到 ~150ms。

**工程踩坑**：Blackwell SM12.x 上 FlashInfer 采样器 CUDA kernel 不兼容，禁用后性能损失不到 2%。"

---

## 背诵版总结

### 核心概念速记

| 概念 | 一句话 | 对应原理 |
|------|--------|---------|
| PagedAttention | KV Cache 分页管理，按需分配，零碎片 | 类比OS虚拟内存 |
| Continuous Batching | 动态混合 prefill/decode，消灭等待空洞 | 每step重新调度 |
| Prefill | 并行处理prompt，计算密集，O(N²) | 一次性算完所有prompt token |
| Decode | 逐token生成，显存带宽密集，O(N) | 每次只处理1个新token |
| TTFT | 首token延迟 = queueing + prefill | 用户体验的"响应感" |
| TPOT | 每token生成延迟 = 单步decode时间 | 决定"流畅感" |
| saturation concurrency | 吞吐不再随并发增加的点 | 再多并发只加延迟不加吞吐 |

### 显存估算速算公式

```
模型权重: params × dtype_bytes (8B × 2 = 16GB BF16)
KV Cache/token: 2 × layers × kv_heads × head_dim × dtype_bytes = 2 × 32 × 8 × 128 × 2 ≈ 128 KB
KV Cache Pool: GPU_memory × gpu_memory_utilization - 模型权重 - overhead ≈ 32 × 0.90 - 16 - 3 ≈ 12.8 GB
```

### 项目数字速记

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-8B, BF16 合并模型, 16GB |
| GPU | RTX 5090 32GB |
| max-model-len | 4096 |
| max-num-seqs | 16 |
| gpu_memory_utilization | 0.90 |
| 极限吞吐 | ~877 tok/s |
| 1并发 TTFT | ~42ms |
| 16并发 TTFT | ~100ms |
| 显存峰值 | 28.86 GB |
| KV Cache per token | ~128 KB |

### 高频面试问答

**Q: vLLM 为什么比 HF generate 快？** PagedAttention（显存利用率 +4x）+ Continuous Batching（调度优化 +3x）+ CUDA graph/kernel 优化（+30-50%）→ 总吞吐提升 10-20x。

**Q: PagedAttention 的 block 是什么？** 固定大小（16 token）的 KV Cache 存储单元。类比 OS 内存页。通过 block table 做逻辑到物理的映射。

**Q: RAG 接入后为什么变慢？** prompt 长度膨胀（200→2000 token），prefill 的 O(N²) 导致 TTFT 增长 7x。优化：精简检索文档、prompt 压缩、prefix cache。

**Q: TTFT 和 TPOT 有什么区别？** TTFT 是等第一个 token 的时间（prefill + queueing），TPOT 是后续每个 token 的时间（decode）。总延迟 = TTFT + (N-1) × TPOT。

**Q: 怎么压测？** asyncio + httpx 并发发流式请求，统计 TTFT、TPOT、tokens/s、p50/p95/p99。找饱和并发点。本项目饱和并发=16。
