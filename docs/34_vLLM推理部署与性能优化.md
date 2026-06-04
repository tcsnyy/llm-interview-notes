# 34. vLLM 推理部署与性能优化（压测、指标、调参、优化全解）

> 重要性：本章是"项目落地能力"的核心考察点。面试官会从"你们怎么部署的"切入，深挖你对推理性能的理解。本章覆盖 vLLM 原理、性能指标体系、参数调优、RAG 慢查询排查、压测脚本全链路。
>
> 关联文件：`13_推理部署与vLLM.md`（基础版）、`37_分布式推理与多GPU部署.md`（多GPU/多实例扩展）、`12_RAG与SafetyRAG详细八股.md`（RAG 检索链路）。

---

## 1. vLLM 是什么？为什么大模型部署都选它？

### 1.1 一句话定义

vLLM 是 UC Berkeley 开源的 **大模型推理服务引擎**，核心创新是 **PagedAttention**——把 KV Cache 像操作系统虚拟内存一样分页管理，解决传统框架的显存碎片和利用率低的问题。

### 1.2 传统推理框架的三大痛点

**痛点一：显存预分配的浪费**

传统 HuggingFace Transformers 在推理时为每个请求**预分配**一块"最大长度"的连续 KV Cache 显存。比如设置 `max_length=4096`，即使实际只生成 50 个 token，也会预留 4096 个 token 的 KV Cache 空间。这批预分配但不使用的空间就是**内部碎片**——就像餐厅给每个客人预留一个 20 人大桌，哪怕只来一个人。

```
预分配显存 = 2 × num_layers × num_kv_heads × head_dim × max_seq_len × dtype_size

Qwen3-8B 单请求预分配 (max_len=4096):
= 2 × 32 × 8 × 128 × 4096 × 2 bytes
≈ 512 MB（即使只生成 50 token 也占这么多）
```

**痛点二：显存碎片化导致无法服务更多并发**

不同请求的序列长度参差不齐，预分配造成大量"用不了又释放不掉"的碎片空间。实际可用并发数远低于理论值。

**痛点三：相同前缀无法共享**

每个请求的 system prompt 内容完全相同（如"你是一个专业的医学助手..."），但传统框架为每个请求各自分配一份 KV Cache。system prompt 越长，浪费越严重。

### 1.3 vLLM 的解决方案：PagedAttention

PagedAttention 把 KV Cache 切成固定大小的 **block**（比如每个 block 存 16 个 token 的 K/V），block 之间通过 **block table** 映射逻辑位置到物理位置。一个请求的 KV Cache 不再要求物理连续，按需分配 block 即可。

```
传统 KV Cache:
请求1: [████████████████░░░░░░░░░░░░░░░░]  预分配 4096，实际用 2000，浪费 2096
请求2: [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  预分配 4096，实际用 500，浪费 3596
请求3: [████████████████████░░░░░░░░░░░░]  预分配 4096，实际用 2500，浪费 1596
                                                            ↑ 碎片总和 = 7288 token 空间

PagedAttention:
显存池: [B0][B1][B2][B3][B4][B5][B6][B7][B8][B9][B10][B11]...  每个 block=16 tokens
请求1:  B0 → B3 → B5 → B9 → ...        (用多少拿多少)
请求2:  B1 → B7                           (同上)
请求3:  B2 → B4 → B6 → B8 → B10 → ...    (同上)
共享 system prompt: B0 被请求1和请求3 共享（只存一份）
                                            ↑ 近乎零浪费
```

**PagedAttention 的四大优势**：

1. **零碎片**：按需分配 block，不存在预分配浪费，显存利用率接近 100%。
2. **Prefix Sharing**：相同前缀的 block 可被多个请求**物理共享**（如 system prompt 全服务只存一份）。
3. **Copy-on-Write**：共享的 block 只有在被某个请求修改时才复制，类似 fork 进程的 COW 语义。
4. **更高并发**：同等显存下可服务的并发请求数提升 2-4x。

> **面试金句**："PagedAttention 的灵感来自操作系统的虚拟内存分页。它把 KV Cache 从'连续物理分配'变成'逻辑映射+按需分配'，让显存利用率从传统框架的 20-40% 提升到接近 100%。这是 vLLM 比 HuggingFace 推理快 10-20x 的根本原因。"

### 1.4 PagedAttention 的 block size 选择

block size 是一个关键超参数。vLLM 默认 block_size=16（每个 block 存 16 个 token 的 K/V）：

| block_size | 优点 | 缺点 |
|-----------|------|------|
| 小 (8) | 碎片更少，内存利用率更高 | block table 更大，查表开销增加 |
| 大 (32) | block table 小，查表快 | 碎片稍多（block 内未用完的空间） |
| 默认 (16) | 工程上验证的最佳平衡点 | — |

**本项目使用默认 block_size=16**，在 RTX 5090 上表现良好。

---

## 2. vLLM vs Transformers Generate vs TGI vs SGLang vs TensorRT-LLM

### 2.1 横向对比表

| 维度 | HuggingFace Transformers | vLLM | TGI (Text Generation Inference) | SGLang | TensorRT-LLM |
|------|-------------------------|------|--------------------------------|--------|--------------|
| **开发者** | HuggingFace | UC Berkeley | HuggingFace | Stanford/UC Berkeley | NVIDIA |
| **KV Cache** | 连续预分配 | PagedAttention | PagedAttention (类似) | RadixAttention | 预分配+reuse |
| **调度策略** | 静态 batching | Continuous Batching | Continuous Batching | Continuous Batching + 结构化引导 | In-flight Batching |
| **前缀缓存** | 无 | Automatic Prefix Caching | 支持 | RadixAttention (前缀树天然共享) | 手动配置 |
| **部署方式** | Python 脚本 | OpenAI-compatible server | REST API + gRPC | 原生 HTTP + 自定义 DSL | Triton Inference Server |
| **量化支持** | bitsandbytes (4/8 bit) | GPTQ, AWQ, FP8, W8A8 | GPTQ, AWQ, bitsandbytes | FP8, GPTQ, AWQ | FP8, INT8, INT4 (极致) |
| **安装复杂度** | `pip install transformers` | `pip install vllm` | Docker 为主 | `pip install sglang` | 需编译引擎 + 模型转换 |
| **性能 (8B 模型吞吐)** | ~20-30 tok/s | ~800-900 tok/s | ~600-800 tok/s | ~800-1000 tok/s | ~900-1100 tok/s |
| **生态成熟度** | 最成熟 | 成熟，社区活跃 | 成熟 | 快速成长 | 企业级但上手门槛高 |
| **适用场景** | 调试/研究 | 通用推理部署 | 生产级 API 服务 | 复杂 prompt 编排 | 极致性能优化 |

### 2.2 各框架适用场景详解

**HuggingFace Transformers `.generate()`**：
- 适用：单条推理调试、模型效果验证、研究实验。
- 不适用：任何需要并发/高吞吐的场景。
- 核心问题：每次 `generate()` 都要重新做 prefill，没有 KV Cache 持久化，没有 batching。

**vLLM**：
- 适用：通用 LLM 部署，快速上线，OpenAI-compatible API。
- 优势：pip install 即用，中文社区资料多，PagedAttention 论文热度高（面试高频考点）。
- 本项目选 vLLM 的原因：部署成本最低、生态最完善、8B 模型单卡 RTX 5090 足够。

**TGI (Text Generation Inference)**：
- 适用：需要完整生产级特性（watermark、safety logits processor、guidance）。
- 优势：HuggingFace 官方支持，与 HF Hub 深度集成。
- 劣势：Docker 部署为主，调试灵活性不如 vLLM 纯 Python 方案。

**SGLang**：
- 适用：复杂 prompt 编排场景（多轮分支、并行生成、约束解码）。
- 优势：RadixAttention 的前缀树共享比 PagedAttention 的 prefix caching 更高效（对多轮对话场景）。
- 劣势：生态较新，部分高级功能尚不稳定。

**TensorRT-LLM**：
- 适用：大规模 GPU 集群、追求极致吞吐的企业级部署。
- 优势：NVIDIA 官方深度优化，FP8/INT8/INT4 推理性能最佳。
- 劣势：需要编译模型引擎（几分钟到几十分钟），调试困难，消费级显卡收益有限。

> **面试金句**："框架选择没有绝对的优劣，要看场景。8B 模型单卡部署，vLLM 的易用性和性能已经足够。如果在 H100 集群上部署 70B+ 模型，追求极致吞吐，TensorRT-LLM 的 FP8 和 in-flight batching 会更合适。如果多轮对话场景很多（如客服系统），SGLang 的 RadixAttention 前缀树共享优势明显。"

### 2.3 为什么本项目选 vLLM（面试回答模板）

"我们的选择基于三个权衡：

**第一，部署效率优先**。医学 LLM Teacher 项目需要快速迭代——训练完新 checkpoint 到部署上线，vLLM 只需修改模型路径重启服务，全程不到 2 分钟。TensorRT-LLM 每次都要重新编译引擎，迭代周期太长。

**第二，硬件约束**。RTX 5090 是消费级显卡，TensorRT-LLM 的很多极致优化需要企业级 GPU（A100/H100）才能发挥。在 RTX 5090 上 vLLM 和 TensorRT-LLM 的性能差距不大（5-10%），不值得为这点收益付出部署复杂度代价。

**第三，面试生态**。PagedAttention 是 LLM 推理方向的必考论文，选 vLLM 意味着我对推理部署的理解从论文到工程是打通的。"

---

## 3. OpenAI-Compatible API、Serve 模式、Offline Inference 模式

### 3.1 vLLM 的三种使用模式

| 模式 | 命令 | 用途 | 接口 |
|------|------|------|------|
| **Online Serving (serve)** | `vllm serve ./model` | 启动 HTTP 服务器 | OpenAI-compatible REST API |
| **Offline Inference** | `from vllm import LLM` | 本地批量推理 | Python API |
| **API Server (旧版)** | `python -m vllm.entrypoints.api_server` | 启动 API 服务器（已废弃）| 旧版 API |

### 3.2 Serve 模式详解

Serve 模式启动一个 HTTP 服务器，提供与 OpenAI API 完全兼容的接口。这也是本项目采用的部署方式。

```bash
# 启动命令
vllm serve ./merged_qwen3_8b_medical \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --chat-template ./chat_template.jinja \
  --served-model-name medical-llm-teacher
```

**Serve 模式提供的 API Endpoints**：

| Endpoint | 完整路径 | 用途 |
|----------|---------|------|
| List Models | `GET /v1/models` | 获取可用模型列表 |
| Chat Completions | `POST /v1/chat/completions` | 对话生成（最常用） |
| Completions | `POST /v1/completions` | 文本续写 |
| Tokenize | `POST /tokenize` | 分词 |
| Detokenize | `POST /detokenize` | 逆分词 |
| Health Check | `GET /health` | 健康检查 |

### 3.3 OpenAI-Compatible API 本质

vLLM 的 HTTP API 兼容 OpenAI Python SDK，意味着任何用 `openai` 库写的代码都可以**只改 `base_url`** 就无缝切换到本地 vLLM：

```python
# 只需改这一行
client = OpenAI(
    base_url="http://localhost:8000/v1",  # 指向本地 vLLM
    api_key="not-needed",                  # 本地部署无需 API Key
)

# 其余代码与调用 OpenAI API 完全一致
response = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[{"role": "user", "content": "解释高血压的病理机制"}],
    temperature=0.3,
    max_tokens=512,
)
```

**兼容的 parameter mapping**：

| OpenAI Parameter | vLLM 支持 | 说明 |
|-----------------|----------|------|
| `model` | Yes | 对应 `--served-model-name` |
| `messages` | Yes | 标准对话格式 |
| `temperature` | Yes | 0.0-2.0 |
| `top_p` | Yes | nucleus sampling |
| `max_tokens` | Yes | 最大生成 token 数 |
| `stop` | Yes | 停止序列 |
| `stream` | Yes | 流式输出 |
| `frequency_penalty` | Yes | 频率惩罚 |
| `presence_penalty` | Yes | 存在惩罚 |
| `n` | Yes | 生成 n 个候选 |
| `logprobs` | Yes | 返回 log 概率 |
| `seed` | Yes | 随机种子 |
| `extra_body` 中的参数 | Yes (vLLM 特有) | `top_k`, `repetition_penalty`, `skip_special_tokens` 等 |

### 3.4 Offline Inference 模式

离线推理模式不使用 HTTP 服务，而是直接在 Python 进程中调用，适合批量处理数据：

```python
from vllm import LLM, SamplingParams

# 初始化引擎（加载一次模型）
llm = LLM(
    model="./merged_qwen3_8b_medical",
    dtype="bfloat16",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)

# 准备 prompt 列表
prompts = [
    "解释高血压的病理机制。",
    "糖尿病分为哪几种类型？",
    "抗生素的作用机理是什么？",
]

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.3,
    top_p=0.9,
    max_tokens=512,
    stop=["</s>", "<|im_end|>"],
)

# 批量推理（内部自动 batching，充分利用 GPU）
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Response: {output.outputs[0].text}")
    print("---")
```

**Offline 模式的适用场景**：

- **Rejected Generation**：DPO 训练前为每条数据生成 rejected response，批量跑完几千条。
- **Knowledge Distillation**：用 Teacher 模型批量生成训练数据。
- **评测批量推理**：对评测集的几百条 prompt 批量打推理结果。
- **Embedding 离线计算**：批量生成文档的 embedding 用于检索。

**Offline vs Serve 对比**：

| 维度 | Offline Inference | Serve |
|------|------------------|-------|
| 延迟 | 更高（批处理延迟） | 更低（即时响应） |
| 吞吐 | 更高（最大化 batching） | 受并发和排队影响 |
| 使用方式 | Python 脚本 | HTTP API（任意语言） |
| 适用场景 | 离线批处理 | 在线服务 |
| 本项目场景 | Rejected generation | 医学教学平台 API |

---

## 4. Merged Model 部署 vs LoRA Serving 区别

### 4.1 两种部署策略

**策略一：Merged Model（本项目采用）**

把 LoRA adapter 的权重通过 `W_merged = W_base + B × A / α` 融回 base model 的对应权重矩阵，得到一个完整的 BF16 模型文件，然后直接部署这个"一体模型"。

```python
# 步骤1: merge
from peft import PeftModel
model = PeftModel.from_pretrained(base_model_4bit, "./qlora_checkpoint")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged_model", safe_serialization=True)

# 步骤2: vLLM 部署 merged model（就像部署普通模型一样）
# vllm serve ./merged_model --dtype bfloat16 ...
```

**策略二：LoRA Serving（vLLM 原生支持）**

不 merge，base model 和 LoRA adapter 分开加载。vLLM 通过 `--enable-lora` 参数在推理时动态应用 LoRA adapter：

```bash
vllm serve ./Qwen3-8B \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --lora-modules medical-lora=./qlora_checkpoint
```

推理时通过 `model` 参数指定使用哪个 LoRA：

```python
response = client.chat.completions.create(
    model="medical-lora",  # 对应 --lora-modules 中的名称
    messages=[{"role": "user", "content": "..."}],
)
```

### 4.2 两种策略对比

| 维度 | Merged Model | LoRA Serving |
|------|-------------|-------------|
| **计算开销** | 无需额外计算 | 每个 token 需要 W = W0 + BA 的加法（微小） |
| **显存占用** | 一份完整模型（16GB BF16） | base model (16GB) + LoRA adapter（50MB × N） |
| **多 LoRA 支持** | 不行，一份 merge = 一个模型 | 一个 base model 可同时服务多个 LoRA |
| **模型管理** | 多个 merge 模型占用多份磁盘 | 一套 base + 多套 LoRA adapter |
| **部署复杂度** | 简单，就是普通模型 | 需要配置 --enable-lora, --max-loras 等 |
| **本项目选择** | **采用** | 未采用 |

### 4.3 为什么不选 LoRA Serving（面试回答）

"我们的场景下只有一个 LoRA adapter（SFT+DPO 训练产物），没有多 LoRA 服务的需求。Merge 成一个模型更简单，不需要管理 base-adapter 的版本对应关系。而且 merge 后推理路径和原生模型完全一致，不需要 W=W0+BA 的额外计算（虽然开销很小，但能省则省）。

如果未来我们有多个医学子方向（内科/外科/儿科）各自训练独立 LoRA，且需要在一个服务上动态切换，那时会考虑启用 LoRA Serving。但目前 single-model 场景下 merge 是最简洁的方案。"

---

## 5. KV Cache、Prefill 阶段、Decode 阶段、PagedAttention 深度解析

### 5.1 自回归生成的本质

LLM 生成文本是按 token 逐个生成的：每次预测下一个 token，然后把这个 token 拼到输入后面，再预测下一个。问题在于——每次预测时，模型都要对整个序列做 self-attention：

```
Step 0: 输入 [t1, t2, t3] → attention over [t1,t2,t3] → 预测 t4
Step 1: 输入 [t1, t2, t3, t4] → attention over [t1,t2,t3,t4]  → 预测 t5
Step 2: 输入 [t1, t2, t3, t4, t5] → attention over [t1,t2,t3,t4,t5] → 预测 t6
```

如果不做任何优化，Step 1 会重新计算 t1, t2, t3 的 Key/Value——这些在 Step 0 已经算过了。这就是 **KV Cache** 要解决的问题：把已经算过的 K 和 V 存下来，后续 step 直接复用。

### 5.2 KV Cache 的物理含义

Transformer 的每一层、每个注意力头在计算 attention 时都需要：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

- **Q (Query)**：当前 token 的"查询"向量，每次不同，必须实时计算。
- **K (Key)**：每个 token 的"键"向量，一旦算出就不再变化——可缓存！
- **V (Value)**：每个 token 的"值"向量，一旦算出就不再变化——可缓存！

KV Cache 就是存储所有历史 token 在各层各头的 K 和 V 向量。每个 decode step 只需计算当前新 token 的 Q，然后拿 Q 去和缓存的 K 做 attention，再乘以缓存的 V。

> **面试金句**："KV Cache 的本质是用空间换时间——多花显存存储已计算的 K/V，避免每个 decode step 都重新计算整个序列的 attention。没有 KV Cache，生成第 N 个 token 要做 O(N²) 的 attention 计算；有了 KV Cache，只需 O(N)。"

### 5.3 Prefill 阶段（预填充）

Prefill 阶段处理**输入 prompt 的所有 token**。这些 token 一次性送入模型做并行前向传播：

**Prefill 做了什么**：
1. 并行计算 prompt 中每个 token 在各层的 K 和 V，**存入 KV Cache**。
2. 在最后一层拿到最后一个 token 的 hidden state，投影到词表，得到**第一个输出 token** 的概率分布。
3. 采样/贪心选择第一个输出 token。

**Prefill 的计算特征**：
- **计算密集型（Compute-bound）**：prompt 的 N 个 token 同时做矩阵乘法，需要大量 FLOPS。
- **耗时与 prompt 长度的平方近似成正比**（attention 的 O(N²) 复杂度）。
- GPU 算力利用率的瓶颈（SM 占用率高），显存带宽不是瓶颈。

```
Prefill 耗时 ≈ α × N² × hidden_size² / GPU_FLOPS
其中 N = prompt token 数
```

### 5.4 Decode 阶段（逐 token 生成）

Decode 阶段从第一个输出 token 开始，逐 token 自回归生成，直到遇到 EOS 或达到 max_tokens：

**Decode 每个 step 做了什么**：
1. 将上一步生成的 1 个 token 送入模型。
2. 计算这个新 token 的 Q、K、V（K/V 追加到 KV Cache）。
3. 用 Q 去 attend 缓存中的所有历史 K/V（包括 prompt 的和之前生成的）。
4. 拿到 hidden state → 投影 → 得到下一个 token 的概率分布。
5. 采样下一个 token。

**Decode 的计算特征**：
- **显存带宽密集型（Memory-bound）**：每个 step 只做 1 个 token 的矩阵乘法（计算量很小），但需要从显存读取**整个 KV Cache**（数十 GB 级别）。
- 瓶颈不在算力，在显存读写速度。
- **耗时与已有的 KV Cache 长度近似成正比**（O(N)）。

```
Decode 每步耗时 ≈ β × (prompt_len + generated_len) × num_layers × num_heads × head_dim / mem_bandwidth
```

### 5.5 Prefill 和 Decode 的 GPU 利用特征对比

| 维度 | Prefill | Decode |
|------|---------|--------|
| **输入 token 数** | N（prompt） | 1（当前生成的 token） |
| **计算量** | O(N²) — 大 | O(N) — 小（N=已有序列长度） |
| **瓶颈类型** | Compute-bound | Memory-bound |
| **GPU SM 占用** | 高（大量并行计算） | 低（大部分时间等显存读取） |
| **对 batch size 敏感** | 是（大 batch 可更好利用算力） | 不太敏感（瓶颈不在算力） |
| **对序列长度敏感** | 是（平方级增长） | 是（线性增长） |
| **优化方向** | 减少 prompt 长度、提高 FLOPS | 提高显存带宽、减少 KV Cache 大小 |

### 5.6 PagedAttention 的 Block Table 映射

PagedAttention 的核心数据结构是 **block table**，每个请求维护一张表：

```
请求 i 的 block_table = [physical_block_id_0, physical_block_id_1, ...]
```

**新 token 到来时**：
1. 检查 block_table 最后一个 physical block 是否还有空位（block 没满）。
2. 如果满了 → 从全局 free block pool 申请一个新 block → 追加到 block_table。
3. 将新 token 的 K/V 写入对应 block 的空位。

**做 attention 时**：
1. 遍历 block_table 中的所有 physical block。
2. 从每个 block 中取出对应 token 范围内的 K/V。
3. GPU kernel 内部做高效的 gather + attention 计算。

**Sequence 结束时**：
1. block_table 中的所有 physical block 归还全局 free pool。
2. 归还的 block 可被新请求复用。

> **面试金句**："PagedAttention 让 KV Cache 的管理从'预分配连续大块'变成'按需分配小块+逻辑映射'，把显存利用率从 20-40% 提升到接近 100%，同时支持 prefix sharing 进一步节省显存。这背后是 operation system virtual memory 思想在 GPU 显存管理上的迁移。"

### 5.7 Continuous Batching 如何调度 Prefill 和 Decode

vLLM 的 scheduler 在每个 step（一次 GPU kernel launch）做决策：

```
当前活跃请求:
  Req1: 正在 decode 第 12 个 token
  Req2: 刚到达，需要 prefill（prompt 长度=300 tokens）
  Req3: 正在 decode 第 5 个 token

Scheduler 决策:
  → Req2 的 prefill 有 300 个 token 要处理，计算量大，单独跑
  → Req1 和 Req3 的 decode 只有 1 个 token 各，计算量小，batch 在一起
  → 下一个 step 再根据各请求状态重新决策
```

**关键点**：vLLM 的 scheduler 会在 prefill 和 decode 之间动态切换，而不是等某个请求完成才处理下一个。这就是 "continuous" 的含义。

---

## 6. KV Cache 显存估算与关键参数对显存的影响

### 6.1 KV Cache 精确估算公式

```
KV Cache (per token) = 2 × num_layers × num_kv_heads × head_dim × dtype_size

参数说明:
- 2: Key + Value 两份
- num_layers: Transformer 层数
- num_kv_heads: KV head 数（GQA 下 < num_attention_heads）
- head_dim: 每个 head 的维度
- dtype_size: BF16 = 2 bytes, FP16 = 2 bytes
```

**Qwen3-8B 的 KV Cache 估算（GQA num_kv_heads=8, head_dim=128, num_layers=32）**：

```
单 token KV Cache = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB
```

这个"128 KB/token"是一个非常重要的基准数字——每个 token（prompt + output）在 KV Cache 中占约 128KB 显存。

### 6.2 不同场景的显存占用

| 场景 | prompt_len | output_len | 总 KV Cache (单请求) | 并发数 | 总 KV Cache |
|------|-----------|------------|---------------------|--------|------------|
| 短问答 | 200 | 100 | 300 × 128KB = 38.4 MB | 16 | 614 MB |
| 中等问答 | 500 | 300 | 800 × 128KB = 102.4 MB | 16 | 1.6 GB |
| RAG 问答 | 2000 | 500 | 2500 × 128KB = 320 MB | 8 | 2.5 GB |
| RAG + 长输出 | 2000 | 1536 | 3536 × 128KB = 452 MB | 4 | 1.8 GB |
| 极限场景 | 3000 | 1096 | 4096 × 128KB = 512 MB | 2 | 1.0 GB |

### 6.3 本项目完整显存预算

| 组件 | 大小 | 说明 |
|------|------|------|
| 模型权重 (BF16) | 16 GB | 8B × 2 bytes |
| KV Cache (16 seq × avg 1500 tokens) | ~3 GB | 16 × 1500 × 128KB |
| CUDA Context + cuBLAS workspace | ~1.5 GB | PyTorch/CUDA 运行时开销 |
| 激活值 (推理) | ~0.5 GB | 远小于训练（无反向传播） |
| 其他（output buffer等） | ~0.5 GB | — |
| **总计** | **~21.5 GB** | 均值 |
| **峰值（长输出 max_tokens=1536）** | **~28.86 GB** | 压测验证值 |

**RTX 5090 32GB × 0.90 = 28.8 GB 可用。** 峰值场景 28.86 GB 已经非常接近上限，这就是为什么 `gpu_memory_utilization=0.90` 和 `max_model_len=4096` 是经过仔细考量的——再多就不安全了。

### 6.4 各参数对显存的影响机制

**max_model_len**：
- 直接影响单个请求可分配的最大 block 数。
- `max_model_len / block_size = max_blocks_per_seq`
- 4096 / 16 = 256 blocks per seq max
- 设置越大 → 单请求最多可占的 KV Cache 越大 → 同样显存可服务的并发数越少。
- 本项目设 4096：兼顾医学场景的长 prompt（RAG context 可达 1500-2500 token）和长输出需求。

**max_num_seqs（并发序列数）**：
- 决定了同一时刻最多有多少条 sequence 在内存中。
- 每增加一条 sequence，需要额外分配 block_table + 至少 1 个 block。
- 不直接设死 KV Cache 总量，而是通过 max_num_seqs × 每请求平均 KV Cache 间接控制。
- 本项目设 16：在 28.8GB 可用显存下，平均每请求 1500 token 的 KV Cache ≈ 3GB，16 × 3 = 48GB? 不对——实际上大部分请求不会同时达到 max_model_len。16 并发时约 60-70% 的 block 被占用。

**gpu_memory_utilization**：
- vLLM 在启动时计算：`可用显存 = total_gpu_memory × gpu_memory_utilization - 模型权重占用`，剩余作为 KV Cache 的 block pool。
- 本项目：`32GB × 0.90 - 16GB ≈ 12.8 GB` 用于 KV Cache block pool。
- `12.8 GB / 128 KB ≈ 100,000 tokens` 的 KV Cache 容量。
- 设太高（>0.95）→ OOM 风险；设太低（<0.80）→ KV Cache pool 太小，并发上不去。

**max_num_batched_tokens**：
- 一个 iteration（一次 GPU kernel launch）中最多处理多少个 token。
- 包括所有请求的 prefill tokens + decode tokens。
- 设太大 → 单次 kernel launch 时间长，其他请求排队；设太小 → GPU 利用率低。
- 本项目默认值与 max_model_len 联动。

**长上下文（RAG 场景）对显存的冲击**：
- RAG 场景下 prompt 从几百 token 膨胀到 2000-3000 token。
- KV Cache 从 100MB/请求 膨胀到 300-400MB/请求。
- 在 12.8GB KV Cache pool 下：从可服务 ~100 条短请求，降到只能服务 ~30-40 条 RAG 请求。
- **这是 RAG 导致系统吞吐下降的根本原因。**

---

## 7. RAG Context 太长为什么会拖慢 Prefill

### 7.1 现象描述

接入 RAG 后，原本 200 token 的 prompt 变成 2000-3000 token（system prompt + user query + retrieved documents）。用户明显感到"首 token 响应变慢"。

### 7.2 根本原因：Prefill 的 O(N²) 复杂度

RAG 增加的是 prompt 长度，而 prompt 在 prefill 阶段处理。Prefill 的 self-attention 计算量是 O(N²)：

```
prompt_len = 200:  attention 矩阵 = 200 × 200 = 40,000 个元素
prompt_len = 2000: attention 矩阵 = 2000 × 2000 = 4,000,000 个元素

2000/200 = 10x prompt 长度增长
4,000,000/40,000 = 100x attention 计算量增长！
```

**结论**：prompt 长度增加 10 倍，prefill 计算量增加约 100 倍（平方级），TTFT 从 42ms 可能变成数百毫秒甚至几秒。

### 7.3 其他附加开销

除了 attention 的 O(N²) 问题，长 prompt 还带来：

1. **KV Cache 显存膨胀**：3000 token × 128KB ≈ 384 MB per request，是短 prompt 的 10-15 倍。
2. **Block 分配开销**：需要分配的 block 数从 ~13 个增加到 ~188 个（block_size=16 下），block table 变大，查表开销增加。
3. **首 token 的 logits 计算**：prompt 越长，最后一个 token 的 hidden state 汇聚的信息量越大，虽然 logits 投影本身开销不变，但前置的 attention 汇聚已经涵盖了全部 N 个 token。
4. **Prefix cache 失效**：如果 RAG 检索到不同文档，prompt 前缀不共享，prefix cache 命中率下降。

### 7.4 量化分析：不同 prompt 长度的 prefill 耗时

| prompt 长度 | prefill 估算耗时 | TTFT (1 并发) | TTFT (16 并发) |
|------------|-----------------|--------------|----------------|
| 200 (无 RAG) | ~20 ms | ~42 ms | ~100 ms |
| 1000 (轻 RAG) | ~80 ms | ~110 ms | ~350 ms |
| 2000 (中 RAG) | ~250 ms | ~300 ms | ~900 ms |
| 3000 (重 RAG) | ~500 ms | ~560 ms | ~1800 ms |

**本项目数据验证**：无 RAG 时 1 并发 TTFT≈42ms；接入 RAG（prompt ~2000 tokens）后 TTFT 掉到 ~300ms，约 7x 恶化，与 O(N²) 理论趋势一致。

### 7.5 缓解策略（面试回答亮点）

> **面试官**："RAG 接入后为什么变慢了？怎么解决？"

详见第 11 节"性能优化"中的"控制 RAG Context 长度"和"Prompt 压缩"部分，以及文件末尾的"面试官问 RAG 接入后为什么变慢 标准回答"。

---

## 8. 推理性能指标体系（完整解释词条）

面试官可能会随意挑一个指标问"这个是什么意思"，需要能准确解释每个指标的定义、计算方式、影响因素、优化方向。

### 8.1 延迟类指标

| 指标 | 全称/英文 | 定义 | 计算方式 | 用户体感 |
|------|----------|------|---------|---------|
| **Latency** | End-to-End Latency | 从发送请求到收到完整响应的总时间 | t_response_complete - t_request_sent | "等了多久拿到完整回答" |
| **TTFT** | Time To First Token | 从发送请求到收到第一个 token 的时间 | t_first_token - t_request_sent | "响应快不快"——最关键的用户体验指标 |
| **TPOT** | Time Per Output Token | 除首个 token 外，每个输出 token 的平均生成时间 | (总时间 - TTFT) / (输出 token 数 - 1) | "生成流畅不流畅" |
| **ITL** | Inter-Token Latency | 相邻两个 token 之间的延迟 | t_token_i - t_token_{i-1} | "打字机卡不卡" |
| **Prefill Latency** | — | prefill 阶段的耗时 | 从开始处理 prompt 到 prefill 完成 | 受 prompt 长度影响最大 |
| **Decode Latency** | — | 每个 decode step 的耗时 | 单个 token 的 decode 时间 ≈ TPOT | 受 KV Cache 大小影响 |
| **Queueing Latency** | — | 请求在队列中等待被调度的时间 | t_scheduled - t_request_sent | 并发高时的主要延迟来源 |
| **p50/p90/p95/p99 Latency** | 百分位延迟 | 50%/90%/95%/99% 的请求延迟低于此值 | 所有请求按延迟排序，取对应百分位 | "稳定性"——p99 代表最差 1% 用户体感 |

### 8.2 吞吐类指标

| 指标 | 全称/英文 | 定义 | 计算方式 |
|------|----------|------|---------|
| **Throughput** | 吞吐量 | 单位时间处理的请求数或生成的 token 数 | requests / second 或 tokens / second |
| **QPS** | Queries Per Second | 每秒处理的请求数 | 总请求数 / 总时间 |
| **tokens/s** | Tokens Per Second | 每秒生成的 token 数（含 prompt + output） | (prompt_tokens + output_tokens) / 总时间 |
| **Prompt tokens/s** | — | 每秒处理的 prompt token 数 | prompt_tokens_processed / prefill_time |
| **Generation tokens/s** | — | 每秒生成的 output token 数 | output_tokens_generated / total_decode_time |
| **Request/s** | Requests Per Second | 每秒完成的请求数 | 同 QPS |
| **Token Throughput** | — | 同 tokens/s，但更关注生成侧 | output_tokens / total_time |

### 8.3 资源利用率类指标

| 指标 | 英文 | 定义 | 理想值 |
|------|------|------|--------|
| **GPU Utilization** | GPU 使用率 | GPU SM（流多处理器）活跃时间的比例 | 推理 decode 阶段通常较低（30-60%），因为 memory-bound |
| **Memory Utilization** | 显存使用率 | 已分配显存 / 总显存 | 80-90%（留余量防 OOM） |
| **Batch Occupancy** | 批处理占用率 | 当前 batch 中活跃的请求数 / max_num_seqs | 高并发时应接近 100% |
| **KV Cache Utilization** | KV Cache 利用率 | 已分配的 block 数 / 总可用 block 数 | 70-90%（太低浪费显存，太高 OOM 风险） |

### 8.4 指标之间的关系（面试必问）

**TTFT 和 TPOT 的分解**：
```
End-to-End Latency = Queueing Latency + TTFT + (num_output_tokens - 1) × TPOT

TTFT ≈ Queueing Latency + Prefill Latency
TPOT ≈ Decode Per-Step Latency（主要由显存带宽决定）
Prefill Latency ∝ prompt_len²（近似）
Decode Latency ∝ (prompt_len + generated_len)（线性）
```

**延迟和吞吐的 trade-off**：
```
低并发 → TTFT 低, GPU 利用率低, 吞吐低
中并发 → TTFT 适中, GPU 利用率中等, 吞吐上升
饱和并发 → TTFT 开始明显上升, GPU 利用率饱和, 吞吐达到平台
过饱和并发 → TTFT 暴涨(p99 爆炸), GPU 利用率略增, 吞吐几乎不变
```

**本项目数据验证**：

| 并发 | TTFT | Throughput | GPU 利用率 |
|------|------|-----------|-----------|
| 1 | 42ms | 95 tok/s | ~30% |
| 4 | 55ms | 362 tok/s | ~60% |
| 8 | 71ms | 643 tok/s | ~80% |
| 16 | 100ms | 877 tok/s | ~90% |
| 32 | 280ms | 892 tok/s | ~92% (排队严重) |

**关键观察**：16 并发是"甜蜜点"——吞吐达到峰值的 98%，TTFT 仍在可接受范围（100ms < 300ms 人类感知阈值）。32 并发时吞吐几乎不增，但 TTFT 翻了近 3 倍。

> **面试金句**："压测不是为了跑出最高的吞吐数字，而是找到延迟可接受前提下的最大吞吐点。对于医学在线问答，100ms TTFT 和 877 tok/s 的吞吐是最好的平衡。超过这个点，延迟恶化远超吞吐收益——这就是'饱和并发'的含义。"

### 8.5 p50/p95/p99 为什么重要（医学场景特化）

在医学在线问答场景中：

- **p50 延迟**：一半用户的体验，"典型"延迟。
- **p95 延迟**：20 个用户中有 1 个遇到的延迟，代表了"偶尔慢"的场景。
- **p99 延迟**：100 个用户中有 1 个遇到的最差延迟，**长尾用户的体验直接影响对服务的信任**。医学场景中，这 1% 的用户可能正在问一个紧急问题。

> **面试金句**："医学场景关注 p95/p99 比平均延迟更重要。一个 p99=2s 的系统，意味着每 100 个医学生中就有一个要等 2 秒才能看到回答——这在教学场景可能还能接受，但在临床辅助场景是不可接受的。我们的目标是 p95 < 300ms, p99 < 500ms。"

---

## 9. vLLM 关键参数详解

### 9.1 模型与路径类

| 参数 | 类型 | 说明 | 本项目配置 |
|------|------|------|-----------|
| `--model` | 位置参数 | HuggingFace model name 或本地路径 | `./merged_qwen3_8b_medical` |
| `--served-model-name` | str | API 中 `model` 参数对应的名称，可设多个（逗号分隔） | `medical-llm-teacher` |
| `--tokenizer` | str | tokenizer 路径（默认与 model 相同） | 默认（与 model 一致） |
| `--trust-remote-code` | flag | 允许执行模型仓库中的自定义代码（Qwen 系列必需） | 使用（Qwen3 有自定义 modeling 代码） |
| `--revision` | str | 指定模型的 git revision/branch | 不使用 |
| `--download-dir` | str | 模型下载缓存目录 | 默认 |

### 9.2 精度与量化类

| 参数 | 类型 | 说明 | 本项目配置 |
|------|------|------|-----------|
| `--dtype` | str | 模型推理精度：auto/half/float16/bfloat16/float/float32 | `bfloat16` |
| `--quantization` | str | 量化方法：awq/gptq/squeezellm/fp8/tpu_int8 | 不使用（BF16 原始精度） |
| `--kv-cache-dtype` | str | KV Cache 的存储精度：auto/fp8/fp8_e5m2/fp8_e4m3 | auto |

**`--dtype` 选择建议**：
- `bfloat16`：推荐（本项目使用），与 BF16 训练的模型一致，精度无损。
- `float16`：如果显卡不支持 BF16 可用，但需要确认模型兼容。
- `auto`：自动从模型 config 读取，通常安全。

**`--quantization` 使用场景**：
```bash
# GPTQ 4-bit 量化模型部署：省一半显存
vllm serve ./model-gptq --quantization gptq --dtype float16

# AWQ 4-bit 量化模型部署
vllm serve ./model-awq --quantization awq --dtype float16
```

> **面试要点**：量化推理（INT4/INT8）可以用更少的显存跑更大的模型，但会有少量精度损失。医学场景对精度敏感，我们选择了 BF16 而非 INT4 推理，保证答案的准确性。

### 9.3 显存与调度类

| 参数 | 类型 | 默认值 | 说明 | 本项目配置 |
|------|------|--------|------|-----------|
| `--max-model-len` | int | 模型 config | 模型能处理的最大 token 数（prompt + output） | `4096` |
| `--gpu-memory-utilization` | float | 0.90 | vLLM 可用的 GPU 显存比例 | `0.90` |
| `--max-num-seqs` | int | 256 | 同一时刻最大的并发序列数 | `16` |
| `--max-num-batched-tokens` | int | 与 max_model_len 相关 | 单个 iteration 中处理的最大 token 数 | 默认 |
| `--max-num-partial-prefills` | int | 1 | 单 iteration 中可 partial prefill 的请求数 | 默认 |

**这些参数如何联动影响显存**：

```
可用显存 = total_gpu_memory × gpu_memory_utilization

KV Cache Block Pool = 可用显存 - 模型权重 - CUDA 开销

单请求最大 block 数 = max_model_len / block_size

理论最大并发 = KV Cache Block Pool / (单请求平均 block 使用数)

实际最大并发 ≤ min(理论最大并发, max_num_seqs)
```

**如何根据显存调整这些参数（面试实操题）**：

```
场景1: 32GB 显卡, 8B 模型 (16GB 权重)
  → gpu_memory_utilization=0.90, max_model_len=4096, max_num_seqs=16
  → KV Cache pool ≈ 12.8 GB, 可容纳约 100K token 的 KV Cache

场景2: 24GB 显卡 (RTX 4090), 8B 模型 (16GB 权重)
  → gpu_memory_utilization=0.90, max_model_len=2048, max_num_seqs=8
  → KV Cache pool ≈ 5.6 GB, 需减少 max_model_len 和 max_num_seqs

场景3: 48GB 显卡 (RTX 6000 Ada), 8B 模型 (16GB 权重)
  → gpu_memory_utilization=0.90, max_model_len=8192, max_num_seqs=32
  → KV Cache pool ≈ 27.2 GB, 充分发挥大显存优势

场景4: 32GB 显卡, INT4 量化 8B 模型 (4GB 权重)
  → gpu_memory_utilization=0.90, max_model_len=8192, max_num_seqs=32
  → KV Cache pool ≈ 24.8 GB, 量化释放的显存可大幅提升并发
```

> **面试金句**："vLLM 参数调优的核心思想是'显存预算制'——先算模型权重占多少，再算 KV Cache pool 剩多少，然后根据业务场景分配 max_model_len（单请求上限）和 max_num_seqs（并发数）。这两者是 trade-off：要支持更长上下文就要牺牲并发数，反之亦然。"

### 9.4 并行类

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tensor-parallel-size` | int | 1 | Tensor Parallel 切分到几张 GPU |
| `--pipeline-parallel-size` | int | 1 | Pipeline Parallel 切分到几组 GPU |

```bash
# 单卡（本项目）
vllm serve ./model --dtype bfloat16

# 2 卡 Tensor Parallel（模型放不进单卡时）
vllm serve ./model --dtype bfloat16 --tensor-parallel-size 2

# 4 卡 Tensor Parallel
vllm serve ./model --dtype bfloat16 --tensor-parallel-size 4
```

**Tensor Parallel 的利弊**：
- 利：让超单卡容量的模型能在多卡上跑。
- 弊：GPU 间通信（all-reduce）引入延迟，每层都要同步一次。对于能放进单卡的模型，加 TP 反而降低吞吐（通信开销 > 并行收益）。
- 本项目不用的原因：8B 模型 16GB 单 RTX 5090 32GB 完全够，加 TP 反而降速。

### 9.5 LoRA 类

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enable-lora` | flag | False | 启用 LoRA serving |
| `--max-loras` | int | 1 | 最多同时加载几个 LoRA adapter |
| `--max-lora-rank` | int | 16 | 支持的最大 LoRA rank |
| `--lora-modules` | list | None | 定义 LoRA 模块：`name=path` 对 |
| `--max-cpu-loras` | int | None | 允许 CPU 缓存的 LoRA 数量 |

```bash
# LoRA Serving 示例
vllm serve ./Qwen3-8B \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 64 \
  --lora-modules medical=./lora_medical surgery=./lora_surgery
```

### 9.6 网络与日志类

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--host` | str | 127.0.0.1 | 绑定的 IP，`0.0.0.0` 表示允许外部访问 |
| `--port` | int | 8000 | HTTP 端口 |
| `--ssl-keyfile` | str | None | HTTPS 私钥 |
| `--ssl-certfile` | str | None | HTTPS 证书 |
| `--disable-log-requests` | flag | False | 禁止打印每个请求的日志（生产环境建议开启） |
| `--disable-log-stats` | flag | False | 禁止打印统计日志 |

```bash
# 生产环境启动（减少日志开销）
vllm serve ./model \
  --host 0.0.0.0 \
  --port 8000 \
  --disable-log-requests \   # 不打印每个请求，减少 I/O
  --dtype bfloat16 \
  ...
```

### 9.7 计算与加速类

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--enforce-eager` | flag | False | 强制使用 eager 模式（禁用 CUDA graph），用于调试 |
| `--disable-custom-all-reduce` | flag | False | 禁用自定义 all-reduce（仅 TP > 1 时相关） |
| `--enable-prefix-caching` | flag | False | 启用自动前缀缓存 |
| `--cpu-offload-gb` | float | 0 | 将多少 GB 的模型权重 offload 到 CPU |
| `--swap-space` | int | 4 | 每个请求的 CPU swap 空间（GB），用于 KV Cache 换出 |

```bash
# 启用前缀缓存（推荐）
vllm serve ./model --enable-prefix-caching ...

# 调试模式（禁用 CUDA graph，方便排查问题）
vllm serve ./model --enforce-eager ...

# CPU offload（显存极度受限时，牺牲速度换容载）
vllm serve ./model --cpu-offload-gb 8 ...
```

### 9.8 本项目完整启动命令及参数解释

```bash
# FlashInfer 兼容性修复（Blackwell SM 12.x）
export VLLM_USE_FLASHINFER_SAMPLER=0

# 启动服务
vllm serve ./merged_qwen3_8b_medical \        # --model: 合并后的模型路径
  --host 0.0.0.0 \                             # 允许外部 IP 访问
  --port 8000 \                                # 端口 8000
  --served-model-name medical-llm-teacher \     # API 中 model 参数的值
  --dtype bfloat16 \                           # BF16 推理精度（与训练一致）
  --max-model-len 4096 \                       # 最大序列长度，控制 KV Cache 上限
  --max-num-seqs 16 \                          # 最大并发序列数（饱和并发点）
  --gpu-memory-utilization 0.90 \              # 使用 90% 显存
  --enable-prefix-caching \                    # 启用前缀缓存（RAG 场景收益大）
  --chat-template ./chat_template.jinja \       # 自定义对话模板
  --trust-remote-code \                        # Qwen3 必须
  --disable-log-requests                       # 生产环境不打印请求日志
```

---

## 10. 性能优化策略（全栈清单）

### 10.1 调度层优化

**Continuous Batching**（vLLM 默认启用）：
- 机制：动态混合 prefill 和 decode 请求，不让 GPU 空闲等待。
- 效果：相比静态 batching，吞吐提升 2-4x。
- 无需额外配置，vLLM 原生支持。

**Chunked Prefill**（vLLM 0.5.0+）：
- 机制：将长 prompt 的 prefill 拆成多个 chunk，与 decode 请求穿插调度。
- 效果：减少长 prompt 对其他请求的阻塞效应，降低 p99 延迟。
- 配置：`--max-num-batched-tokens` 控制每个 chunk 大小。

### 10.2 显存层优化

**减少 max-model-len**：
```bash
# 短问答场景：2048 足够，比 4096 省一半 KV Cache
--max-model-len 2048
```

**限制 max_new_tokens（API 层面）**：
```python
# 客户端限制最大输出长度，防止恶意长输出耗尽 KV Cache
response = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[...],
    max_tokens=256,  # 限制输出 ≤ 256 tokens
)
```

**BF16 推理**（已完成）：
- BF16 比 FP32 省 50% 显存，精度几乎无损。

**量化推理**（显存极度受限时的备选）：
```bash
# GPTQ 4-bit：模型从 16GB → 4GB，同样的显存可大幅提升并发
vllm serve ./model-gptq --quantization gptq --dtype float16
```

### 10.3 Prompt 层优化

**控制 RAG Context 长度**：
```python
# 方案1: 限制检索文档数量
retriever.search(query, top_k=3)  # 只取 Top-3 文档，而非 Top-10

# 方案2: 对检索到的文档做截断
def truncate_docs(docs, max_chars_per_doc=500):
    return [doc[:max_chars_per_doc] for doc in docs]

# 方案3: 使用文档摘要而非原文
docs = [summarizer(doc) for doc in retrieved_docs]  # 摘要比原文短 5-10x
```

**Prompt 压缩**：
```python
# 方案1: LLMLingua 等 prompt 压缩工具
from llmlingua import PromptCompressor
compressor = PromptCompressor(model_name="Qwen/Qwen3-8B")
compressed_prompt = compressor.compress_prompt(
    long_prompt, rate=0.5  # 压缩到 50%
)

# 方案2: 使用小模型做 prompt 摘要
summary = small_model.generate(f"请总结以下内容的关键信息：\n{long_context}")
short_prompt = f"基于以下信息回答问题：\n{summary}\n\n问题：{query}"
```

**Prefix Cache / Prompt Cache**（vLLM 原生支持）：
```bash
# 启用自动前缀缓存：相同的 system prompt 只需计算一次
vllm serve ./model --enable-prefix-caching
```
- 原理：如果两个请求的 prompt 前 N 个 token 相同，后续请求可以直接复用前一个请求已算好的 KV Cache block。
- 收益：system prompt 越长受益越大（如 RAG 场景中 system prompt + instruction 模板部分可跨请求共享）。
- 注意：RAG 检索到的文档各不相同，这部分无法缓存。所以 prefix cache 对 RAG 的 system prompt 部分有效，对文档部分无效。

### 10.4 缓存层优化

**RAG 检索缓存**：
```python
# 缓存检索结果，避免相同 query 反复检索
from functools import lru_cache

@lru_cache(maxsize=1000)
def retrieve_with_cache(query: str):
    return retriever.search(query)
```

**Embedding 缓存**：
```python
# 文档 embedding 提前计算并持久化，不每次实时计算
# 使用 FAISS + 预计算 embedding
import numpy as np
doc_embeddings = np.load("./precomputed_embeddings.npy")  # 离线算好
index = faiss.IndexFlatIP(dimension)
index.add(doc_embeddings)
```

**Rerank 缓存**：
```python
# Reranker 只对 Top-k 检索结果做重排，而不对全库
initial_results = retriever.search(query, top_k=100)       # 快速粗筛
reranked = reranker.rerank(query, initial_results[:20])     # 只重排前 20
```

### 10.5 并发与延迟平衡

**控制并发避免 p99 爆炸**：
```python
# 在 API Gateway 层做并发控制
# 方案1: 信号量限制
import asyncio
semaphore = asyncio.Semaphore(16)  # 限制最大 16 并发

async def handle_request(request):
    async with semaphore:
        return await vllm_client.generate(request)

# 方案2: 请求队列 + 超时
from asyncio import Queue
request_queue = Queue(maxsize=32)  # 最多排队 32 个
```

**增大 batch 提升吞吐（离线场景）**：
```python
# 离线推理时，batch 越大 GPU 利用率越高
from vllm import LLM
llm = LLM(model="./model")
# 一次性给 100 条 prompt，vLLM 内部自动调度 batching
outputs = llm.generate(prompts_100, sampling_params)
```

### 10.6 部署层优化

**Warmup 预热**：
```python
# vLLM 启动后先发几个预热请求，让 CUDA kernel 编译完成
import requests

def warmup(base_url, model_name, n_warmup=5):
    """发送预热请求，触发 CUDA kernel 编译和 KV Cache 初始化"""
    for i in range(n_warmup):
        requests.post(f"{base_url}/v1/chat/completions", json={
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        })
    print("Warmup complete.")

warmup("http://localhost:8000", "medical-llm-teacher")
```

**流式输出降低感知延迟**：
```python
# 流式输出让用户立刻看到第一个 token，无需等全部生成完
stream = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[...],
    stream=True,  # 关键参数
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

> **面试金句**："流式输出的核心价值是降低感知延迟——用户不用等全部回答生成完才看到内容。TTFT 从端到端延迟变成了用户感知的唯一等待时间。但总延迟不变（甚至因为 SSE framing 略有增加）。"

### 10.7 完整优化优先级清单

| 优先级 | 优化手段 | 收益 | 代价 | 本项目是否采用 |
|--------|---------|------|------|--------------|
| P0 | Continuous Batching | 吞吐 +2-4x | 无 | 是（vLLM 默认） |
| P0 | 减少 max-model-len | 显存省 20-50% | 长文本截断 | 是（4096） |
| P0 | 限制 max_new_tokens | 防 OOM | 输出被截断 | 是（API 层限制） |
| P1 | Prefix Caching | TTFT -20-30% | 少量显存 | 是 |
| P1 | 流式输出 | 感知延迟 -80% | 总延迟不变 | 是 |
| P1 | 控制 RAG context 长度 | TTFT -50-70% | RAG 召回略降 | 是 |
| P2 | Embedding 缓存 | Prefill 前的延迟 -50% | 存储空间 | 可选 |
| P2 | Prompt 压缩 | Prefill 延迟 -30-50% | 信息损失 | 可选 |
| P3 | 量化部署 (INT4) | 显存省 75% | 精度略降 | 未采用 |
| P3 | Tensor Parallel | 支持更大模型 | 通信开销 | 本项目不需要 |

---

## 11. 代码示例合集

### 11.1 vLLM 启动命令（merged model 部署）

```bash
#!/bin/bash
# start_vllm_server.sh
# 医学 LLM Teacher vLLM 服务启动脚本

# 1. 修复 Blackwell 架构兼容性
export VLLM_USE_FLASHINFER_SAMPLER=0

# 2. 设置 CUDA 可见设备（多卡时指定用哪张卡）
export CUDA_VISIBLE_DEVICES=0

# 3. 启动 vLLM 服务
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

# 4. 等待服务就绪
echo "Waiting for vLLM server to be ready..."
until curl -s http://localhost:8000/health > /dev/null; do
    sleep 2
done
echo "vLLM server is ready!"
```

### 11.2 OpenAI-Compatible API 调用示例

```python
"""
openai_client_example.py
使用 OpenAI Python SDK 调用本地 vLLM 服务
"""
from openai import OpenAI

# 初始化客户端（指向本地 vLLM）
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # 本地部署无需 API Key
)

# ==================== 标准调用 ====================
response = client.chat.completions.create(
    model="medical-llm-teacher",
    messages=[
        {
            "role": "system",
            "content": (
                "你是一个专业的医学教学助手。请以准确、易懂的方式回答医学问题。\n"
                "要求：\n"
                "1. 回答基于循证医学证据\n"
                "2. 不确定的内容请明确说明\n"
                "3. 不要给出具体的用药剂量建议"
            ),
        },
        {"role": "user", "content": "请解释II型糖尿病的发病机制和主要治疗策略。"},
    ],
    temperature=0.3,       # 低温度保证医学回答的稳定性
    top_p=0.9,
    max_tokens=512,
    stop=["</s>", "<|im_end|>"],
    extra_body={
        "repetition_penalty": 1.1,
        "top_k": 50,
    },
)

print("=" * 60)
print("完整回答：")
print(response.choices[0].message.content)
print("=" * 60)
print(f"Usage: prompt_tokens={response.usage.prompt_tokens}, "
      f"completion_tokens={response.usage.completion_tokens}")
```

### 11.3 Python Requests 调用示例（不依赖 openai 库）

```python
"""
requests_example.py
使用 requests 库直接调用 vLLM API（无需 SDK）
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# ==================== 非流式调用 ====================
def chat_completion(prompt: str, system_prompt: str = None, max_tokens: int = 512):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "medical-llm-teacher",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "stream": False,
    }

    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# 测试
answer = chat_completion("什么是高血压？")
print(answer)

# ==================== 健康检查 ====================
def health_check():
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    return resp.status_code == 200

# ==================== 模型列表 ====================
def list_models():
    resp = requests.get(f"{BASE_URL}/v1/models", timeout=5)
    return resp.json()

print(f"Health: {health_check()}")
print(f"Models: {list_models()}")
```

### 11.4 流式输出调用示例

```python
"""
streaming_example.py
vLLM 流式输出 (Server-Sent Events)
"""
from openai import OpenAI
import time

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

def stream_generate(prompt: str, max_tokens: int = 512):
    """流式生成并实时打印，同时统计延迟指标"""
    t_start = time.time()
    first_token_time = None
    token_count = 0
    full_response = ""

    stream = client.chat.completions.create(
        model="medical-llm-teacher",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
        stream=True,
    )

    print("=" * 60)
    print("Streaming response:")
    print("-" * 60)

    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            if first_token_time is None:
                first_token_time = time.time() - t_start
                print(f"\n[TTFT: {first_token_time*1000:.0f}ms]\n")
            full_response += content
            token_count += 1
            print(content, end="", flush=True)

    t_end = time.time()
    total_time = t_end - t_start
    tpot = (total_time - first_token_time) / max(token_count - 1, 1) if token_count > 1 else 0

    print("\n" + "-" * 60)
    print(f"Metrics:")
    print(f"  TTFT:       {first_token_time*1000:.0f} ms")
    print(f"  Total Time: {total_time*1000:.0f} ms")
    print(f"  Tokens:     {token_count}")
    print(f"  TPOT:       {tpot*1000:.0f} ms/token")
    print(f"  Tokens/s:   {token_count / total_time:.1f}")
    print("=" * 60)

    return {
        "ttft_ms": first_token_time * 1000,
        "total_ms": total_time * 1000,
        "tokens": token_count,
        "tpot_ms": tpot * 1000,
        "tokens_per_sec": token_count / total_time,
    }

# 测试
stream_generate("请用通俗易懂的语言解释什么是抗生素耐药性。", max_tokens=256)
```

### 11.5 批量压测脚本

```python
"""
benchmark_batch.py
批量压测脚本：测不同并发下的 TTFT、TPOT、Throughput
"""
import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import List
import httpx

BASE_URL = "http://localhost:8000"
MODEL_NAME = "medical-llm-teacher"
PROMPT = "请详细解释高血压的发病机制、分类标准、常见并发症以及一线治疗药物的选择原则。"

@dataclass
class RequestResult:
    ttft: float        # Time to First Token (seconds)
    total_time: float  # End-to-end (seconds)
    tokens: int        # Number of generated tokens
    success: bool
    error: str = ""

    @property
    def tpot(self) -> float:
        """Time Per Output Token (seconds)"""
        if self.tokens <= 1:
            return 0.0
        return (self.total_time - self.ttft) / (self.tokens - 1)

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens / self.total_time if self.total_time > 0 else 0


async def send_single_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int = 256,
) -> RequestResult:
    """发送单次流式请求并统计延迟指标"""
    t_start = time.time()
    first_token_time = None
    token_count = 0

    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # 贪心解码，保证可复现
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=120,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # 去掉 "data: " 前缀
                    if data_str == "[DONE]":
                        break
                    if first_token_time is None:
                        first_token_time = time.time() - t_start
                    try:
                        data = json.loads(data_str)
                        if data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            token_count += 1
                    except json.JSONDecodeError:
                        continue

        total_time = time.time() - t_start
        return RequestResult(
            ttft=first_token_time or total_time,
            total_time=total_time,
            tokens=token_count,
            success=True,
        )
    except Exception as e:
        return RequestResult(
            ttft=0, total_time=0, tokens=0,
            success=False, error=str(e),
        )


async def run_benchmark(
    concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32],
    max_tokens: int = 256,
    requests_per_level: int = 20,
):
    """多并发压测主函数"""
    results_summary = []

    async with httpx.AsyncClient() as client:
        for concurrency in concurrency_levels:
            print(f"\n{'='*60}")
            print(f"Testing concurrency = {concurrency}")
            print(f"{'='*60}")

            # 发送 concurrency 个并发请求
            tasks = [
                send_single_request(client, PROMPT, max_tokens)
                for _ in range(concurrency)
            ]

            batch_start = time.time()
            results: List[RequestResult] = await asyncio.gather(*tasks)
            batch_elapsed = time.time() - batch_start

            # 只统计成功的请求
            success_results = [r for r in results if r.success]
            failed_count = len(results) - len(success_results)

            if not success_results:
                print(f"  All {concurrency} requests failed!")
                continue

            # 计算聚合指标
            avg_ttft = sum(r.ttft for r in success_results) / len(success_results)
            avg_tpot = sum(r.tpot for r in success_results) / len(success_results)
            avg_total = sum(r.total_time for r in success_results) / len(success_results)
            total_tokens = sum(r.tokens for r in success_results)
            throughput = total_tokens / batch_elapsed  # tokens/s

            # 百分位延迟（使用总延迟）
            sorted_times = sorted([r.total_time for r in success_results])
            n = len(sorted_times)

            def percentile(p):
                idx = int(n * p / 100)
                idx = min(idx, n - 1)
                return sorted_times[idx] * 1000  # ms

            summary = {
                "concurrency": concurrency,
                "success_count": len(success_results),
                "failed_count": failed_count,
                "avg_ttft_ms": avg_ttft * 1000,
                "avg_tpot_ms": avg_tpot * 1000,
                "avg_total_ms": avg_total * 1000,
                "total_tokens": total_tokens,
                "throughput_tok_s": throughput,
                "batch_elapsed_s": batch_elapsed,
                "p50_ms": percentile(50),
                "p90_ms": percentile(90),
                "p95_ms": percentile(95),
                "p99_ms": percentile(99),
            }
            results_summary.append(summary)

            # 实时输出
            print(f"  Success: {len(success_results)}, Failed: {failed_count}")
            print(f"  TTFT (avg):    {avg_ttft*1000:.0f} ms")
            print(f"  TPOT (avg):    {avg_tpot*1000:.0f} ms")
            print(f"  Total (avg):   {avg_total*1000:.0f} ms")
            print(f"  Throughput:    {throughput:.0f} tok/s")
            print(f"  p50: {percentile(50):.0f}ms, p95: {percentile(95):.0f}ms, "
                  f"p99: {percentile(99):.0f}ms")
            print(f"  Batch elapsed: {batch_elapsed:.1f}s")

    return results_summary


# ==================== 运行压测 ====================
if __name__ == "__main__":
    print("Starting vLLM Benchmark...")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT[:80]}...")

    summary = asyncio.run(run_benchmark(
        concurrency_levels=[1, 2, 4, 8, 16, 32],
        max_tokens=256,
    ))

    # 打印汇总表
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Conc':>5} {'TTFT(ms)':>9} {'TPOT(ms)':>9} {'Total(ms)':>10} "
          f"{'Thru(t/s)':>10} {'p50(ms)':>8} {'p95(ms)':>8} {'p99(ms)':>8}")
    print("-" * 80)
    for s in summary:
        print(f"{s['concurrency']:>5} {s['avg_ttft_ms']:>9.0f} {s['avg_tpot_ms']:>9.0f} "
              f"{s['avg_total_ms']:>10.0f} {s['throughput_tok_s']:>10.0f} "
              f"{s['p50_ms']:>8.0f} {s['p95_ms']:>8.0f} {s['p99_ms']:>8.0f}")
```

### 11.6 并发压测脚本（简化版）

```python
"""
benchmark_simple.py
简化版并发压测：快速测 TTFT 和 Throughput
"""
import asyncio
import time
import httpx

BASE_URL = "http://localhost:8000"
MODEL = "medical-llm-teacher"

async def quick_benchmark(concurrency: int = 16, requests_count: int = 30):
    """
    快速压测指定并发数
    连续发送 requests_count 个请求，保持 concurrency 个同时在跑
    """
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def worker(prompt: str):
        async with semaphore:
            t0 = time.time()
            first_token = None
            token_n = 0

            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", f"{BASE_URL}/v1/chat/completions",
                    json={
                        "model": MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 256,
                        "temperature": 0.0,
                        "stream": True,
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and "[DONE]" not in line:
                            if first_token is None:
                                first_token = time.time() - t0
                            token_n += 1

            elapsed = time.time() - t0
            return {
                "ttft": first_token or elapsed,
                "total": elapsed,
                "tokens": token_n,
            }

    prompt = "请详细解释抗生素的分类及其作用机制。"

    t_start = time.time()
    tasks = [worker(prompt) for _ in range(requests_count)]
    results = await asyncio.gather(*tasks)
    t_total = time.time() - t_start

    # 统计
    ttfts = [r["ttft"] * 1000 for r in results]
    ttfts.sort()
    total_tokens = sum(r["tokens"] for r in results)
    throughput = total_tokens / t_total

    print(f"Concurrency limit: {concurrency}")
    print(f"Total requests:    {requests_count}")
    print(f"Total time:        {t_total:.1f}s")
    print(f"Throughput:        {throughput:.0f} tok/s")
    print(f"Avg TTFT:          {sum(ttfts)/len(ttfts):.0f} ms")
    print(f"p50 TTFT:          {ttfts[len(ttfts)//2]:.0f} ms")
    print(f"p95 TTFT:          {ttfts[int(len(ttfts)*0.95)]:.0f} ms")
    print(f"p99 TTFT:          {ttfts[int(len(ttfts)*0.99)]:.0f} ms")

if __name__ == "__main__":
    asyncio.run(quick_benchmark(concurrency=16, requests_count=30))
```

### 11.7 结果统计脚本

```python
"""
stats_analyzer.py
压测结果统计分析：从日志/JSON 中读取请求指标并生成报告
"""
import json
import statistics
from typing import List, Dict

def analyze_benchmark_results(results_file: str):
    """
    读取压测结果 JSON 文件，生成统计报告

    预期 JSON 格式（每行一个请求指标）:
    {"ttft_ms": 42.5, "total_ms": 850.3, "tokens": 256, "success": true}
    """
    records: List[Dict] = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    success_records = [r for r in records if r.get("success", True)]
    failed_records = [r for r in records if not r.get("success", True)]

    if not success_records:
        print("No successful requests to analyze!")
        return

    # ========== 延迟统计 ==========
    ttfts = sorted([r["ttft_ms"] for r in success_records])
    totals = sorted([r["total_ms"] for r in success_records])

    def p(arr, percentile):
        idx = int(len(arr) * percentile / 100)
        return arr[min(idx, len(arr) - 1)]

    print("=" * 70)
    print("LATENCY STATISTICS")
    print("=" * 70)
    print(f"  Total requests:  {len(records)} ({len(success_records)} success, "
          f"{len(failed_records)} failed)")
    print(f"  Failure rate:    {len(failed_records)/len(records)*100:.1f}%")
    print()

    print(f"  {'Metric':<18} {'Mean':>8} {'Min':>8} {'p50':>8} {'p90':>8} "
          f"{'p95':>8} {'p99':>8} {'Max':>8}")
    print(f"  {'-'*18} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, arr in [("TTFT (ms)", ttfts), ("E2E Total (ms)", totals)]:
        print(f"  {name:<18} "
              f"{statistics.mean(arr):>8.1f} "
              f"{min(arr):>8.1f} "
              f"{p(arr, 50):>8.1f} "
              f"{p(arr, 90):>8.1f} "
              f"{p(arr, 95):>8.1f} "
              f"{p(arr, 99):>8.1f} "
              f"{max(arr):>8.1f}")

    # ========== 吞吐统计 ==========
    total_tokens = sum(r.get("tokens", 0) for r in success_records)
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in success_records)
    total_output_tokens = sum(r.get("tokens", 0) for r in success_records)

    # 假设所有请求在 t0 到 t1 之间完成
    min_start = min(r.get("start_time", 0) for r in success_records)
    max_end = max(r.get("end_time", r.get("total_ms", 0) / 1000) for r in success_records)
    duration = max_end - min_start if max_end > min_start else 1

    # 计算 TPOT
    tpots = []
    for r in success_records:
        tokens = r.get("tokens", 0)
        ttft = r.get("ttft_ms", 0)
        total = r.get("total_ms", 0)
        if tokens > 1:
            tpot = (total - ttft) / (tokens - 1)
            tpots.append(tpot)

    print()
    print("=" * 70)
    print("THROUGHPUT STATISTICS")
    print("=" * 70)
    print(f"  Total time window:       {duration:.1f} s")
    print(f"  Total output tokens:     {total_output_tokens}")
    print(f"  Requests per second:     {len(success_records) / duration:.2f}")
    print(f"  Tokens per second:       {total_output_tokens / duration:.1f}")
    print(f"  Avg tokens per request:  {total_output_tokens / len(success_records):.0f}")
    if tpots:
        print(f"  Avg TPOT:                {statistics.mean(tpots):.1f} ms")

    # ========== 稳定性评估 ==========
    print()
    print("=" * 70)
    print("STABILITY ASSESSMENT")
    print("=" * 70)

    p99 = p(totals, 99)
    p50 = p(totals, 50)
    cv = statistics.stdev(totals) / statistics.mean(totals) * 100 if len(totals) > 1 else 0

    print(f"  Coefficient of Variation: {cv:.1f}%")
    print(f"  p99/p50 ratio:            {p99/p50:.1f}x")
    print(f"  Tail latency (p99 - p50): {p99 - p50:.0f} ms")

    if cv < 20:
        print("  Stability: GOOD (CV < 20%)")
    elif cv < 40:
        print("  Stability: FAIR (20% <= CV < 40%)")
    else:
        print("  Stability: POOR (CV >= 40%) - consider reducing concurrency")

    # ========== 异常值检测 ==========
    if len(totals) > 0:
        q1 = p(totals, 25)
        q3 = p(totals, 75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr
        outliers = [t for t in totals if t > upper_bound]
        if outliers:
            print(f"\n  Outliers detected: {len(outliers)} requests > {upper_bound:.0f} ms")
            print(f"  Max outlier: {max(outliers):.0f} ms")


if __name__ == "__main__":
    # 假设压测脚本输出到 benchmark_results.jsonl
    analyze_benchmark_results("benchmark_results.jsonl")
```

### 11.8 面试官问"vLLM 为什么快"标准回答

> **面试官**："vLLM 比 HuggingFace generate 快在哪里？为什么工业界都选它？"

**标准回答（简洁版，适合面试开场）**：

"vLLM 的快来自三个层面的优化：

**第一层：内存管理——PagedAttention。** 传统框架为每个请求预分配连续的最大长度 KV Cache，造成严重的显存碎片和浪费（利用率只有 20-40%）。PagedAttention 把 KV Cache 切成固定大小的 block，按需分配、物理离散、逻辑连续，把显存利用率提升到接近 100%。同样显存下能服务的并发请求数是传统框架的 2-4 倍。

**第二层：调度优化——Continuous Batching。** 传统框架的静态 batching 等一个 batch 所有请求都完成了才能接收新请求，GPU 存在大量等待空洞。vLLM 在每个 step 动态决定哪些请求 prefill、哪些 decode、哪些加入/退出 batch，最大化 GPU 利用率。吞吐再提升 2-3x。

**第三层：工程优化。** CUDA graph 加速（消除 kernel launch overhead）、自定义 CUDA kernel（高效的 PagedAttention kernel、融合的 RMSNorm+Residual）、prefix caching（相同前缀共享 KV Cache）、FP8 KV Cache 等。这些工程优化加起来再提升 30-50%。

综合起来，vLLM 的吞吐比 HuggingFace generate 高 10-20 倍。在我们的项目中，RTX 5090 单卡跑 Qwen3-8B，vLLM 达到 877 tok/s，而 HuggingFace generate 只有约 30-40 tok/s。"

### 11.9 面试官问"RAG 接入后为什么变慢"标准回答

> **面试官**："你们接入 RAG 之后推理变慢了吗？为什么？怎么解决的？"

**标准回答**：

"确实变慢了。我们的 RAG 接入后，TTFT 从 42ms 增加到约 300ms，大约 7 倍的恶化。原因和解决方法如下：

**原因拆解**：

第一，prefill 阶段的 **O(N²) attention 复杂度**是罪魁祸首。无 RAG 时 prompt 约 200 token，attention 计算量约 40K 元素。接入 RAG 后 prompt 膨胀到 2000 token（system prompt + query + 3 篇检索文档），attention 计算量约 4M 元素——prompt 长度 10x，但 attention 计算量 100x。TTFT 从 42ms 飙到 300ms+。

第二，**KV Cache 显存压力**。2000 token 的 prompt 单请求 KV Cache 约 250MB（vs 原来 25MB）。12.8GB 的 KV Cache pool 原来能服务数百条短请求，现在只能服务约 50 条 RAG 请求。并发能力下降。

第三，**prefix cache 命中率降低**。RAG 检索到的文档各不相同，系统 prompt 和指令模板虽然能共享，但文档部分无法缓存。prefix caching 的收益被稀释。

**我们的解决方案**：

1. **限制检索文档数量和长度**：从 Top-10 降到 Top-3，每篇文档截断到 500 字符，prompt 从 3000 token 压缩到约 1200 token。
2. **使用文档摘要替代原文**：对检索到的文档先用小模型做摘要，摘要比原文短 5-8x。
3. **启用 prefix caching**：system prompt 和 instruction 模板部分可跨请求共享，节省约 200 token 的 prefill 计算。
4. **分层缓存**：在 RAG 检索层加了 query 缓存（相同 query 复用检索结果），减少重复检索。

**效果**：优化后 TTFT 从 300ms 降到约 150ms，虽然仍比无 RAG 的 42ms 慢，但在医学教学场景中 150ms 的响应完全可接受，而 RAG 带来的事实准确性提升远超这个延迟代价。"

### 11.10 面试官问"如何降低医疗问答 TTFT"标准回答

> **面试官**："医学问答场景对响应速度要求很高，你怎么降低 TTFT？"

**标准回答**：

"降低 TTFT 需要从三个层面入手——prompt 层、调度层、部署层：

**Prompt 层优化（投入产出比最高）**：

1. **精简 system prompt**：从 300 token 压缩到 100 token 以内，去掉冗余的格式化指令，用最简洁的语言描述角色。
2. **控制检索文档量**：RAG 只取 Top-3 而非 Top-10，每个文档限制长度。
3. **使用 prompt 压缩工具**：如 LLMLingua 把 prompt 压缩到原来的 50%，只保留关键信息。
4. **预处理用户输入**：用关键词提取替代原始问题作为检索 query，减少检索延迟。

**调度层优化**：

1. **启用 prefix caching**：system prompt 只需要 prefill 一次，后续请求直接复用 KV Cache。
2. **降低最大并发数**：如果医学场景对延迟敏感度高于吞吐，可以把 max_num_seqs 设在 TTFT 足够低的值（如 8 而非 16），牺牲吞吐换延迟。
3. **prefill 请求优先级**：给 prefill 请求更高的调度优先级（但 vLLM 原生不支持，需要自定义 scheduler）。

**部署层优化**：

1. **使用流式输出**：虽然总延迟不变，但用户感知的等待从端到端延迟变成 TTFT，TTFT 42ms 用户完全感知不到。
2. **warmup 预热**：服务启动后先发几个预热请求编译 CUDA kernel，避免首请求慢。
3. **负载均衡 + 多实例**：如果请求量大，部署多个 vLLM 实例，Nginx 做 least-connections 负载均衡，降低单实例排队。
4. **RAG 请求和非 RAG 请求分流**：简单问答走快速通道（不检索），复杂医学问题走 RAG 通道。
5. **量化部署**：如果显存紧张，可用 AWQ 4-bit 量化把模型从 16GB 压到 4GB，释放的显存用于更大的 KV Cache pool，提升并发。

**在本项目中的实践**：
- max_num_seqs=16 是我们找到的平衡点：TTFT ~100ms（低于人类感知阈值 300ms），吞吐 ~877 tok/s。
- 启用 prefix caching 减少 system prompt 的重复 prefill。
- 流式输出确保用户立刻看到响应。
- RAG 场景限制检索 Top-3 文档。

如果面试官追问'TTFT 还能更低吗'，我会回答：可以部署多实例 + 负载均衡，将单实例的并发降低到 4-8，TTFT 可降到 50-70ms。代价是 GPU 利用率和吞吐会下降（因为每个实例的 batch 更小）。这本质上是 latency vs throughput 的 trade-off，医学场景下 100ms 已经足够好了。"

---

## 面试1分钟回答

"vLLM 是我们医学 LLM Teacher 项目的推理部署方案。核心优势是 PagedAttention——把 KV Cache 像操作系统虚拟内存一样分页管理，按需分配、物理离散、逻辑连续，显存利用率从传统框架的 20-40% 提升到接近 100%。加上 Continuous Batching 动态混合 prefill/decode 调度，比 HuggingFace generate 快 10-20 倍。

我们部署 Qwen3-8B BF16 合并模型在 RTX 5090 单卡上，max-model-len=4096, max-num-seqs=16, gpu_memory_utilization=0.90。极限吞吐 ~877 tok/s，1 并发 TTFT~42ms，16 并发 TTFT~100ms。16 并发是甜点——延迟可接受前提下的最大吞吐。部署中遇到 FlashInfer 在 Blackwell SM12.x 不兼容的问题，通过 VLLM_USE_FLASHINFER_SAMPLER=0 解决，吞吐影响不到 2%。”

---

## 面试3分钟回答

"我们用 vLLM 部署医学 LLM Teacher 项目，这涉及几个关键决策和优化：

**为什么选 vLLM**：PagedAttention 解决了传统 KV Cache 预分配的碎片问题，Continuous Batching 消灭了静态 batching 的等待空洞。相比 TGI/SGLang/TensorRT-LLM，vLLM 部署最简单、生态最完善、对于 8B 级别模型在消费级显卡上性能已足够。TensorRT-LLM 在企业级集群中更有优势。

**部署配置**：LoRA Merge 后得到完整 BF16 模型（16GB），而不是 LoRA Serving，因为单 LoRA 场景 merge 更简洁。关键参数：max-model-len=4096 控制单请求 KV Cache 上限，max-num-seqs=16 控制最大并发，gpu_memory_utilization=0.90 留 10% 余量防 OOM。显存预算：权重 16GB + KV Cache pool 12.8GB（约 100K token 容量）+ CUDA 开销 ~3GB，峰值场景（长输出 max_tokens=1536）显存峰值 28.86GB。

**性能指标**：极限吞吐 877 tok/s，饱和并发 16。TTFT 从 1 并发的 42ms 涨到 16 并发的 100ms。32 并发时吞吐几乎不涨（892 tok/s）但 TTFT 暴涨到 280ms——这就是饱和并发。医学场景我们关注 p95/p99 延迟，目标是 p95 < 300ms, p99 < 500ms。

**RAG 场景优化**：RAG context 从 200 膨胀到 2000 token，prefill 的 O(N²) 复杂度导致 TTFT 从 42ms 飙到 ~300ms（7x 恶化）。优化方案：限制检索 Top-3、文档截断、启用 prefix caching、query 缓存。优化后 TTFT 回到 ~150ms。

**工程踩坑**：Blackwell SM12.x 上 FlashInfer 采样器 CUDA kernel 不兼容，禁用后性能损失不到 2%。chat template 中显式插入空 think 块，防止 Qwen3 生成 `<think>` 标签泄漏。"

---

## 背诵版总结

### 核心概念速记

| 概念 | 一句话 | 对应原理 |
|------|--------|---------|
| PagedAttention | KV Cache 分页管理，按需分配，零碎片 | 类比 OS 虚拟内存 |
| Continuous Batching | 动态混合 prefill/decode，消灭等待空洞 | 每 step 重新调度 |
| Prefill | 并行处理 prompt，计算密集，O(N²) | 一次性算完所有 prompt token |
| Decode | 逐 token 生成，显存带宽密集，O(N) | 每次只处理 1 个新 token |
| TTFT | 首 token 延迟 = queueing + prefill | 用户体验的"响应感" |
| TPOT | 每 token 生成延迟 = 单步 decode 时间 | 决定"流畅感" |
| saturation concurrency | 吞吐不再随并发增加的点 | 再多并发只加延迟不加吞吐 |
| prefix caching | 相同前缀的 KV Cache 物理共享 | system prompt 只算一次 |

### 显存估算速算公式

```
模型权重: params × dtype_bytes (8B × 2 = 16GB BF16)
KV Cache/token: 2 × layers × kv_heads × head_dim × dtype_bytes
              = 2 × 32 × 8 × 128 × 2 ≈ 128 KB
KV Cache Pool: GPU_memory × gpu_memory_utilization - 模型权重 - overhead
              ≈ 32 × 0.90 - 16 - 3 ≈ 12.8 GB
KV Cache 容纳 token 数: 12.8 GB / 128 KB ≈ 100,000 tokens
```

### 参数决策速查

| 场景 | 调整方向 |
|------|---------|
| 显存紧张 | 减小 max-model-len, 减小 max-num-seqs, 考虑量化 |
| TTFT 太高 | 减小并发, 精简 prompt, 启用 prefix cache, 流式输出 |
| 吞吐不足 | 增大 max-num-seqs (别超过饱和点), batch 离线推理 |
| p99 爆炸 | 降并发, 加请求队列, 设置 max_num_seqs 上限 |
| RAG 变慢 | 限制文档数/长度, prompt 压缩, prefix cache, query 缓存 |

### 项目数字速记

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-8B, BF16 合并模型, 16GB |
| GPU | RTX 5090 32GB |
| max-model-len | 4096 |
| max-num-seqs | 16 |
| gpu_memory_utilization | 0.90 |
| 极限吞吐 | ~877 tok/s |
| 1 并发 TTFT | ~42ms |
| 16 并发 TTFT | ~100ms |
| 显存峰值 | 28.86 GB |
| FlashInfer | Blackwell SM12.x 不兼容，VLLM_USE_FLASHINFER_SAMPLER=0 |
| KV Cache per token | ~128 KB |

### 高频面试问答

**Q: vLLM 为什么比 HF generate 快？**
A: PagedAttention（显存利用率 +4x）+ Continuous Batching（调度优化 +3x）+ CUDA graph/kernel 优化（+30-50%）→ 总吞吐提升 10-20x。

**Q: PagedAttention 的 block 是什么？**
A: 固定大小（16 token）的 KV Cache 存储单元。类比 OS 内存页。通过 block table 做逻辑到物理的映射。

**Q: RAG 接入后为什么变慢？**
A: prompt 长度膨胀（200→2000 token），prefill 的 O(N²) 导致 TTFT 增长 7x。优化：精简检索文档、prompt 压缩、prefix cache。

**Q: TTFT 和 TPOT 有什么区别？**
A: TTFT 是等第一个 token 的时间（prefill + queueing），决定"响应快不快"。TPOT 是后续每个 token 的时间（decode），决定"生成流畅不流畅"。总延迟 = TTFT + (N-1) × TPOT。

**Q: 怎么压测？**
A: asyncio + httpx 并发发流式请求，统计 TTFT、TPOT、tokens/s、p50/p95/p99。找饱和并发点（吞吐不再随并发增长的点）。本项目饱和并发=16。
