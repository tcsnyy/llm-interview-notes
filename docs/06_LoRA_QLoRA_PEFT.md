# 06 — LoRA / QLoRA / PEFT 面试八股

---

## 面试 1 分钟 / 3 分钟回答版本

### 1 分钟版（电梯 pitch）

LoRA（Low-Rank Adaptation）是一种参数高效微调方法，核心思想是在预训练权重矩阵 W 旁边并联两个低秩矩阵 A 和 B，训练时只更新 A/B，不更新原始 W。前向计算为 h = Wx + (alpha/r) * BAx。QLoRA 在 LoRA 基础上引入 4-bit NormalFloat 量化、双重量化和 Paged Optimizer，使得在单张 RTX 5090（32GB）上就能微调 Qwen3-8B。训练完成后执行 merge_and_unload 将 BA 合并回 W 的精度，得到标准模型权重。在我的医学项目中，SFT 使用 rank=16、alpha=32，target_modules 覆盖了所有 attention 线性层和 MLP 的 gate/up/down proj，约 5.4M 可训练参数，仅占总参数的 0.07%。

### 3 分钟版（含项目细节）

LoRA 出自微软 2021 年的论文，核心假设是模型微调时权重更新矩阵 ΔW 是低秩的，因此可以用两个低秩矩阵 A(d×r) 和 B(r×k) 的乘积来近似。前向传播公式为 h = Wx + (alpha/r)·BAx，其中 alpha 是缩放因子（通常取 rank 的 1~2 倍），scaling = alpha/r 用于补偿初始化带来的梯度变化。

在我的医学 LLM Teacher 项目中，基于 Qwen3-8B 做 QLoRA SFT。配置为：rank=16、alpha=32、dropout=0.05，target_modules 覆盖 7 类线性层——q_proj、k_proj、v_proj、o_proj（attention 部分）以及 gate_proj、up_proj、down_proj（MLP 部分）。之所以全加，是因为 MLP 存储了大量世界知识，在医学微调中不应被忽略。QLoRA 使用的 4bit NormalFloat 量化对正态分布的权重信息损失最小，Double Quantization 将量化常数再量化一次，Paged Optimizer 用统一内存避免 OOM。最终可训练参数约占总参数 0.07%，单卡 RTX 5090 32GB 即可完成训练。

DPO 阶段直接从 SFT LoRA checkpoint 加载继续训练，生成新的 adapter 权重。项目使用 sigmoid loss type，beta=0.1。训练完成后使用 merge_and_unload 将 LoRA adapter 合并回 base model 的 BF16 精度，保存为 safetensors 格式供推理使用。

---

### Q: PEFT 是什么？LoRA 的核心思想与公式是什么？⭐⭐⭐⭐⭐

![LoRA 低秩分解示意图](images/LoRA%E4%BD%8E%E7%A7%A9%E5%88%86%E8%A7%A3%E7%A4%BA%E6%84%8F%E5%9B%BE.png)

PEFT（Parameter-Efficient Fine-Tuning）是一类在微调大模型时**只更新极少量参数**的方法总称。与 Full Fine-Tuning（全部参数参与梯度计算与更新）不同，PEFT 通过注入可训练的 adapter 模块、prefix embeddings、或低秩分解矩阵，在保持模型主体冻结的前提下实现任务适配。

PEFT 家族的典型代表：

| 方法 | 核心思路 | 推理时开销 |
|------|----------|------------|
| **LoRA** | 在权重矩阵旁并联低秩分解矩阵 BA | 可合并为 0 |
| **QLoRA** | LoRA + 4bit 量化 base model | 可合并为 0 |
| Adapter | 在 Transformer 层间插入小瓶颈网络 | 有额外计算 |
| Prefix Tuning | 在输入前拼接可学习前缀向量 | 占用 seq len |
| Prompt Tuning | 学习软 prompt embedding | 占用 seq len |
| IA³ | 学习 rescaling vector | 极小 |

LoRA 是当前生态最成熟、通用性最强、工业落地最广的 PEFT 方法。

**LoRA 核心公式**：核心思想来自一个经验观察：**预训练大模型在适配下游任务时，权重更新矩阵 ΔW 是低秩的**。

原始全量微调的前向传播：

```
h = W·x    （W 冻结，微调时实际计算 h = (W + ΔW)·x）
```

LoRA 将 ΔW 分解为两个低秩矩阵的乘积：

```
ΔW = B · A

其中：
  A: d × r   (d 为输入维度, r 为 rank)
  B: r × k   (k 为输出维度)
```

因此 LoRA 前向传播为：

```
h = W·x + (α/r) · B·A·x

         ┌──────────┐
    x ──▶│    W     │─────────▶ (+)──▶ h
         │ (frozen) │            ▲
         └──────────┘            │
         ┌────┐  ┌────┐         │
    x ──▶│ A  │─▶│ B  │─────────┘
         │d×r │  │r×k │   ×(α/r)
         └────┘  └────┘
```

缩放因子 `α/r` 的作用：当 rank r 变化时，α 固定可以让学习率对 rank 不敏感。常见做法是取 α = 2r（如 r=16 则 α=32），此时 scaling = 2。

**为什么低秩假设成立**：

1. **Intrinsic Dimension 理论**：大模型参数虽多，但在特定任务上的有效自由度（intrinsic dimension）远低于参数总量，LoRA 的 rank r 正是对这个本征维度的近似。
2. **Aghajanyan et al. (2020)** 研究表明，预训练语言模型在 fine-tuning 时，参数更新的本征维度可以小到几百维。
3. 在实践中，rank=8~64 通常就能获得接近全量微调的效果。

---

### Q: LoRA 的矩阵维度怎么定义？参数量怎么算？rank、alpha、scaling 的关系是什么？⭐⭐⭐⭐

**维度定义**：以 Qwen3-8B 的一个 attention 线性层为例（假设 hidden_size=4096, r=16）：

```
q_proj:  W ∈ R^(4096 × 4096)   # 权重矩阵
         A ∈ R^(4096 × 16)      # LoRA A 矩阵：dowm-projection
         B ∈ R^(16 × 4096)      # LoRA B 矩阵：up-projection
```

A 矩阵将 4096 维输入降维到 16 维，B 矩阵将 16 维升回 4096 维，两者的乘积 B·A 是一个 4096×4096 的矩阵，恰好可以加到原始 W 上。

**参数量计算**：

单个 LoRA 模块（一个 target linear）的可训练参数：

```
params_one_module = d × r + r × k
                  = 4096 × 16 + 16 × 4096
                  = 131,072
```

其中 A 矩阵 d × r = 65,536 参数，B 矩阵 r × k = 65,536 参数，总计约 0.13M / 模块。

在 Qwen3-8B 中，总参数量约 8B，每个 Transformer 层有 7 个 target linear（q_proj、k_proj、v_proj、o_proj、gate_proj、up_proj、down_proj），共 32~36 层。实际可训练参数约 5.4M —— 占总参数 0.07%。之所以实际只有约 5.4M 而非 30M，是因为 Qwen3 的 attention 使用了 GQA（Grouped Query Attention），k_proj 和 v_proj 的维度可能小于 q_proj。

**rank、alpha、dropout、scaling 的关系**：

| 超参数 | 含义 | 项目取值 |
|--------|------|----------|
| `r` (rank) | 低秩维度，决定 adapter 的表达能力 | 16 |
| `alpha` | 缩放因子，控制 adapter 输出的幅度 | 32 |
| `lora_dropout` | adapter 层的 dropout，正则化 | 0.05 |
| `scaling` | α/r，实际应用于 BAx 前的系数 | 2.0 |

rank 越大表达能力越强但参数越多，医学任务需要一定容量来记住术语和诊疗逻辑。alpha=32, r=16 时 scaling=2，让 LoRA 输出的梯度与原始权重梯度在同一数量级。scaling 让不同 rank 的超参互相独立，调 rank 不调 lr。

在 PEFT 库中，前向传播实际计算：

```python
result = (self.lora_B(self.lora_A(self.dropout(x)))) * self.scaling[self.active_adapter]
```

scaling 放在 B 之后还是 A 之后本质等价（线性变换的可交换性），但 PEFT 实现中放在 B·A 整体之后。

---

### Q: 为什么 LoRA 的 B 矩阵要初始化为 0？⭐⭐⭐

LoRA 论文明确要求：**A 用 Kaiming 均匀 / 高斯分布随机初始化，B 初始化为全零**。

数学直觉：

```
初始状态：B = 0  →  BA = 0  →  h = Wx + 0 = Wx
```

在训练的第一步，LoRA adapter 的输出为 0，模型的输出完全等于预训练模型的输出。这保证了 **训练从原始模型的能力基线开始**，不会因为随机初始化的 adapter 引入噪声而破坏预训练权重已有的能力。

如果 B 也随机初始化，BA 不为零，相当于给 W 加了一个随机扰动，可能在训练初期导致 loss 跳变甚至模型"忘记"已学会的知识（catastrophic forgetting 在微调初期的微观表现）。

**对称性破缺**：

```
A 随机初始化 + B 零初始化 → 前向传播时 BA=0（不改变输出）
                         → 反向传播时 B 的梯度 ≠ 0（因为 A 随机，loss 通过 B 回传时有非零输入）
                         → B 从 0 开始被优化，A 从随机开始被优化
                         → 两者对称性被打破，不会出现 A=B 的退化情况
```

如果 A 和 B 都初始化为 0，则前向/反向都为零，永远无法训练。如果都用随机初始化，虽然也能训练，但起点不在预训练基线上，效果通常不如 B=0 的方案。

> **核心总结**："B 初始化为零保证训练初始时刻 LoRA adapter 的输出为零，模型行为等于预训练模型，不会引入随机扰动。同时 A 的随机初始化确保反向传播时 B 有非零梯度，即训练的起点是基座模型的能力水平，终点是任务适配能力，中间是平滑过渡。"

---

### Q: LoRA 应该加在哪些层？Attention 和 MLP 怎么选？⭐⭐⭐⭐

在医学 LLM Teacher 项目中，LoRA 加在了 **所有线性层**：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",   # Query 投影
        "k_proj",   # Key 投影
        "v_proj",   # Value 投影
        "o_proj",   # Output 投影
        "gate_proj",# MLP 门控
        "up_proj",  # MLP 上投影
        "down_proj",# MLP 下投影
    ],
)
```

各模块的位置与作用：

```
Transformer Block
├── Attention
│   ├── q_proj:  Query = q_proj(hidden)    → 计算注意力 query
│   ├── k_proj:  Key = k_proj(hidden)      → 计算注意力 key
│   ├── v_proj:  Value = v_proj(hidden)    → 计算注意力 value
│   ├── o_proj:  Output = o_proj(attn_out) → 注意力输出投影
│
└── MLP (SwiGLU in Qwen3)
    ├── gate_proj:  gate = gate_proj(hidden) → SwiGLU 的 sigmoid 门
    ├── up_proj:    up = up_proj(hidden)     → 非线性变换
    └── down_proj:  output = down_proj(gate * up) → 降回 hidden_size
```

**只加 Attention（保守方案）**：参数量更少（4 个模块 vs 7 个模块），训练更快、显存更省，适合只需要调整"注意力模式"的任务（对话风格、格式遵循）。但 MLP 存储了大量事实性/世界知识，不调 MLP 可能导致新领域知识注入不足。

**加 Attention + MLP（激进方案，项目采用）**：覆盖模型全部前向路径，表达能力最强。MLP 的 gate_proj/up_proj/down_proj 是知识存储的关键层，在医学领域注入疾病、药物、诊疗逻辑等知识时不可忽略。实际经验表明，医疗 SFT 中不加 MLP 的 LoRA 在事实性问答上准确率明显下降。代价是参数量增加约 75%，但 QLoRA 下仅增加几百 MB 显存，换来了更完整的知识注入能力。

> **核心总结**："在对话式任务或格式调整中只加 attention 就够了，但在医学微调中，大量新术语、药物名称、诊断标准需要被模型'记住'——这些事实性信息主要存储在 MLP 的前馈权重中。所以我选择对 attention 和 MLP 都加 LoRA。"

---

### Q: Rank 怎么选？医疗 SFT 场景为什么用 r=16？⭐⭐⭐

通用选 rank 原则：

| rank | 参数量级 | 适用场景 |
|------|----------|----------|
| 4~8 | 极少 | 简单分类、情感分析、格式调整 |
| 16~32 | 适中 | 对话微调、指令遵循、领域适配 |
| 64~128 | 较多 | 多任务复杂微调、接近全量微调效果 |

医疗场景为什么选 r=16：

1. **医学知识密度高**：一个 prompt 中可能包含疾病名、药物名、检查指标、诊疗指南，需要足够的参数容量来编码这些新知识
2. **数据量中等**：项目有约 10K 条 SFT 数据，rank 太小会欠拟合，rank 太大会过拟合
3. **经验法则**：r 应小于有效数据量的开方。10K 数据 sqrt ≈ 100，r=16 在安全范围内
4. **QLoRA 显存约束**：r=16 在 8B 模型上约 5~30M 可训练参数，QLoRA 下额外显存 < 1GB

r 的选择是表达能力和泛化性的权衡。医学 SFT 需要模型学习大量专业术语和诊疗逻辑——这些信息密度远高于通用对话，所以不能用 r=4。同时项目有 ~10K 条数据，r=64 可能过拟合。r=16 配合 alpha=32（scaling=2）在实验中表现稳定。

**注意**：
- **LoRA+ 论文**指出更多 rank 不一定更好，反而可能引入噪声
- **rsLoRA** 提出 scaling 应该用 α/√r 而非 α/r，在较大 rank 时更稳定
- 实际项目中可以通过在验证集上观察 loss 和指标曲线来决定

---

### Q: QLoRA 的四大技术组件是什么？显存怎么分解的？⭐⭐⭐⭐⭐

QLoRA（Quantized LoRA）出自 Dettmers et al. 2023，核心贡献是让普通研究者也能在消费级 GPU 上微调大模型。论文中在单张 48GB GPU 上微调了 65B 的 Guanaco 模型。

**1. 4-bit NormalFloat (NF4)**：假设权重服从零均值正态分布，在正态分布的高概率区域分配更多量化值，在尾部区域分配更少量化值，信息论上最优。NF4 的 16 个量化值（非均匀分布）：

```
[-1.0, -0.696192800, -0.525073051, -0.394917488,
 -0.284441381, -0.184773430, -0.091050036, 0.0,
  0.079580299, 0.160930202, 0.246112301, 0.337915241,
  0.440709829, 0.562617003, 0.722956836, 1.0]
```

每个量化值在标准正态分布下覆盖等概率区间，最小化量化误差的期望。

**2. Double Quantization（双重量化）**：第一重量化将 FP32 → 4bit NF4（每 4bit 权重需要 1 个 32bit 量化常数）；第二重量化将 32bit 量化常数 → 8bit FP8。每个 4bit 参数额外节省约 0.373 bit 的存储开销，在 8B 模型上节省约 0.5GB 显存。

**3. Paged Optimizer（分页优化器）**：当 GPU 显存不足时，将优化器状态暂时移到 CPU RAM（统一内存），需要时再换回 GPU。本质是 CUDA 的统一虚拟寻址（UVA）—— GPU 和 CPU 共享地址空间。

**4. BF16 Compute Dtype（BF16 计算精度）**：权重存为 4bit NF4，量化常数存为 8bit，但 LoRA 权重和中间激活使用 BF16。前向传播时将 4bit NF4 反量化为 BF16，在 BF16 精度下计算 forward/backward，只更新 LoRA adapter（始终在 BF16）。BF16 相比 FP16 指数位更多（8 vs 5），动态范围更大（约 10^38 vs 10^4），不会出现 FP16 常见的梯度上溢问题。

**QLoRA 显存分解（以 Qwen3-8B + r=16 为例）**：

```
组件                    精度        显存
──────────────────────────────────────────
Base Model Weights      4bit NF4    ~4 GB
量化常数                8bit FP8    ~0.5 GB
LoRA Adapter (可训练)   BF16        ~0.02 GB （5.4M × 2 bytes）
梯度 (仅 LoRA)          BF16        ~0.02 GB
Optimizer States        32bit FP32  ~0.06 GB (AdamW: m + v，各 5.4M × 4)
激活值                  BF16        ~5-8 GB (取决于 batch/seq_len)
──────────────────────────────────────────
总计                                 ~12-15 GB
→ 单卡 RTX 5090 32GB 绰绰有余
```

---

### Q: 4bit 训练和 4bit 部署有什么区别？⭐⭐⭐

这是面试中非常容易被追问的概念辨析。

| 维度 | 4bit 训练 (QLoRA) | 4bit 部署 (GPTQ/AWQ) |
|------|-------------------|----------------------|
| **目的** | 降低训练显存 | 降低推理显存 + 加速推理 |
| **权重状态** | 部分可训（LoRA），base 冻结 | 全部冻结 |
| **计算精度** | 反量化到 BF16 后计算 | 直接 INT4 计算（或反量化为 FP16） |
| **量化方法** | NF4（非均匀，正态分布假设） | GPTQ（均匀 INT4，逐层校准）、AWQ（激活感知） |
| **反量化开销** | 有（每次 forward 反量化） | 有但可融合进 kernel |
| **代表性工具** | bitsandbytes + PEFT | AutoGPTQ、vLLM、TGI |
| **最终产物** | 合并后的 BF16 模型（用于推理） | INT4 模型权重 |

> **核心总结**："QLoRA 的 4bit 是为了训练——冻结的 base model 权重压缩到 4bit 减少显存，但计算仍在 BF16；训练完 merge_and_unload 后得到的是 BF16 精度的完整模型。推理时的 4bit 部署（如 GPTQ/AWQ）是为了降低推理延迟和显存——权重永久量化到 INT4 且直接在这上面计算。两者目标不同、量化方法不同、前后处理不同。我们项目训练用 QLoRA，部署用合并后的 BF16 模型——因为医疗场景对精度要求极高，不敢在推理侧做 4bit 量化造成信息损失。"

---

### Q: merge_and_unload 的底层逻辑是什么？合并后得到什么文件？⭐⭐⭐⭐

**合并公式**：

```python
# 对每个加了 LoRA 的 linear 层：
W_merged = W_base + (alpha / r) * B @ A

# 其中 W_base 需要在合并前反量化为 BF16（如果 QLoRA 训练）
```

**合并的本质**：计算 ΔW = B·A（将 A(d×r) 和 B(r×k) 矩阵相乘得到与原始 W 同维度的 ΔW），乘以 scaling = α/r，矩阵加法 W_base + scaled ΔW，替换层的权重，删除 adapter 层（移除 LoRA 的前向 hook/forward 方法，恢复为标准 nn.Linear）。

**合并后的权重精度**：训练时 base weight 是 4bit NF4（存储）→ 反量化为 BF16（计算），LoRA adapter 是 BF16，合并后的 W 是 BF16（完整精度）。合并后保存的是 BF16 或 FP32 权重，不再有量化信息。这意味着合并后的模型和用全量 BF16 SFT 得到的模型在格式上完全一致。

**为什么不直接保存 4bit + adapter 来做推理？**
1. 每次推理需要额外反量化 + BA 计算，增加 latency
2. 4bit + adapter 格式不是所有推理框架都支持
3. 合并后的 BF16 模型可以直接用 vLLM/TGI/transformers 标准加载
4. 医疗场景对推理精度要求高，BF16 比 4bit 更可靠

**合并后的文件清单**（标准 HuggingFace transformer 模型）：

```
my_lora_merged_model/
├── config.json                  # 模型架构配置（来自 base model）
├── generation_config.json       # 生成配置（来自 base model）
├── model-00001-of-0000X.safetensors  # 合并后的权重（分片）
├── model-00002-of-0000X.safetensors
├── ...
├── model.safetensors.index.json # 权重索引（多分片时的映射）
├── tokenizer.json               # Tokenizer 词表
├── tokenizer_config.json        # Tokenizer 配置
├── special_tokens_map.json      # 特殊 token 映射
└── vocab.json / merges.txt      # BPE 词表和合并规则（如适用）
```

**注意**：合并后的模型**不包含** `adapter_config.json`（因为 adapter 已被合并，不再是 PEFT 格式）和 `adapter_model.safetensors`。

---

### Q: safetensors、adapter_config.json、adapter_model.safetensors 分别是什么？⭐⭐⭐

**safetensors**：HuggingFace 开发的张量序列化格式，替代 .bin / .pt。优势：安全（不执行任意 Python 代码，避免 pickle 安全漏洞）、快速（支持零拷贝加载 mmap）、懒加载（只加载实际用到的 tensor）、跨框架（PyTorch、TensorFlow、JAX 都可读写）。格式为 `[header_size: 8 bytes][JSON header: N bytes][tensor data: M bytes]`。

**adapter_config.json**：PEFT adapter 的元数据配置文件，记录 LoRA 的所有超参数：

```json
{
  "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
  "bias": "none",
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "r": 16,
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ],
  "task_type": "CAUSAL_LM",
  "peft_type": "LORA",
  "fan_in_fan_out": false,
  "inference_mode": true
}
```

用途：加载 adapter 时 PEFT 库根据此配置重建 LoRA 模块结构，合并时根据此配置知道哪些层需要合并，迁移 adapter 到不同 base model 时需要匹配。

**adapter_model.safetensors**：只包含 LoRA adapter 的权重张量。内容示例（Qwen3-8B, r=16, 7 target modules, 32 layers）包含每层每个 target module 的 lora_A.weight 和 lora_B.weight。文件大小估算：Qwen3-8B 中 target_modules 覆盖约 65-75% 的前向路径，以 70% 计，LoRA 参数 ≈ 7.6B × 0.7 × (2r/d)，对于 r=16, d=4096 约 40M 参数，40M × 2 bytes (BF16) ≈ 80 MB。

---

### Q: 如何验证 LoRA 合并成功？⭐⭐⭐

三种方法：

**方法一：参数比较法**——逐参数验证合并前后差异，理论上只有 target_modules 有变化。

**方法二：推理验证法**——用同一 prompt 分别在 base_model（no adapter）、peft_model（with adapter, not merged）、merged_model 三种模式下推理，output_peft 和 output_merged 应该完全一致。

**方法三：权重直接验证**——手动计算 W_merged = W_base + scaling * B @ A 并与实际合并后的权重对比（项目中通常在合并后用第三种方法抽样验证前几层）。

```python
# 方法三核心代码
W_expected = W_base + scaling * (lora_B @ lora_A)
W_actual = merged_model.get_parameter(f"{module_name}.weight")
assert torch.allclose(W_expected, W_actual, atol=1e-5)
```

---

### Q: DPO 继续从 SFT LoRA checkpoint 训练要注意什么？⭐⭐⭐

**1. 加载 SFT adapter 而非 base model**：必须保持 peft model 格式（base frozen + adapter trainable），如果调用了 merge_and_unload() 则无法继续 LoRA 训练。

```python
# 正确：从 SFT LoRA checkpoint 继续训练
model = PeftModel.from_pretrained(model, sft_lora_path, is_trainable=True)
```

**2. DPO 的 reference model 加载**：reference model 应该是 SFT 模型（冻结），policy model 在此基础上训练。ref_model 可以 merge_and_unload 后冻结省显存，或保持 QLoRA 格式但 requires_grad=False。

**3. 不要重新初始化 LoRA**：直接加载 SFT adapter 并设为可训练，不要在 SFT checkpoint 上又加一个新 LoRA。

**4. 学习率与 beta 调整**：SFT 阶段 lr 通常较高（如 2e-4），DPO 阶段 lr 应降低（如 5e-5 或 1e-5），因为 adapter 已有基础且 DPO 的 loss landscape 更陡峭。beta 通常取 0.1~0.5。

**5. 保存 DPO checkpoint**：训练结束后保存新的 adapter（adapter_config.json + adapter_model.safetensors + trainer_state.json），这是 DPO 训练后在 SFT adapter 基础上更新过的 LoRA 权重。

---

### Q: 单卡 RTX 5090 为什么适合 QLoRA？⭐⭐

RTX 5090 的规格：

| 规格 | 数值 | 意义 |
|------|------|------|
| 显存 | 32GB GDDR7 | 足够容纳 8B 模型 4bit + LoRA + 激活值 |
| 带宽 | ~1.5 TB/s | 反量化 + 前向传播不构成瓶颈 |
| BF16 吞吐 | 高（Tensor Cores 5th gen） | BF16 计算密集的 LoRA 前向/反向高效 |
| FP4 支持 | 原生支持 | 未来可能直接用硬件 4bit 加速量化训练 |

与 A100/H100 的对比：

| 维度 | RTX 5090 (32GB) | A100 (80GB) | H100 (80GB) |
|------|-----------------|-------------|-------------|
| 价格 | ~$1,600 | ~$10,000+ | ~$25,000+ |
| 显存 | 32GB | 80GB | 80GB |
| 全量微调 8B | 困难 | 轻松 | 轻松 |
| QLoRA 微调 8B | 完美 | 轻松 | 轻松 |

> **核心总结**："RTX 5090 32GB 是 QLoRA 的'甜点'卡。8B 模型 4bit 量化后约 4-5GB，加上 ~5M LoRA 参数、优化器状态、激活值，总计约 12-15GB，远低于 32G 上限。4090 到 5090 的最大变化是显存从 24G 升到 32G，这多出来的 8G 刚好让 8B 模型的 QLoRA 微调从'勉强能跑'变成'运行舒适区'。"

---

## 代码实现示例

### 示例 1：LoRA Linear 前向传播

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    LoRA 线性层的简化实现。
    对应 transformers 中 q_proj / k_proj / v_proj / o_proj 等。

    维度说明：
        in_features  = 4096   # hidden_size
        out_features = 4096   # hidden_size (attention 输出维度可能不同)
        r            = 16     # rank
    """

    def __init__(
        self,
        in_features: int,   # 输入维度，例如 4096
        out_features: int,  # 输出维度，例如 4096
        r: int = 16,        # LoRA rank
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        # --- 原始权重（冻结，不更新梯度）---
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # 假设已经加载了预训练权重
        self.linear.weight.requires_grad = False

        # --- LoRA A 矩阵：in × r，降维 ---
        # 形状: [r, in_features]  —— PEFT 库习惯存为 [r, in]
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        # 随机初始化 A（Kaiming uniform）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # --- LoRA B 矩阵：r × out，升维 ---
        # 形状: [out_features, r]  —— PEFT 库习惯存为 [out, r]
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        # B 初始化为 0 —— 保证训练开始时 BAx = 0

        # --- 缩放因子 ---
        self.scaling = lora_alpha / r  # 32 / 16 = 2.0

        # --- Dropout ---
        self.lora_dropout = nn.Dropout(p=lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, in_features]
               e.g. [4, 2048, 4096]

        Returns:
            output: [batch_size, seq_len, out_features]
                    e.g. [4, 2048, 4096]
        """
        # --- 原始前向（冻结权重）---
        result = self.linear(x)  # [B, S, out_features]

        # --- LoRA 前向 ---
        # x: [B, S, in] → lora_A: [r, in] → [B, S, r] → lora_B: [out, r] → [B, S, out]
        lora_out = self.lora_dropout(x)           # dropout first
        lora_out = F.linear(lora_out, self.lora_A)   # x @ A^T = [B, S, in] @ [in, r] → [B, S, r]
        lora_out = F.linear(lora_out, self.lora_B)   # [B, S, r] @ [r, out] → [B, S, out]
        lora_out = lora_out * self.scaling           # × (alpha / r)

        # --- 合并输出 ---
        return result + lora_out


# ============================================================
# 维度追踪示例
# ============================================================
def trace_dimensions():
    batch_size, seq_len, hidden_size = 4, 2048, 4096
    r = 16

    x = torch.randn(batch_size, seq_len, hidden_size)
    layer = LoRALinear(hidden_size, hidden_size, r=r)

    output = layer(x)
    print(f"输入:  {x.shape}")       # [4, 2048, 4096]
    print(f"输出:  {output.shape}")  # [4, 2048, 4096]
    print(f"lora_A: {layer.lora_A.shape}")  # [16, 4096]
    print(f"lora_B: {layer.lora_B.shape}")  # [4096, 16]
    print(f"scaling: {layer.scaling}")       # 2.0

    # 梯度验证：B 初始为 0，前向时 LoRA 贡献为 0
    with torch.no_grad():
        base_out = layer.linear(x)
        assert torch.allclose(output, base_out), \
            "B=0 初始状态应保证 LoRA 输出为 0！"


if __name__ == "__main__":
    trace_dimensions()
```

### 示例 2：LoRA 权重合并（merge）

```python
import torch
import torch.nn as nn


def merge_lora_weights(
    linear: nn.Linear,          # 原始线性层（含预训练权重）
    lora_A: torch.Tensor,       # LoRA A 矩阵: [r, in_features]
    lora_B: torch.Tensor,       # LoRA B 矩阵: [out_features, r]
    scaling: float,             # alpha / r
) -> nn.Linear:
    """
    将 LoRA adapter 权重合并到原始 nn.Linear 层中。

    数学公式：
        W_merged = W_base + scaling * (B @ A)

    参数维度：
        W_base:    [out_features, in_features]  e.g. [4096, 4096]
        lora_A:    [r, in_features]             e.g. [16, 4096]
        lora_B:    [out_features, r]            e.g. [4096, 16]
        B @ A:     [out_features, in_features]  与 W_base 同维度
        scaling:   float                         e.g. 2.0
    """
    # 1. 计算 ΔW
    delta_W = lora_B @ lora_A  # [out, r] @ [r, in] → [out, in]

    # 2. 合并
    with torch.no_grad():
        linear.weight.data = linear.weight.data + scaling * delta_W

    # 3. 此时 linear 不再需要 LoRA 分支 — 变成了标准 nn.Linear
    return linear


# ============================================================
# 使用示例
# ============================================================
def demo_merge():
    in_features, out_features, r = 4096, 4096, 16
    alpha = 32
    scaling = alpha / r  # 2.0

    # 模拟 pretrained weight
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = torch.randn(out_features, in_features) * 0.02

    # 模拟 LoRA 训练后的 adapter（实际来自 checkpoint）
    lora_A = torch.randn(r, in_features) * 0.02   # A 是训练出来的
    lora_B = torch.randn(out_features, r) * 0.02   # B 是训练出来的

    # 保存合并前的权重用于验证
    W_before = linear.weight.data.clone()

    # 合并
    linear = merge_lora_weights(linear, lora_A, lora_B, scaling)

    # 验证：合并后的权重应该等于 W_before + scaling * B @ A
    W_expected = W_before + scaling * (lora_B @ lora_A)
    assert torch.allclose(linear.weight.data, W_expected, atol=1e-6), \
        "merge 计算错误！"
    print("✓ 合并验证通过")


if __name__ == "__main__":
    demo_merge()
```

### 示例 3：PEFT 官方 merge_and_unload

```python
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def peft_merge_and_unload_demo():
    """
    PEFT 官方库的 merge_and_unload 使用流程。
    这是实际项目中最常用的合并方式。
    """

    # ============ 步骤 1：加载 Base Model（QLoRA 场景） ============
    from transformers import BitsAndBytesConfig

    # QLoRA 量化配置（与训练时一致）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 4bit 加载
        bnb_4bit_compute_dtype=torch.bfloat16, # 计算精度 BF16
        bnb_4bit_use_double_quant=True,        # 双重量化
        bnb_4bit_quant_type="nf4",            # NormalFloat4
    )

    base_model_name = "Qwen/Qwen2.5-7B-Instruct"  # 项目的 base model
    lora_checkpoint = "./output/lora_sft/checkpoint-1000"

    # 加载 base model（4bit 量化）
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ============ 步骤 2：加载 LoRA Adapter ============
    model = PeftModel.from_pretrained(model, lora_checkpoint)
    # 此时 model 是 PeftModelForCausalLM，内部 7 类 target_modules 各有 A/B 矩阵

    print(f"合并前模型类型: {type(model)}")
    # 输出: <class 'peft.peft_model.PeftModelForCausalLM'>

    # ============ 步骤 3：合并并卸载 ============
    model = model.merge_and_unload()
    # merge_and_unload 内部做的事：
    #   1. 遍历所有 LoRA 层
    #   2. 将 4bit 权重反量化为 BF16
    #   3. 计算 W_new = W_base + scaling * (B @ A)
    #   4. 替换层的权重复制为 W_new
    #   5. 删除 lora_A / lora_B / lora_dropout 等参数
    #   6. 返回标准 transformers 模型对象

    print(f"合并后模型类型: {type(model)}")
    # 输出: <class 'transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM'>

    # ============ 步骤 4：保存合并后的模型 ============
    save_path = "./output/lora_sft_merged"
    model.save_pretrained(save_path, safe_serialization=True)
    # → 保存为 model-00001-of-0000X.safetensors

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(save_path)
    # → 保存 tokenizer.json, tokenizer_config.json, special_tokens_map.json 等

    # ============ 步骤 5：加载合并后的模型（验证） ============
    loaded = AutoModelForCausalLM.from_pretrained(
        save_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # 此时 loaded 是标准模型，可以直接推理
    print("✓ merge_and_unload + save + reload 完成")


if __name__ == "__main__":
    # 伪代码演示，实际运行需要模型和 checkpoint
    print("PEFT merge_and_unload 流程演示")
    print("=" * 60)
    peft_merge_and_unload_demo()
```

### 示例 4：QLoRA 加载模型

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def qlora_load_model_example():
    """
    医学 LLM Teacher 项目中 QLoRA 加载模型的标准流程。
    """

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # ============ Step 1: 4bit 量化配置 ============
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                    # 核心：4bit 加载
        bnb_4bit_compute_dtype=torch.bfloat16, # 前向/反向计算用 BF16
        bnb_4bit_use_double_quant=True,        # 双重量化 —— 节省约 0.5GB
        bnb_4bit_quant_type="nf4",            # NormalFloat4 —— 对正态分布最优
        # 以下为可选参数：
        # llm_int8_threshold=6.0,             # 只有 8bit 量化时才需要
        # llm_int8_has_fp16_weight=False,
    )

    # ============ Step 2: 加载 4bit 量化模型 ============
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",                    # 自动分配层到 GPU/CPU
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,               # Qwen 系列需要
    )

    # ============ Step 3: 准备模型用于 k-bit 训练 ============
    # prepare_model_for_kbit_training 做了什么：
    #   1. 冻结所有非 LoRA 的参数（requires_grad = False）
    #   2. 将 LayerNorm / RMSNorm 转为 FP32（稳定性）
    #   3. 对梯度检查点做适配
    model = prepare_model_for_kbit_training(model)

    # 验证：base model 参数是否冻结
    for name, param in model.named_parameters():
        if "lora" not in name:
            assert not param.requires_grad, \
                f"非 LoRA 参数 {name} 应该被冻结！"

    # ============ Step 4: LoRA 配置 ============
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,                                   # rank
        lora_alpha=32,                          # alpha（α/r = 2）
        lora_dropout=0.05,                      # dropout
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",      # Attention
            "gate_proj", "up_proj", "down_proj",          # MLP
        ],
        bias="none",                            # 不对 bias 做 LoRA
        # modules_to_save 可选：训练额外的"头"层
        # modules_to_save=["lm_head", "embed_tokens"],
    )

    # ============ Step 5: 注入 LoRA Adapter ============
    model = get_peft_model(model, lora_config)

    # 打印可训参数统计
    model.print_trainable_parameters()
    # 输出示例:
    # trainable params: 5,406,720 || all params: 7,615,733,760 || trainable%: 0.0710

    return model


def verify_4bit_loading(model):
    """
    验证模型确实以 4bit 加载。
    """
    import bitsandbytes as bnb

    # 检查量化状态
    for name, module in model.named_modules():
        if hasattr(module, "weight") and hasattr(module.weight, "quant_state"):
            print(f"[4bit] {name}: 量化类型={module.weight.quant_state}")
        elif isinstance(module, bnb.nn.Linear4bit):
            print(f"[4bit] {name}: bitsandbytes Linear4bit")

    # 手动计算 4bit 部分的显存
    total_4bit_bytes = 0
    total_bits = 0
    for name, param in model.named_parameters():
        if "lora" not in name and param.requires_grad == False:
            total_bits += param.numel() * 4  # 4bit per param
            total_4bit_bytes += param.numel() * 0.5  # 4 bit = 0.5 byte
    print(f"4bit 权重总大小: {total_4bit_bytes / 1e9:.2f} GB")


if __name__ == "__main__":
    print("QLoRA 加载模型流程")
    print("=" * 60)
    # model = qlora_load_model_example()  # 实际运行需要模型文件
    print("代码结构演示完成")
```

---

## 面试追问与回答策略

### Q: "LoRA rank 越大越好吗？"

不是。rank 增大意味着更多的可训练参数和更强的表达能力，但也更容易过拟合小数据集。同时更大的 rank 会让 ΔW 的实际秩增加，但预训练权重的 fine-tuning 更新本身是低秩的——rank 超过"真正的本征维度"后，额外容量带来的是噪声而非信号。在医学项目中，r=16 配合 10K 条 SFT 数据取得了良好效果。实践中建议从 r=8 起步，根据验证集指标逐步增大，通常在 r=32 到 r=64 之间会看到收益饱和。

### Q: "为什么不直接全量微调，要用 LoRA？"

核心原因是资源约束和实际收益的权衡。Qwen3-8B 全量微调用 AdamW 至少需要 ~60-80GB 显存（模型 16GB + 梯度 16GB + 优化器 32GB + 激活值），单卡 RTX 5090 32GB 根本跑不了。而 QLoRA 将可训练参数降低到 0.07%，只用 12-15GB 就能训练。除此之外，LoRA 还有个工程优势——可以保存多个任务 adapter（如内科、外科、放射科），推理时动态切换，而全量微调每个任务都需要一份完整模型。

### Q: "LoRA 和 QLoRA 有什么区别？"

LoRA 的 base model 保持原始精度（FP16/BF16），训练时 base model 不量化。QLoRA 在此基础上将 base model 量化到 4bit NF4，并引入双重量化和分页优化器来进一步节省显存。技术上 QLoRA 是 LoRA 的超集——如果量化位宽设为 16 bit，QLoRA 退化为 LoRA。效果上，QLoRA 论文表明在合理配置下，4bit QLoRA 可以达到与全量 BF16 LoRA 相当甚至更好的效果，因为 4bit 量化本身有一定的正则化作用。

### Q: "merge_and_unload 后模型的推理速度会变快吗？"

会比未合并前快。合并前每次 forward 需要：反量化 base weight → 计算 Wx → 计算 BAx → 相加。合并后只需：直接计算 W_merged x。省去了反量化和 LoRA 分支的计算开销。但相比直接部署 BF16 模型，合并后的速度是一样的——本质上合并后的模型就是一个标准 BF16 模型。

### Q: "如果 Qwen3-8B 用 rank=16 LoRA，哪些层的参数量最大？"

q_proj 和 o_proj 的 LoRA 参数最多（hidden_size × hidden_size 的两个投影，各 4096×16+16×4096 = 131K 参数/模块）。k_proj 和 v_proj 因为 GQA（Grouped Query Attention）的实现，投影维度通常小于 hidden_size（如 Qwen3-8B 中 key/value 的 head_dim 可能较小），参数略少。MLP 的 gate/up/down_proj 中，down_proj 的参数量和 q_proj 同量级。所以按参数量排序大致是：q_proj ≈ o_proj ≈ down_proj > up_proj > k_proj ≈ v_proj, gate_proj。

---

## 背诵版总结

```
┌─────────────────────────────────────────────────────────────┐
│                  LoRA / QLoRA 背诵版总结                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 【核心公式】h = Wx + (α/r)·BAx                              │
│                                                             │
│ 【三个关键设计】                                              │
│  1. ΔW = BA（低秩分解）：A∈R^{d×r}, B∈R^{r×k}               │
│  2. B 初始化为 0：训练从预训练基线开始, 不引入随机扰动           │
│  3. Scaling = α/r：让不同 rank 的学习率互相独立                 │
│                                                             │
│ 【QLoRA 四大技术】                                            │
│  1. NF4 量化 — 正态分布最优 4bit                               │
│  2. Double Quantization — 量化常数再压缩                        │
│  3. Paged Optimizer — 显存不够切 CPU                         │
│  4. BF16 Compute — 计算用高精度                               │
│                                                             │
│ 【项目配置（医学 LLM Teacher）】                                │
│  - Base: Qwen3-8B, QLoRA 4bit NF4                           │
│  - r=16, α=32, dropout=0.05, scaling=2                       │
│  - target_modules: q/k/v/o/gate/up/down (全部 7 类)          │
│  - 可训练参数: ~5.4M (占总参数 0.07%)                          │
│  - 单卡 RTX 5090 32GB                                        │
│                                                             │
│ 【merge_and_unload 本质】                                      │
│  W_merged = W_base + (α/r)·B·A → 保存为 BF16 safetensors      │
│  合并后 = 标准模型 = 可直接用 vLLM/TGI 推理                      │
│                                                             │
│ 【safetensors vs adapter 文件】                               │
│  adapter_model.safetensors → 只有 LoRA A/B 权重               │
│  model-000xx.safetensors   → 合并后的完整模型权重               │
│  adapter_config.json       → LoRA 超参数元数据                 │
│                                                             │
│ 【DoRA (Weight-Decomposed LoRA)】                             │
│  将权重分解为方向+幅度，只对方向做低秩更新                        │
│  效果接近全量微调，比标准 LoRA 更好。了解即可。                    │
│                                                             │
│ 【DPO 从 SFT LoRA 继续训练】                                   │
│  1. 加载 SFT adapter（is_trainable=True）                     │
│  2. 不要重新初始化 LoRA                                       │
│  3. lr 降低（1e-5），beta 取 0.1                              │
│  4. ref model = SFT 合并后模型（冻结）                         │
│                                                             │
│ 【参数量计算】                                                 │
│  单模块: d×r + r×k                                           │
│  全模型: Σ(target 层数 × 2dr) = 总参数 × (2r/d) × 覆盖率      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

*文档生成日期：2026-06-04*
*基于医学 LLM Teacher 项目实际配置编写（Qwen3-8B + QLoRA + r=16/α=32）*
