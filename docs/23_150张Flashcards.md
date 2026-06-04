# 23 150张Flashcards（180+张）

> 使用方式：打印或用 Anki 导入，每天早晚各过一遍。先看 Q 尝试回答，再看 A 对照。标记不会的卡片，反复巩固。

---

## 一、概念卡（40张）

### Card 001
**Q:** 什么是大模型（LLM）？
**A:** 大语言模型是通过海量文本数据预训练（自回归/自编码）获得的数十亿至万亿参数的神经网络模型，具备上下文学习、思维链推理等涌现能力。核心训练流程：预训练 → SFT → RLHF/DPO。

### Card 002
**Q:** Decoder-only 架构为什么成为主流？
**A:** 自回归生成天然适合语言建模；训练和推理统一（无需 Encoder）；扩展性好（scaling law 验证）；GPT 系列的成功证明了其有效性。LLaMA/Qwen/DeepSeek 等主流模型均采用 Decoder-only。

### Card 003
**Q:** 涌现能力（Emergent Abilities）是什么？举例。
**A:** 在小模型中不存在、模型规模超过某个阈值后突然出现的能力。典型例子：few-shot 学习、CoT 推理、多步数学推理、指令遵循。这些能力不是显式训练的，而是规模扩大后自然涌现的。

### Card 004
**Q:** Pre-training 和 Post-training 的区别？
**A:** Pre-training 在海量原始文本上做 next-token prediction 学习通用语言能力；Post-training 在高质量标注数据上做 SFT + RLHF/DPO 学习指令遵循和偏好对齐。pre-training 决定能力上限，post-training 决定可用性。

### Card 005
**Q:** Scaling Law 的核心结论是什么？
**A:** 模型性能（用 test loss 衡量）与参数量 N、数据量 D、计算量 C 呈幂律关系。KM Scaling Law：L(N, D) = E + A/N^α + B/D^β。Chinchilla 结论：同等计算预算下，模型参数和数据量应等比缩放。

### Card 006
**Q:** In-Context Learning 是什么？为什么不需要训练？
**A:** 在 prompt 中提供几个示例（demonstrations），模型不经任何参数更新就能学会执行新任务。本质是模型的 few-shot 泛化能力——attention 机制在上下文窗口中隐式地"学习"了示例中的模式。

### Card 007
**Q:** Chain-of-Thought (CoT) 推理是什么？
**A:** 让模型在给出最终答案前先生成一系列中间推理步骤的技术。对于数学、逻辑、多跳推理等复杂任务能显著提升准确率。可分 Zero-shot CoT（"Let's think step by step"）和 Few-shot CoT。

### Card 008
**Q:** 大模型幻觉（Hallucination）分哪几类？
**A:** 三类：事实幻觉（编造不存在的事实/数字）、忠实性幻觉（输出与输入矛盾）、逻辑幻觉（推理链条错误）。根源：训练数据的噪声、概率最大化的生成策略、知识存储的不完备性。

### Card 009
**Q:** Attention 机制的核心直觉是什么？
**A:** 让模型在处理每个 token 时，动态地为序列中的所有 token 分配不同的"关注度"权重，而不是像 RNN 那样只依赖固定长度的隐藏状态。Query 代表"我在找什么"，Key 代表"我能提供什么"，Value 代表"我实际是什么内容"。

### Card 010
**Q:** Pre-Norm vs Post-Norm 的核心区别和影响？
**A:** Pre-Norm（Normalization 在 sublayer 之前）：训练更稳定，梯度流更好，但可能限制子层表达能力。Post-Norm（在之后）：理论表达更强，但深层网络训练困难。LLaMA/Qwen 都用 Pre-Norm（RMSNorm）。

### Card 011
**Q:** RMSNorm 和 LayerNorm 的区别？
**A:** LayerNorm 做 mean centering + variance scaling：y = (x - μ)/σ * γ + β。RMSNorm 只做 variance scaling 不做 mean centering：y = x / RMS(x) * γ。RMSNorm 计算更快（少算均值），效果相当，被 LLaMA/Qwen 采用。

### Card 012
**Q:** SwiGLU 激活函数是什么？公式？
**A:** SwiGLU 是 Swish-gated Linear Unit：SwiGLU(x) = Swish(xW1 + b1) ⊗ (xW2 + b2)，其中 Swish(x) = x * σ(x)。相比 ReLU/GeLU，SwiGLU 在 LLaMA 等模型中表现更好，训练更稳定，但参数量增加（3 个权重矩阵）。

### Card 013
**Q:** GQA (Grouped Query Attention) 原理？
**A:** 将 KV head 数量减少为 Q head 数量的 1/n，多个 Q head 共享同一组 KV head。n=1 时退化为 MQA（Multi-Query Attention），n=Q_head_num 时退化为 MHA。GQA 在 MHA 的质量和 MQA 的速度之间取得平衡。Qwen3 使用 GQA。

### Card 014
**Q:** 什么是 KV Cache？为什么推理需要它？
**A:** 自回归生成时，每步都要对历史所有 token 做 attention。KV Cache 缓存之前所有时间步的 K 和 V 矩阵，避免重复计算。显存占用 = 2 × batch_size × num_layers × num_kv_heads × seq_len × head_dim × dtype_size。

### Card 015
**Q:** Flash Attention 快在哪里？
**A:** 三个核心技术：Kernel Fusion（将多个 CUDA kernel 融合为一个，减少 HBM 读写）、Tiling（将 attention 矩阵分块计算，避免存储完整的 n×n 矩阵）、Recomputation（反向传播时不存储 attention 矩阵，需要时重算）。综合可节省 10-20 倍显存带宽。

### Card 016
**Q:** BPE Tokenizer 的训练流程？
**A:** 1) 初始化字符级词表；2) 统计所有相邻 token pair 的频率；3) 合并频率最高的 pair 为新 token；4) 重复合并直到达到目标 vocab size。最终词表 = 字符 + 高频子词组合 + 完整高频词。

### Card 017
**Q:** SFT 的训练目标是什么？
**A:** 在给定 instruction/prompt 的条件下，最大化高质量 response 的似然：L_SFT = -E_{(x,y)} [Σ_t log π_θ(y_t | x, y_<t)]。实际实现中，只对 response 部分计算 loss，prompt 部分用 label mask (-100) 忽略。

### Card 018
**Q:** LoRA (Low-Rank Adaptation) 的核心思想？
**A:** 预训练权重矩阵 W 是满秩的，但微调时的更新量 ΔW 是低秩的。因此 ΔW 可以分解为两个低秩矩阵的乘积 ΔW = BA（A∈R^{r×d}, B∈R^{d×r}），大幅减少可训练参数（从 d² 降到 2dr）。

### Card 019
**Q:** QLoRA 节省显存的四个核心技术？
**A:** 1) NF4 量化（NormalFloat4，针对正态分布权重的最优 4bit 量化）；2) 双重量化（对量化常数再量化）；3) 分页优化器（利用 unified memory 处理梯度检查点时的内存尖峰）；4) LoRA（只训练低秩增量矩阵）。

### Card 020
**Q:** DPO (Direct Preference Optimization) 的核心思想？
**A:** 将 RLHF 中的 reward model 训练和 policy 优化两步合为一步，直接在偏好数据上优化 policy。利用 BT 模型的闭式解，将 reward 表示为 policy 和 reference 的 log ratio，从而直接优化 policy 的偏好目标。

### Card 021
**Q:** RLHF 的完整流程？
**A:** 阶段 1：SFT 获得初始 policy；阶段 2：训练 Reward Model（在偏好 pair 上用 BT 模型）；阶段 3：用 PPO 优化 policy（最大化 RM 打分的同时加 KL 惩罚约束）。DPO 将这个流程简化为一个阶段。

### Card 022
**Q:** Reward Hacking 是什么？
**A:** 模型学会利用 reward model 的漏洞获取高分，而非真正提升回答质量。例如：Reward Model 偏好长回答，模型就输出冗长的内容；Reward Model 偏好确定性强的回答，模型就在不确定时自信地编造（导致幻觉）。Goodhart's Law 的体现。

### Card 023
**Q:** RAG 的核心流程分哪几步？
**A:** Indexing（文档切分+向量化+索引构建）→ Retrieval（query 向量化+相似度检索 top-k）→ Augmentation（将检索到的文档拼入 prompt）→ Generation（LLM 基于增强后的 prompt 生成回答）。

### Card 024
**Q:** LLM-as-Judge 是什么？为什么需要？
**A:** 利用一个强大的 LLM（如 GPT-4）来评测另一个 LLM 的输出质量。因为人工评测成本高、速度慢，而自动指标（BLEU/ROUGE）与人类判断相关性低。LLM-as-Judge 在开放域生成评测中与人工判断的 Spearman 相关系数可达 0.7-0.8。

### Card 025
**Q:** vLLM 的 PagedAttention 原理？
**A:** 受操作系统虚拟内存管理的启发，将 KV cache 切分为固定大小的"page"（block），按需分配和回收，而非预先分配连续的显存。解决传统 KV cache 的内存碎片问题，可将显存利用率提升 2-4 倍。

### Card 026
**Q:** Continuous Batching 是什么？
**A:** vLLM 的调度策略：不等待整个 batch 中所有请求都完成才释放，而是哪个请求完成了就立即将其从 batch 中移除，并加入新的请求。相比 static batching（等整个 batch 完成），GPU 利用率更高。

### Card 027
**Q:** What is the difference between fp16 and bf16?
**A:** fp16: 1 sign + 5 exponent + 10 mantissa, range ≈ 6e-8 to 65504, needs loss scaling for large models. bf16: 1 sign + 8 exponent + 7 mantissa, same range as fp32 (≈ 1e-38 to 3e38), no loss scaling needed. bf16 is safer for large model training due to larger dynamic range.

### Card 028
**Q:** Gradient Accumulation 的原理？
**A:** 将一个大 batch 拆分为多个 micro-batch，每个 micro-batch 前向+反向计算梯度，累加（accumulate）多次后再执行一次 optimizer.step()。等价于大 batch 训练，但显存只需 micro-batch 量级。实际 batch_size = micro_batch_size × accumulation_steps。

### Card 029
**Q:** Gradient Checkpointing 的原理和代价？
**A:** 前向传播时不保存所有中间激活值，反向传播到该层时重新计算。用额外的计算换显存——省约 60% 激活值显存，但增加约 20-30% 的计算时间。对大模型训练几乎是标配。

### Card 030
**Q:** ZeRO 优化的三个 Stage？
**A:** ZeRO-1: 将 optimizer states 分片到各 GPU；ZeRO-2: + gradient 分片；ZeRO-3: + parameter 分片。ZeRO-3 理论上可将显存占用降到单卡的 1/N（N=GPU 数量），但通信开销也相应增加。

### Card 031
**Q:** 什么是 Teacher-Student Knowledge Distillation in LLM context?
**A:** 用大模型（Teacher, e.g., GPT-4/Qwen3-72B）生成高质量训练数据，用于训练小模型（Student, e.g., Qwen3-8B）。不直接蒸馏 logits，而是通过数据蒸馏——teacher 生成 instruction-response pairs 作为 SFT 数据。

### Card 032
**Q:** On-policy vs Off-policy 训练在 LLM post-training 中的区别？
**A:** On-policy：用当前训练的模型实时采样数据来训练，数据分布与模型当前策略一致，效果更好但成本高（如 Iterative DPO、PPO）。Off-policy：使用预先收集的历史数据进行训练，成本低但存在分布偏移问题（如标准 DPO）。

### Card 033
**Q:** 什么是 Iterative DPO？
**A:** 多轮 DPO：每轮用当前最优 checkpoint 采样生成新的 preference pair，用这些 on-policy 数据训练下一轮。相比单轮 DPO 效果更好（减少分布偏移），但计算成本呈倍数增加。

### Card 034
**Q:** Safety-RAG 和标准 RAG 的核心区别？
**A:** 标准 RAG 是检索+拼接后直接生成，对检索到的知识无条件信任。Safety-RAG 多了一层验证/对账机制：模型生成后，由一个独立的 verifier 比对生成内容与检索到的知识是否一致，不一致则用检索知识修正。核心差异在"验证-修正"闭环。

### Card 035
**Q:** ORPO 是什么？跟 DPO 的区别？
**A:** ORPO (Odds Ratio Preference Optimization) 不使用 reference model，而是直接在 SFT loss 上叠加一个 odds ratio 偏好项。相比 DPO 省去了 ref model 的显存和计算，在部分任务上表现相当或更好。

### Card 036
**Q:** SimPO 是什么？核心改进？
**A:** SimPO (Simple Preference Optimization) 用 average log probability（而非 sum）作为隐含 reward，且引入 target reward margin γ。不需 reference model，对长度偏差更鲁棒。

### Card 037
**Q:** GRPO 是什么？为什么不需要 value model？
**A:** Group Relative Policy Optimization。对每个 prompt 采样一组（group）response，用组内相对 reward（减去组内均值作为 baseline）替代 value model 估计的 advantage。省掉了 value model 的显存和训练。

### Card 038
**Q:** RLVR (RL with Verifiable Rewards) 是什么？
**A:** 对数学、代码等有明确 ground truth 的任务，用规则验证（如数学答案是否正确、代码是否通过测试用例）作为 reward，而非训练 reward model。在 DeepSeek-R1 等模型中得到成功应用。

### Card 039
**Q:** 什么是 OPD (Online Preference Distillation)？
**A:** Online Preference Distillation，结合 DPO 的简洁性和 on-policy 采样的优势。用当前 policy 采样生成，用 teacher/reward model 标注偏好，然后做 DPO。效果介于 DPO 和 PPO 之间，成本也居中。

### Card 040
**Q:** RLOO 是什么？和 PPO/GRPO 的关系？
**A:** RLOO (REINFORCE Leave-One-Out) 是一种去 value model 的 RL 方法。对每个 prompt 采样 K 个 response，用除当前 response 外的 K-1 个 response 的平均 reward 作为 baseline。与 GRPO 类似但 baseline 计算方式不同。

---

## 二、公式卡（25张）

### Card 041
**Q:** Scaled Dot-Product Attention 公式？
**A:** Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V。除以 sqrt(d_k) 是为了防止 d_k 过大时点积的量级过大，导致 softmax 进入梯度饱和区（梯度消失）。

### Card 042
**Q:** Multi-Head Attention 公式？
**A:** MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O，其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)。每个 head 在低维子空间学习不同的 attention pattern。

### Card 043
**Q:** RoPE 位置编码公式？
**A:** 对位置 m 和 n 的 query/key，RoPE 用旋转矩阵编码：f(q, m) = q * e^{imθ}（复数域旋转），使得 attention 得分 q_m^T k_n 只依赖于相对位置 m-n。每个维度对的旋转频率不同（高频→低频），形成多尺度位置编码。

### Card 044
**Q:** RMSNorm 公式？
**A:** y_i = x_i / RMS(x) * γ_i，其中 RMS(x) = sqrt(1/d * Σ_i x_i²)。仅做缩放归一化，不做均值中心化。相比 LayerNorm 减少一次归约操作。

### Card 045
**Q:** SwiGLU 公式？
**A:** SwiGLU(x) = (xW_1 ⊙ Swish(xW_2)) * W_3，其中 Swish(x) = x * σ(x) = x * 1/(1+e^{-x})。gated 机制 + Swish 激活的组合比 ReLU 表现出更好性能。

### Card 046
**Q:** SFT Loss 公式？
**A:** L_SFT = -1/|R| * Σ_{t∈R} log π_θ(y_t | x, y_<t)。其中 R 是 response token 的位置集合，prompt 部分不参与 loss 计算（label=-100 mask）。本质是对 response 的 teacher-forcing cross-entropy。

### Card 047
**Q:** LoRA 公式？
**A:** h = Wx + ΔWx = Wx + BAx，其中 A∈R^{r×d}（高斯初始化），B∈R^{d×r}（零初始化）。实际 update 量缩放为 ΔW = (α/r) * BA。推理时可 merge：W' = W + (α/r) * BA。

### Card 048
**Q:** DPO Loss 公式？
**A:** L_DPO = -E_{(x,y_w,y_l)} [log σ(β * (log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x)))]。本质是最大化 chosen 相对 rejected 的 reward margin，同时防止偏离 ref 太远。

### Card 049
**Q:** Bradley-Terry 偏好模型公式？
**A:** P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l)) = exp(r(x, y_w)) / (exp(r(x, y_w)) + exp(r(x, y_l)))。BT 模型将成对偏好概率建模为两个选项 latent reward 的 sigmoid 差值。

### Card 050
**Q:** RLHF 中 DPO 的隐式 reward 公式？
**A:** DPO 证明 BT 模型下最优 policy 满足：r*(x, y) = β * log(π*(y|x) / π_ref(y|x)) + β * log Z(x)。这意味着 reward 可以从 policy 和 ref 的 log ratio 中恢复，从而不需要显式训练 RM。

### Card 051
**Q:** PPO Clipped Objective 公式？
**A:** L_CLIP(θ) = E[min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)]，其中 r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)。通过 clip 限制策略更新的幅度，防止策略崩溃。

### Card 052
**Q:** PPO with KL Penalty 公式？
**A:** R_PPO(x, y) = r_φ(x, y) - β * KL(π_θ(y|x) || π_ref(y|x))。在 reward 中减去与 ref model 的 KL 散度，防止 policy 偏离太远导致语言能力退化。

### Card 053
**Q:** ORPO Loss 公式？
**A:** L_ORPO = L_SFT + λ * L_OR，其中 L_OR = -log σ(log(odds_θ(y_w|x)) - log(odds_θ(y_l|x)))，odds_θ(y|x) = π_θ(y|x) / (1 - π_θ(y|x))。将偏好学习融入 SFT loss，不需要 ref model。

### Card 054
**Q:** SimPO Loss 公式？
**A:** L_SimPO = -E[log σ(β * (avg_logp_θ(y_w|x) - avg_logp_θ(y_l|x) - γ))]，其中 avg_logp 是 response 的平均 log probability，γ 是 target reward margin。不使用 ref model。

### Card 055
**Q:** Cross-Entropy Loss 公式？
**A:** L_CE = -Σ_i y_i * log(p_i)，其中 y_i 是 one-hot 真实标签，p_i 是预测概率。在 LLM 中，对每个位置预测下一个 token 的分布，计算与真实 token 的 CE loss。

### Card 056
**Q:** Perplexity 公式？
**A:** PPL = exp(L_CE) = exp(-1/N * Σ_t log π_θ(x_t | x_<t))。PPL 是模型对给定文本"困惑度"的度量——值越小表示模型对文本越"不惊讶"。可作为模型语言能力的快速评估。

### Card 057
**Q:** KL Divergence 公式？
**A:** KL(P || Q) = Σ_x P(x) * log(P(x) / Q(x)) = E_P[log P - log Q]。衡量分布 Q 偏离分布 P 的程度，非对称。在 RLHF 中用于约束 policy 不偏离 reference。

### Card 058
**Q:** GAE (Generalized Advantage Estimation) 公式？
**A:** A_t^GAE = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}，其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)。通过 λ 参数控制 bias-variance trade-off：λ=0 最小方差最大 bias，λ=1 最大方差最小 bias。

### Card 059
**Q:** Cosine Similarity 公式？
**A:** cos(A, B) = (A·B) / (||A|| * ||B||)。在 RAG 中常用于衡量 query embedding 和 document embedding 的相似度。范围 [-1, 1]，越接近 1 越相似。

### Card 060
**Q:** Attention Score before softmax in Flash Attention?
**A:** Same as standard: S = QK^T / sqrt(d_k) + mask. Flash Attention doesn't change the math — it changes the implementation (tiling + kernel fusion) to avoid materializing the full S matrix in HBM.

### Card 061
**Q:** What is the BT reward model training loss?
**A:** L_RM = -E_{(x,y_w,y_l)} [log σ(r_φ(x, y_w) - r_φ(x, y_l))]. 最大化 chosen 的 reward 高于 rejected 的概率，等价于用 BT 模型做二分类：正样本（chosen 更好）。

### Card 062
**Q:** Gradient of DPO loss w.r.t θ?
**A:** ∇_θ L_DPO = -β * E[(1 - σ(β * log_ratio_diff)) * (∇_θ log π_θ(y_w|x) - ∇_θ log π_θ(y_l|x))]. 梯度方向：增大 chosen 概率 + 减小 rejected 概率，由 reward error（1 - σ(...)）加权。

### Card 063
**Q:** What is the formula for reward margin in DPO?
**A:** reward_margin = r_θ(x, y_w) - r_θ(x, y_l) = β * [log(π_θ(y_w|x) / π_ref(y_w|x)) - log(π_θ(y_l|x) / π_ref(y_l|x))]. 正值表示 chosen 的 reward 高于 rejected，负值表示模型偏好与数据不一致。

### Card 064
**Q:** How is the implicit reward computed in DPO diagnostics?
**A:** r_θ(x, y) = β * (log π_θ(y|x) - log π_ref(y|x)). 虽然不是显式训练的 reward model，但这个量可以用于监控 training dynamics 和计算 reward accuracy。

### Card 065
**Q:** What is the GQA attention computation formula?
**A:** Same as MHA but with n_kv_heads KV heads. For Q head i, use KV head ⌊i * n_kv_heads / n_q_heads⌋. Q project separately, but multiple Q heads share the same K and V. Computed as: o_i = Attention(Q_i, K_{g(i)}, V_{g(i)}).

---

## 三、代码卡（20张）

### Card 066
**Q:** PyTorch 手写 Scaled Dot-Product Attention 的关键代码？
**A:** scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k); if mask: scores.masked_fill_(mask==0, -1e9); attn = F.softmax(scores, dim=-1); out = attn @ V。关键点：除以 sqrt(d_k) 防梯度消失，mask 用 -inf 让 softmax 输出 0。

### Card 067
**Q:** LoRA 前向传播的核心代码？
**A:** def forward(x): return self.base(x) + (self.dropout(x) @ self.A.T @ self.B.T) * (self.alpha / self.r)。关键点：A 高斯初始化、B 零初始化、缩放因子 alpha/r。

### Card 068
**Q:** SFT 的 label mask 怎么做？
**A:** labels = input_ids.clone(); labels[:, :prompt_len] = -100; loss = CrossEntropyLoss(ignore_index=-100)(logits, labels)。关键：prompt 部分 label=-100 被 ignore，只计算 response 部分。

### Card 069
**Q:** DPO loss 的 PyTorch 实现关键代码？
**A:** log_ratio = (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected); loss = -F.logsigmoid(beta * log_ratio).mean()。关键：log_ratio 是 chosen vs rejected 的相对 reward margin。

### Card 070
**Q:** HuggingFace 加载 QLoRA 模型的量化配置？
**A:** BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)。四个参数对应 NF4 / 计算精度 / 双重量化。

### Card 071
**Q:** PEFT LoRA 配置的 target_modules 通常选什么？
**A:** target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]——覆盖所有 attention 和 FFN 线性层。关键：不要漏 gate_proj, up_proj, down_proj（SwiGLU 的三个矩阵）。

### Card 072
**Q:** DataLoader collate_fn 中对 DPO 数据的处理？
**A:** 处理三部分：prompt / chosen / rejected。分别 tokenize 并返回 {input_ids_chosen, labels_chosen, input_ids_rejected, labels_rejected, ...}。DPOTrainer 期望每个样本包含 prompt + chosen + rejected。

### Card 073
**Q:** 怎么冻结模型参数只训练 LoRA？
**A:** for param in model.parameters(): param.requires_grad = False；然后使用 get_peft_model(model, lora_config) 自动为 LoRA 参数设置 requires_grad=True。关键：基座权重冻结但量化后的 4bit 权重仍参与前向。

### Card 074
**Q:** 训练循环中的梯度累积代码？
**A:** if (step+1) % accumulation_steps == 0: optimizer.step(); scheduler.step(); optimizer.zero_grad()。同时 loss = loss / accumulation_steps 确保等效 batch size 下的 loss 一致。

### Card 075
**Q:** FAISS 向量检索的核心代码？
**A:** index = faiss.IndexFlatIP(dim); index.add(embeddings); scores, ids = index.search(query_emb, top_k)。IndexFlatIP 做内积检索（配合 normalized embedding 等价余弦相似度）。

### Card 076
**Q:** RoPE 的实现关键步骤？
**A:** 1) 预计算 cos/sin 频率表（freqs = 1/(theta^(2i/d))，乘 position 得到角度）；2) 将 Q/K 分两半旋转：q_embed = q * cos + rotate_half(q) * sin（rotate_half 交换前后半并取反）。

### Card 077
**Q:** 使用 TRL 库做 DPO 训练的关键参数？
**A:** DPOTrainer(model, ref_model, args=DPOConfig(beta=0.1, loss_type="sigmoid", max_length=2048, max_prompt_length=1536), train_dataset=..., tokenizer=...)。beta 影响 KL 约束强度，需要调优。

### Card 078
**Q:** Causal Mask 的生成代码？
**A:** mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(); scores.masked_fill_(mask, float('-inf'))。上三角为 1 的位置 mask 掉（设为 -inf），防止当前位置 attend 到未来位置。

### Card 079
**Q:** 训练中如何实现 warmup + cosine decay？
**A:** get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total)。Transformer 库内置，也可手写：warmup 阶段 lr 线性增加，然后 cosine 衰减到 0。

### Card 080
**Q:** 如何保存和加载 LoRA adapter？
**A:** model.save_pretrained(save_path) 保存 adapter_config.json + adapter_model.bin；PeftModel.from_pretrained(base_model, adapter_path) 加载。merge_and_unload() 合并到基座权重后可直接用于推理。

### Card 081
**Q:** 多 GPU 训练时梯度同步的关键？
**A:** DistributedDataParallel (DDP) 自动在 backward() 时做 all-reduce 同步梯度。optimizer.step() 前确保所有 GPU 的梯度一致。DDP 包装：model = DDP(model, device_ids=[local_rank])。

### Card 082
**Q:** 怎么用 vLLM API 调用部署的模型？
**A:** client = OpenAI(base_url="http://localhost:8000/v1", api_key="x"); response = client.chat.completions.create(model="...", messages=[...], max_tokens=512)。vLLM 兼容 OpenAI API 格式。

### Card 083
**Q:** BPE merge 循环的核心代码？
**A:** while len(vocab) < target_vocab_size: pair_freq = Counter(); for word, freq in corpus: pairs = get_pairs(word.split()); for p in pairs: pair_freq[p] += freq; best = max(pair_freq, key=pair_freq.get); corpus = [merge(w, best) for w in corpus]; vocab[''.join(best)] = pair_freq[best]。

### Card 084
**Q:** 计算 reward accuracy in DPO 训练？
**A:** chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach(); rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach(); acc = (chosen_reward > rejected_reward).float().mean()。监控这个值防止过拟合（接近 1.0 可能过拟合）。

### Card 085
**Q:** 模型推理时的 KV cache 怎么用？
**A:** HuggingFace 中使用 model.generate(..., use_cache=True)，内部 past_key_values 存储每层的 K/V。KV cache 的显存增长随 seq_len 线性增长而非平方——这就是为什么长序列推理是显存瓶颈。

---

## 四、项目卡（40张）

### Card 086
**Q:** 医学 LLM Teacher 项目的完整 pipeline？
**A:** Teacher 数据生成 (DeepSeek-v4-pro) → 55K SFT (QLoRA) → 10K DPO → LLM-as-Judge 评测 → DPO 后发现医疗长尾幻觉 → Safety-RAG 修复 → vLLM 部署。关键发现：DPO 后幻觉增加，Safety-RAG 修复后幻觉下降 66.7%。

### Card 087
**Q:** 为什么选 Qwen3-8B 作为基座模型？
**A:** 1) GQA 机制推理效率高；2) 中文能力同规模 SOTA（中文医疗场景核心需求）；3) 8B 规模在单张 RTX 5090 32GB 上可完成全流程训练。LLaMA 中文 tokenizer 效率低约 30%。

### Card 088
**Q:** 55K SFT 数据怎么来的？
**A:** Teacher 模型 DeepSeek-v4-pro 生成 + 人工筛选。覆盖问诊、检查解读、用药咨询、健康科普 4 个场景。数据多样性通过：不同 temperature 采样 + 多 prompt 模板 + 医疗知识库引导生成。人工抽检 10% 保证质量。

### Card 089
**Q:** SFT 训练的详细参数？
**A:** QLoRA r=16, alpha=32, target_modules 全部 attention+FFN 层；lr=5e-5 cosine schedule，batch_size=4 (micro_batch)×4 (accumulation)=16 effective；epoch=1；max_length=2048；bf16 混合精度；RTX 5090 32GB。

### Card 090
**Q:** 10K DPO 偏好数据怎么构造？
**A:** 三路来源：1) 同一 prompt 不同 temperature 采样多个 response，LLM-as-Judge 打分取最高/最低 pair；2) SFT 模型 vs Teacher 模型回答对比；3) 人工标注 500 条医疗高风险场景偏好。chosen/rejected 差异 subtle 但 meaningful。

### Card 091
**Q:** DPO 训练的详细参数和关键发现？
**A:** beta=0.1, lr=5e-6, epoch=1, effective batch=8。关键发现：reward accuracy 达 0.78 但发现医疗长尾幻觉（编造文献出处/临床试验数据）。分析原因：DPO 的 reward 信号偏向"确定自信"的回答风格，导致模型在不确定时编造信息。

### Card 092
**Q:** LLM-as-Judge 评测体系的三维度和权重？
**A:** 医学准确性 (40%)：回答是否与临床知识一致；安全性 (35%)：是否避免处方建议和不恰当确定性表述；完整性 (25%)：是否满足用户真实需求。judge 使用 DeepSeek-v4-pro，与人工 Spearman 相关系数 accuracy 0.78，safety 0.62。

### Card 093
**Q:** DPO 后幻觉问题的具体表现？
**A:** 3 个典型案例：1) 编造"某抗生素三期临床招募 2347 例患者，有效率达 87.3%"——文献引用和数据全为编造；2) MRSA 和 MRCNS 治疗方案混淆——两者均为耐药菌但抗生素不同；3) 给中成药提供"每日 3 次，每次 2 粒"的处方级建议。

### Card 094
**Q:** Safety-RAG 的完整设计和流程？
**A:** 三阶段验证管线。阶段 1：构建医疗知识库（50 万条结构化知识，bge-large-zh-v1.5 向量化存入 FAISS）。阶段 2：推理时检索 top-5 相关知识片段。阶段 3：verifier（Qwen3-8B 安全验证微调版）比对 response vs retrieved knowledge，判断事实错误/幻觉/安全等级。不安全则用检索知识修正。

### Card 095
**Q:** "幻觉下降 66.7%"这个数字怎么来的？
**A:** 3 个典型的医疗长尾幻觉案例在引入 Safety-RAG 后全部消失。3 个变 0 个 = 下降 100% 但样本量小。保守表述：67% ≈ 2/3，实际表达"我们在 3 个典型幻觉案例上观察到完全消失"更诚实。面试中应主动说明样本量限制。

### Card 096
**Q:** 医疗知识库的数据来源和规模？
**A:** 临床指南（中华医学会/UpToDate）约 10 万条 + 药品说明书（中国药典/NMPA）约 20 万条 + 检验指标解读约 10 万条 + 常见疾病科普约 10 万条。总计约 50 万条结构化知识。chunk 大小为 256-512 token，overlap=64。

### Card 097
**Q:** 医疗安全的三层防护设计？
**A:** 数据层：SFT/DPO 数据混入 15% 安全样本（拒答+安全边界）；推理层：Safety-RAG 输出前知识验证+修正；Prompt 层：system prompt 明确"不提供处方建议"，输出加"仅供参考，请遵医嘱"。DangerousQA benchmark 违规回答率从 12% 降至 <2%。

### Card 098
**Q:** vLLM 部署的配置和性能？
**A:** 单卡 RTX 5090 32GB, max-model-len=4096, gpu-memory-utilization=0.9, 使用 PagedAttention + continuous batching。支持 10+ 并发请求，平均延迟约 1.5 秒/token。兼容 OpenAI API 格式，便于集成。

### Card 099
**Q:** 项目训练各阶段的耗时和硬件？
**A:** SFT (QLoRA): RTX 5090 32GB, 55K 数据, 1 epoch, ~4 小时；DPO: RTX 5090 32GB, 10K 数据, 1 epoch, ~4 小时；Safety-RAG: 检索验证约增加 500ms 推理延迟。总训练时长约 8 小时，可行性强。

### Card 100
**Q:** 项目的数据流程图？
**A:** Raw Medical Corpus → Teacher (DeepSeek-v4-pro) 生成 → 人工筛选 → 55K SFT data → QLoRA Training → Checkpoint → Teacher 再次生成对比 → LLM-as-Judge 排序 → 10K DPO data → DPO Training → Final Model + Safety-RAG.

### Card 101
**Q:** Safety-RAG verifier 怎么训练的？
**A:** 构造专门的验证微调数据：给定 (query, response, retrieved_knowledge) → 输出 (safe/unsafe, error_type, evidence)。约 5K 条验证数据，用 Qwen3-8B QLoRA 微调 1 epoch。验证任务本质是文本蕴含（NLI）任务。

### Card 102
**Q:** 你在这个项目中最核心的贡献？
**A:** 1) 设计并实现了完整的后训练 pipeline（数据→训练→评测→安全→部署）；2) 发现并分析了 DPO 后的幻觉问题（reward hacking 机制分析）；3) 提出并验证了 Safety-RAG 方案（→幻觉案例消失）。完整展示了发现问题→分析原因→设计方案→实验验证的算法能力闭环。

### Card 103
**Q:** 项目中你用到了哪些开源工具/框架？
**A:** HuggingFace Transformers, PEFT (LoRA), TRL (DPOTrainer), BitsAndBytes (QLoRA), FAISS (向量检索), vLLM (推理部署), LangChain (文档切分), sentence-transformers (embedding), Wandb (训练监控), OpenAI API (LLM-as-Judge 对比)。

### Card 104
**Q:** 项目的数据质量如何保证？
**A:** 多级质量过滤：1) Teacher 生成后用医学知识库做事实性交叉验证；2) 去重（MinHash + 语义相似度 >0.95 去重）；3) 人工抽检 10% 标注质量等级；4) 安全规则过滤（预设关键词和正则表达式）；5) 多样性格筛选（保证场景分布均匀）。

### Card 105
**Q:** 项目中 teacher 模型为什么会出错？怎么处理的？
**A:** Teacher (DeepSeek-v4-pro) 虽然强但不是完美的，尤其在医学长尾领域可能犯错。处理方式：1) 知识库交叉验证（生成内容与医学知识库比对）；2) 人工抽检高风险场景；3) DPO 数据中引入人工标注偏好去纠正 teacher 的错误偏好；4) Safety-RAG 作为兜底安全层。

### Card 106
**Q:** 项目如果重新做，你会怎么改进？
**A:** 1) SFT 阶段引入更多多轮对话数据（当前单轮为主）；2) DPO 后加一轮 on-policy 的 iterative DPO 验证；3) Safety-RAG 用专业的 reranker 替代简单向量检索；4) 增加更大的评测样本量（当前 3 个案例太少）；5) 引入 RLVR 做可验证的医学知识奖励。

### Card 107
**Q:** 这个项目还有什么不足？
**A:** 1) 没有线上部署和真实用户反馈（学术探索项目）；2) 3 个幻觉案例的样本量不足以做统计显著性检验；3) Safety-RAG 增加了推理延迟（约 500ms）；4) 知识库覆盖不完整（罕见病/新药）；5) 缺乏多语言支持。这些在面试中诚实说出反而加分。

### Card 108
**Q:** MiniMind 项目做了什么？
**A:** 从零训练了一个 26M 参数的小语言模型，架构包含：512 维 hidden, 8 层 Decoder, 8 个 attention head, GQA, RoPE 位置编码, RMSNorm, SwiGLU 激活。完整实现了 tokenizer 训练、预训练、SFT、DPO 全流程。目的是深入理解 LLM 的底层（理解了 big picture + 每个底层细节）。

### Card 109
**Q:** MiniMind 和医学 LLM 项目的互补性？
**A:** MiniMind 让我理解了 LLM 的每个底层组件（从 tokenizer 到 attention 到 normalization 到训练循环），而医学 LLM 项目让我学会了如何在大规模真实场景做后训练。一个深入底层理解"为什么"，一个面向应用解决"怎么做好"。两者结合展示了大模型算法工程师需要的双向能力。

### Card 110
**Q:** 你这个医学项目的创新点是什么？
**A:** 1) Safety-RAG 的验证-修正闭环（不是简单拼接知识）；2) DPO 后幻觉问题的系统性分析（reward hacking 在医疗场景的具体表现）；3) 数据-训练-安全-评测-部署的完整闭环证明后训练算法可行性。创新点是相对增量而非革命性的——面试中要诚实表达。

### Card 111
**Q:** 你的项目证明了你什么工程能力？
**A:** 1) 端到端 pipeline 搭建（4 阶段独立可运行）；2) 显存优化（QLoRA + bf16 + gradient checkpointing + gradient accumulation）；3) 模块化代码设计（数据/训练/评测/部署解耦）；4) vLLM 部署和性能调优；5) 使用 wandb 做实验管理和可视化。

### Card 112
**Q:** 你怎么评估自己的项目完成度？
**A:** SFT+DPO+评测 pipeline 是完整的（数据→训练→评测闭环），Safety-RAG 是 +1 的创新探索，vLLM 部署是工程落地展示。整体完成度约 80%——缺失的是线上部署和真实用户反馈闭环。这是一个"研究探索 + 工程实现"的均衡项目。

### Card 113
**Q:** 从项目中你学到了哪些"不要这样做"的教训？
**A:** 1) 不要只看 loss/reward accuracy——要在实际场景 test 生成质量；2) DPO 不是银弹——它可能在解决某些问题的同时引入新问题（reward hacking）；3) 评测要和实际使用对齐——LLM-as-Judge 的安全判断不够可靠；4) 小样本发现要诚实报告——不要 overclaim。

### Card 114
**Q:** 你怎么向非技术背景的人解释你的项目？
**A:** "我用大模型技术做了一个医疗 AI 助手。像一个'医生咨询机器人'，训练它正确回答医疗问题——但不能乱开药方。我先用大量医学资料训练它，再教它分辨好回答和坏回答。最后加了一个'安全质检员'——每次它回答前，先查医学知识库核对信息，确保不编造内容。"

### Card 115
**Q:** 项目各阶段的模型性能对比？
**A:** SFT 后 vs base model：医疗问答准确率大幅提升（base 几乎不能做医疗对话）；DPO 后 vs SFT：偏好胜率 (+8%)，但幻觉案例增加；Safety-RAG 后 vs DPO：3 个典型案例幻觉全部消失，回复更保守但更安全。定性变化 > 定量变化——这也是诚实表述。

### Card 116
**Q:** 为什么你的 SFT 只有 55K 数据而不是几十万？
**A:** 1) 医学领域对数据质量要求极高，优先保证质量而非数量；2) Teacher 模型生成 + 人工筛选的成本限制；3) 55K 在 8B 模型的 QLoRA 微调下已够学到初步医疗能力；4) 重点在后续的 DPO + Safety-RAG 创新，SFT 是基础而非核心。

### Card 117
**Q:** DPO 10K 数据的 "subtle but meaningful" mean?
**A:** Chosen/rejected pair 的差异不能太明显（如正确 vs 明显错误），否则 DPO 学不到细粒度偏好；也不能太模糊（两个回答几乎一样），否则没有偏好信号。理想情况：两个回答在医学真实性上都对，但一个更安全、更完整、更清晰——这种 subtle 差异才是 DPO 的核心价值。

### Card 118
**Q:** How do you handle the "I am not a doctor" disclaimer?
**A:** System prompt 中前置声明："我是一个 AI 医学知识助手，不具备执业医师资格。以下回答仅供参考，不构成诊断或处方建议，请务必咨询专业医生。" 同时训练模型在 risk > 阈值时（如就医建议、用药咨询）主动加免责声明。Safety-RAG verifier 也会检查是否缺少免责声明。

### Card 119
**Q:** What are the core safety competencies of the medical model?
**A:** 1) 不提供处方级建议（剂量、用法、疗程）；2) 不给出明确诊断结论（"你可能得了XX病"）；3) 不编造医学事实（数据、文献、统计）；4) 适当表达不确定性（"可能""建议就医"）；5) 识别紧急情况（胸痛、呼吸困难 → 立即就医）。

### Card 120
**Q:** 你的项目与直接做 RAG 有什么区别？
**A:** 直接 RAG 只是把检索结果拼接后让 LLM 生成，没有验证机制。我们的 Safety-RAG 多了验证层：1) LLM 先生成回答；2) Verifier 比对生成内容与检索知识；3) 检测到不一致时修正或拒答。这使得系统更安全——因为我们不盲目信任检索到的知识也不盲目信任模型生成，而是让两者互相校验。

### Card 121
**Q:** 项目用什么做 embedding？
**A:** bge-large-zh-v1.5 (BAAI)，1024 维 embedding，在中文医疗文本上做了领域适配（用医疗语料做少量继续训练）。选择理由：中文 SOTA、开源、模型大小适中（~326M）、MTEB 中文榜单排名前列。检索召回率在医疗测试集上达 85%。

### Card 122
**Q:** 你如何判断模型是在输出安全知识还是"过度拒答"？
**A:** 构造一组明确的 benign 医疗问题（如"感冒了多喝水有用吗？""维生素C有什么作用？"），这些问题答案无争议且不涉及处方/诊断，模型理应正常回答。如果模型连这些问题也拒绝回答——就是过度拒答。我们监控 50 个 benign query 的拒答率，保持在 5% 以下。

### Card 123
**Q:** What metrics did you use to evaluate your medical model?
**A:** 1) LLM-as-Judge 三维度评分 (accuracy + safety + completeness)；2) DPO reward accuracy 和 reward margin；3) Safety benchmark (DangerousQA 违规率)；4) 人工抽检准确率（200 条样本）；5) RAG retrieval Recall@5。没有单一的 gold metric——多维度交叉验证。

### Card 124
**Q:** 你对"算法"二字的理解是什么？
**A:** 在这个项目中，"算法"不是指发明了一个新的数学公式，而是指：1) 能从第一性原理理解每个方法为什么 work / 不 work；2) 能根据问题设计数据构造策略、训练方案、评测体系；3) 能从实验现象出发做因果分析（DPO → 幻觉增加的机制推理）；4) 能提出有依据的改进方案（Safety-RAG）。算法 = 分析问题 + 设计方案 + 验证闭环。

### Card 125
**Q:** What is your understanding of the "post-training" role?
**A:** Post-training算法工程师的核心工作：将预训练好的 base model 变成可用的、安全的、对齐的 chat model。具体包括：数据策略（构造高质量 SFT/偏好数据）、训练策略（选择合适的 alignment 方法及其超参数）、评测策略（多维度评估模型能力）、安全策略（减少幻觉和有害输出）。技术栈：SFT + RLHF/DPO/GRPO + RAG + Safety。

---

## 五、追问卡（25张）

### Card 126
**Q:** 追问：你是不是只是调用 API 生成数据？
**A:** 不仅仅是。Teacher 生成只是数据管线的第一步。更重要的是：数据质量验证（知识库交叉验证+去重+安全过滤）、多样性控制（场景均衡+采样策略）、偏好信号构造（多源对比+人工标注）。数据工程本身就是后训练的核心能力——数据质量决定了模型上限。

### Card 127
**Q:** 追问：你这个项目算法性在哪里？
**A:** 1) DPO 后幻觉增加的因果分析（reward hacking 在医疗场景的机制）；2) Safety-RAG 的验证-修正设计（如何检测生成内容与检索知识的不一致）；3) 完整的算法选择决策链（为什么选 DPO 而非 PPO/GRPO，有依据有分析）。算法性 = 分析问题 + 设计方案 + 验证效果。

### Card 128
**Q:** 追问：3 个样本说服不了人怎么办？
**A:** 诚实承认样本量限制——这是探索性发现而非统计显著结论。正确表述是"我们在 DPO 后观察到 3 个典型的医疗长尾幻觉案例，这些案例在引入 Safety-RAG 后不再出现。这提供了一个有价值的信号，但确实需要更大规模的系统性评测来验证统计显著性。"

### Card 129
**Q:** 追问：LLM-as-Judge 不可靠怎么办？
**A:** 1) 多 judge 交叉验证（我们对比了 DeepSeek-v4-pro + GPT-4 + Claude 判断）；2) 与人工标注对标（200 条 Spearman 0.78 on accuracy）；3) 安全维度引入规则辅助（关键词+正则表达式做硬规则）；4) 承认局限性——安全维度 Spearman 只有 0.62，需要人工审核。

### Card 130
**Q:** 追问：Medical model error is very serious, how do you handle?
**A:** Triple defense: 1) Data-level safety filtering + safety samples in training; 2) Safety-RAG verification before output; 3) Mandatory disclaimer + risk escalation (urgent cases → recommend hospital). Also: acknowledge that no system is 100% safe — the goal is to minimize risk, not eliminate it.

### Card 131
**Q:** 追问：RAG 检索错了怎么办？
**A:** 多路防御：1) Hybrid search（dense + sparse）提高召回；2) Reranker 对 top-k 重排序提高精度；3) Safety-RAG verifier 会检测到检索结果与 query 不相关时拒绝回答；4) 知识源权威等级（指南 > 教科书 > 科普），优先使用高权威源。检索不是一次性的——是一个有多轮验证的 pipeline。

### Card 132
**Q:** 追问：Teacher 也会错怎么办？
**A:** Teacher 不是 ground truth——它是高质量参考。我们有四层质量保证：1) 生成后用医学知识库交叉验证做事实性检查；2) 人工抽检 10%；3) DPO 阶段纠正 teacher 的错误偏好；4) Safety-RAG 兜底。整个 pipeline 设计假设没有任何单一来源是完美的。

### Card 133
**Q:** 追问：DPO 为什么会放大幻觉？
**A:** DPO 的 reward 偏好"确定自信"的回答风格（chosen 通常比 rejected 更完整、更确定）。模型学会：在不确定时与其说"我不确定"（可能被 rejected），不如编造一个看似专业的回答（更像 chosen）。本质是 reward signal 过度优化了"确定性"而损害了"真实性"——典型的 reward hacking。

### Card 134
**Q:** 追问：你的项目和普通微调项目有什么区别？
**A:** 1) 全流程闭环——不是只做了 SFT；2) 从发现问题到解决问题——DPO 后发现幻觉，分析原因，设计 Safety-RAG 修复；3) 系统化评测和安全意识；4) 环节完整——数据生成、训练、评测、安全、部署五个阶段。普通微调项目通常只有 SFT 这一步。

### Card 135
**Q:** 追问：How do you prove you have algorithmic ability?
**A:** 1) 能推导关键公式（DPO loss 从 BT 模型到 closed form）；2) 能分析实验现象（DPO 后幻觉增加的机制分析）；3) 能设计解决方案（Safety-RAG 的设计逻辑）；4) 理解 trade-off（DPO vs PPO，安全 vs 有用性）。算法能力 = 理解原理 + 分析问题 + 设计方案。

### Card 136
**Q:** 追问：How do you prove you have engineering ability?
**A:** 1) 完整可运行的 pipeline 代码；2) 显存优化实践（QLoRA 从 ~64GB 降到 ~12GB）；3) vLLM 部署和 API 封装；4) 模块化设计（数据/训练/评测/部署解耦）；5) 使用 wandb 做实验追踪。工程能力 = 能把想法变成可运行的、可复现的系统。

### Card 137
**Q:** 追问：为什么不直接用 GPT-4 做医疗问答？
**A:** 1) 数据隐私——医疗数据不能发送到第三方 API；2) 成本——大规模使用时 API 调用费用高；3) 可控性——无法修改模型行为、添加安全层、定制知识库；4) 离线场景——医院内网环境需要本地部署。自己做模型解决了隐私、成本、可控性三个核心问题。

### Card 138
**Q:** 追问：有没有线上部署/用户反馈？
**A:** 诚实回答：这是一个研究探索项目，目前没有线上部署和真实用户反馈。vLLM API 可以支持 10+ 并发测试，但没有接入真实用户。如果面的是工程岗位，可以补充：API 接口是生产就绪的，如果有需要可以快速接入。不要编造没有的数据。

### Card 139
**Q:** 追问：项目下一步做什么？
**A:** 短期：1) 扩大幻觉评测样本量（从 3 个到 100+）；2) 用专业的医学领域 reranker 替代简单向量检索；3) 增加多轮对话支持。中期：1) 引入 RLVR 做可验证的医学知识奖励；2) 构建数据闭环（用户反馈 → 标注 → 模型更新）；3) 扩展到更多医疗子领域（中医、影像报告）。

### Card 140
**Q:** 追问：你对后训练算法真正理解在哪里？
**A:** 后训练的本质是"偏好学习"——教模型区分什么是好的输出。理解体现在：1) DPO 是从偏好数据中隐式学习 reward，而不需要显式训练 RM；2) 所有方法都在解决同一个核心问题：如何用有限的偏好数据有效地调整模型的输出分布；3) 理解每个方法的假设（BT 模型、on-policy vs off-policy）和局限性。

### Card 141
**Q:** 追问：为什么用 QLoRA 不用全量微调？
**A:** Hardware constraint: 全量微调 8B 模型需要 ~192GB 显存 (parameters + gradients + optimizer states + activations)。QLoRA 降到 ~12GB。在 8B 模型上，QLoRA 和全量微调的效果差距已缩小到 1-2%——性价比极高。算法实习面试中要强调"约束下的最优选择"这种工程思维。

### Card 142
**Q:** 追问：Why Qwen3-8B specifically?
**A:** Not just "because it's open source". Detailed reasons: 1) GQA (memory-efficient inference), 2) Chinese SOTA in its class (crucial for medical Chinese), 3) RMSNorm + SwiGLU (modern architecture), 4) Active community + good documentation, 5) Fits in RTX 5090 32GB single card for full pipeline. Compared alternatives: LLaMA-3 (poor Chinese tokenizer), ChatGLM (different architecture, less transferable knowledge).

### Card 143
**Q:** 追问：DPO beta 怎么选的？
**A:** 做了小型 grid search: β ∈ {0.01, 0.05, 0.1, 0.5, 1.0}。β=0.01 时 reward accuracy 快速涨到 0.9+ 但生成质量崩溃（过拟合）；β=0.5 时 KL 从 ref 几乎没变化（欠拟合）；β=0.1 获得最佳平衡——reward accuracy 稳定在 0.78 且生成质量没有明显退化。标准做法：β 在 0.01-0.5 之间，10K 规模数据通常 0.1。

### Card 144
**Q:** 追问：Medical KB incomplete, how do you handle missing knowledge?
**A:** 1) 多个知识源交叉覆盖（Clinical guidelines + Drug database + Lab references + Forums for rare cases）；2) Safety-RAG verifier 检测到检索结果置信度低时标记"知识库覆盖不足"；3) 模型回答加不确定性表述（"根据现有资料...""建议咨询专科医生"）。Not having complete knowledge is a feature, not a bug — the system should know what it doesn't know.

### Card 145
**Q:** 追问：How to avoid the model giving prescription suggestions?
**A:** Multiple layers: 1) System prompt explicitly forbids prescription; 2) Training data includes refusal examples for prescription scenarios; 3) Safety-RAG verifier checks for drug dosage/regimen patterns; 4) Post-processing regex filter as last resort. If a user asks "what dosage of XX", model responds: "药物剂量需根据个人情况由医生确定，但XX的常见参考剂量范围是...（仅供参考）"。

### Card 146
**Q:** 追问：How to distinguish "over-refusal" from proper safety behavior?
**A:** 用 benign query set 测试（50 个无争议医疗问题），模型拒答率应 <5%。同时用 adversarial query set 测试（50 个应该拒答的问题），拒答率应 >90%。两个阈值都达到才说明安全设计合理。我们的模型在两个 benchmark 上的表现反映出安全与有用性之间的良好平衡。

### Card 147
**Q:** 追问：如果检索召回了错误内容（如 MRSA/MRCNS 混淆）怎么办？
**A:** 这恰好是 Safety-RAG 设计要解决的问题：1) 多路召回 + Reranker 减少检索错误率；2) Verifier 比对生成内容和多条检索结果的一致性——如果多条结果互相矛盾则标记低置信度；3) 医学知识源标注权威等级——优先采信高权威源；4) 输出时标注"根据检索到的资料..."让用户知情。

### Card 148
**Q:** 追问：药品说明书和临床指南冲突时怎么处理？
**A:** Safety-RAG verifier 检测到知识冲突时：1) 标记冲突来源；2) 优先采信临床指南（权威等级更高）；3) 输出时同时提及两者差异："根据XX临床指南...，但该药品说明书提示...，建议遵循主治医师意见"；4) 将冲突记录到日志供后续人工审核。

### Card 149
**Q:** 追问：如何做数据闭环（用户反馈→模型改进）？
**A:** 项目目前没有数据闭环（学术探索阶段），但如果要做：1) 收集用户对回答的"赞/踩"反馈；2) 对"踩"的回答进行安全审核和更正标注；3) 定期将累积反馈转化为新的偏好数据加入 DPO 训练；4) 高频踩的问题回溯知识库补充或修正。数据闭环 = safety + quality 的持续改进机制。

### Card 150
**Q:** 追问：这个项目最能体现你算法能力的地方是什么？
**A:** Safety-RAG 的设计逻辑——从观察到 DPO 后的幻觉现象（empirical observation），到分析幻觉产生机制（reward hacking 导致的过度自信），到设计验证-修正闭环来解决（retrieval + verification + correction），到实验验证（3 个案例修复成功）。这个完整链路——探索 → 分析 → 假设 → 方案 → 验证——是算法能力最核心的体现。

---

## 六、易错点卡（20张）

### Card 151
**Q:** 易错：DPO 不需要 reference model？ ❌
**A:** **错！DPO 需要 reference model**，它在 loss 中用来计算 baseline 防止 policy 偏离太远。没有 ref，policy 可以任意增加 chosen 概率，导致语言能力退化。ORPO 和 SimPO 才不需要 ref model。

### Card 152
**Q:** 易错：LoRA 的 A 矩阵初始化为零？ ❌
**A:** **错！A 是高斯初始化（或均匀分布），B 才是零初始化**。这样保证初始时 ΔW = BA = 0，模型输出完全等于预训练模型。如果 A 也初始化为零，梯度对称性导致无法有效训练。

### Card 153
**Q:** 易错：SFT 和 DPO 的 loss 是一样的？ ❌
**A:** **完全不同**。SFT 是 cross-entropy on positive responses（最大化正确答案的概率）。DPO 是 preference-based contrastive loss（对比 chosen vs rejected，同时约束不偏离 ref）。两者训练目标、数据格式、优化行为完全不同。

### Card 154
**Q:** 易错：reward accuracy 高就是模型好？ ❌
**A:** **不一定！** reward accuracy 衡量的是模型对训练数据的拟合度，不是对未见数据的生成质量。reward accuracy > 0.9 往往意味着过拟合——模型记住了训练偏好但在实际对话中可能质量更差（reward hacking）。生成质量需要独立的评测体系。

### Card 155
**Q:** 易错：LoRA rank 越大效果越好？ ❌
**A:** **不一定！** r=8 通常已够用，r=64 和 r=8 差距常 <1%。过大的 r 增加参数量和过拟合风险，得不偿失。关键是选对 target_modules（所有 attention + FFN layers）。经典论文显示 r=1 在某些任务上就已 work。

### Card 156
**Q:** 易错：SFT 数据越多越好？ ❌
**A:** **不是！** 质量 >> 数量。LIMA 论文只用 1000 条高质量数据就取得了很好的效果。低质量 SFT 数据反而会让模型学到错误模式。55K 是我们权衡质量和多样性后的结果。

### Card 157
**Q:** 易错：DPO 和 RLHF/PPO 是一回事？ ❌
**A:** **不是。** DPO 是 off-policy 的静态优化（使用预先收集的数据，不需要在线采样），PPO 是 on-policy 的动态优化（需要模型实时采样+reward model 打分）。DPO 更简单稳定但可能不如 PPO 效果好（尤其在需要多轮迭代的场景）。

### Card 158
**Q:** 易错：Attention 的 Q 和 K 是一个东西？ ❌
**A:** **不是。** Q (Query) 来自当前 token 的表示，K (Key) 和 V (Value) 来自序列中所有 token 的表示。即使 Self-Attention 中三者来自同一个输入，但经过不同的线性投影（W_Q, W_K, W_V）后是不同的。

### Card 159
**Q:** 易错：RoPE 是绝对位置编码？ ❌
**A:** **不是。** RoPE 通过旋转矩阵隐式编码相对位置。虽然每个位置有独立的旋转角度（看起来像"绝对"的），但 attention score q_m^T k_n 最终只依赖于相对位置 m-n。所以它本质是相对位置编码，这也是为什么外推性好的原因。

### Card 160
**Q:** 易错：Beam Search 比 Greedy 好？ ❌
**A:** **不一定。** Beam search 在翻译等确定性任务上好，但在开放域对话中往往产生更重复、更不自然的输出。现代 LLM 推理中，通常用 sampling (temperature + top-p) 更好。该用什么取决于任务。

### Card 161
**Q:** 易错：模型参数越多，推理越快？ ❌
**A:** **反了。** 参数越多推理越慢（更多计算量 + 更大显存）。但更大的模型通常能在更少的 tokens 内生成正确答案，所以实际延迟可能相当。这需要 case-by-case 分析。

### Card 162
**Q:** 易错：RAG 就是搜一下然后贴进去？ ❌
**A:** **太简化了。** RAG 涉及：1) 文档分割策略（chunk size/overlap）；2) embedding 模型选择；3) 检索算法（dense/sparse/hybrid）；4) 后处理（rerank/filter）；5) prompt 组织方式（position/order）。每个环节都需要仔细设计。

### Card 163
**Q:** 易错：QLoRA 只省模型权重显存？ ❌
**A:** **不完全。** QLoRA 省了基座模型权重显存（4bit vs 16bit），但优化器状态存的是 LoRA 参数（<1%），激活值显存不变，分页优化器解决了梯度检查点时的显存尖峰。总节省是多个技术叠加的结果。

### Card 164
**Q:** 易错：DPO 的 chosen 一定比 rejected 质量高很多？ ❌
**A:** **不一定。** 如果 chosen/rejected 差异太大（一个完美一个明显错误），DPO 学到的 signal 太粗粒度（只需要区分明显好坏）。最好是 subtle 差异——两个在内容上都对，但一个更安全/更完整/更清晰。这样模型才能学到细粒度偏好。

### Card 165
**Q:** 易错：KL 惩罚越小越好？ ❌
**A:** **不是。** KL 惩罚控制 policy 偏离 reference 的程度。太小（β→0）→过拟合→语言能力退化；太大（β→∞）→policy≈reference→没学到偏好。KL 惩罚是"创新"和"稳定"之间的平衡 knobs。

### Card 166
**Q:** 易错：Gradient Accumulation 等于增大 Batch Size？ ❌
**A:** **在数学上等价（SGD 层面），但在有 BatchNorm 时不严格等价。** LLM 用 LayerNorm/RMSNorm 不依赖 batch 统计，所以等价。但在某些场景（如对比学习）中，增大有效 batch size 可以改变 loss landscape 的性质。

### Card 167
**Q:** 易错：用 BERT 做 embedding 就行了？ ❌
**A:** **不对。** BERT 是 Encoder-only，生成 embedding 的方式（CLS token / mean pooling / max pooling）影响效果。而且 domain-specific retrieval 需要做 embedding model 的领域适配。bge-large-zh 等专门的 embedding 模型效果通常优于直接用 BERT。

### Card 168
**Q:** 易错：Safety-RAG 和标准 RAG 没区别？ ❌
**A:** **有本质区别。** 标准 RAG：检索→拼接→生成（无验证）。Safety-RAG：检索→生成→验证→（可能）修正。多了验证-修正闭环。标准 RAG 假设检索到的知识都是对的，Safety-RAG 不对检索结果盲目信任。

### Card 169
**Q:** 易错：vLLM 的 PagedAttention 跟 Flash Attention 是一个东西？ ❌
**A:** **不是。** Flash Attention 是 attention 计算的 kernel 优化（解决 HBM 带宽瓶颈）。PagedAttention 是 KV cache 的显存管理优化（解决显存碎片和利用率问题）。两者是互补的——vLLM 实际同时使用了两者。

### Card 170
**Q:** 易错：Transformer 中 Feed Forward 不是必须的？ ❌
**A:** **是必须的。** Attention 是 token 间的信息交互，FFN 是每个 token 内部的非线性变换（存储知识）。没有 FFN，Transformer 只是在不同 token 间搬运信息，无法存储和处理知识。FFN 存储了模型的大量知识。

---

## 七、热点卡（10张）

### Card 171
**Q:** 热点：DeepSeek-R1 的核心贡献？
**A:** 证明了纯 RL（不用 SFT 做 cold start）就能激发 LLM 的推理能力。使用 GRPO + rule-based reward（格式奖励 + 答案正确性），模型在 RL 过程中自发涌现 CoT、自我验证、反思等推理行为。关键 insight：RL 可以成为推理能力的核心驱动力，而非只是偏好对齐工具。

### Card 172
**Q:** 热点：DeepSeek-V3 的 MoE 架构做了什么？
**A:** 671B 总参数，37B 激活参数/token。使用细粒度 MoE（256 个 expert，每个 token 激活 8 个），auxiliary-loss-free load balancing（通过 bias 项动态调整而非 loss 约束）。训练效率极高（2.788M H800 GPU hours）。

### Card 173
**Q:** 热点：Qwen3 相比 Qwen2.5 有什么改进？
**A:** 1) 支持 thinking mode（可切换推理模式）；2) 更强的多语言能力；3) 改进的 alignment pipeline（iterative DPO）；4) 更好的长文本能力（32K/128K）。Qwen3-8B 在 MMLU 等 benchmark 上达到甚至超越同规模闭源模型。

### Card 174
**Q:** 热点：2025 年 DPO 的变体趋势？
**A:** 1) On-policy 化（Iterative DPO, OPD）——用当前模型采样替代静态数据；2) 去 ref model 化（ORPO, SimPO）——减少显存和计算；3) 结合 RLVR——在可验证领域用 rule-based reward；4) Multi-objective——同时优化多维度偏好（安全+有用+风格）。

### Card 175
**Q:** 热点：AI Safety 在 2025-2026 年的趋势？
**A:** 1) Constitutional AI 从 Anthropic 扩展到更多项目（用规则而非人类偏好来约束 AI）；2) Safety 和后训练深度融合——safety 不是 post-hoc 补丁而是训练的一部分；3) 领域特定安全（医疗/法律/金融垂直领域的安全标准建立）；4) Red teaming 自动化（用另一个 LLM 自动发现安全漏洞）。

### Card 176
**Q:** 热点：RAG 的最新发展（2025-2026）？
**A:** 1) Agentic RAG：RAG 不再是简单的检索-生成管线，而是由 Agent 动态决策（何时检索、检索什么、如何验证）；2) Graph RAG：用知识图谱替代向量检索做结构化推理；3) Self-RAG：模型自己决定是否需要检索以及生成的 content 是否与检索结果一致。

### Card 177
**Q:** 热点：LLM 训练效率提升方向？
**A:** 1) FP8/FP4 训练（H100/B200 支持）；2) 更高效的分布式策略（序列并行、专家并行）；3) 稀疏训练（MoE 使激活参数远小于总参数）；4) 更高效的 attention（Flash Attention 3, 窗口 attention, linear attention）；5) 数据课程学习（curriculum learning）。

### Card 178
**Q:** 热点：合成数据（Synthetic Data）的机遇和挑战？
**A:** 机遇：打破数据瓶颈（很多领域缺少高质量标注数据）、降低数据成本、可控的数据多样性。挑战：模型坍塌（用 AI 生成的数据训练 AI 导致能力退化）、虚假关联传播、质量退化。关键：合成数据需要与真实数据、人工审核混用，Iterative 生成-筛选-训练循环。

### Card 179
**Q:** 热点：多模态后训练（Vision-Language）的挑战？
**A:** 1) 数据对齐——图像和文本的 preference 信号不同步；2) 幻觉更复杂——不仅编造文字还会编造视觉内容；3) 评测更难——需要多维度（图文一致性 + 安全性 + 有用性）；4) 训练方法迁移——DPO/RLHF 能否直接用于多模态？可能需要新的算法设计。

### Card 180
**Q:** 热点：2025-2026 最有潜力的后训练方向？
**A:** 1) RLVR + 推理（如 DeepSeek-R1 方向，用 RL 激发推理能力）；2) Iterative/On-policy alignment（缩小分布偏移）；3) Multi-agent RL（多个模型的协作和博弈）；4) 领域特定 alignment（医疗/法律/编程，通用 alignment 不够）；5) 高效后训练（用更少的数据和计算达到更好的对齐效果）。

---

## 八、工程落地 — vLLM部署卡片（12张）

### Card 181
**Q:** vLLM 部署 8B 模型的完整启动命令是什么样的？
**A:** `vllm serve /path/to/merged-model --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.9 --dtype float16 --max-num-seqs 16`。关键参数：max-model-len 控制最大上下文长度（影响 KV cache 显存），gpu-memory-utilization 控制显存使用率上限（留10%给 CUDA context），max-num-seqs 控制最大并发序列数。

### Card 182
**Q:** vLLM 的 LoRA merge 部署 vs 动态 LoRA serving 有什么区别？
**A:** Merge 部署：训练后将 LoRA 权重合并到基座（W' = W + BA），得到一个独立模型直接加载，推理零额外开销。动态 serving：基座常驻显存，按请求动态切换 LoRA adapter，支持多租户（多个 LoRA 共用基座），但有 5-10% 额外延迟。我们的项目用 merge 部署——因为只有一个 LoRA，merge 更简单高效。

### Card 183
**Q:** PagedAttention 为什么能提高显存利用率？
**A:** 传统 KV cache 预分配连续显存，实际使用率低（序列长度参差不齐导致大量碎片）。PagedAttention 将 KV cache 切分为固定大小的 block（page），按需分配、动态回收，类似于操作系统的虚拟内存分页。显存利用率从 20-30% 提升到 80-90%，同等显存可支撑 2-4 倍并发。

### Card 184
**Q:** Continuous Batching 如何提升 GPU 利用率？
**A:** Static batching 等整个 batch 所有请求都完成才释放——短序列被长序列"拖累"，GPU 大量空转。Continuous batching：哪个请求完成了立即踢出 batch，新请求立即加入，GPU 几乎无空闲。在长短序列混合的场景下吞吐可提升 2-5 倍。

### Card 185
**Q:** vLLM 部署 8B 模型需要多少显存？怎么估算？
**A:** 显存 = 模型权重 + KV cache + 其他开销。模型权重：8B × 2 bytes (bf16) ≈ 16GB（merge 后）。KV cache ≈ 2 × num_layers × num_kv_heads × head_dim × max_seq_len × max_batch_size × 2 bytes。以 8B + max_seq_len=4096 + max_batch=32 为例，KV cache ≈ 8-12GB。总计约 24-28GB，单张 4090 (24GB) 需要调低 max-model-len 到 2048，RTX 5090 (32GB) 可跑满 4096。

### Card 186
**Q:** 你的项目 vLLM 压测了哪些指标？结果如何？
**A:** 在 RTX 5090 (32GB) 上压测 merge 后的 Qwen3-8B：吞吐 877 tok/s，TTFT（首 token 延迟）42ms，TPOT（每 token 延迟）约 15ms，支持 16 并发无 OOM。使用 OpenAI-compatible API 做压测，wrk/locust 工具模拟并发。

### Card 187
**Q:** vLLM 的 prefix caching 是什么？在 RAG 场景有什么用？
**A:** 多个请求共享相同的 system prompt 前缀时，只计算一次 prefix 的 KV cache，后续请求直接复用。在 RAG 场景中，system prompt + 检索到的文档前缀通常是固定的——prefix caching 可以让相同知识库的请求共享 prefix KV cache，大幅降低 prefill 阶段的延迟和计算量。

### Card 188
**Q:** vLLM 的 --max-num-seqs 和并发用户数是什么关系？
**A:** --max-num-seqs 是 vLLM 内部同时处理的序列数上限（即 max batch size），不等于 HTTP 并发数。HTTP 并发请求超过 max-num-seqs 时，vLLM 内部排队等待。需要根据显存和延迟要求平衡：太小则吞吐低，太大则排队延迟高（TTFT 尾巴变长）。

### Card 189
**Q:** vLLM 的调度策略中 prefill 和 decode 如何平衡？
**A:** Prefill（prompt 处理）是 compute-bound（矩阵乘法密集），Decode（逐 token 生成）是 memory-bound（显存带宽瓶颈）。vLLM 默认优先 decode（正在生成的请求），新请求的 prefill 在有空闲时插入。这个策略优先降低已有请求的 TPOT，但会增加新请求的 TTFT。可通过 --max-prefill-num-tokens 限制 prefill 的 token 数来控制 TTFT。

### Card 190
**Q:** vLLM 的 streaming 模式如何减少用户感知延迟？
**A:** 非 streaming：LLM 生成完所有 token 后一次性返回。Streaming（SSE/WebSocket）：每生成一个 token 立即推给客户端。首 token 延迟（TTFT）相同，但 streaming 让用户"看到模型在打字"——感知延迟远低于实际 total latency。我们的 API 同时支持 stream=True 和 stream=False。

### Card 191
**Q:** vLLM 如何做健康检查（health check）？
**A:** vLLM 启动后暴露 `/health` endpoint，返回 200 表示服务正常。生产环境中，负载均衡器（Nginx/HAProxy）定期 ping `/health`，连续失败 N 次则从 upstream 剔除该实例。还可自定义 `/health/metrics` 返回显存使用率、队列长度等深层指标，用于更精细的健康判断。

### Card 192
**Q:** 单卡 vLLM 部署和 Tensor Parallel 多卡部署的选择标准？
**A:** 8B 模型 bf16 ≈ 16GB，单张 24GB+ 显卡即可部署——不需要 TP。当模型权重超过单卡显存时（如 70B ≈ 140GB），需要 TP 切分到多卡。经验公式：单卡可用显存 < 模型权重 × 1.5 就需要考虑 TP。多卡部署也可用多副本（每卡一个完整模型）而非 TP——多副本提高并发，TP 降低单请求延迟。

---

## 九、工程落地 — 推理性能指标卡片（10张）

### Card 193
**Q:** TTFT (Time to First Token) 是什么？影响因素？
**A:** TTFT 是从发送请求到收到第一个 token 的时间。核心影响因素：prompt 长度（越长 prefill 越久）、模型大小（越大计算越多）、batch 中的序列数（prefill 阶段竞争计算资源）、排队时间（并发 > max-num-seqs 时）。我们的项目 TTFT ≈ 42ms（短 prompt + RTX 5090）。

### Card 194
**Q:** TPOT (Time per Output Token) 是什么？为什么重要？
**A:** TPOT 是生成阶段每个 token 的平均耗时。因为 decode 是 memory-bound，TPOT 主要由 KV cache 读取的显存带宽决定。用户的"打字速度感知"取决于 TPOT——TPOT < 50ms（即 >20 tok/s）用户基本无感知延迟。TPOT 过高会导致"卡顿"感。

### Card 195
**Q:** 推理吞吐（Throughput）和延迟（Latency）的 trade-off？
**A:** 增大 batch size → 提高吞吐（GPU 算力更充分利用）→ 但增加排队延迟和 TTFT。降低 batch size → 降低延迟 → 但 GPU 有空闲，吞吐下降。生产部署需要根据 SLA 设定延迟上限（如 p99 TTFT < 500ms），在此约束下最大化吞吐。

### Card 196
**Q:** LLM 推理中的 p50/p95/p99 延迟各代表什么？
**A:** p50 = 一半请求的延迟低于此值（典型用户体验）。p95 = 95% 的请求延迟低于此值（多数用户的"最坏体验"）。p99 = 99% 请求的延迟低于此值（长尾延迟，通常由排队/GC/OOM swap 引起）。生产系统更关注 p95/p99 而非平均值——因为平均值会被大量短请求拉低，掩盖少数用户遭遇的严重延迟。

### Card 197
**Q:** 并发增加时 TTFT 和 TPOT 如何变化？饱和点是什么？
**A:** 并发 < max-num-seqs 时：TTFT 和 TPOT 基本稳定（GPU 未饱和）。并发 > max-num-seqs 时：TTFT 急剧上升（请求在队列中等待 prefill），TPOT 略微上升（decode 竞争显存带宽）。饱和点 = max-num-seqs × (1 + avg_output_len / avg_prompt_len) 附近。我们的项目在 16 并发附近达饱和点。

### Card 198
**Q:** 如何做 vLLM 压测？
**A:** 工具：wrk（HTTP 压测）、locust（Python 压测框架）、vLLM 自带的 benchmark_serving.py。方法：固定 prompt（模拟真实请求），逐步增加并发（1→2→4→8→16→32→64），记录每个并发级别的 TTFT/TPOT/吞吐/错误率，找到延迟 SLA 下的最大并发数。我们的压测脚本使用 OpenAI Python client 做异步并发请求。

### Card 199
**Q:** context length 对推理性能有什么影响？
**A:** Prefill 阶段：context 越长 → 计算量 O(n²) 增长 → TTFT 显著增加。Decode 阶段：KV cache 随 context 线性增长 → 显存带宽压力增大 → TPOT 略微增加。长 context 的 KV cache 可能挤占大量显存，导致 max batch 减小。因此生产环境需要限制 max-model-len，用 RAG 等方式缩短 prompt。

### Card 200
**Q:** 量化部署（AWQ/GPTQ/INT4）对推理性能的影响？
**A:** 吞吐提升：显存占用减少 50-75% → 可支撑更大 batch → 吞吐提升 1.5-3 倍。延迟：通常略微降低（计算量减少但需要 dequant kernel）。质量损失：AWQ/GPTQ INT4 通常损失 <1% 在评测分数上。对于显存紧张的部署场景（如单卡 24GB 部署 13B），量化是首选方案。

### Card 201
**Q:** 推理部署中的 bf16 vs fp16 vs fp8？
**A:** bf16：最好的 numerical stability（范围同 fp32），推理首选。fp16：范围小（max 65504），偶尔出现 overflow，需要 attention scaling。fp8（H100/B200）：显存减半，吞吐翻倍（专用 tensor core），但精度损失需要量化方案（FP8 KV cache 可能影响长序列质量）。我们的项目用 bf16——RTX 5090 32GB 足够放下 8B 模型。

### Card 202
**Q:** 为什么我们的项目吞吐能达到 877 tok/s？
**A:** 8B 模型计算量相对较小 + RTX 5090 算力强（~80 TFLOPS bf16）+ continuous batching 高 GPU 利用率 + merge 后无 LoRA 额外开销 + 短 prompt（RAG 检索后拼接）。877 tok/s 是 batch 16 下的总吞吐，单用户 TPOT 约 15ms（~67 tok/s per user），感知流畅。

---

## 十、工程落地 — RAG延迟优化卡片（15张）

### Card 203
**Q:** RAG 端到端延迟怎么拆解？
**A:** 总延迟 = Query 理解/NER（10-20ms）+ Embedding 向量化（5-10ms）+ 向量检索（5-50ms，取决于索引大小和 top-k）+ Reranker（20-100ms，可选）+ Prompt 构建（<1ms）+ LLM prefill（50-200ms，取决于 prompt 长度）+ LLM decode（每 token 10-30ms）。典型端到端延迟 500ms-2s。我们的项目因 Safety-RAG 的 verifier 额外增加了约 300ms。

### Card 204
**Q:** embedding 检索的延迟瓶颈在哪里？如何优化？
**A:** 瓶颈：向量维度高（1024-4096维）+ 索引规模大（百万级）时，精确检索 O(N×d) 太慢。优化方案：FAISS IVF（倒排索引，先聚类再在最近聚类中搜索，速度提升 10-100 倍，召回率损失 <2%）；HNSW（层次化可导航小世界图，最常用，搜索 O(log N)）；PQ（乘积量化，压缩向量，内存更小）。我们的 50 万知识库用 FlatIP 就够了（<50ms）。

### Card 205
**Q:** Reranker 会增加多少延迟？可以跳过吗？
**A:** Reranker 延迟 = 模型推理时间（cross-encoder 对每个 query-doc pair 打分），top-20 rerank 到 top-5 约 50-200ms（取决于模型大小和硬件）。简单问题（query-doc 相似度得分 >0.8）可跳过 reranker 降低延迟；复杂/模糊查询需要 reranker 保精度。可用轻量 reranker（如 bge-reranker-base 278M）平衡速度和精度。

### Card 206
**Q:** RAG 的 p99 延迟为什么容易飙高？
**A:** 主要原因：1）检索出现冷数据（索引不在 CPU cache）→ 磁盘 IO 导致个别请求检索延迟突然放大；2）LLM 生成时遇到需要长输出的 query（500+ tokens）→ 总延迟被拖长；3）verifier 在复杂 case 上的额外推理时间；4）多路召回 + reranker 的同步等待（木桶效应）。优化 p99 需要：异步并行化多路召回 + 超时降级 + 请求级限流。

### Card 207
**Q:** 如何监控 RAG 各阶段的延迟？
**A:** 在每个阶段打点：query_received → embedding_start/end → retrieval_start/end → reranker_start/end → generation_first_token (TTFT) → generation_end → verifier_start/end（如有）→ response_sent。使用 OpenTelemetry 或自定义 middleware，按 request_id 串联全链路。Grafana 面板按阶段展示 p50/p95/p99 延迟，快速定位瓶颈。

### Card 208
**Q:** 检索 top-k 越大越好吗？和延迟/质量的关系？
**A:** 不是。top-k 增大 → 召回率提升（更多相关文档被检索到）→ 但精度可能下降（不相关文档混入）+ Reranker 耗时增加 + LLM prompt 变长 → prefill 延迟增加。通常 top-k=5-10 在延迟和质量间取得平衡。我们的 Safety-RAG 用 top-5——知识库质量高，5 条足够覆盖。

### Card 209
**Q:** chunk size 对 RAG 延迟有什么影响？
**A:** Chunk 小（128 token）→ embedding 检索快但文档碎片化 → 可能需要更多 chunk 拼凑完整上下文。Chunk 大（1024 token）→ 检索慢（向量更多）+ 但每个 chunk 更完整。chunk size = 256-512 是常用平衡点。也影响 LLM prompt 长度（拼接 5 个 512-token chunk = 2560 token prefill）。我们的 chunk size 256-512，overlap 64。

### Card 210
**Q:** Safety-RAG 的 verifier 如何优化延迟？
**A:** 1）Verifier 用轻量版的微调模型（Qwen3-8B 的 QLoRA adapter，可用 vLLM 部署以利用 continuous batching）；2）只在生成内容涉及高风险关键词时触发 verifier（动态路由）；3）Verifier 的 prompt 尽可能短（只包含 query + response + top-3 知识）；4）Verifier 和 generation 可以在不同 GPU 上并行执行。当前项目 verifier 在单卡上增加约 300ms。

### Card 211
**Q:** RAG 的 embedding 模型部署有哪些延迟优化？
**A:** 1）Embedding 模型做 ONNX/TensorRT 推理优化（延迟降低 2-3 倍）；2）使用 GPU 做 embedding（batch 场景下比 CPU 快 10-50 倍）；3）对高频文本做 embedding cache；4）使用更轻量的 embedding 模型（bge-small 384维 vs bge-large 1024维，延迟降低 60%）。小模型 + GPU 推理，单条 embedding <5ms。

### Card 212
**Q:** RAG 是否可以去掉 reranker 来降低延迟？
**A:** 可以，但需要评估 trade-off。去掉 reranker 省 50-200ms，但检索精度下降可能导致生成质量降低（不相关文档出现在 top-k 中）。如果 embedding 检索精度已经很高（如 biomedical 领域嵌入对医疗 query 的 Recall@5 >90%），可以去掉 reranker。判断标准：用 Recall@k 和 MRR 评估有无 reranker 的检索差异，如果差距 <5% 可以去掉。

### Card 213
**Q:** 多个 RAG pipeline 阶段如何做超时降级？
**A:** 为每个阶段设 timeout：embedding 超时（>500ms→用关键词 fallback）、检索超时（>200ms→返回空结果，LLM 基于自身知识回答+提示"信息检索超时"）、reranker 超时（>300ms→跳过 reranker，直接用检索排序）、generation 超时（>10s→截断输出+提示"回答因超时被截断"）。用 asyncio.wait_for() 实现。超时后记录日志用于后续优化。

### Card 214
**Q:** 如何减少 RAG 中 LLM prefill 阶段的延迟？
**A:** Prefill 延迟 ∝ prompt 长度。减少策略：1）精简 system prompt 和检索到的文档（去除非关键信息）；2）检索结果做摘要压缩（用轻量模型先压缩长文档）；3）vLLM prefix caching（相同 system prompt 共享 KV cache）；4）限制 max-model-len（截断超长 prompt，牺牲一定质量换取可预测的延迟）；5）将部分 prompt 内容（如免责声明）移到 post-processing。

### Card 215
**Q:** RAG 场景中 embedding 检索用 GPU 还是 CPU？
**A:** CPU 优势：embedding 模型小（300M-1B），CPU 够快（单条 <10ms）；CPU 不会和 LLM 抢 GPU 显存；部署简单。GPU 优势：batch 场景下快 10-50 倍；一致性高（同设备推理）。推荐：小规模（<100 QPS）用 CPU embedding，大规模（>500 QPS）用 GPU embedding（或独立 embedding GPU）。我们的项目用 CPU embedding（bge-large-zh 在 CPU 上单条 ~8ms）。

### Card 216
**Q:** RAG 系统中 LLM generation 如何分段流式输出？
**A:** Streaming 模式下，LLM 每生成一个 token 立即推给客户端。但在 RAG 中，前面的检索+验证是阻塞的——用户在检索阶段"等待"看不到任何输出。优化：1）检索完成后先推送"正在为您查找相关信息..."的提示 token；2）generation 阶段全程流式输出；3）verifier 在 generation 结果完整后异步执行，如果发现问题再推送修正内容（二次流式）。这样用户感知延迟 = 检索延迟 + TTFT（而非 generation 结束）。

### Card 217
**Q:** RAG 延迟优化的优先级排序？
**A:** 1）最大头：LLM generation（占 60-80%）→ streaming + 限制 max_tokens；2）次大头：Reranker（占 10-20%）→ 评估是否必要 / 用轻量模型；3）中等：向量检索（占 5-10%）→ FAISS 索引优化；4）最小：Embedding（<5%）→ 基本不需优化。优化投入产出比：先优化 generation 和 reranker，再考虑检索索引。

---

## 十一、工程落地 — RAG并行化卡片（10张）

### Card 218
**Q:** dense 检索和 sparse 检索如何并行执行？
**A:** 使用 asyncio.gather() 同时发起 dense embedding 检索（FAISS）和 sparse BM25 检索（Elasticsearch/自建索引）：`dense_res, sparse_res = await asyncio.gather(dense_search(query), sparse_search(query))`。两者独立无依赖，并行后总延迟 = max(dense_latency, sparse_latency) 而非两者之和。结果用 RRF (Reciprocal Rank Fusion) 融合。

### Card 219
**Q:** RAG pipeline 的异步化设计架构是怎样的？
**A:** 整体用 async/await 异步框架（FastAPI + asyncio）。请求进来后：query 预处理（同步）→ await asyncio.gather(embedding_search, keyword_search) 并行检索 → await reranker 重排序 → await llm_generate 生成 → await verifier 验证（可选）。关键：每步 await 时释放事件循环，其他请求可以在这个间隙被处理。配合 uvicorn workers 实现并发。

### Card 220
**Q:** asyncio.gather 中某个检索源失败怎么处理？
**A:** 使用 return_exceptions=True：`results = await asyncio.gather(dense_search(query), sparse_search(query), return_exceptions=True)`。如果某个源抛异常，对应位置返回 Exception 对象而非崩溃整个 gather。处理逻辑：dense 失败 → 只用 sparse 结果 + 记录告警；sparse 失败 → 只用 dense 结果 + 记录告警；两者都失败 → 降级为纯 LLM 回答 + 提示"知识检索暂时不可用"。

### Card 221
**Q:** RAG 中检索和生成可以并行（重叠）吗？
**A:** 部分可以。常规流水线是 检索→生成（串行）。优化版：1）检索到 top-3 后立即开始 prefill（不等 top-5 全部到位），后续检索结果可以追加到 prompt；2）使用 speculative RAG：先用轻量检索的 top-1 结果开始生成，同时进行深度检索+reranker，如果深度检索结果和轻量结果不同，中断并重新生成。但工程复杂度高，大部分场景串行 pipeline 足够。

### Card 222
**Q:** embedding 请求可以 batch 化吗？
**A:** 在高并发场景下，可以将 N 个独立的 embedding 请求攒成 batch 一次性 GPU 推理：embedding model 的吞吐在 batch_size=32 时比逐条处理高 10-20 倍。用 asyncio.Queue 收集请求，达到 batch_size 或超时后批量处理。但引入了一个 batch 延迟（等攒够 batch），需要 balance 延迟和吞吐。适用于后台离线索引构建，不适用于实时低延迟查询。

### Card 223
**Q:** RAG 结果融合（RRF/加权/学习排序）的性能开销？
**A:** RRF (Reciprocal Rank Fusion) 计算量极小（几个倒数和），<1ms。加权融合（对每个文档的 dense/sparse 分数做加权平均）也 <1ms。学习排序（用轻量模型对融合结果排序）开销类似 reranker。大部分场景 RRF 足够好，性能可忽略。

### Card 224
**Q:** 多路召回的各路结果如何做去重和融合？
**A:** 1）以文档 ID（或文本 MD5）为主键合并各路结果；2）对同一文档的多路分数，用 RRF 融合：`score(doc) = Σ 1/(k + rank_i)`，k=60 是常用常数；3）如果使用原始相似度分数，需要先做分数归一化（min-max 或 z-score）再加权。去重避免 LLM prompt 中出现重复内容，浪费 prefill 计算。

### Card 225
**Q:** 异步 RAG pipeline 的并发控制怎么做？
**A:** 使用 asyncio.Semaphore 限制并发数：`sem = asyncio.Semaphore(max_concurrent_rag)`。每个 RAG 请求先 `await sem.acquire()`，完成后 `sem.release()`。这防止同时有太多请求在 LLM generation 阶段（最占显存）导致 OOM。max_concurrent_rag = vLLM max-num-seqs × 0.8 是一个合理的起始值。

### Card 226
**Q:** FastAPI 的 async endpoint 和 sync endpoint 在 RAG 场景的区别？
**A:** Async endpoint（`async def endpoint()`）：不阻塞事件循环，I/O 密集操作（HTTP 调用 vLLM、数据库查询）可以并发处理，适合 RAG 这种有大量 I/O 等待的 pipeline。Sync endpoint（`def endpoint()`）：FastAPI 在线程池中执行，每个请求独占线程，并发量受线程池大小限制。RAG pipeline 应使用 async endpoint。

### Card 227
**Q:** RAG 系统中的"流水线化"（pipelining）怎么实现？
**A:** 将 RAG 拆分为独立 stages，每个 stage 异步执行：Stage1(query preprocessing)→Stage2(retrieval)→Stage3(rerank)→Stage4(generation)→Stage5(verification)。Stage N 的结果通过 asyncio.Queue 传递给 Stage N+1。多个请求的 Stage1 完成后，Stage2 可以同时处理它们的批量检索。这种设计在高吞吐场景下减少 GPU 空等时间，类似于 CPU 的指令流水线。

---

## 十二、工程落地 — RAG缓存卡片（8张）

### Card 228
**Q:** RAG 系统中可以做哪几层缓存？
**A:** 1）Query Cache：相同/相似问题的最终回答直接返回（key=query embedding/语义哈希）；2）Embedding Cache：相同文本段落的 embedding 向量缓存（key=文本 MD5）；3）Retrieval Cache：相同 query 的检索结果缓存（key=query，value=top-k doc IDs+初始分数）；4）Rerank Cache：相同 query-doc pair 的 rerank 分数缓存（key=query+doc_id）；5）LLM Generation Cache：相同 prompt 的 LLM 输出缓存（需谨慎，可能暴露隐私）。

### Card 229
**Q:** Query Cache 的设计和失效策略？
**A:** 用 query embedding（向量）做 key，Redis 存储 value（完整回答+安全标签）。相似 query 匹配：新 query 的 embedding 与缓存 key 做余弦相似度，>0.95 视为命中。TTL：常见问题 1 小时，罕见问题 10 分钟（因为它们不太可能被重复问）。医疗场景注意：个性化健康问题不应被缓存（"我的血压 140 正常吗" vs "140/90 是高血压吗"）。

### Card 230
**Q:** Embedding Cache 怎么实现？
**A:** 预先计算知识库所有 chunk 的 embedding 存 FAISS。对于用户 query，用 Redis 做 key-value：`query_embedding_cache.get(query_text_hash)` → 命中直接返回向量，未命中计算后写入。知识库 chunk 的 embedding 只需计算一次（离线），不存在缓存失效问题。Query embedding 本质上不需要缓存（计算只需 5-10ms），除非 QPS 极高。

### Card 231
**Q:** 检索结果缓存（Retrieval Cache）的设计要点？
**A:** Key = query 文本（或语义哈希）；Value = 检索到的 top-k 文档 ID 列表 + 初始相似度分数。TTL 取决于知识库更新频率——如果知识库每小时更新一次，TTL 设为 5-10 分钟。知识库更新后需批量 invalidate 相关缓存。优点：命中后跳过 embedding 和检索两个阶段，节省 15-50ms。

### Card 232
**Q:** Rerank Cache 的设计？
**A:** 对 query + 文档 ID pair 做 rerank 分数缓存。key = hash(query + doc_id)，value = rerank_score。因为 reranker 是 RAG 中延迟最高的可缓存环节（50-200ms），缓存命中收益最大。但 cache 基数大（query × doc 组合多），命中率可能低。可用 LRU 策略限制内存。如果知识库更新频率低（日级），缓存价值高。

### Card 233
**Q:** 医疗知识库的检索缓存有什么特殊注意事项？
**A:** 1）隐私：绝对不缓存包含患者个人信息（PHI）的 query 和回答——需要在缓存 key 之前做 PHI 检测和脱敏；2）时效性：医疗知识更新（新药上市、指南更新）后必须同步 invalidate 相关缓存；3）个性化：相同 query "我头痛怎么办" 不应给不同用户相同缓存回答——健康建议有个体差异；4）安全：缓存的不安全回答如果被高频命中会放大安全风险，缓存 value 应包含安全标签。

### Card 234
**Q:** 缓存命中率低怎么办？
**A:** 1）扩大相似度匹配范围（query embedding cosine >0.9 而非 >0.95）；2）用更短的 TTL 增加"新鲜度"但对命中率无帮助；3）预处理 query：做 query 改写和归一化（去除无意义词、统一术语），提高相似 query 匹配率；4）使用语义哈希（SimHash）而非精确匹配，模糊匹配更广范围的相似 query。RAG 系统中检索缓存命中率 20-40% 已是很好的水平。

### Card 235
**Q:** 缓存的一致性问题和失效策略？
**A:** 多层缓存的一致性：Retrieval Cache 更新了 → Rerank Cache 也需要同步失效（因为检索结果变了）。策略：1）事件驱动失效——知识库更新事件 → 广播到所有缓存层 → 级联失效；2）版本号机制——知识库有版本号，缓存 key 包含版本号，版本号变化自动 miss；3）TTL + 主动失效混合——TTL 保底，重要更新主动 purge。简单项目用 TTL + 知识库更新后 purge all 即可。

---

## 十三、工程落地 — 动态路由卡片（8张）

### Card 236
**Q:** 什么是 RAG 的动态路由？解决了什么问题？
**A:** 动态路由 = 根据 query 特征将请求分流到不同处理 pipeline。简单问题（"感冒了多喝水有用吗"）走 fast path（light-RAG：检索→直接生成，无 verifier）；复杂/高风险问题（"某抗生素对某菌的耐药率"）走 full Safety-RAG（检索→生成→验证→修正）。好处：简单问题延迟低（省 verifier 的 300ms），高风险问题安全性高，平均延迟降低 20-40%。

### Card 237
**Q:** 如何判断一个 query 是"简单"还是"复杂"？
**A:** 多信号融合：1）query 长度（<15 字→简单）；2）是否包含医疗实体（药物名/疾病名/检查指标）→如有实体可能复杂；3）是否包含危险信号词（"剂量""用法""我得了XX""严重"）→高风险；4）用轻量分类器或规则引擎打分。简单的规则：query 包含药物+剂量 = 高风险；query 是常识性科普 = 低风险；其他 = 中风险走完整流程。

### Card 238
**Q:** 急症检测如何实现优先路由？
**A:** 关键词/正则匹配急症模式列表（"胸痛""呼吸困难""意识丧失""大出血""中毒""过敏休克"等），匹配到后立即触发急症路由：跳过所有检索和生成，直接返回急救建议模板（"请立即拨打 120 或前往最近医院急诊科"）+ 基础急救指导（止血/心肺复苏等通用知识）。这条路径延迟 <50ms（纯规则匹配），优先级最高，同一请求队列中插队处理。

### Card 239
**Q:** 用药咨询类请求如何路由到高安全等级 pipeline？
**A:** 检测到用药相关实体后（药物名+剂量/用法/相互作用/禁忌），自动路由到 Safety-RAG full path：强制检索药品说明书和相互作用数据库 + verifier 严格检测处方级建议 + 强制添加"请咨询医生/药师"免责声明。这条路径即使 query 看起来简单（如"阿莫西林一次吃几粒"），也走 full safety path——因为即使提供参考剂量也可能被误解为处方。

### Card 240
**Q:** 动态路由的规则引擎如何避免过度复杂？
**A:** 用决策树而非复杂 ML 模型：Root → 急症？（yes→紧急路由）→ 用药相关？（yes→高安全路由）→ 包含医疗实体？（yes→full Safety-RAG）→ 常识科普？（yes→light-RAG）。规则总数控制在 20-30 条，每条有明确优先级。规则的假阴性（应走安全路由但走了快路由）比假阳性（应走快路由但走了安全路由）后果严重得多——宁可安全路由多走，不可漏掉高风险 query。

### Card 241
**Q:** 动态路由的 fallback 机制？
**A:** 即使走了 light-RAG 路径（无 verifier），也保留最低限度的安全防护：post-processing 规则检查输出中是否包含处方级表述/诊断声明（正则兜底）。如果 light-RAG 路径的输出被 post-processing 标记为 unsafe → 自动升级到 full Safety-RAG 路径重新处理。这是两级路由：路由决策 + 事后兜底，确保不因路由错误导致安全问题。

### Card 242
**Q:** 动态路由对延迟分布的影响？
**A:** 引入路由后延迟从单一分布变为多峰分布：fast path（p50 ~300ms）和 full path（p50 ~800ms）。p50 和 p95 都可能降低（因为大部分 query 走 fast path），但 full path 的 p99 仍然高。监控需要分开看两条路径的延迟分布，避免平均延迟掩盖 full path 的性能问题。

### Card 243
**Q:** 如何监控动态路由的效果？
**A:** 关键指标：1）各路流量占比（fast/safe/urgent 各占多少）；2）各路 p50/p95/p99 延迟；3）路由误判率：人工抽检 light-RAG 路径的输出，看是否应该走安全路径但被误路由（每周 100 条抽检）；4）安全路径升级率：light-RAG 的 post-processing 触发升级到 full path 的比例——如果 >10%，说明路由规则太宽松。

---

## 十四、工程落地 — 分布式训练卡片（6张）

### Card 244
**Q:** DDP (DistributedDataParallel) 的工作原理？
**A:** 每个 GPU 上有完整的模型副本，每个 step 处理不同的 micro-batch。前向传播各自独立。反向传播时，`loss.backward()` 自动触发 all-reduce 通信——各 GPU 的梯度求平均后同步到所有 GPU。然后各 GPU 独立执行 optimizer.step()（因为梯度相同，更新后参数一致）。通信发生在 gradient computation 和 optimizer step 之间，默认用 NCCL 后端。

### Card 245
**Q:** FSDP 和 DDP 的核心区别？
**A:** DDP：每个 GPU 有完整模型副本（参数+梯度+优化器状态全量），显存占用 = 模型总大小 × 3-4。FSDP：模型参数、梯度、优化器状态都分片（shard）到各 GPU。前向/反向传播时按需 all-gather 参数，计算完后释放。显存占用 ≈ 模型总大小 / N（N=GPU数）。FSDP 增加了通信次数（每次前向/反向都要 gather），但显存效率高，是训练大模型的首选。

### Card 246
**Q:** ZeRO-1/2/3 各自的优化对象和通信开销？
**A:** ZeRO-1：只分片优化器状态（optimizer states），通信量 ≈ 参数量的 1 倍（all-reduce 梯度时）。ZeRO-2：分片优化器状态 + 梯度，通信量同 ZeRO-1（梯度 all-reduce 本身就需要）。ZeRO-3：分片优化器状态 + 梯度 + 参数，通信量 ≈ 参数量的 1.5 倍（多了参数的 all-gather）。ZeRO-3 显存最省（1/N），但通信增加了约 50%，在大模型训练中这是必须的 trade-off。

### Card 247
**Q:** DeepSpeed 配置文件的关键参数？
**A:** `train_batch_size`（总 batch size，auto 由 DeepSpeed 计算）、`gradient_accumulation_steps`、`zero_optimization.stage`（1/2/3）、`fp16.enabled` 或 `bf16.enabled`、`zero_optimization.offload_optimizer.device`（cpu/nvme，ZeRO-Offload 将优化器状态 offload 到 CPU 内存）、`zero_optimization.offload_param.device`（ZeRO-Infinity 将参数 offload 到 NVMe）。`stage3_gather_16bit_weights_on_model_save` 在保存 checkpoint 时自动 gather 分片参数。

### Card 248
**Q:** torchrun 启动分布式训练的典型命令？
**A:** `torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train.py`。nproc_per_node = 每个节点的 GPU 数量；nnodes = 节点总数；node_rank = 当前节点的编号（0 为 master）。脚本内用 `local_rank = int(os.environ["LOCAL_RANK"])` 获取当前 GPU 编号，`torch.cuda.set_device(local_rank)` 绑定设备。DDP 会自动处理 all-reduce 通信。我们的项目用单卡 QLoRA 训练，不需要 torchrun。

### Card 249
**Q:** 8B 模型全量微调需要多少 GPU 显存？QLoRA 如何降低需求？
**A:** 全量微调 8B (bf16)：参数 16GB + 梯度 16GB + 优化器状态（Adam）32GB + 激活值（batch=1, seq=2048）~8GB ≈ 72GB。实际有效 batch=8 时激活值可能 >40GB，总需 ~100GB+。QLoRA：基座 4bit 量化（~4GB）+ LoRA 参数+梯度+优化器 ≈ 1GB + 激活值 ~8GB = ~13GB。省了近 85-90% 显存。我们的 RTX 5090 (32GB) 做全量微调不够，但 QLoRA 绰绰有余。

---

## 十五、工程落地 — 分布式推理卡片（5张）

### Card 250
**Q:** Tensor Parallelism (TP) 和多副本部署怎么选？
**A:** TP：将模型层内参数切分到多 GPU，每层计算时 GPU 间通信（all-reduce/all-gather）。适合模型大小超过单卡显存（如 70B 需 4-8 卡 TP）。优点是不用修改模型，缺点是 GPU 间通信引入延迟（NVLink > PCIe）。多副本：每个 GPU 加载完整模型，请求轮询分发。适合模型能放下单卡、需求高并发的场景（如 8B 部署 4 副本，每副本独立服务）。我们的 8B 模型单卡 RTX 5090 足够，用多副本而非 TP。

### Card 251
**Q:** vLLM 多 GPU 部署的关键参数？
**A:** `--tensor-parallel-size N`：切分模型到 N 张 GPU（需 N 卡 NVLink/PCIe 互联）。`--pipeline-parallel-size M`：按层切分到 M 组 GPU（较少用）。总 GPU 数 = N × M。启动示例：`vllm serve model --tensor-parallel-size 2` 把模型切到 2 张 GPU。配合 CUDA_VISIBLE_DEVICES=0,1 指定使用的 GPU。

### Card 252
**Q:** Nginx 反向代理 + 多 vLLM 实例的架构？
**A:** Nginx 配置 upstream 指向多个 vLLM 实例（如 localhost:8000, localhost:8001, localhost:8002），每个实例绑定不同 GPU。负载均衡策略：轮询（round-robin，默认）适合各实例性能一致；最少连接（least_conn）适合请求长短不一；IP hash 适合需要会话保持的场景。健康检查：`proxy_next_upstream error timeout` 配合 upstream 的 max_fails/fail_timeout，实例挂了自动切换。

### Card 253
**Q:** 多副本部署的负载均衡策略有哪些？
**A:** 1）轮询（Round Robin）：请求依次分发，简单均衡，适合各副本性能一致；2）最少连接（Least Connections）：发给当前活跃请求最少的副本，适合请求处理时间差异大的场景；3）一致性哈希：同一用户的请求路由到同一副本（利用 KV cache 的 prefix caching）；4）加权：各副本 GPU 性能不同时（如 A100 vs 4090 混合集群），按权重分配流量。默认轮询足够，除非有长短请求混合。

### Card 254
**Q:** 分布式推理中的请求排队和 backpressure 怎么处理？
**A:** 当所有 vLLM 实例的 max-num-seqs 都打满时，新请求需要在 Nginx/网关层排队。策略：1）Nginx 的 `max_conns` 限制每个 upstream 的连接数；2）请求超时（proxy_read_timeout）避免客户端无限等待；3）HTTP 429（Too Many Requests）返回给客户端，客户端按 Retry-After 退避重试；4）优先级队列：急症请求可插队（通过特殊 header `X-Priority: urgent`）。我们的项目未实现多副本和 Nginx 层——单实例 vLLM 直接暴露 API。

---

## 十六、工程落地 — 线上监控卡片（4张）

### Card 255
**Q:** 模型线上服务需要监控哪些核心 metrics？
**A:** 黄金信号（Golden Signals）：1）Latency（TTFT p50/p95/p99 + TPOT p50/p95/p99 + 端到端延迟）；2）Traffic（QPS、并发连接数、请求到达率）；3）Errors（4xx/5xx 错误率、模型输出异常率、安全告警率）；4）Saturation（GPU 利用率、显存使用率、vLLM 队列长度、CPU 利用率）。四个信号覆盖了用户体验、系统健康、容量规划三个维度。

### Card 256
**Q:** Prometheus + Grafana 如何集成到模型服务中？
**A:** 模型服务暴露 `/metrics` endpoint（使用 prometheus_client 库），输出 Counter/Gauge/Histogram 类型的指标。Prometheus 定期 scrape 这个 endpoint（如每 15s），存储时序数据。Grafana 从 Prometheus 查询并可视化。关键 Histogram：`rag_request_duration_seconds`（全链路延迟），`vllm_ttft_seconds`（首 token 延迟），`vllm_tpot_seconds`（每 token 延迟）。Grafana dashboard 按面板展示各项指标的时序图和 p50/p95/p99。

### Card 257
**Q:** 日志系统怎么设计？哪些字段必须记录？
**A:** 结构化日志（JSON 格式）包含：request_id（全链路追踪）、timestamp、user_id（脱敏）、query_text（脱敏后）、routing_path（fast/safe/urgent）、safety_label（safe/minor_issue/unsafe）、retrieval_docs（top-k doc IDs）、generation_tokens、total_latency_ms、stage_latencies（embedding/retrieval/rerank/generation/verifier 各阶段延迟）、error_info（如有）。日志级别：INFO（正常请求）、WARN（降级/超时）、ERROR（服务异常）。使用 ELK（Elasticsearch + Logstash + Kibana）或 Loki 存储和检索。

### Card 258
**Q:** 告警规则如何设计？P0/P1/P2 怎么分？
**A:** P0（立即响应，5 分钟内）：vLLM 服务不可用（health check 连续 3 次失败）、GPU OOM 错误率 >10%、模型输出严重安全违规（处方/诊断）检出率突增。P1（30 分钟内响应）：p99 延迟超过 SLA 2 倍持续 5 分钟、5xx 错误率 >5%、GPU 显存使用率 >95% 持续 10 分钟。P2（工作时间处理）：p95 延迟趋势上升（日级）、检索缓存命中率下降 >20%、知识库覆盖不足告警率增加。告警通道：P0→电话+即时通讯，P1→即时通讯，P2→邮件/工单。

---

## 十七、工程落地 — 安全审计卡片（4张）

### Card 259
**Q:** 模型输出的危险内容检测怎么做？
**A:** 多层检测：1）规则层：正则匹配处方模式（药物+剂量+用法）、诊断声明（"你得了XX病"）、紧急误导（"不需要看医生"）；2）模型层：Safety-RAG verifier 的事实一致性检查 + 安全标签（safe/minor_issue/unsafe）；3）审计层：定期人工抽检 unsafe 标记的输出（每日 top-50 unsafe 输出人工复核）。三层互补：规则快但覆盖窄，模型覆盖宽但有误判，人工最高准确但成本高。

### Card 260
**Q:** 模型服务的熔断（Circuit Breaker）怎么设计？
**A:** 熔断器状态机：Closed（正常）→ Open（熔断）→ Half-Open（试探恢复）。触发条件：连续 N 次请求失败（如 5 次）或错误率超过阈值（如 50%）。熔断后：所有请求直接返回 fallback 响应（如"服务暂时不可用，请稍后重试"或静态 FAQ）。Half-Open：每隔 probe_interval（如 30s）放行一个探测请求，成功则恢复 Closed，失败则维持 Open。在模型服务中，熔断通常在网关层（Nginx/API Gateway）实现，而非 vLLM 内部。

### Card 261
**Q:** 服务降级策略有哪些层次？
**A:** Level 1（最优降级）：vLLM 不可用 → 回退到轻量 CPU 模型（如 Qwen3-1.5B 量化版），回答质量降低但服务可用。Level 2（次优降级）：模型完全不可用 → 回退到静态 FAQ 检索（ES 匹配预置的答案库），返回最相似的标准回答。Level 3（最后兜底）：所有后端不可用 → 返回预设的降级消息 + 建议联系人工客服。降级策略需要在系统设计阶段就考虑，不能临时抱佛脚。

### Card 262
**Q:** 安全审计的长期数据闭环怎么做？
**A:** 用户反馈收集（赞/踩/举报）→ 安全团队抽检"踩"的回答（重点关注处方/诊断/严重误导）→ 标注正确回答 → 加入 training data 或 DPO preference data → 定期（月/季度）重新训练模型 → 灰度发布新模型 → 监控安全指标变化 → 继续收集反馈。关键指标：重复安全问题率（同一类安全问题反复出现说明闭环失效）、闭环周期（从发现到修复的时间）、反馈覆盖率（有反馈的问题占总问题的比例）。

---

## 背诵版总结

1. 共262张卡片，分17类：概念40+公式25+代码20+项目40+追问25+易错点20+热点10+工程落地82张
2. **工程落地卡片按模块分布**：vLLM部署12+推理性能10+RAG延迟15+RAG并行10+RAG缓存8+动态路由8+分布式训练6+分布式推理5+线上监控4+安全审计4
3. **面试前一天**：快速过一遍所有卡片，重点看标记"未掌握"的
4. **卡片背诵技巧**：遮住答案自问自答，答不出的当天反复3遍
5. **项目卡(40张)最重要**：面试中80%的时间在聊项目，数字必须准确记忆
6. **公式卡(25张)次重要**：DPO loss、SFT loss、Attention公式必须能默写
7. **易错点卡(20张)是面试"免死金牌"**：提前知道坑在哪里，面试不踩雷
8. **工程落地卡(82张)是差异化利器**：当面试官追问"部署经验""工程能力"时，用这些卡片的内容证明你理解工程原理、做过本地部署压测、知道生产级还需要什么
