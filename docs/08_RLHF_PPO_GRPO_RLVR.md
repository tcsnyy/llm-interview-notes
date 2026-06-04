# 08 RLHF / PPO / GRPO / RLVR

> **项目背景声明（面试必须明确说）**：本项目（医学 LLM Teacher）中**没有实现 PPO/GRPO/RLVR，也没有训练 Reward Model**。项目采用 SFT → DPO → LLM-as-Judge → Safety-RAG 路线。但面试中需要理解这些概念的原理、适用场景以及与 SFT/DPO 的区别。以下内容基于论文阅读、开源实现（MiniMind 简化版 PPO/GRPO）和面试准备整理。

---

## 面试 1 分钟 / 3 分钟回答版本

### 1 分钟版本

RLHF 是三阶段管线——SFT 学指令格式，Reward Model 学人类偏好，PPO 用强化学习优化 policy。PPO 需要同时跑 4 个模型（policy、reference、reward、critic），非常重。DPO 直接拿偏好数据做对比学习，省掉了 RM 和 RL。GRPO 是 PPO 的简化版：去掉 value model/critic，用同一 prompt 下的一组采样的相对好坏来算 advantage，特别适合 reasoning 场景。RLVR 更进一步，直接用规则验证器（代码能否跑通、数学答案对不对）作为 reward 信号，不需要人工标注偏好。DeepSeek-R1 就是 GRPO + RLVR 的典型。我们项目选择 SFT+DPO 是因为：医学标注成本高、安全风险大、DPO 稳定可控。

### 3 分钟版本

**RLHF 全景**：InstructGPT 提出经典三阶段——SFT 让模型学会指令格式，RM 在偏好标注上训练一个打分器，PPO 用 KL 约束下的 reward 信号优化 policy。PPO 是 clipped surrogate objective，限制每步更新幅度，加上 KL penalty 防止 reward hacking。

**PPO 的工程成本**：需要同时加载 policy、reference、reward model、critic（value model）四个模型，显存爆炸。训练不稳定，reward 容易跑飞。所以后来有了 DPO，直接在 preference pair 上做对比学习，把 reward 建模隐式化，一条 loss 搞定。

**GRPO 的创新**：DeepSeekMath 提出。核心 idea：不再训练单独的 value model，而是对同一个 prompt 采 N 个 response，用这组 response 的 reward 的均值和标准差做归一化，得到 group-relative advantage。省掉价值模型后，训练省一半显存，也更稳定。特别适合 reasoning：因为 reasoning 效果好坏的差距足够大，group 内部的相对比较就够了。

**RLVR**：Reinforcement Learning with Verifiable Rewards。不用神经网络 reward，直接用规则验证——数学题对答案、代码跑测试用例、逻辑推理验证格式。DeepSeek-R1 用 rule-based reward 做大规 GRPO，发现模型自主涌现了长 chain-of-thought 和自我反思行为——这就是 R1 零样本 RL 直接带 reasoning 能力的原因。

**为什么我们项目没做**：医学 QA 缺乏可验证 ground truth，rule-based reward 只能覆盖选择题、结构化输出等有限场景；训练 RM 需要大量医生标注，成本太高；DPO 在资源有限下更实用。

---

### Q: RLHF 三阶段是什么？⭐⭐⭐⭐⭐

![PPO 训练流程图](images/PPO%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

**阶段一（SFT）**：在高质量指令-回答对上微调基座模型，目的是让模型学会遵循指令、输出符合人类偏好的格式。本项目中：使用 HuatuoGPT2-SFT-GPT4-140K 子集 + Teacher 生成答案，最终 55,000 条 SFT 数据。

**阶段二（RM 训练）**：RM 是一个独立训练的模型（通常从 SFT 模型初始化），输入 prompt+response，输出一个标量 reward。训练数据为人工对同一 prompt 的多个 response 进行排序（A > B > C），经典 loss 为 pairwise ranking loss：

$$\mathcal{L}_{RM} = -\log\sigma(r_{chosen} - r_{rejected})$$

RM 的要求：准确捕捉人类偏好的细粒度差异；不能只学会长度偏好等表面特征。

**阶段三（PPO 强化学习）**：用 PPO 算法优化 policy（即 SFT 模型），最大化 RM 打分，同时用 KL 散度约束不让模型偏离 SFT 模型太远。PPO 本质是 actor-critic 方法。优化目标：$\max_\pi \mathbb{E}[RM(prompt, response) - \beta \cdot D_{KL}(\pi_\theta \parallel \pi_{ref})]$。

---

### Q: PPO 的核心机制是什么？⭐⭐⭐⭐

**Clipped Surrogate Objective**（PPO 的核心创新）：用 clip 限制每次策略更新的幅度。

$$\mathcal{L}_{CLIP} = \mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 是新旧策略的概率比。当 advantage A_t > 0（这个 action 好），ratio 不要超过 1+ε，防止过度利用；当 advantage A_t < 0（这个 action 不好），ratio 不要低于 1-ε，防止过度惩罚。

**KL Penalty**：在 RLHF 中，PPO 的 reward 不是裸的 RM 分数，而是加了 KL 惩罚：R_total = RM(prompt, response) - β * KL(π_θ || π_ref)。如果没有 KL penalty，模型会迅速学会 exploit RM 的漏洞（reward hacking），输出高 reward 但质量极差的文本。

**Reward Hacking & Reward Overoptimization**：
- Reward Hacking：模型学会了骗 RM 而不是真正提高质量，如发现 RM 偏爱长文本就输出超长废话。
- Reward Overoptimization：RM 的分数持续上升，但真实质量（人工评估）反而下降。因为 RM 只是一个代理（proxy），不可能完美捕获人类偏好——过度优化代理指标必然偏离真实目标（Goodhart's Law）。

**Advantage / Value Model / Critic / GAE**：
- Value Model / Critic：估计从当前状态出发的期望累积回报 V(s)，输入 prompt+已生成的 token 前缀，输出期望 reward。
- Advantage：A(s, a) = Q(s, a) - V(s)，衡量某个 action 比平均水平好多少。
- GAE (Generalized Advantage Estimation)：一种平衡 bias-variance 的 advantage 估计方法：A_GAE = Σ (γλ)^l * δ_{t+l}，δ_t = r_t + γV(s_{t+1}) - V(s_t)。

**PPO 的不稳定性和工程成本**：Reward 信号稀疏（整个 response 只有一个标量），RM 和 policy 互相博弈容易发散，超参数敏感，四个模型协同训练任何一个出问题都会连锁反应。显存：同时加载 4 个模型即使 LoRA 也需要大量显存，训练时间比 SFT 慢 5-10 倍。

---

### Q: PPO vs DPO 的核心对比？⭐⭐⭐⭐⭐

| 维度 | PPO (RLHF) | DPO |
|------|------------|-----|
| 需要 RM | 是，需单独训练 | 否，隐式建模 |
| 需要 Critic | 是 | 否 |
| 训练模型数 | 4 (policy, ref, RM, critic) | 2 (policy, ref) |
| 训练稳定性 | 不稳定 | 较稳定 |
| 显存需求 | 极高 | 中等 |
| 在线采样 | 是（需要 policy 实时生成） | 否（离线数据） |
| 偏好数据格式 | 标注打分/排序 | chosen/rejected pair |
| 适合场景 | 大规模、有充足资源 | 资源受限、偏好数据明确 |
| Reward Hacking | 有（需 KL 约束） | 风险较低（隐式 reward） |

**关键理解**：DPO 把 RLHF 的目标函数重新参数化，直接从偏好对中学习，等价于在 Bradley-Terry 偏好模型下优化 KL 约束的 reward 最大化。DPO 不是"不做 RLHF"，而是"换了一种方式做 RLHF"。

---

### Q: RLAIF 与 Constitutional AI 是什么？⭐⭐

**RLAIF (RL from AI Feedback)**：用 AI（另一个 LLM）替代人类提供偏好反馈。Anthropic 的 Constitutional AI 是代表作。优势：成本低、可规模化、避免人类标注的不一致性。风险：AI 反馈的偏差会传递到训练中。

**Constitutional AI**（Anthropic 提出，分两阶段）：

**阶段一（监督学习）**：用 harmful prompt 让模型生成 response，让模型按 Constitution（一组安全原则）自我修正 response，用修正后的数据做 SFT。

**阶段二（RL）**：用 AI 反馈（基于 Constitution 的评估）训练偏好模型，用 RL（PPO）优化，让模型输出符合 Constitution 的 response。

Constitution 示例：不要支持暴力、不要歧视、不要提供危险信息、尊重隐私等。

---

### Q: GRPO 是什么？为什么不需要 Value Model？⭐⭐⭐⭐⭐

GRPO 由 DeepSeekMath（2024）提出，是 PPO 的简化变体。核心创新：**去掉 Value Model（Critic），用 Group Relative Reward 替代**。

**GRPO 算法流程**：
1. 对每个 prompt，从当前 policy 采样 G 个 response（如 G=4 或 8）
2. 用 Reward Model（或 rule-based verifier）给每个 response 打分
3. 计算每个 response 的 advantage：A_i = (r_i - mean(r_group)) / std(r_group)
4. 用 PPO-style clipped objective 更新 policy：L_GRPO = -E[min(r_t * A_i, clip(r_t, 1-ε, 1+ε) * A_i) - β * KL(π_θ || π_ref)]

**为什么不需要 Value Model**：传统的 PPO 需要 critic 估计 V(s) 来算 advantage。GRPO 的思路是：如果对同一个 prompt 采样足够多的 response，那么这些 response 的平均 reward 就是该 prompt 的期望 reward 的近似（蒙特卡洛估计）。数学直觉：V(prompt) ≈ mean({RM(prompt, response_i) for i=1..G})。当 G 足够大时，group 平均 reward 趋近于真实期望 reward。

**GRPO vs PPO 核心区别**：

| 维度 | PPO | GRPO |
|------|-----|------|
| Critic/Value Model | 需要 | 不需要 |
| Advantage 计算 | GAE（需要 critic） | Group relative normalization |
| 在线模型数 | 4 | 3 (policy, ref, RM) |
| 采样需求 | 每步采样，逐 token 更新 | 每 prompt 采样 G 个完整 response |
| 显存 | 极高 | 中等（省一个 critic） |
| 稳定性 | 较差 | 较好 |
| Reasoning 适配 | 一般 | 优秀 |

**GRPO 为什么适合 Reasoning**：Reasoning 的 reward 方差大（正确和错误答案差异足够大），不需要 per-token reward（reasoning 的评估在完整输出后进行），节省的显存可用于更长输出，Group 采样天然适合 exploration（同一 prompt 采多种推理路径）。

---

### Q: RLVR 是什么？DeepSeek-R1 怎么用的？⭐⭐⭐⭐⭐

RLVR（Reinforcement Learning with Verifiable Rewards）是 DeepSeek-R1 技术报告中描述的训练方法。核心：用**基于规则的验证器**替代 Reward Model，提供确定性的 reward 信号。

**Verifiable Reward 的类型**：
- **Outcome Reward（结果奖励）**：数学题最终答案是否匹配、代码题是否通过测试用例、选择题选项是否正确。
- **Process Reward（过程奖励）**：推理步骤是否正确、中间计算是否准确、逻辑链是否完整。
- **Rule-Based Reward 的实现**：数学提取 \boxed{...} 中的答案与 ground truth 比较；代码用 sandbox 执行检查 pass/fail；格式检查是否包含思考过程标签。

**DeepSeek-R1 与 RLVR**：数学领域最典型——通过规则验证器做 GRPO 让模型涌现长 CoT 和自我反思。R1 技术报告的核心发现：
1. **R1-Zero**：直接在 DeepSeek-V3-Base 上做纯 RL（GRPO + rule-based reward），不经过任何 SFT。模型自主学会了自我检查、反思、回溯、探索替代方案。
2. **R1**：先收集少量高质量 cold-start 数据做 SFT，再做 RLVR，效果更好。
3. 推理能力的涌现来自 RL 训练过程中模型发现"多想一想"能提高正确率，reward 信号驱动了这一行为。

GRPO/RLVR 不是因为有人推才火，而是因为 R1 证明了这条路能做出真正的 reasoning 能力。在此之前，reasoning 主要靠 prompt engineering（CoT、ToT）和多模型投票（self-consistency），RLVR 证明模型可以自己学出 reasoning 策略。

---

### Q: 其他 RL 变体有哪些？⭐⭐

**RLOO (REINFORCE Leave-One-Out)**：和 GRPO 类似也用 group sampling 去掉 critic，区别在于 advantage 用 "除自己之外的 group 平均作为 baseline"（A_i = r_i - mean(r_{j != i})），理论上比 GRPO 的全局 group 归一化更无偏。

**REINFORCE**：最基础的 policy gradient 方法：∇J = E[∇log π(a|s) * R]，没有 critic，没有 baseline，方差极大，原始形式几乎无法训练 LLM。

**REINFORCE++**：REINFORCE 的改进版，加了多采样求平均作为 baseline、PPO-style clipping、KL 正则化。可以理解为 GRPO 的一个变体。

**DAPO（Dynamic Sampling Policy Optimization）**：ByteDance 2025（arXiv:2503.14476）对 GRPO 的四项改进：① Clip-Higher（上下界不对称 clip，上界放松鼓励探索，防止 entropy collapse）；② Dynamic Sampling（过滤掉 group 内所有回答 reward 相同的 prompt，因为这些 prompt 不产生有效梯度）；③ Token-level Policy Gradient Loss（token 级别而非 sequence 级别计算 loss，避免长度偏置）；④ Overlong Reward Shaping（对超长回答做软惩罚而非硬截断）。

**Dr. GRPO**：论文"Dr. GRPO: Removing the Entropy Collapse"（arXiv:2503.02471）。解决 GRPO 训练中 entropy 快速坍缩的问题。核心修改：移除 advantage 计算中的 std normalization。原始 GRPO：A_i = (r_i - mean(r)) / std(r)，Dr. GRPO：A_i = r_i - mean(r)（不除 std）。除以 std 在 group 内方差小时会放大 advantage 幅度，加速 entropy collapse。

---

### Q: OPD 和 GRPO 的区别？⭐⭐

| 维度 | OPD | GRPO |
|------|-----|------|
| 本质 | 蒸馏 | 强化学习 |
| 需要在线采样 | 否 | 是 |
| 优化目标 | 模仿 teacher policy 的输出分布 | 最大化 reward |
| 稳定性 | 高 | 中等 |
| 探索能力 | 无（只模仿） | 有（RL 探索） |
| Reasoning 提升 | 有限 | 显著（R1 证明） |

OPD 相当于"让好老师教你"，GRPO 相当于"让规则奖励你自己探索"。在 reasoning 提升上，GRPO 的探索-奖励机制更有效。

---

### Q: 医学问答是否适合 GRPO/RLVR？⭐⭐⭐

**适合的场景**：医学选择题/考试题（有明确 ground truth，适合 RLVR）、结构化诊断输出（可用规则检查）、临床路径推理（可通过医学知识图谱验证）、医学代码生成（可通过执行结果验证）。

**不适合的场景**：开放式医学咨询（没有唯一正确答案）、共情沟通/医患对话（质量在于语气和共情）、复杂鉴别诊断（正确答案可能不唯一）、多模态诊断（涉及影像、检验结果的综合判断）。

**核心挑战**：医学 QA 的根本问题是**缺乏可验证 ground truth**。不像数学有确定答案，医学问题的"正确答案"往往是概率性的、上下文相关的、因人而异的。这是本项目没有直接做 PPO/GRPO 的核心原因之一。

---

### Q: 为什么本项目没有直接做 PPO/GRPO？⭐⭐⭐⭐

**直接原因（面试中必须讲清楚）**：

1. **医学标注成本极高**：训练 RM 需要大量医生标注偏好，成本远超通用领域。一对医学偏好标注可能需要数分钟到数十分钟，且需要专业背景。

2. **医学安全风险不可控**：RL 训练有不可预测性。在生产环境中，reward hacking 可能导致模型输出看似正确但实际危险的医学建议。DPO 通过直接对比学习更可控。

3. **缺乏可靠的自动化 reward 信号**：大部分医学 QA 无法设计 rule-based reward，而神经网络 RM 在医学领域存在严重 hallucination 风险。

4. **资源约束**：PPO 需要大量显存和时间，团队资源有限，DPO 路线性价比更高。

5. **DPO 已能覆盖大部分对齐需求**：在 preference pair 数据质量有保障的情况下，DPO 能达到接近 PPO 的效果，且工程复杂度低得多。

**技术路线选择**：SFT 打基础 → DPO 做偏好对齐 → LLM-as-Judge 做质量评估 → Safety-RAG 兜底安全。这是一条务实可行的路线，在学术界和工业界都有广泛采用。

---

### Q: 如果后续扩展医学 RLVR，应该找哪些可验证信号？⭐⭐

**可直接做 Rule-Based Reward 的任务**：
1. 医学考试/选择题（执业医师考试题库、USMLE、MedMCQA）—— 答案是否正确
2. 医学计算题（药物剂量计算、BMI 计算、补液量计算）—— 计算结果是否在允许误差范围内
3. 结构化信息提取（疾病名、药名、症状的 F1 分数）
4. 医学代码生成（ICD/CPT 代码是否正确）
5. 医学知识图谱推理（推理链中的实体关系是否可验证）

**需要设计的 Semi-Verifiable Signal**：诊断一致性（多个模型/多轮推理给出相同诊断）、引文准确性（PubMed ID 是否存在）、禁忌症检查（药物推荐是否与患者病史冲突）。

**无法做 Rule-Based 的任务**：共情质量/沟通技巧、复杂病例的综合判断、前沿医学知识。

---

---

## PPO 深度解析

### Q: PPO 的 ratio 是什么？为什么需要重要性采样？star:5

PPO 中 ratio = $\frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$，即新策略和旧策略在同一状态下产生同一动作的概率比。

**为什么需要 ratio？** 因为 PPO 用旧策略 $\pi_{old}$ 采样的数据（advantage $\hat{A}$）来更新新策略 $\pi_{new}$。这是重要性采样：用旧分布的数据估计新分布的期望。ratio 修正了采样分布不一致的问题——如果新策略认为某动作的概率比旧策略高（ratio > 1），就放大该样本的权重；反之收缩。

PPO-Clip 的目标函数：$\mathcal{L} = \min(r \cdot \hat{A}, \text{clip}(r, 1-\epsilon, 1+\epsilon) \cdot \hat{A})$
其中 $\epsilon$ 通常为 0.1~0.2。Clip 机制防止单步更新过大——当 ratio 超出 $[1-\epsilon, 1+\epsilon]$ 时直接截断，不再给梯度。这相当于一个**信任域**约束。

### Q: GAE 是什么？PPO 中 advantage 是怎么算的？star:5

**Advantage** $A = Q(s,a) - V(s)$，即"做这个动作比平均好多少"。

**GAE (Generalized Advantage Estimation)**：用 λ 参数在 bias 和 variance 之间插值：
$$A^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$
其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD error。

- λ=0：只用单步 TD error，bias 高但 variance 低
- λ=1：等价于蒙特卡洛 return，无 bias 但 variance 高
- 典型值 λ=0.95，折中效果好

PPO 中 advantage 通常用 GAE 计算，因为它在理论上更稳定，实践中效果优于直接使用 return。

### Q: GRPO 中 group sampling 和 reward normalization 是怎么做的？star:5

对每个 prompt $x$，采样 $G$ 个回答 $\{y_1...y_G\}$（通常 $G$=4~64）。对每个回答打分 $\{r_1...r_G\}$，然后组内标准化：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$$

这就是**不需要 Value Model 的关键**——组内相对排序直接作为优势估计的代理。如果一个回答比组平均好，它的 advantage 为正；差则为负。

如果一组内所有回答都很差（reward 都低），标准化后仍然会有正有负——这是潜在问题。差的回答之间对比可能学到错误方向。因此 GRPO 通常需要 reward model 质量较高、group size 够大来降低噪声。

### Q: GRPO 为什么不需要 Critic / Value Model？比 PPO 省多少？star:4

PPO 需要 4 个模型（Actor + Critic + Reward Model + Reference Model），GRPO 只需要 2 个（Policy + Reference Model）。去掉 Critic 的原因是：**group relative reward 本身就是优势函数的无偏估计**——不需要额外训练一个神经网络来预测状态价值。

显存对比（以 8B 模型为例）：
- PPO：~4 × 16GB = 64GB（4个模型）
- GRPO：~2 × 16GB = 32GB（2个模型）+ 采样显存

GRPO 也因此训练更快、超参更少、工程实现更简单——这也是 DeepSeek-R1 选择 GRPO 的重要原因。

### Q: DAPO / Dr. GRPO / VAPO 这些 GRPO 变体主要解决什么问题？⭐⭐⭐

| 变体 | 论文 | 核心改进 | 解决什么问题 |
|------|------|----------|--------------|
| **DAPO** | ByteDance 2025 (arXiv:2503.14476) | ①Clip-Higher：放宽上界 clip，增强低概率 token 探索；②Dynamic Sampling：过滤 group 内 reward/accuracy 全同、无有效梯度的 prompt；③Token-level PG Loss：缓解长回答在 loss 聚合中的长度偏置；④Overlong Reward Shaping / Soft Overlong Punishment：对超长截断样本做软惩罚 | 长 CoT RL 中的 entropy collapse、无效样本、长度偏置、超长回答 |
| **Dr. GRPO** | arXiv:2503.02471 | 移除 advantage 的 std normalization：原 GRPO `A_i=(r_i-mean(r))/std(r)`，改为 `A_i=r_i-mean(r)` | 避免 group 内 reward 方差很小时 advantage 被异常放大，缓解训练不稳定与 entropy collapse |
| **VAPO** | ByteDance/Seed 2025 (arXiv:2504.05118) | 使用 value-model-based RL 框架，并针对 value bias、不同序列长度、稀疏 reward 设计稳定训练机制 | 长 CoT 推理中的稀疏奖励、信用分配困难、value 估计偏差和训练稳定性问题 |

> **面试注意**：DAPO、Dr. GRPO、VAPO 是 2025 年围绕 GRPO / reasoning RL 的真实论文，适合重点了解。GSPO 也是真实存在，核心是 sequence-level importance ratio / clipping；SAPO 缩写存在多个不同版本，面试中不要随意展开，除非能明确对应论文。

当前项目中未实现这些变体，面试中了解原理即可。


### Q: PPO 的完整 loss 由哪几部分组成？star:5

PPO 训练涉及 4 个 loss：

1. **Policy Loss（策略损失）**：$\mathcal{L}_{policy} = -\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) \hat{A}_t)$
   其中 $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$ 是 ratio，$\hat{A}_t$ 是 advantage。

2. **Value Loss（价值损失）**：$\mathcal{L}_{value} = (V_\theta(s_t) - R_t)^2$
   Critic 预测的 value 和实际 return 的 MSE。

3. **Entropy Loss（熵损失）**：$\mathcal{L}_{entropy} = -\mathbb{E}[\mathcal{H}(\pi_\theta(\cdot|s))]$
   鼓励策略保持一定的随机性，防止过早坍塌到确定性策略。系数通常很小（~0.01）。

4. **KL Penalty**：$-\beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$
   在 RLHF 中额外约束 policy 不要偏离 reference model 太远，防止 reward hacking。

**总 loss**：$\mathcal{L} = \mathcal{L}_{policy} + c_1 \mathcal{L}_{value} - c_2 \mathcal{L}_{entropy} + c_3 \mathcal{L}_{KL}$

### Q: PPO 的 Clip 机制和 KL 约束有什么区别？都限制更新幅度，有什么不同？star:4

| | Clip 机制 | KL 约束 |
|---|---------|--------|
| 作用方式 | 直接截断 ratio 在 $[1-\epsilon, 1+\epsilon]$ 内 | 在 loss 中加惩罚项 |
| 限制对象 | 单步 policy update 幅度 | policy 与 reference 的距离 |
| 硬/软约束 | 硬截断（超出范围无梯度） | 软约束（超出范围有惩罚） |
| 目的 | 训练稳定性（信任域） | 防止偏离 SFT 分布过远 |

**关键区别**：Clip 是"同一轮训练中别更新太快"，KL 是"别跑离 SFT 模型太远"。它们是两个正交的约束——Clip 控制优化步长，KL 控制优化终点。PPO 中两者同时使用：Clip 保证每步稳定，KL 保证最终结果不崩塌。

### Q: GRPO 中，如果一个 group 内所有回答的 reward 都很低，模型会怎么更新？star:4

组内标准化后：$\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$

即使所有 reward 都低（如都是 1-2/10 分），标准化后仍会有人"相对好"（正 advantage）和"相对差"（负 advantage）。模型会把"相对好"的差回答往上推、"相对差"的更差回答往下拉。

**这是 GRPO 的潜在风险**：在差的 group 里做"矮子里拔将军"——学到的可能是"如何在差回答中不那么差"，而非"如何产出好回答"。缓解方法：
1. **增大 group size**：更多样本 → 更大概率有人碰巧做好
2. **Reward model 质量要够高**：能区分"差"和"极差"
3. **设绝对阈值**：如果 group mean < 某个阈值（如 <1/10），跳过该 group 的更新
4. **混入高质量样本**：在 batch 里穿插已知好回答作为 anchor

### Q: GRPO 和 DPO 能不能结合使用？先 DPO 再 GRPO 合理吗？star:3

可以结合。典型流程：**SFT → DPO → GRPO**。

DPO 先做偏好对齐——让模型在已有采样数据上学到"什么是好回答"，建立基本的 reward 感知。GRPO 再在 DPO 模型基础上做 on-policy 探索——用 verifiable reward 或 reward model 进一步优化。

**为什么这个顺序合理**：DPO 提供更好的初始化策略（比纯 SFT 更能区分好坏），GRPO 需要策略有一定质量才能生成有意义的 group（否则所有回答都很差，group 内比较无意义）。

**两者目标不冲突**：DPO 优化的是静态偏好对（已有数据），GRPO 优化的是动态采样反馈（自己生成 → 被打分 → 组内对比）。DPO 解决"你知道什么是好的"，GRPO 解决"你能持续产出更好的"。


## 背诵版总结

```
【RLHF 三阶段】SFT 学格式 → RM 学偏好 → PPO 优化
【PPO 核心】clipped objective + KL penalty, 需要 4 个模型
【Reward Hacking】模型骗 RM 得高分但质量差, KL penalty 防这个
【DPO vs PPO】DPO 省 RM+critic, 离线做, 更稳, 本项目选了 DPO
【GRPO 核心】去掉 value model, 用 group 内归一化算 advantage
【GRPO 公式】A_i = (r_i - mean) / std, 同一 prompt 采 G 个
【RLVR 核心】用规则验证取代神经网络 RM, 不要偏好标注
【R1 为什么火】GRPO+RLVR 让模型自发涌现 reasoning 能力
【OPD vs GRPO】OPD 是蒸馏, GRPO 是 RL 探索
【本项目为什么没做 PPO/GRPO】
  - 医学标注成本高、安全风险大、缺乏自动化 reward
  - DPO 在资源有限下够用、安全可控
【医学 RLVR 适合什么】选择题、计算题、结构化提取、代码生成
【医学 RLVR 不适合什么】开放式咨询、共情沟通、复杂鉴别诊断
```

---

## 面试可能的追问及回答要点

**Q: 你们的 DPO 和 PPO 本质区别是什么？**
DPO 是离线对比学习，PPO 是在线与 RM 博弈。DPO 相当于把 RLHF 的 reward 建模隐式化，直接用 chosen/rejected pair 的对比信号更新 policy。我们在医学场景选 DPO，因为更稳定、更安全、更省资源。

**Q: GRPO 真的不需要任何形式的 critic 吗？**
不需要单独的 critic 网络。GRPO 的 group mean 充当了 implicit baseline——它是对 V(prompt) 的蒙特卡洛估计。当采样数 G 足够大（通常 G>=4），这个估计是有效的。

**Q: 如果让你现在给医学项目加 GRPO，你怎么设计？**
选医学选择题做 RLVR 试点。用执业医师题库，把选项正确性作为 rule-based reward。用 medical safety classifier 做额外的安全约束 reward。先在 1B 小模型上验证方案再 scale。

**Q: RLVR 的 reward 会不会过于稀疏？**
确实稀疏。但 reasoning 模型的好处是输出长、包含中间步骤，即使只有最终 reward，中间步骤也能通过 credit assignment 学到有用的策略。R1 证明稀疏 reward 在 reasoning 上足够有效。
