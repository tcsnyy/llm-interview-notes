# 07 — DPO 偏好优化 面试八股

---

## 面试 1 分钟 / 3 分钟回答版本

### 1 分钟版

DPO（Direct Preference Optimization）是一种无需显式训练 Reward Model 的对齐方法。核心思路是：将 RLHF 中 "先训 Reward Model 再 PPO 优化" 的两阶段流程，转化为一个直接在偏好数据上优化 policy 的分类问题。DPO loss 本质上是让模型对 chosen 回答赋予更高的隐式奖励，对 rejected 回答赋予更低的奖励，同时用 reference model（通常是 SFT 模型）的 logprob 做 KL 约束。在我的医学项目中，chosen 使用 Teacher 模型（DeepSeek-v4-pro）的高质量答案，rejected 使用 SFT 模型自己采样的低质量回答，约 10K 对数据在 SFT LoRA checkpoint 上继续训练，beta=0.1、loss_type=sigmoid。DPO 后模型在多数医学问答上表现提升，但部分长尾问题出现了更自信的幻觉，最终引入 Safety-RAG 修复。

### 3 分钟版

DPO 是 Stanford 在 2023 年提出的方法，发表于 NeurIPS 2023。它统一了 Reward Modeling 和 Policy Optimization 两个阶段，核心洞察来自 Bradley-Terry 偏好模型与 RLHF 目标函数之间的数学等价性。

传统 RLHF 流程是：收集人类偏好 → 训练 Reward Model → 用 PPO 在 Reward Model 的引导下优化 policy，同时用 KL 散度约束不偏离太远。这个过程需要额外训练一个 Reward Model、需要在线采样、需要处理 PPO 的四个 loss 项和 reward hacking 问题，极其不稳定且资源消耗巨大。

DPO 的巧妙之处在于数学推导：将 Bradley-Terry 模型下的最优 policy 显式解代入偏好优化目标，消去 Reward Model，得到一个仅依赖 policy model 和 reference model 输出 logprob 的 loss 函数。这个 loss 就是一个二分类交叉熵——最大化 chosen 相对于 rejected 的对数优势。

在我的医学项目中，数据构造方式有其特殊性：chosen 由 DeepSeek-v4-pro 作为 teacher 生成高质量医学回答——不是简单地让学生抄标准答案，而是纠正模型自己会犯的实际错误。

DPO 训练后效果提升明显，但在长尾问题（罕见病、偏门药物副作用）上出现了更自信的幻觉——模型学会了"说得更肯定"但事实是错误的。根本原因是 DPO 优化的是"人类偏好"而非"事实正确性"。最终引入 Safety-RAG 在推理阶段做事实性校验来弥补。

---

### Q: DPO 是什么？它解决了 RLHF 的什么问题？⭐⭐⭐⭐⭐

![DPO 训练流程图](images/DPO%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B%E5%9B%BE.png)

DPO（Direct Preference Optimization）是一种直接使用人类偏好数据来优化语言模型的方法，它将 RLHF 的 Reward Model 训练和 PPO 策略优化两个阶段合并为一个稳定的监督式 loss，无需显式训练 Reward Model。

直觉理解：

- **RLHF 的做法（两阶段）**：先训练一个"判卷老师"（Reward Model），再让 AI 学生反复答题，判卷老师打分，学生根据分数用 PPO 调整策略。问题：判卷老师可能有偏见，学生可能"耍小聪明"（reward hacking）。
- **DPO 的做法（一阶段）**：直接告诉 AI 学生"这种答法比那种答法更好，你直接往更好的方向调吧"。不需要判卷老师，直接比较两种回答的好坏，用交叉熵让模型偏好好的回答。

DPO 解决的 RLHF 痛点：

| RLHF 痛点 | DPO 解决方案 |
|-----------|-------------|
| 需要单独训练 Reward Model（额外 GPU + 数据） | 不需要 Reward Model |
| Reward Model 可能被 hacking | 优化目标直接绑定 policy 的 logprob，无法投机取巧 |
| PPO 训练不稳定（4 个 loss 项需要调权重） | 单一交叉熵 loss，梯度稳定 |
| 需要在线采样（PPO 要对当前 policy 的输出打分） | 完全离线（offline）训练，只需要偏好对 |
| 样本效率低（PPO 需要大量 rollout） | 数据利用率高，10K 对即可见效 |
| 工程复杂度极高（多个模型协同、分布式训练） | 单卡/双卡即可跑 |

---

### Q: DPO vs SFT、DPO vs PPO-RLHF 有什么区别？⭐⭐⭐⭐

**DPO vs SFT**：

| 维度 | SFT | DPO |
|------|-----|-----|
| **训练目标** | 最大化 chosen 回答的似然 | 最大化 chosen 相对 rejected 的偏好优势 |
| **数据需求** | 单条 (prompt, answer) | 三元组 (prompt, chosen, rejected) |
| **优化信号** | "这个回答是正确的" | "这个回答比那个更好" |
| **对负面样本** | 完全忽略 | 主动惩罚 rejected |
| **KL 约束** | 无（可能偏离预训练知识） | 有（reference model 约束） |
| **典型问题** | 过拟合、多样性降低 | 可能放大模型已有的错误倾向 |

关键区别：SFT 是"告诉我该说什么"，DPO 是"告诉我什么比什么更好"。SFT 只看到"好答案"，DPO 同时看到"好答案"和"坏答案"的对比。因此 DPO 理论上能学会更精细的"边界感"——不仅知道什么是好的，还知道什么是需要避免的。

**DPO vs PPO/RLHF**：

| 维度 | PPO-RLHF | DPO |
|------|----------|-----|
| **Reward Model** | 必须单独训练 | 不需要 |
| **训练方式** | 在线采样 + PPO 梯度 | 离线，直接 loss 反向传播 |
| **Loss 复杂度** | 4 个 loss term | 1 个 loss（binary cross-entropy） |
| **稳定性** | 极不稳定，需要大量调参 | 稳定，类似 SFT |
| **计算开销** | 高（需同时加载 Policy、Reward、Value、Reference 4 个模型） | 中（需同时加载 Policy 和 Reference 两个模型） |
| **Reward Hacking** | 常见 | 理论上不会（因为没有可以 hack 的 RM） |
| **样本效率** | 低（需要在线生成） | 高（离线数据即可） |
| **收敛保证** | 无理论保证 | 有（凸优化问题，在 logprob 空间） |

**选型建议**：已有高质量偏好数据、计算资源有限、需要快速迭代、对训练稳定性要求高时用 DPO。需要在线探索、动态调整 reward、多轮迭代优化时 PPO 可能更好。

---

### Q: DPO 需要 Reward Model 吗？Reference Model 呢？⭐⭐⭐⭐

**Reward Model：不需要但隐含**。DPO 不需要显式训练 Reward Model，但它通过数学等价性隐含地将 policy model 的 logprob 差作为 reward：

$$r(x, y) = \beta \cdot \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

这意味着 DPO 中 policy model 本身就在充当 Reward Model 的角色——它对 chosen 和 rejected 的相对评分就是隐式奖励。

**Reference Model：必须要有**。Reference Model（π_ref）在 DPO 中是必需的，作用类似于 PPO 中的 KL 约束锚点：正则化防止 catastrophic forgetting，标准化消除绝对 logprob 的尺度差异，定义"进步"的基线。

Reference Model 通常选 SFT 模型，因为 SFT 模型已经学会了"给出合理回答"，DPO 在此基础上调整偏好。

> **核心总结**："DPO 不需要 Reward Model——这是它最大的卖点。但它必须要有 Reference Model，通常是 SFT checkpoint。Reference Model 在 loss 里充当 KL 约束的锚点，防止 policy model 在优化偏好的过程中'跑偏'。beta 参数控制这个约束的强度——beta 越大，policy 被拉得越靠近 reference。"

---

### Q: DPO Loss 的公式是什么？怎么推导的？⭐⭐⭐⭐⭐

**最终公式**：

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l) \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)}\right)\right]$$

其中 $\pi_\theta$ 是 Policy Model，$\pi_{ref}$ 是 Reference Model（冻结的 SFT 模型），$y_w$ 是 Chosen 回答，$y_l$ 是 Rejected 回答，$\sigma$ 是 Sigmoid 函数，$\beta$ 是温度参数。

**直观理解**：记 $r_{chosen} = \beta \cdot \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}$，$r_{rejected} = \beta \cdot \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$，$margin = r_{chosen} - r_{rejected}$，$loss = -\log \sigma(margin)$。当 margin 很小（chosen 不比 rejected 更受偏好）时 loss 高，当 margin 很大（chosen 远好于 rejected）时 loss 趋近于 0。

**数学推导概要**（三步走）：

**Step 1：Bradley-Terry 偏好模型**。假设人类偏好遵循 P(y_w ≻ y_l | x) = σ( r(x, y_w) - r(x, y_l) )，即 chosen 被偏好的概率等于两个隐式奖励 r 之差过 sigmoid。

**Step 2：RLHF 优化目标**。标准 RLHF 目标是 $\max_\pi \mathbb{E}_{y\sim\pi}[ r(x, y) ] - \beta \cdot D_{KL}( \pi(\cdot|x) \parallel \pi_{ref}(\cdot|x) )$。这个带 KL 约束的优化问题有闭式解：$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp( r(x, y) / \beta )$，其中 $Z(x)$ 是配分函数。

**Step 3：消去 Reward Model**。从闭式解反解 $r$：$r(x, y) = \beta \cdot \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \cdot \log Z(x)$。代入 Bradley-Terry 偏好模型，$Z(x)$ 项在差值中消去。将 π* 替换为当前训练的 π_θ，取负对数似然作为 loss，即得到 DPO loss。

**关键**：Z(x) 在 chosen 和 rejected 的差值中消去是 DPO 推导最精妙的一步。如果 Z(x) 不能消去，DPO 就无法仅用 policy 的 logprob 来计算 loss。


**从 Bradley-Terry 到 DPO 的完整推导**：

Bradley-Terry 偏好模型：人类偏好 $y_w \succ y_l$ 的概率为
$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$
即奖励差越大，chosen 胜出的概率越高。

RLHF 中，最优策略和奖励的关系为（从 KL 约束优化推导）：
$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

反解出奖励：$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$

代入 Bradley-Terry，消去 $Z(x)$，将 $\pi^*$ 替换为当前策略 $\pi_\theta$：
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

这就是 DPO 的优雅之处——**不需要显式训练 reward model，直接最大化 chosen 和 rejected 在 policy 下的 log probability 差异，同时用 reference model 做正则化**。

---

### Q: Beta 参数怎么选？太大或太小会怎样？⭐⭐⭐

Beta ($\beta$) 是 DPO loss 中唯一的关键超参数。它源于 RLHF 目标中的 KL 散度约束系数：$\max_\pi \mathbb{E}[r(x, y)] - \beta \cdot D_{KL}(\pi \parallel \pi_{ref})$。

- **Beta 太大（β → +∞）**：KL 约束极强，policy 被强制贴近 reference，几乎不更新，DPO 基本退化。
- **Beta 太小（β → 0）**：KL 约束极弱，policy 可以随意偏离 reference，可能过拟合偏好数据，出现 catastrophic forgetting，也可能对数据集的偏好噪声过拟合。

典型 Beta 取值：

| Beta | 效果倾向 | 适用场景 |
|------|---------|---------|
| 0.01 | 极激进，容易过拟合 | 几乎不使用 |
| 0.1 | 标准值，适中的 KL 约束 | 大多数 DPO 训练（**项目取值**） |
| 0.5 | 保守，更贴近 reference | 对稳定性要求高的场景 |
| 1.0 | 很保守 | 偏向 reference 的多任务微调 |

医学项目用 beta=0.1 的原因：SFT 后模型已有良好的医学基础知识，DPO 的目标是"把好回答的倾向再拉高一点"不需要大范围修改，偏好数据 10K 对质量较好但难免有噪声。beta=0.1 提供一个温和的正则化，让模型有空间去偏好 chosen，但不会偏到忘记原有医学知识。

> **核心总结**："Beta 是 DPO 的'方向盘'。beta 太大，模型被 reference 拉得太紧，DPO 基本等于没训练；beta 太小，模型可能为了讨好偏好数据而忘记 SFT 阶段学到的知识。我们医学项目用 0.1，实践经验是 beta 在 0.01~0.5 之间按验证集指标做网格搜索，0.1 几乎是'最稳的默认值'。"

---

### Q: Chosen / Rejected 数据怎么构造？格式是什么？⭐⭐⭐⭐

DPO 数据每个样本是一个三元组：

```json
{
  "prompt": "患者，男，65岁，高血压病史10年，最近一周出现胸闷、气短。请给出可能的诊断和下一步检查建议。",
  "chosen": "根据患者年龄、高血压病史及胸闷气短症状，高度怀疑冠状动脉粥样硬化性心脏病（冠心病）合并心功能不全。建议：1. 心电图及动态心电图监测；2. 心脏超声评估心功能；3. 心肌酶谱及BNP/NT-proBNP检测；4. 必要时行冠脉CTA或冠脉造影。同时排查高血压性心脏病及肺部疾病。",
  "rejected": "可能是感冒了，多喝水休息一下就好。如果不见好转可以去社区医院看看。"
}
```

**数据构造的黄金法则**：Prompt 相同（chosen 和 rejected 必须来自同一个 prompt），Chosen > Rejected（标注者/Teacher 模型认为 chosen 明显更优），Rejected 不能是无意义噪声（必须是"看起来像回事但实际有问题的回答"），Chosen 必须是高质量的（垃圾进垃圾出）。

数据来源分类：人工标注（质量最高但贵且慢）、Teacher 模型生成（快速可扩展但可能有 bias）、模型自身采样（反映模型真实弱点但需要筛选）、对抗生成（针对特定短板但泛化性存疑）。

---

### Q: Chosen 与 Rejected 长度差问题怎么处理？Sum vs Mean logprob？⭐⭐⭐

DPO 训练中一个常见陷阱：模型可能学会"生成更长的回答"而非"生成更好的回答"。如果 chosen 系统性地比 rejected 长（常见情况），DPO loss 如果直接用 sum logprob，长回答天然有更低的 logprob。

**解决方案**：

**方案 A：用 mean logprob（项目采用）**——logp_chosen = sequence_logprob(chosen_ids).mean()，消除长度对 loss 的影响。

**方案 B：长度惩罚 / 归一化**——在 reward 计算时引入长度正则项。

**方案 C：数据筛选**——确保 chosen 和 rejected 的长度分布尽量接近。

| 维度 | Sum | Mean |
|------|-----|------|
| 长度偏差 | 天然偏向短回答 | 长度无关 |
| 数学意义 | sequence 的联合对数概率 | per-token 的期望对数概率 |
| DPO 论文 | 使用 sum | 也有讨论 |
| 实践推荐 | 如果 chosen/rejected 长度接近用 sum | 长度相差大用 mean |

在医学场景中，好的回答（chosen）通常确实比坏的回答（rejected）长——因为好的诊断需要详细的分析推理。这是合理的长短差异，不是 bias。但为了防止模型学到"更长的废话"，项目使用 mean logprob。

---

### Q: 为什么 Rejected 不能太弱也不能太强？最佳 rejected 是什么水平？⭐⭐⭐

**Rejected 太弱**（如随机文本、无关回答）：chosen 和 rejected 差距巨大，loss 很快降到 0，梯度信号消失，模型几乎没学到任何东西。类比让一个高中生和随机数发生器比赛数学——赢了毫无意义。

**Rejected 太强**（如与 chosen 质量差不多的回答）：chosen 和 rejected 的 logprob 差异极小，DPO 无法可靠区分，可能把 chosen 的一些随机波动错当成"偏好"。类比让两个奥林匹克金牌选手比试——胜负可能由随机因素决定。

**最佳 rejected 水平：刚好比 chosen 差一档，但不是垃圾。** 理想 rejected：看起来像一个合理的回答（格式正确、语法通顺），但存在实质性问题（事实错误、推理漏洞、不够全面），是模型自身可能犯的错误类型。这样梯度信号有意义，训练针对性强，不会 trivial。

我们项目的 rejected 来自 SFT 后的 Qwen3-8B 模型自身采样。采样时使用适中的 temperature（0.7~0.8），让 rejected 是"SFT 模型在真实场景下可能给出的答案"。这样 rejected 既有合理的格式（不是垃圾），又有 SFT 模型的真实缺陷（如不够专业、遗漏关键信息、诊断不完整），梯度信号非常有针对性。

---

### Q: 为什么 Rejected 用 SFT 模型采样？Chosen 用 Teacher 模型有什么好处？⭐⭐⭐⭐

**Rejected 用 SFT 模型采样的五大好处**：

1. **分布一致**：rejected 来自 policy model 自身的分布，梯度信号直接指导 policy 往哪里改进
2. **针对性强**：rejected 的错误正是 policy model 当前会犯的错误
3. **自举效应**：DPO 训练后 policy 更新，可以再次采样生成新的 rejected，做迭代训练（Iterative DPO）
4. **训练稳定**：chosen 和 rejected 的 logprob scale 在同一数量级
5. **数据高效**：不需要人工标注，利用模型的自身能力来挖掘弱点

如果用 GPT2/随机生成 rejected，chosen vs rejected 的差异是"模型能力"差异而非"回答质量"差异，DPO 学的是"更像大模型"而非"医学回答更好"。

**Chosen 用 Teacher 模型的好处**：

在医学 LLM Teacher 项目中，Teacher 是 DeepSeek-v4-pro，一个更大、更强的商业化医学 LLM。用 Teacher 而非人工标注的好处：质量高且稳定、可大规模生成（10K+）、风格统一、经过医学训练、完全可复现。

深层好处：chosen 的质量上限决定了 DPO 的上限，Teacher 模型确定了"理想回答"的方向；DPO 实际上将 Teacher 模型的知识偏好通过偏好比较的方式迁移到 Student（Qwen3-8B）；本质是一种知识蒸馏。

潜在风险：Teacher 模型自身的 bias 会传递到 Student，如果 Teacher 在某些领域也有错误 Student 会学到这些错误，Teacher 的风格可能与最终期望的风格不完全一致。

> **核心总结**："我们使用 DeepSeek-v4-pro 生成 chosen、SFT 模型采样 rejected，这是一个'自我对抗'的数据构造策略——SFT 模型采样出的 rejected 代表了它现阶段容易犯的错误，DPO 在 loss 优化过程中逐步修正自己最真实的弱点。"

---

### Q: Reference Model 与 Policy Model 分别是什么？怎么配置？⭐⭐⭐

**Policy Model (π_θ)**：当前正在训练的模型，参数 θ 参与梯度更新。在项目中加载 base model (Qwen3-8B 4bit QLoRA) + SFT LoRA adapter，LoRA adapter 参数可训练 (is_trainable=True)。前向传播时同时计算 chosen logprob 和 rejected logprob，反向传播时只更新 LoRA adapter 参数。

**Reference Model (π_ref)**：冻结的 SFT 模型，参数不参与梯度更新，仅用于计算 KL 约束的基线。加载同样的 base model + SFT LoRA adapter，通常合并后冻结 (merge_and_unload + requires_grad=False)，只做前向传播（计算 logprob）。合并后冻结的好处：标准 nn.Linear 前向传播更快，不需要 LoRA 分支的额外计算，显存中只有一份 BF16 权重。

两者关系可视化：

```
                      ┌─────────────┐
  DPO Training        │  Preference  │
  Data                │   Dataset    │
  (chosen, rejected)  └──────┬──────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │  Policy Model  │           │ Reference Model│
     │  π_θ (trainable)│          │ π_ref (frozen) │
     │ Base + SFT     │           │ Base + SFT     │
     │ LoRA adapter   │           │ merged + frozen│
     │ (可更新)        │           │ (不可更新)      │
     └───────┬────────┘           └───────┬────────┘
             │                            │
             │  log π_θ(chosen|x)         │  log π_ref(chosen|x)
             │  log π_θ(rejected|x)       │  log π_ref(rejected|x)
             │                            │
             └──────────┬─────────────────┘
                        ▼
              ┌─────────────────┐
              │   DPO Loss      │
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │ Backward: 只更新 │
              │ LoRA adapter    │
              └─────────────────┘
```

---

### Q: Sequence Logprob 怎么计算？Prompt 部分是否计入？⭐⭐⭐

对于一个 token 序列 y = (y1, y2, ..., yL)，sequence logprob 是 log π(y|x) = Σ_{t=1}^{L} log π(y_t | x, y_{<t})。

计算方法：将 prompt + response 拼接后做 forward 获取 logits，取每个位置的 log_softmax，gather 出 response 部分每个 target token 的 logprob。

**Prompt 部分不计入 logprob**。DPO 优化的是 π_θ(response | prompt)，而不是 π_θ(prompt, response)。因为 prompt 是用户/系统给的，不是模型生成的；对于同一个 prompt，prompt 部分的 logprob 在 chosen 和 rejected 中相消。如果计入 prompt 部分反而可能引入不必要的噪声。

---

### Q: DPO 训练的四大监控指标是什么？⭐⭐⭐

**1. DPO Accuracy**：chosen 的隐式 reward > rejected 的隐式 reward 的比例。训练开始时约 0.5（随机），逐渐上升到 0.7~0.85，超过 0.95 可能过拟合。

**2. Reward Margin**：chosen 和 rejected 隐式 reward 的差距。随着训练逐渐增大，从接近 0 到 ~0.5~2.0。如果 margin 爆炸（> 10）说明可能过拟合。

**3. Chosen Reward**：policy model 相对于 reference model 对 chosen 回答的"偏好提升"。应该为正（policy 比 reference 更喜欢 chosen），如果为负意味着 DPO 在反向优化。

**4. Rejected Reward**：policy model 相对于 reference model 对 rejected 回答的"偏好程度"。应该为负（policy 比 reference 更不喜欢 rejected），如果持续为正意味着 DPO 方向反了。

理想的训练曲线：chosen_reward 逐渐上升，rejected_reward 为负并逐渐下降，margin 逐步扩大，accuracy 从 0.5 上升到 0.75~0.82。

> **核心总结**："DPO 训练主要监控四个指标。正常训练中，chosen reward 应该为正并逐渐增大，rejected reward 为负并逐渐减小，margin 逐步扩大。如果 rejected reward 也在增大（变正），说明模型在同时提升对所有输出的概率——这是无效训练。如果 margin 爆炸增长，说明模型在过拟合，需要加大 beta 或减小 lr。"

---

### Q: DPO Loss 不下降 / Reward 异常怎么诊断和修复？⭐⭐⭐

**问题 1：DPO Loss 一直震荡不下降**。可能原因：学习率太大、chosen 和 rejected 质量太接近、beta 太小、数据噪声大。修复：降低学习率（2e-4 → 5e-5），增大 beta（0.01 → 0.1），人工抽查数据质量。

**问题 2：Chosen Reward 和 Rejected Reward 同时增大**。模型学到了"对所有回答都提概率"而不是"区分好坏"。修复：增大 beta，检查 reference model 是否正确加载（没有被更新），检查 loss 实现中是否有 sign 错误。

**问题 3：DPO Accuracy 很高但生成质量下降**。过拟合了。修复：早停（early stopping），增加 warmup steps，增大 dropout。

**问题 4：Loss 迅速降到 0 附近就不再变化**。chosen 和 rejected 差距太大（如 rejected 是随机文本），模型轻松达到 margin >> 0。修复：替换 rejected 为更难的样本。

> **排查优先级**：第一看 lr 是否太大（DPO 对 lr 比 SFT 敏感得多，通常用 5e-6 到 5e-5）；第二看 chosen 和 rejected 的区分度；第三看 reference model 是否被错误地更新了；第四看 beta 是否合适。

---

### Q: DPO 训练显存为什么比 SFT 高？怎么省？⭐⭐⭐

DPO 比 SFT 更吃显存，因为需要同时持有 policy model 和 reference model。Naive 方式下 Reference Model 以 BF16 全精度存储（8B 模型约 16GB），加上 policy model 的量化权重和双倍的激活值，很容易超出 32GB 上限。

省显存技巧：

**方法一（推荐）**：Reference Model 也使用 QLoRA 加载（但冻结），显存仅约 4-5 GB。

**方法二**：顺序计算——先计算 reference logprob（no_grad），卸载到 CPU，再正常训练 policy，省去同时驻留的开销。

**方法三**：policy model 开启 gradient checkpointing，激活值显存减少约 30-50%。

使用这些技巧后，8B + DPO 在 RTX 5090 32GB 上可以在 batch_size=4、seq_len=2048 下稳定运行。

---

### Q: DPO vs IPO vs KTO vs ORPO vs SimPO vs CPO 有什么区别？⭐⭐⭐

| 方法 | 全称 | 核心改动 | 是否需要 reference | 论文年份 |
|------|------|---------|-------------------|---------|
| **DPO** | Direct Preference Optimization | 直接将 RLHF 转化为二分类 loss | 需要 | 2023 |
| **IPO** | Identity Preference Optimization | 用平方 loss 替代 sigmoid，margin 更严格 | 需要 | 2023 |
| **KTO** | Kahneman-Tversky Optimization | 不要求成对数据，单条数据即可 | 需要 | 2024 |
| **ORPO** | Odds Ratio Preference Optimization | 将 DPO loss 和 SFT loss 联合优化 | **不需要** | 2024 |
| **SimPO** | Simple Preference Optimization | 用序列平均 logprob 差作 reward，加长度惩罚 | **不需要** | 2024 |
| **CPO** | Contrastive Preference Optimization | 直接用 chosen - rejected logprob 的 contrastive loss | **不需要** | 2024 |

**IPO**：用平方 loss (β·Δ - 1/(2τ))² 替代 sigmoid + 交叉熵，不会 early saturation，理论上能学得更深。**KTO**：不需要成对数据，只需要 (prompt, answer, label)，数据收集成本大幅降低。**ORPO**：不需要 reference model，SFT + 偏好联合优化，显存减半。**SimPO**：不需要 reference model + 内置长度惩罚，额外 margin γ 确保最小差距。**CPO**：极简实现，但没有 KL 约束容易跑偏。

**选型建议**：有成对偏好数据选 DPO/IPO，有单条评分数据选 KTO，显存紧张选 SimPO/ORPO，数据质量极高不需要 KL 约束选 CPO。

---

### Q: DPO 为什么会放大幻觉？医疗场景有什么特有风险？⭐⭐⭐⭐⭐

这是 DPO 一个关键缺陷。

**核心机制**：DPO 优化的是"人类偏好"，而非"事实正确性"。人类偏好自信肯定的语气、详细有条理的回答、流畅专业的表达，但不一定偏好事实正确。如果 chosen 因为"看起来更专业"而被偏好，DPO 学到的是"说得很肯定 = 好"，模型学会对不准确的内容也说得很肯定，幻觉被强化。

**数学直觉**：DPO 的隐式 reward $r(x, y) = \beta \cdot \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$，不包含任何事实性校验。模型只需要让 logprob 变高就能获得更高的 reward。如果 chosen 中包含事实错误但语气自信、条理清晰，DPO 仍然会奖励这种模式。

**长尾问题更容易中招**：高频问题（如高血压、糖尿病）训练数据中 chosen 包含正确答案，DPO 学到正确方向。长尾问题（如罕见病、偏门药物）训练数据中 chosen 可能也有不准确之处（teacher 也不是完美的），但 chosen 的格式/语气/结构更好，DPO 学到的是"格式好 = 好"。测试时模型对不准确的回答也说得头头是道。

**医疗场景的特有风险**：SFT 模型的错误是犹豫的不确定的错误（"可能是感冒，建议观察"——错误但低风险），DPO 后的错误是确定性的、看起来专业的错误（"根据你的症状描述，可以确诊为病毒性上呼吸道感染，建议口服复方氨酚烷胺..."——更危险的自信误诊）。此外，医学伦理要求准确性 > 流畅性、安全性 > 肯定性，而 DPO 可能学会相反的偏好。

> **核心总结**："DPO 有一个根本性的局限：它优化的是'人类偏好'而非'事实正确性'。在常见问题上这个风险较小，但在长尾问题上，chosen 本身可能也不完美，DPO 就可能强化模型的幻觉倾向。这在我们医学项目中得到了验证——DPO 后长尾罕见病问题上的幻觉率反而上升了。医疗是 DPO 的高风险场景，因为医学容错率极低——在其他领域，DPO 让模型更自信是好事；在医疗领域，DPO 让模型对错误答案更自信可能是致命的。"

---

### Q: DPO 是不是一定提升安全性？⭐⭐⭐

DPO 不一定提升安全性，甚至可能降低安全性。DPO 优化的是"偏好"而非"正确性"或"安全性"。安全性能不能提升完全取决于偏好数据中是否包含了安全相关的偏好信号。

安全性提升的前提：训练数据中的 chosen 在安全维度上确实优于 rejected，安全性相关的信号强度足以被 DPO 捕捉到，数据分布覆盖了安全风险的高发场景。

安全性下降的情况（最危险）：DPO 让模型对所有输出更自信包括不安全的输出，模型学会了"打包票"式回答丢失了必要的谨慎，对未见过的安全场景模型可能"自信地胡说"。

> **分层防护策略**：第一层，在数据构造阶段确保 chosen 符合安全标准（如必须包含免责声明、不给出明确的药物剂量建议）；第二层，在 DPO 训练中监控安全相关的指标；第三层，DPO 训练后接入 Safety-RAG 在推理阶段做事实性校验——这是最可靠的一层，因为它不依赖模型参数化的"记忆"。

---

### Q: DPO 数据质量怎么保证？⭐⭐⭐

分三步走：

**第一步（Chosen 质量控制）**：Teacher 模型（DeepSeek-v4-pro）生成 chosen，然后人工抽查 5-10% 确保没有明显的事实错误、伦理问题或糟糕格式。规则过滤包括：长度 >= 50 tokens、包含专业术语、无明显格式错误。

**第二步（Rejected 质量控制）**：SFT 模型采样生成（temperature=0.7~0.8），规则过滤：长度 20-1000 tokens、不包含特殊控制 token、格式完整性。质量过滤：不能是纯通用万能回答、不能是纯格式错误（太容易区分）。去重：与 chosen 的相似度不能太低也不能太高。

**第三步（对比质量检查）**：确保 chosen 和 rejected 有实质性差距而非仅格式差异，验证 rejected 确实存在实质性问题而非仅是表述方式不同。还需要做数据分布分析——确保 chosen/rejected 的长度分布接近，各主题领域平衡覆盖。

---

### Q: 项目复盘：DPO 为何在长尾医学问题上产生幻觉？⭐⭐⭐⭐

现象：DPO 训练后，常见病问答质量提升明显，但长尾问题（罕见病、少见药物交互作用）出现更严重的幻觉，模型会编造不存在的症状、药物、剂量信息。

根因分析：

1. **数据分布不均**：高频问题 (~7K 对) 的 chosen 质量高、rejected 有明显错误；长尾问题 (~3K 对) 的 chosen 质量参差不齐（teacher 模型对罕见病也可能不完美），rejected 和 chosen 的差距主要是"格式"而非"事实"，DPO 学到的是"输出更详细的长回答"而非"纠正事实"。

2. **Teacher 模型本身在长尾问题上并不完美**：DeepSeek-v4-pro 虽然很强，但对罕见病的了解可能也不全面。如果 chosen 中包含不准确信息，DPO 合法地把"不准确但格式好的回答"当作偏好目标。

3. **DPO 的隐式奖励只依赖 logprob**：没有任何机制检查"chosen 是否事实正确"。模型可以通过"说得更详细、更肯定"来获得更高 reward，无需确保内容是正确的。

4. **Mean logprob 可能加剧了问题**：虽然使用 mean logprob 避免了长度偏差，但长回答中每个 token 的预测难度更低（有更多 context），更长意味着更大比例的 token 是容易预测的功能词。

> **核心反思**："这让我意识到 DPO 不是银弹——它对数据质量的依赖比 SFT 更高。最终我们引入 Safety-RAG 在推理阶段做知识检索来弥补。"

---

### Q: 为什么 DPO 后还需要 Safety-RAG？⭐⭐⭐

DPO 优化的是"模型表达知识的方式"——让回答更专业、更有条理；Safety-RAG 解决的是"模型知道什么"——确保输出的内容是事实正确的。在医学场景，这两者缺一不可：DPO 让回答好看，RAG 让回答正确。没有 RAG 的 DPO 可能在长尾问题上把错误的回答包装得更加逼真，这是医学场景最危险的 failure mode。

RAG 解决了 DPO 解决不了的问题：保证事实正确性（特别是长尾/新的知识）、纠正在 chosen 中就已存在的错误、防止模型对幻觉"过于自信"、覆盖训练数据中从未出现的新疾病/新药物。

---

### Q: 你手写过 DPO loss 吗？⭐⭐⭐⭐

手写过的。DPO loss 本质是一个二分类交叉熵，核心就是计算 policy model 和 reference model 在 chosen 和 rejected 上的 logprob，然后套公式。

---

## 代码实现示例

### 示例 1：Sequence Logprob 计算

```python
import torch
import torch.nn.functional as F


def compute_sequence_logprob(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,        # [batch_size, seq_len] 或 [1, seq_len]
    attention_mask: torch.LongTensor,   # [batch_size, seq_len]
    label_ids: torch.LongTensor,        # [batch_size, response_len] — 要计算 logprob 的 token
    label_mask: torch.LongTensor,       # [batch_size, response_len] — 1=计入, 0=不计入 (padding)
    prompt_len: int,                    # prompt 的 token 数量
) -> torch.Tensor:
    """
    计算给定 response (label_ids) 在模型下的 per-token log probability。

    维度追踪 (single batch):
        input_ids:    [1, P+R]     # P = prompt_len, R = response_len
        outputs:      [1, P+R, V]  # V = vocab_size
        log_probs:    [1, P+R, V]
        
        对 response 的第 i 个 token (0-indexed 在 response 内):
            position_in_input = prompt_len + i - 1  # 模型的预测位置
            target_token_id   = label_ids[0, i]     # 实际的 token
        
        所以：
            logp[:, i] = log_probs[0, prompt_len + i - 1, label_ids[0, i]]

    更高效的做法是一次性 gather：
        log_probs[:, prompt_len-1 : prompt_len+R-1, :] — shape [1, R, V]
        用 label_ids (shape [1, R]) 索引最后一个维度：
        → gather 出 [1, R] 的 per-token logprob
    """
    batch_size = input_ids.size(0)
    response_len = label_ids.size(1)

    # Step 1: 前向传播获取 logits
    with torch.no_grad() if not model.training else torch.enable_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # [batch_size, P+R, vocab_size]

    # Step 2: 计算 log_softmax
    log_probs = F.log_softmax(logits, dim=-1)  # [batch_size, P+R, vocab_size]

    # Step 3: 提取 response 部分的 logprob
    response_log_probs_full = log_probs[:, prompt_len - 1 : prompt_len + response_len - 1, :]
    # shape: [batch_size, response_len, vocab_size]

    # Step 4: Gather 出实际 token 的 logprob
    per_token_logp = response_log_probs_full.gather(
        dim=-1,
        index=label_ids.unsqueeze(-1)   # [batch_size, response_len, 1]
    ).squeeze(-1)  # [batch_size, response_len]

    # Step 5: 应用 label_mask（屏蔽 padding token）
    per_token_logp = per_token_logp * label_mask  # padding 位置变为 0

    # Step 6: 聚合 — mean（推荐）或 sum
    num_valid_tokens = label_mask.sum(dim=-1).clamp(min=1)  # [batch_size]
    sequence_logp = per_token_logp.sum(dim=-1) / num_valid_tokens  # [batch_size]

    return sequence_logp, per_token_logp  # 返回聚合值和 per-token 值（可用于分析）


# ============================================================
# 维度追踪示例
# ============================================================
def trace_logprob_computation():
    """演示 logprob 计算的维度变化"""
    batch_size, prompt_len, response_len = 2, 128, 64
    vocab_size = 151936  # Qwen2/3 词表大小
    seq_len = prompt_len + response_len

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    label_ids = input_ids[:, prompt_len:]
    label_mask = torch.ones(batch_size, response_len)

    class MockModel:
        class MockOutput:
            def __init__(self):
                self.logits = torch.randn(batch_size, seq_len, vocab_size)
        def __call__(self, input_ids, attention_mask):
            return self.MockOutput()
        @property
        def training(self):
            return False

    model = MockModel()
    seq_logp, per_token_logp = compute_sequence_logprob(
        model, input_ids, attention_mask, label_ids, label_mask, prompt_len
    )

    print(f"input_ids shape:        {input_ids.shape}")          # [2, 192]
    print(f"label_ids shape:        {label_ids.shape}")           # [2, 64]
    print(f"per_token_logp shape:   {per_token_logp.shape}")     # [2, 64]
    print(f"sequence_logp shape:    {seq_logp.shape}")           # [2]
    print(f"sequence_logp values:   {seq_logp}")


if __name__ == "__main__":
    trace_logprob_computation()
```

### 示例 2：DPO Loss 完整实现

```python
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


def dpo_loss(
    policy_chosen_logp: torch.Tensor,    # [batch_size] — π_θ(chosen|x) 的 mean logprob
    policy_rejected_logp: torch.Tensor,  # [batch_size] — π_θ(rejected|x) 的 mean logprob
    ref_chosen_logp: torch.Tensor,       # [batch_size] — π_ref(chosen|x) 的 mean logprob
    ref_rejected_logp: torch.Tensor,     # [batch_size] — π_ref(rejected|x) 的 mean logprob
    beta: float = 0.1,                   # KL 约束系数
    loss_type: str = "sigmoid",          # loss 类型：sigmoid / hinge / ipo
    reference_free: bool = False,        # 是否无 reference（退化为 CPO-like）
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    DPO Loss 的完整实现。

    数学公式（sigmoid loss_type）：
        L = -E[ log σ(
            β · [log π_θ(chosen|x) - log π_ref(chosen|x)]
          - β · [log π_θ(rejected|x) - log π_ref(rejected|x)]
        ) ]

    返回：
        loss:   标量，用于 backward
        metrics: dict，包含 accuracy / margin / rewards 等监控指标
    """
    # ===== Step 1: 计算隐式 reward =====
    if reference_free:
        chosen_reward = beta * policy_chosen_logp
        rejected_reward = beta * policy_rejected_logp
    else:
        chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
        rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)

    # ===== Step 2: 计算 reward margin =====
    reward_margin = chosen_reward - rejected_reward  # [batch_size]

    # ===== Step 3: 根据 loss_type 计算 loss =====
    if loss_type == "sigmoid":
        loss = -F.logsigmoid(reward_margin).mean()
    elif loss_type == "hinge":
        loss = torch.clamp(1.0 - reward_margin, min=0.0).mean()
    elif loss_type == "ipo":
        target = 1.0 / (2.0 * beta)
        loss = ((reward_margin - target) ** 2).mean()
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # ===== Step 4: 计算监控指标（不参与 backward） =====
    with torch.no_grad():
        accuracy = (reward_margin > 0).float().mean().item()
        avg_margin = reward_margin.mean().item()
        avg_chosen_reward = chosen_reward.mean().item()
        avg_rejected_reward = rejected_reward.mean().item()
        reward_gap = avg_chosen_reward - avg_rejected_reward
        logp_gap_chosen = (policy_chosen_logp - ref_chosen_logp).mean().item()
        logp_gap_rejected = (policy_rejected_logp - ref_rejected_logp).mean().item()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "reward_margin": avg_margin,
        "chosen_reward": avg_chosen_reward,
        "rejected_reward": avg_rejected_reward,
        "reward_gap": reward_gap,
        "logp_gap_chosen": logp_gap_chosen,
        "logp_gap_rejected": logp_gap_rejected,
    }

    return loss, metrics


# ============================================================
# 维度追踪示例
# ============================================================
def trace_dpo_loss():
    """演示 DPO loss 的计算过程"""
    batch_size = 4

    policy_chosen_logp = torch.tensor([-2.1, -1.8, -3.0, -2.5], dtype=torch.float32)
    policy_rejected_logp = torch.tensor([-3.5, -2.2, -2.9, -4.0], dtype=torch.float32)
    ref_chosen_logp = torch.tensor([-2.5, -2.0, -3.2, -2.8], dtype=torch.float32)
    ref_rejected_logp = torch.tensor([-3.0, -2.1, -3.0, -3.5], dtype=torch.float32)

    loss, metrics = dpo_loss(
        policy_chosen_logp=policy_chosen_logp,
        policy_rejected_logp=policy_rejected_logp,
        ref_chosen_logp=ref_chosen_logp,
        ref_rejected_logp=ref_rejected_logp,
        beta=0.1,
        loss_type="sigmoid",
    )

    print("=== DPO Loss 计算示例 ===")
    print(f"Loss:          {metrics['loss']:.4f}")
    print(f"Accuracy:      {metrics['accuracy']:.2%}")
    print(f"Reward Margin: {metrics['reward_margin']:.4f}")
    print(f"Chosen Reward: {metrics['chosen_reward']:.4f}")
    print(f"Rejected Reward: {metrics['rejected_reward']:.4f}")

    print("\n逐样本分析:")
    for i in range(batch_size):
        chosen_r = 0.1 * (policy_chosen_logp[i] - ref_chosen_logp[i])
        rejected_r = 0.1 * (policy_rejected_logp[i] - ref_rejected_logp[i])
        m = chosen_r - rejected_r
        print(f"  Sample {i}: "
              f"R_chosen={chosen_r:.4f}, "
              f"R_rejected={rejected_r:.4f}, "
              f"margin={m:.4f}, "
              f"correct={'✓' if m > 0 else '✗'}")


if __name__ == "__main__":
    trace_dpo_loss()
```

### 示例 3：Chosen / Rejected Batch Logprob 计算

```python
import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_dpo_batch_logprobs(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    对 DPO 的一个 batch 计算 chosen 和 rejected 的 logprob。

    batch 结构（假设已经 padding 和 concatenate）：
        {
            "input_ids_chosen":   [B, max_seq_len_chosen],
            "attention_mask_chosen": [B, max_seq_len_chosen],
            "label_ids_chosen":   [B, max_response_len_chosen],
            "label_mask_chosen":  [B, max_response_len_chosen],
            "prompt_len_chosen":  [B],
            "input_ids_rejected": [B, max_seq_len_rejected],
            ...
        }
    """
    # ===== 计算 chosen logprob =====
    outputs_chosen = model(
        input_ids=batch["input_ids_chosen"],
        attention_mask=batch["attention_mask_chosen"],
    )
    logits_chosen = outputs_chosen.logits  # [B, S_c, V]
    log_probs_chosen = F.log_softmax(logits_chosen, dim=-1)
    per_token_logp_chosen = gather_response_logprob(
        log_probs=log_probs_chosen,
        label_ids=batch["label_ids_chosen"],
        prompt_lens=batch["prompt_len_chosen"],
    )
    valid_tokens_chosen = batch["label_mask_chosen"].sum(dim=-1).clamp(min=1)
    chosen_logp = (per_token_logp_chosen * batch["label_mask_chosen"]).sum(dim=-1)
    chosen_logp = chosen_logp / valid_tokens_chosen  # [B] — mean logprob

    # ===== 计算 rejected logprob =====
    outputs_rejected = model(
        input_ids=batch["input_ids_rejected"],
        attention_mask=batch["attention_mask_rejected"],
    )
    logits_rejected = outputs_rejected.logits
    log_probs_rejected = F.log_softmax(logits_rejected, dim=-1)
    per_token_logp_rejected = gather_response_logprob(
        log_probs=log_probs_rejected,
        label_ids=batch["label_ids_rejected"],
        prompt_lens=batch["prompt_len_rejected"],
    )
    valid_tokens_rejected = batch["label_mask_rejected"].sum(dim=-1).clamp(min=1)
    rejected_logp = (per_token_logp_rejected * batch["label_mask_rejected"]).sum(dim=-1)
    rejected_logp = rejected_logp / valid_tokens_rejected

    return {
        "chosen_logp": chosen_logp,       # [batch_size]
        "rejected_logp": rejected_logp,   # [batch_size]
        "per_token_chosen": per_token_logp_chosen,
        "per_token_rejected": per_token_logp_rejected,
    }


def gather_response_logprob(
    log_probs: torch.Tensor,      # [B, S, V]
    label_ids: torch.Tensor,      # [B, R]
    prompt_lens: torch.Tensor,    # [B]
) -> torch.Tensor:
    """从完整的 log_probs 中 gather 出 response 部分每个目标 token 的 logprob。"""
    batch_size, seq_len, vocab_size = log_probs.shape
    device = log_probs.device
    response_len = label_ids.size(1)

    batch_indices = torch.arange(batch_size, device=device) \
        .unsqueeze(1).expand(-1, response_len)  # [B, R]
    start_positions = prompt_lens - 1  # [B]
    seq_indices = start_positions.unsqueeze(1) + \
                  torch.arange(response_len, device=device).unsqueeze(0)  # [B, R]

    per_token_logp = log_probs[batch_indices, seq_indices, label_ids]  # [B, R]
    return per_token_logp


# ============================================================
# 维度追踪
# ============================================================
if __name__ == "__main__":
    print("DPO Batch Logprob 计算维度追踪:")
    B, P, R, V = 2, 10, 5, 100
    log_probs = torch.randn(B, P + R, V)
    label_ids = torch.randint(0, V, (B, R))
    prompt_lens = torch.tensor([P, P])
    result = gather_response_logprob(log_probs, label_ids, prompt_lens)
    print(f"  输入 log_probs:  {log_probs.shape}")    # [2, 15, 100]
    print(f"  输入 label_ids:  {label_ids.shape}")     # [2, 5]
    print(f"  输出 per_token:  {result.shape}")        # [2, 5]
```

### 示例 4：DPO Metrics 计算

```python
import torch
from typing import Dict


def compute_dpo_metrics(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, float]:
    """计算 DPO 训练的监控指标。"""
    metrics = {}

    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp)
    margins = chosen_rewards - rejected_rewards

    accuracy = (margins > 0).float().mean().item()
    metrics["accuracy"] = accuracy
    metrics["chosen_reward_mean"] = chosen_rewards.mean().item()
    metrics["chosen_reward_std"] = chosen_rewards.std().item()
    metrics["rejected_reward_mean"] = rejected_rewards.mean().item()
    metrics["rejected_reward_std"] = rejected_rewards.std().item()
    metrics["margin_mean"] = margins.mean().item()
    metrics["margin_std"] = margins.std().item()
    metrics["chosen_logp_diff_mean"] = (policy_chosen_logp - ref_chosen_logp).mean().item()
    metrics["rejected_logp_diff_mean"] = (policy_rejected_logp - ref_rejected_logp).mean().item()

    both_positive = ((chosen_rewards > 0) & (rejected_rewards > 0)).float().mean().item()
    metrics["both_rewards_positive_ratio"] = both_positive
    negative_margin_ratio = (margins < 0).float().mean().item()
    metrics["negative_margin_ratio"] = negative_margin_ratio
    max_abs_reward = max(chosen_rewards.abs().max().item(), rejected_rewards.abs().max().item())
    metrics["max_abs_reward"] = max_abs_reward
    metrics["reward_explosion_warning"] = max_abs_reward > 10.0

    return metrics


if __name__ == "__main__":
    B = 8
    torch.manual_seed(42)
    policy_chosen_logp = torch.randn(B) * 0.5 - 1.5
    policy_rejected_logp = torch.randn(B) * 0.5 - 2.5
    ref_chosen_logp = torch.randn(B) * 0.3 - 2.0
    ref_rejected_logp = torch.randn(B) * 0.3 - 2.2

    metrics = compute_dpo_metrics(
        policy_chosen_logp, policy_rejected_logp,
        ref_chosen_logp, ref_rejected_logp, beta=0.1,
    )

    print("=== DPO Training Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, bool):
            print(f"  {key}: {'⚠ YES' if value else '✓ OK'}")
        else:
            print(f"  {key}: {value:.4f}")
```

### 示例 5：简化 DPO Training Step

```python
import torch
from transformers import Trainer
from typing import Dict


class SimpleDPOTrainingStep:
    """简化的 DPO 训练单步流程。"""

    def __init__(
        self,
        policy_model: torch.nn.Module,
        ref_model: torch.nn.Module,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
    ):
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def training_step(self, batch: Dict[str, torch.Tensor]):
        # Step 1: 计算 reference model 的 logprob（no_grad）
        with torch.no_grad():
            ref_outputs = compute_dpo_batch_logprobs(self.ref_model, batch)
            ref_chosen_logp = ref_outputs["chosen_logp"]
            ref_rejected_logp = ref_outputs["rejected_logp"]

        # Step 2: 计算 policy model 的 logprob
        policy_outputs = compute_dpo_batch_logprobs(self.policy_model, batch)
        policy_chosen_logp = policy_outputs["chosen_logp"]
        policy_rejected_logp = policy_outputs["rejected_logp"]

        # Step 3: 计算 DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logp=policy_chosen_logp,
            policy_rejected_logp=policy_rejected_logp,
            ref_chosen_logp=ref_chosen_logp,
            ref_rejected_logp=ref_rejected_logp,
            beta=self.beta,
            loss_type=self.loss_type,
        )

        # Step 4: 反向传播
        loss.backward()
        return loss, metrics


class DPOTrainer(Trainer):
    """将 DPO training step 集成到 HuggingFace Trainer 中的简化示例。"""

    def __init__(self, ref_model, beta=0.1, loss_type="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.no_grad():
            ref_chosen_logp = compute_sequence_logprob(
                self.ref_model, inputs["input_ids_chosen"],
                inputs["attention_mask_chosen"], inputs["label_ids_chosen"],
                inputs["label_mask_chosen"], inputs["prompt_lens"],
            )[0]
            ref_rejected_logp = compute_sequence_logprob(
                self.ref_model, inputs["input_ids_rejected"],
                inputs["attention_mask_rejected"], inputs["label_ids_rejected"],
                inputs["label_mask_rejected"], inputs["prompt_lens"],
            )[0]

        policy_chosen_logp = compute_sequence_logprob(
            model, inputs["input_ids_chosen"],
            inputs["attention_mask_chosen"], inputs["label_ids_chosen"],
            inputs["label_mask_chosen"], inputs["prompt_lens"],
        )[0]
        policy_rejected_logp = compute_sequence_logprob(
            model, inputs["input_ids_rejected"],
            inputs["attention_mask_rejected"], inputs["label_ids_rejected"],
            inputs["label_mask_rejected"], inputs["prompt_lens"],
        )[0]

        loss, metrics = dpo_loss(
            policy_chosen_logp=policy_chosen_logp,
            policy_rejected_logp=policy_rejected_logp,
            ref_chosen_logp=ref_chosen_logp,
            ref_rejected_logp=ref_rejected_logp,
            beta=self.beta,
            loss_type=self.loss_type,
        )

        return (loss, metrics) if return_outputs else loss


if __name__ == "__main__":
    print("DPO Training Step 示例代码")
    print("=" * 60)
    print("以上代码展示了 DPO 训练的完整流程:")
    print("  1. 计算 reference model 的 chosen/rejected logprob（no_grad）")
    print("  2. 计算 policy model 的 chosen/rejected logprob（需要 grad）")
    print("  3. 用 dpo_loss() 计算 loss")
    print("  4. loss.backward() 更新 policy model")
    print("  5. 记录 accuracy / margin / rewards 等指标")
```

---

## 背诵版总结

```
┌──────────────────────────────────────────────────────────────────────┐
│                       DPO 偏好优化 背诵版总结                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ 【DPO 一句话】不需要 Reward Model，直接用偏好数据优化 policy 的分类 loss    │
│                                                                      │
│ 【核心公式】                                                           │
│   L = -E[ log σ( β·Δ_chosen - β·Δ_rejected ) ]                       │
│   Δ = log[π_θ(y|x) / π_ref(y|x)]                                     │
│                                                                      │
│ 【DPO vs RLHF】                                                       │
│   RLHF: 训 RM → PPO 优化（4 个 loss, 不稳定, 需采样）                     │
│   DPO:  直接偏好 loss（1 个 loss, 稳定, 离线数据）                         │
│                                                                      │
│ 【需要什么】                                                           │
│   ✗ 不需要 Reward Model                                               │
│   ✓ 必须要有 Reference Model（通常是 SFT 模型）                           │
│   ✓ 需要成对偏好数据 (prompt, chosen, rejected)                           │
│                                                                      │
│ 【关键参数】                                                           │
│   beta:  KL 约束强度，0.1 常用默认值                                      │
│   loss_type: sigmoid (标准) / ipo (平方loss) / hinge (硬间隔)            │
│   beta 太大 → 过于贴近 reference, 不学                                    │
│   beta 太小 → 过拟合偏好数据, 忘记 SFT 知识                                  │
│                                                                      │
│ 【Logprob 计算】                                                       │
│   ✓ 只算 response 部分 logprob (prompt 作为条件)                        │
│   ✓ mean logprob 消除长度 bias（医学项目使用）                            │
│   ✓ gather：log_probs[pos][target_token_id]                            │
│                                                                      │
│ 【数据铁律】                                                           │
│   chosen: 高质量（Teacher 模型 / 人工标注）                               │
│   rejected: 不能太弱（不是垃圾），不能太强（与 chosen 有实质差距）              │
│   rejected: 用 SFT 模型采样 → 反映真实弱点 → 梯度信号有意义                   │
│                                                                      │
│ 【项目配置（医学 LLM Teacher）】                                         │
│   chosen: DeepSeek-v4-pro 生成                                           │
│   rejected: SFT 模型自身采样（temperature 0.7~0.8）                      │
│   beta: 0.1, loss_type: sigmoid                                       │
│   ~10K DPO 对, 从 SFT LoRA checkpoint 继续训练                           │
│                                                                      │
│ 【四大监控指标】                                                        │
│   1. Accuracy： (chosen_reward > rejected_reward) 的比例                │
│   2. Reward Margin： chosen_reward - rejected_reward                  │
│   3. Chosen Reward： 应为正（policy 比 ref 更偏好 chosen）               │
│   4. Rejected Reward：应为负（policy 比 ref 更不喜欢 rejected）            │
│                                                                      │
│ 【DPO 的局限性】                                                        │
│   ✗ 优化"偏好"而非"正确性" → 可能放大幻觉                                    │
│   ✗ 依赖数据中的偏好定义 → chosen 的缺点会被学到                                │
│   ✗ 长尾/未见场景 → 模型可能自信地犯错                                       │
│   ✗ 训练更吃显存（需要 ref model）                                        │
│                                                                      │
│ 【DPO 变体选型】                                                        │
│   有 reference: DPO → IPO（更严格）→ KTO（不需成对数据）                    │
│   无 reference: SimPO（长度惩罚）→ ORPO（SFT+偏好联合）→ CPO（极简）         │
│                                                                      │
│ 【KTO (Kahneman-Tversky Optimization)】                                │
│   不需要 chosen/rejected 成对数据，只需单条回答的好/坏标签                    │
│   利用前景理论中的损失厌恶，对"坏回答"施加更强惩罚                             │
│                                                                      │
│ 【CPO (Contrastive Preference Optimization)】                           │
│   最简 DPO 变体：连 reference model 都省了                                 │
│   loss = -log σ(β·[log π(chosen) - log π(rejected)])                  │
│   快但可能过拟合，适合小规模实验                                            │
│                                                                      │
│ 【显存优化】                                                           │
│   1. ref model 也做 QLoRA 加载 → 4GB 而非 16GB                          │
│   2. 顺序计算：先 ref（no_grad）→ offload CPU → 再 policy                 │
│   3. gradient_checkpointing                                            │
│                                                                      │
│ 【医学项目失败经验】                                                      │
│   DPO 后长尾罕见病幻觉加重 → 根因：chosen 在长尾上也有错误                       │
│   修复：DPO + Safety-RAG → RAG 弥补事实性，DPO 弥补表达质量                    │
│                                                                      │
│ 【面试金句】                                                           │
│   "DPO 优化的不是正确性，是偏好。你自己定义的好，模型就会往那个方向跑。"              │
│   "Rejected 不是垃圾，是镜像——要照出模型自己的缺点。"                           │
│   "DPO + RAG 是互补关系：DPO 管怎么说，RAG 管说什么。"                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

*文档生成日期：2026-06-04*
*基于医学 LLM Teacher 项目实际配置编写（Qwen3-8B + QLoRA + DPO, beta=0.1, sigmoid loss）*
*项目中未找到的部分（如 IPO/KTO/SimPO 的具体实现）已标注"面试需要了解"*
