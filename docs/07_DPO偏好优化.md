# 07 — DPO 偏好优化 面试八股

---

## 目录

1. [面试 1 分钟 / 3 分钟回答版本](#面试-1-分钟--3-分钟回答版本)
2. [DPO 是什么：直觉与核心思路](#dpo-是什么直觉与核心思路)
3. [DPO vs SFT / DPO vs PPO-RLHF 对比](#dpo-vs-sft--dpo-vs-ppo-rlhf-对比)
4. [DPO 是否需要 Reward Model / Reference Model](#dpo-是否需要-reward-model--reference-model)
5. [DPO Loss 公式与完整数学推导](#dpo-loss-公式与完整数学推导)
6. [Beta 参数详解](#beta-参数详解)
7. [Chosen / Rejected 数据格式](#chosen--rejected-数据格式)
8. [Chosen 与 Rejected 长度差问题](#chosen-与-rejected-长度差问题)
9. [为什么 Rejected 不能太弱也不能太强](#为什么-rejected-不能太弱也不能太强)
10. [Rejected 用 SFT 模型采样的好处](#rejected-用-sft-模型采样的好处)
11. [Chosen 用 Teacher 模型答案的好处](#chosen-用-teacher-模型答案的好处)
12. [Reference Model 与 Policy Model 的定义](#reference-model-与-policy-model-的定义)
13. [Sequence Logprob 计算与 Token Logprob Gather](#sequence-logprob-计算与-token-logprob-gather)
14. [Response Logprob：Sum vs Mean](#response-logprobsum-vs-mean)
15. [Prompt 部分是否计入 Logprob](#prompt-部分是否计入-logprob)
16. [DPO 训练指标：Accuracy / Reward Margin / Chosen & Rejected Reward](#dpo-训练指标accuracy--reward-margin--chosen--rejected-reward)
17. [DPO Loss 不下降 / Reward 异常的诊断与修复](#dpo-loss-不下降--reward-异常的诊断与修复)
18. [DPO 训练显存分析与 Reference Model 省显存技巧](#dpo-训练显存分析与-reference-model-省显存技巧)
19. [DPO vs IPO vs KTO vs ORPO vs SimPO vs CPO](#dpo-vs-ipo-vs-kto-vs-orpo-vs-simpo-vs-cpo)
20. [DPO 为什么会放大幻觉](#dpo-为什么会放大幻觉)
21. [医疗场景 DPO 的特有风险](#医疗场景-dpo-的特有风险)
22. [项目复盘：DPO 为何在长尾医学问题上产生幻觉](#项目复盘dpo-为何在长尾医学问题上产生幻觉)
23. [为什么 DPO 后还需要 Safety-RAG](#为什么-dpo-后还需要-safety-rag)
24. [面试必答："DPO 是不是一定提升安全性"](#面试必答dpo-是不是一定提升安全性)
25. [面试必答："DPO 数据质量怎么保证"](#面试必答dpo-数据质量怎么保证)
26. [面试必答："你手写过 DPO loss 吗"](#面试必答你手写过-dpo-loss-吗)
27. [代码实现示例](#代码实现示例)
28. [背诵版总结](#背诵版总结)

---

## 面试 1 分钟 / 3 分钟回答版本

### 1 分钟版

> DPO（Direct Preference Optimization）是一种无需显式训练 Reward Model 的对齐方法。核心思路是：将 RLHF 中 "先训 Reward Model 再 PPO 优化" 的两阶段流程，转化为一个直接在偏好数据上优化 policy 的分类问题。DPO loss 本质上是让模型对 chosen 回答赋予更高的隐式奖励，对 rejected 回答赋予更低的奖励，同时用 reference model（通常是 SFT 模型）的 logprob 做 KL 约束。在我的医学项目中，chosen 使用 Teacher 模型（MiMo-v2.5-pro）的高质量答案，rejected 使用 SFT 模型自己采样的低质量回答，约 10K 对数据在 SFT LoRA checkpoint 上继续训练，beta=0.1、loss_type=sigmoid。DPO 后模型在多数医学问答上表现提升，但部分长尾问题出现了更自信的幻觉，最终引入 Safety-RAG 修复。

### 3 分钟版

> DPO 是 Stanford 在 2023 年提出的方法，发表于 NeurIPS 2023。它统一了 Reward Modeling 和 Policy Optimization 两个阶段，核心洞察来自 Bradley-Terry 偏好模型与 RLHF 目标函数之间的数学等价性。
>
> 传统 RLHF 流程是：收集人类偏好 → 训练 Reward Model → 用 PPO 在 Reward Model 的引导下优化 policy，同时用 KL 散度约束不偏离太远。这个过程需要额外训练一个 Reward Model、需要在线采样、需要处理 PPO 的四个 loss 项和 reward hacking 问题，极其不稳定且资源消耗巨大。
>
> DPO 的巧妙之处在于数学推导：将 Bradley-Terry 模型下的最优 policy 显式解代入偏好优化目标，消去 Reward Model，得到一个仅依赖 policy model 和 reference model 输出 logprob 的 loss 函数。这个 loss 就是一个二分类交叉熵——最大化 π_θ(chosen|x) 相对于 π_ref(chosen|x) 的对数优势，最小化 π_θ(rejected|x) 的相对优势。
>
> 在我的医学项目中，数据构造方式有其特殊性：chosen 由 MiMo-v2.5-pro（一个更大的医学 LLM）生成，代表"应该被学习的理想答案"；rejected 由 SFT 后的 Qwen3-8B 自身采样，代表"当前模型的真实错误倾向"。这种构造方式使 rejected 的分布与 policy model 当前输出分布接近，梯度信号更有意义——不是奖励一个遥不可及的"完美答案"，而是纠正模型自己会犯的实际错误。
>
> DPO 训练后效果提升明显，但在长尾问题（罕见病、偏门药物副作用）上出现了更自信的幻觉——模型学会了"说得更肯定"但事实是错误的。这里根本原因是 DPO 优化的是"人类偏好"而非"事实正确性"——chosen 被偏好可能是因为更肯定、更详细，而非事实更正确。最终我引入 Safety-RAG 在推理阶段做事实性校验来弥补。

---

## DPO 是什么：直觉与核心思路

### 一句话定义

> DPO（Direct Preference Optimization）是一种直接使用人类偏好数据来优化语言模型的方法，它将 RLHF 的 Reward Model 训练和 PPO 策略优化两个阶段合并为一个稳定的监督式 loss，无需显式训练 Reward Model。

### 直觉理解

想象你要训练一个模型区分"好回答"和"坏回答"：

```
RLHF 的做法（两阶段）：
  阶段 1：训练一个"判卷老师"（Reward Model），告诉他什么是好、什么是坏
  阶段 2：让 AI 学生反复答题，判卷老师打分，学生根据分数用 PPO 调整策略
  问题：判卷老师可能有偏见，学生可能"耍小聪明"（reward hacking）

DPO 的做法（一阶段）：
  直接告诉 AI 学生："这道题的这种答法比那种答法更好，你直接往更好的方向调吧"
  不需要判卷老师，直接比较两种回答的好坏，用交叉熵让模型偏好好的回答
```

### DPO 解决了 RLHF 的什么问题

| RLHF 痛点 | DPO 解决方案 |
|-----------|-------------|
| 需要单独训练 Reward Model（额外 GPU + 数据） | 不需要 Reward Model |
| Reward Model 可能被 hacking（找个极端高分但质量差的输出） | 优化目标直接绑定 policy 的 logprob，无法投机取巧 |
| PPO 训练不稳定（4 个 loss 项需要调权重） | 单一交叉熵 loss，梯度稳定 |
| 需要在线采样（PPO 要对当前 policy 的输出打分） | 完全离线（offline）训练，只需要偏好对 |
| 样本效率低（PPO 需要大量 rollout） | 数据利用率高，10K 对即可见效 |
| 工程复杂度极高（多个模型协同、分布式训练） | 单卡/双卡即可跑 |

---

## DPO vs SFT / DPO vs PPO-RLHF 对比

### DPO vs SFT

| 维度 | SFT | DPO |
|------|-----|-----|
| **训练目标** | 最大化 chosen 回答的似然 P(y_chosen\|x) | 最大化 chosen 相对 rejected 的偏好优势 |
| **数据需求** | 单条 (prompt, answer) | 三元组 (prompt, chosen, rejected) |
| **优化信号** | "这个回答是正确的" | "这个回答比那个更好" |
| **对负面样本** | 完全忽略 | 主动惩罚 rejected |
| **KL 约束** | 无（可能偏离预训练知识） | 有（reference model 约束） |
| **典型问题** | 过拟合、多样性降低、可能学偏 | 可能放大模型已有的错误倾向 |

**关键区别**：SFT 是"告诉我该说什么"，DPO 是"告诉我什么比什么更好"。SFT 只看到"好答案"，DPO 同时看到"好答案"和"坏答案"的对比。因此 DPO 理论上能学会更精细的"边界感"——不仅知道什么是好的，还知道什么是需要避免的。

### DPO vs PPO/RLHF

| 维度 | PPO-RLHF | DPO |
|------|----------|-----|
| **Reward Model** | 必须单独训练 | 不需要 |
| **训练方式** | 在线采样 + PPO 梯度 | 离线，直接 loss 反向传播 |
| **Loss 复杂度** | 4 个 loss term（policy_loss + value_loss + entropy_bonus + KL_penalty） | 1 个 loss（binary cross-entropy） |
| **稳定性** | 极不稳定，需要大量调参 | 稳定，类似 SFT |
| **计算开销** | 高（需同时加载 Policy、Reward、Value、Reference 4 个模型） | 中（需同时加载 Policy 和 Reference 两个模型） |
| **Reward Hacking** | 常见（Policy 学会讨好 Reward 而非真正变好） | 理论上不会（因为没有可以 hack 的 RM） |
| **样本效率** | 低（需要在线生成） | 高（离线数据即可） |
| **收敛保证** | 无理论保证 | 有（凸优化问题，在 logprob 空间） |

### 什么时候用 DPO 而非 PPO

- 已有高质量偏好数据（如人工标注的对比对）
- 计算资源有限
- 需要快速迭代实验
- 对训练稳定性要求高

### 什么时候 PPO 可能更好

- 需要在线探索（让模型自己生成 + 打分 + 学习）
- 需要在训练过程中动态调整 reward
- 多轮迭代优化（DPO 是 one-shot 离线训练）

---

## DPO 是否需要 Reward Model / Reference Model

### Reward Model：不需要但隐含

DPO **不需要显式训练** Reward Model。但它通过数学等价性**隐含地**将 policy model 的 logprob 差作为 reward：

```
r(x, y) = β · log[ π_θ(y|x) / π_ref(y|x) ]
```

这意味着 DPO 中 **policy model 本身就在充当 Reward Model** 的角色——它对 chosen 和 rejected 的相对评分就是隐式奖励。这也是 DPO 被称为"Direct"的原因：直接在 policy 上定义了偏好，跳过了中间的 RM 步骤。

### Reference Model：必须要有

Reference Model（π_ref）在 DPO 中是**必需的**，作用类似于 PPO 中的 KL 约束锚点：

```
DPO Loss 中明确包含 π_ref：
  L = -E[ log σ( β·log[π_θ(y_w|x)/π_ref(y_w|x)] - β·log[π_θ(y_l|x)/π_ref(y_l|x)] ) ]
                                            ↑                         ↑
                                     chosen 的相对增益          rejected 的相对增益
```

**Reference Model 的作用**：
1. **正则化**：防止 policy model 偏离原始能力太远（catastrophic forgetting）
2. **标准化**：用 π_ref 做分母，消除绝对 logprob 的尺度差异
3. **定义"进步"的基线**：DPO 优化的是相对 reference 的相对提升，而不是绝对提升

**Reference Model 通常选什么**：SFT 模型。因为 SFT 模型已经学会了"给出合理回答"，DPO 在此基础上调整偏好。

### 面试话术

> "DPO 不需要 Reward Model——这是它最大的卖点。但它必须要有 Reference Model，通常是 SFT checkpoint。Reference Model 在 loss 里充当 KL 约束的锚点，防止 policy model 在优化偏好的过程中'跑偏'。beta 参数控制这个约束的强度——beta 越大，policy 被拉得越靠近 reference。"

---

## DPO Loss 公式与完整数学推导

### 最终公式

```
L_DPO(π_θ; π_ref) = -E_{(x, y_w, y_l) ~ D} [
    log σ(
        β · log [π_θ(y_w | x) / π_ref(y_w | x)]
      - β · log [π_θ(y_l | x) / π_ref(y_l | x)]
    )
]
```

其中：
- `π_θ`：Policy Model（当前正在训练的模型）
- `π_ref`：Reference Model（冻结的 SFT 模型）
- `y_w`：Chosen（winning）回答
- `y_l`：Rejected（losing）回答
- `σ`：Sigmoid 函数 σ(z) = 1/(1+e^{-z})
- `β`：温度参数，控制与 reference model 的距离

### 直观理解

记：

```
reward_chosen  = β · log[π_θ(y_w|x) / π_ref(y_w|x)]   # policy 相对 reference 对 chosen 的偏向
reward_rejected = β · log[π_θ(y_l|x) / π_ref(y_l|x)]  # policy 相对 reference 对 rejected 的偏向

margin = reward_chosen - reward_rejected              # 偏好差距
loss = -log σ(margin)                                  # 让 margin 尽可能大（→ +∞）
```

当 margin 很小（chosen 不比 rejected 更受偏好）时：
- σ(margin) ≈ 0.5 → loss ≈ -log(0.5) ≈ 0.693（高 loss）

当 margin 很大（chosen 远好于 rejected）时：
- σ(margin) ≈ 1.0 → loss ≈ -log(1.0) ≈ 0（低 loss）

### 数学推导概要

DPO 的数学推导分三步走：

**Step 1：Bradley-Terry 偏好模型**

假设人类偏好遵循 Bradley-Terry 模型：

```
P(y_w ≻ y_l | x) = σ( r(x, y_w) - r(x, y_l) )
```

即 chosen 被偏好的概率等于两个隐式奖励 r 之差过 sigmoid。

**Step 2：RLHF 优化目标**

标准 RLHF 目标是最大化奖励同时保持与 reference 接近：

```
max_π  E_{y~π}[ r(x, y) ] - β · KL( π(·|x) || π_ref(·|x) )
```

这个带 KL 约束的优化问题有闭式解：

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x, y) / β )
```

其中 Z(x) 是配分函数（只依赖 x 不依赖 y）。

**Step 3：消去 Reward Model**

从闭式解反解 r：

```
r(x, y) = β · log[ π*(y|x) / π_ref(y|x) ] + β · log Z(x)
```

代入 Bradley-Terry 偏好模型，Z(x) 项在差值中消去：

```
P(y_w ≻ y_l | x) = σ( β · log[π*(y_w|x)/π_ref(y_w|x)] - β · log[π*(y_l|x)/π_ref(y_l|x)] )
```

将 π* 替换为当前训练的 π_θ，取负对数似然作为 loss，即得到 DPO loss。

**关键**：Z(x) 在 chosen 和 rejected 的差值中消去是 DPO 推导最精妙的一步。如果 Z(x) 不能消去，DPO 就无法仅用 policy 的 logprob 来计算 loss。

---

## Beta 参数详解

### Beta 是什么

Beta (β) 是 DPO loss 中唯一的关键超参数。它源于 RLHF 目标中的 KL 散度约束系数：

```
max_π  E[r(x, y)] - β · KL(π || π_ref)
```

### Beta 太大：β → +∞

```
效果：KL 约束极强 → policy 被强制贴近 reference → 几乎不更新

数学表现：
  reward = β · log(π_θ/π_ref) → 即使 log(π_θ/π_ref) 很小，β 放大后也会导致 penalty
  loss 基本不下降 → DPO ≈ 退化

适用场景：极度保守的场景，如只需要微小调整
```

### Beta 太小：β → 0

```
效果：KL 约束极弱 → policy 可以随意偏离 reference → 可能过拟合偏好数据

数学表现：
  reward = β · log(π_θ/π_ref) → β 很小则 reward 信号很弱
  log(π_θ/π_ref) 可以变得很大 → policy 的分布与 reference 完全不同
  → 可能出现 catastrophic forgetting（忘记预训练/SFT 知识）
  → 可能对数据集的偏好噪声过拟合

适用场景：数据质量极高且希望激进调整
```

### 典型 Beta 取值

| Beta | 效果倾向 | 适用场景 |
|------|---------|---------|
| 0.01 | 极激进，容易过拟合 | 几乎不使用 |
| 0.1 | 标准值，适中的 KL 约束 | 大多数 DPO 训练（**我们的项目取值**） |
| 0.5 | 保守，更贴近 reference | 对稳定性要求高的场景 |
| 1.0 | 很保守 | 偏向 reference 的多任务微调 |

### 我们的项目为什么用 beta=0.1

```
医学 LLM Teacher 项目：
  - SFT 后模型已有良好的医学基础知识
  - DPO 的目标是"把好回答的倾向再拉高一点"，不需要大范围修改
  - 偏好数据 10K 对，质量较好但难免有噪声
  - beta=0.1 提供一个温和的正则化：
      * 让模型有空间去偏好 chosen
      * 但不会偏到忘记原有医学知识
      * 在验证集上保持了较低的 hallucination 率
```

### 面试话术

> "Beta 是 DPO 的'方向盘'。beta 太大，模型被 reference 拉得太紧，DPO 基本等于没训练；beta 太小，模型可能为了讨好偏好数据而忘记 SFT 阶段学到的知识。我们医学项目用 0.1，这是 DPO 论文和社区中最常用的值。实践经验是：beta 在 0.01~0.5 之间按验证集指标做网格搜索，0.1 几乎是'最稳的默认值'。"

---

## Chosen / Rejected 数据格式

### 三元组格式

DPO 数据每个样本是一个三元组：

```json
{
  "prompt": "患者，男，65岁，高血压病史10年，最近一周出现胸闷、气短。请给出可能的诊断和下一步检查建议。",
  "chosen": "根据患者年龄、高血压病史及胸闷气短症状，高度怀疑冠状动脉粥样硬化性心脏病（冠心病）合并心功能不全。建议：1. 心电图及动态心电图监测；2. 心脏超声评估心功能；3. 心肌酶谱及BNP/NT-proBNP检测；4. 必要时行冠脉CTA或冠脉造影。同时排查高血压性心脏病及肺部疾病。",
  "rejected": "可能是感冒了，多喝水休息一下就好。如果不见好转可以去社区医院看看。"
}
```

### 数据构造的黄金法则

1. **Prompt 相同**：chosen 和 rejected 必须来自同一个 prompt，否则比较没有意义
2. **Chosen > Rejected**：标注者/Teacher 模型认为 chosen 明显更优
3. **Rejected 不能是无意义噪声**：必须是"看起来像回事但实际有问题的回答"
4. **Chosen 必须是高质量的**：垃圾进垃圾出，"好"的标准由 chosen 定义

### 数据来源分类

| 来源 | 优点 | 缺点 |
|------|------|------|
| 人工标注 | 质量最高、最接近真实偏好 | 贵、慢 |
| Teacher 模型生成 | 快速、可扩展 | 可能包含 teacher 的 bias |
| 模型自身采样 | 反映模型真实弱点 | 需要筛选、质量可能低 |
| 对抗生成 | 针对特定短板 | 泛化性存疑 |

---

## Chosen 与 Rejected 长度差问题

### 问题描述

DPO 训练中一个常见陷阱：**模型可能学会"生成更长的回答"而非"生成更好的回答"**。

```
原因：
  - 如果 chosen 系统性地比 rejected 长（这是常见情况）
  - chosen 有更多 token，每个 token 的 logprob 可能看起来更分散
  - DPO loss 如果直接用 sum logprob（按 token 求和），长回答天然有更大（更负）的 logprob
  - 模型可能会学到"生成长回答"的捷径，而不是真正提升内容质量
```

### 解决方案

**方案 A：用 mean logprob（平均 per-token logprob）**

```python
# 而非 sum logprob（按 token 求和）
logp_chosen = sequence_logprob(chosen_ids).mean()  # 归一化到 per-token
logp_rejected = sequence_logprob(rejected_ids).mean()
```

这样可以消除长度对 loss 的影响。

**方案 B：长度惩罚 / 归一化**

在 reward 计算时引入长度正则项。

**方案 C：数据筛选**

确保 chosen 和 rejected 的长度分布尽量接近（并非必须严格相等，但不应该有系统性偏差）。

### 我们的项目如何处理的

> 在医学场景中，好的回答（chosen）通常确实比坏的回答（rejected）长——因为好的诊断需要详细的分析推理。这是合理的长短差异，不是 bias。但为了防止模型学到"更长的废话"，我在 DPO loss 中使用的是 mean logprob（即 per-token 平均对数概率），这样使 chosen 和 rejected 在 loss 中的尺度相等。

---

## 为什么 Rejected 不能太弱也不能太强

### Rejected 太弱（如随机文本、无关回答）

```
问题：
  - chosen 和 rejected 差距巨大，loss 很快降到 0
  - 梯度信号消失 → 模型几乎没学到任何东西
  - model 发现"只要我不输出明显垃圾，我就赢了"→ 但没有学到什么是更好的

类比：
  让一个高中生和随机数发生器比赛数学 → 赢了毫无意义
```

### Rejected 太强（如与 chosen 质量差不多的回答）

```
问题：
  - chosen 和 rejected 的 logprob 差异极小
  - DPO 无法可靠区分 → 可能把 chosen 的一些随机波动错当成"偏好"
  - 可能学到错误的方向

类比：
  让两个奥林匹克金牌选手比试 → 胜负可能由随机因素决定
```

### Rejected 应该是什么水平

**最佳 rejected 水平：刚好比 chosen 差一档，但不是垃圾。**

```
理想 rejected：
  ✓ 看起来像一个合理的回答（格式正确、语法通顺）
  ✓ 但存在实质性问题（事实错误、推理漏洞、不够全面）
  ✓ 是模型自身可能犯的错误类型

为什么：
  - 梯度信号有意义：chosen 和 rejected 有明确的质量差距可以优化
  - 训练针对性强：rejected 反映的是实际会出现的问题
  - 不会 trivial：需要模型真正区分好坏，而非仅排除垃圾
```

### 项目中的实践

> 我们项目的 rejected 来自 SFT 后的 Qwen3-8B 模型自身采样。采样时使用适中的 temperature（0.7~0.8），让 rejected 是"SFT 模型在真实场景下可能给出的答案"。这样 rejected 既有合理的格式（不是垃圾），又有 SFT 模型的真实缺陷（如不够专业、遗漏关键信息、诊断不完整），梯度信号非常有针对性。

---

## Rejected 用 SFT 模型采样的好处

### 为什么不用随机/弱模型生成 rejected

```
如果用 GPT2/随机生成 rejected：
  → chosen vs rejected 的差异是"模型能力"差异，不是"回答质量"差异
  → DPO 学的是"更像大模型"，而非"医学回答更好"
  → 学到的"偏好"在推理时没有帮助

如果用人类写的"明显错误"回答：
  → 模型太容易区分，梯度信号弱
  → 实际推理时用户不会输入那么明显的错误
  → 不反映真实分布
```

### 用 SFT 模型采样 rejected 的五大好处

1. **分布一致**：rejected 来自 policy model 自身的分布，梯度信号直接指导 policy 往哪里改进
2. **针对性强**：rejected 的错误正是 policy model 当前会犯的错误
3. **自举效应**：DPO 训练后 policy 更新 → 可以再次采样生成新的 rejected → 迭代训练（Iterative DPO）
4. **训练稳定**：chosen 和 rejected 的 logprob scale 在同一数量级
5. **数据高效**：不需要人工标注，利用模型的自身能力来挖掘弱点

### 面试话术

> "我们项目的 rejected 由 SFT 模型自身采样生成，这是一个'自我对抗'的数据构造策略。SFT 模型采样出的 rejected 代表了它现阶段容易犯的错误——可能是诊断不够全面、忽略了关键症状、或者推理有逻辑跳跃。用这些 rejected 来做 DPO，模型在 loss 优化过程中逐步修正自己最真实的弱点。这种做法的效果远好于用随机弱回答或人工构造的明显错误来做 rejected。"

---

## Chosen 用 Teacher 模型答案的好处

### 什么是 Teacher 模型

在医学 LLM Teacher 项目中，Teacher 是 MiMo-v2.5-pro，一个更大、更强的商业化医学 LLM。它的回答作为 chosen（"标准答案"）。

### 为什么用 Teacher 而非人工标注 chosen

| 维度 | Teacher 模型 | 人工标注 |
|------|-------------|---------|
| 质量 | 高且稳定（大模型医学能力强） | 高但方差大 |
| 规模 | 可大规模生成（10K+） | 慢且贵 |
| 一致性 | 风格统一 | 不同标注者风格不同 |
| 专业性 | 经过医学训练的 LLM | 依赖标注者医学水平 |
| 可复现 | 完全可复现 | 受标注者状态影响 |

### 用 Teacher 的深层好处

1. **定义"好"的标准**：chosen 的质量上限决定了 DPO 的上限。Teacher 模型确定了"理想回答"的方向。
2. **知识蒸馏效应**：DPO 实际上将 Teacher 模型的知识偏好通过偏好比较的方式迁移到 Student（Qwen3-8B）
3. **降低标注成本**：10K 对数据如果全部人工标注，成本巨大且周期长
4. **质量可控**：可以对 Teacher 的输出做后处理和筛选

### 潜在风险

- Teacher 模型自身的 bias 会传递到 Student
- 如果 Teacher 在某些领域也有错误，Student 会学到这些错误
- Teacher 的风格可能与最终期望的风格不完全一致

### 面试话术

> "我们用 MiMo-v2.5-pro 作为 Teacher 来生成 chosen 答案，本质是一种知识蒸馏。Teacher 模型在医学领域的训练更充分、参数量更大，它的回答质量远高于我们正在训练的 8B 模型。chosen 用 Teacher 答案，rejected 用 SFT 模型采样，DPO 的优化方向就是让 8B 模型往 Teacher 的方向靠。这本质上是让一个小模型学习一个大模型的偏好分布。"

---

## Reference Model 与 Policy Model 的定义

### Policy Model (π_θ)

```
定义：当前正在训练的模型，参数 θ 参与梯度更新。

在项目中的角色：
  - 加载 base model (Qwen3-8B 4bit QLoRA) + SFT LoRA adapter
  - LoRA adapter 参数可训练 (is_trainable=True)
  - 前向传播时同时计算 chosen logprob 和 rejected logprob
  - 反向传播时只更新 LoRA adapter 参数

保存方式：
  - 训练结束后保存 adapter_model.safetensors (DPO 微调后的 LoRA 权重)
```

### Reference Model (π_ref)

```
定义：冻结的 SFT 模型，参数不参与梯度更新，仅用于计算 KL 约束的基线。

在项目中的角色：
  - 加载同样的 base model + SFT LoRA adapter
  - 通常合并后冻结 (merge_and_unload + requires_grad=False)
  - 只做前向传播（计算 logprob），不参与反向传播
  - 提供 chosen 和 rejected 在 SFT 模型下的 logprob

为什么要合并后冻结：
  - 合并后的模型是标准 nn.Linear，前向传播更快
  - 不需要 LoRA 分支的额外计算
  - 显存中只有一份 BF16 权重，而非 4bit + LoRA
```

### 两者关系可视化

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
     │                │           │                │
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
              │ L = -log σ(     │
              │   β·Δ_chosen    │
              │  -β·Δ_rejected  │
              │ )               │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Backward: 只更新 │
              │ LoRA adapter    │
              └─────────────────┘
```

---

## Sequence Logprob 计算与 Token Logprob Gather

### 什么是 Sequence Logprob

对于一个 token 序列 y = (y1, y2, ..., yL)，sequence logprob 是整个序列在模型下的对数概率：

```
log π(y|x) = Σ_{t=1}^{L} log π(y_t | x, y_{<t})
```

即每个 token 的条件对数概率之和。

### 如何在代码中计算

```python
# Step 1: 将 prompt + response 拼接后做 forward
input_ids = torch.cat([prompt_ids, response_ids], dim=-1)  # [1, P+R]
outputs = model(input_ids)  # logits: [1, P+R, V]

# Step 2: 取每个位置的 log_softmax
log_probs = torch.log_softmax(outputs.logits, dim=-1)  # [1, P+R, V]

# Step 3: gather 出 response 部分每个 target token 的 logprob
# response_ids: [1, R]，每个位置是实际输出的 token id
# 我们需要用 response_ids 做索引，从 log_probs 中取出对应的 logprob

# 注意：log_probs[..., :-1, :] 对应预测下一个 token 的分布
# response_ids[..., 1:] 是每个位置的实际 token
# 位置 i 的 logprob = log_probs[:, prompt_len + i - 1, response_ids[:, i]]
```

### Token Logprob Gather 的精确理解

```
假设 prompt = "诊断：", response = "冠心病"

Token 序列：           [P1,  P2,  ...,  Pp,  R1,  R2,  ...,  Rr]
                                    ↑     ↑    ↑         ↑
对 response 的 logprob 计算需要：    模型在位置 p-1 预测 R1 的概率
                                         在位置 p   预测 R2 的概率
                                         ...       ...
                                         在位置 p+r-2 预测 Rr 的概率

所以 gather 时：
  - 从 logits/log_probs 中取 [:prompt_len-1 : prompt_len+response_len-1] 这 R 个位置
  - 每个位置取对应的 token id 的 logprob
  - 这 R 个值就是 response 中每个 token 的 logprob
```

### 为什么需要 log_softmax

```
log_softmax 将 logits 转为对数概率，范围 (-∞, 0]：
  - 0 表示概率 1（完美预测）
  - -∞ 表示概率 0（不可能）
  - 实际值通常在 -1 ~ -10 之间

如果用 softmax + log：
  softmax 可能 underflow（概率太小变 0 → log(0) = -inf）
  log_softmax 数值稳定，直接在 log 空间计算
```

---

## Response Logprob：Sum vs Mean

这是 DPO 实现中非常重要的选择，直接影响训练效果。

### Sum Logprob

```python
logp_response = token_logprobs.sum()  # 所有 token 的 logprob 求和
```

**特点**：
- 长回答的 logprob 更负（绝对值更大），天然处于劣势
- 如果 chosen 比 rejected 长，chosen 的 sum logprob 会更低
- 模型可能学到"生成更短的回答"来获得更高的 reward

### Mean Logprob

```python
logp_response = token_logprobs.mean()  # 平均 per-token logprob
```

**特点**：
- 消除长度影响，只比较 per-token 质量
- 长回答和短回答在同一个尺度上比较
- 更公平

### 深度比较

| 维度 | Sum | Mean |
|------|-----|------|
| 长度偏差 | 天然偏向短回答 | 长度无关 |
| 数学意义 | sequence 的联合对数概率 | per-token 的期望对数概率 |
| DPO 论文 | 使用 sum | 也有讨论 |
| 实践推荐 | 如果 chosen/rejected 长度接近用 sum | 长度相差大用 mean |

### 项目中的选择

```
医学 LLM Teacher 项目：使用 mean logprob

原因：
  1. 医学回答中，好的诊断（chosen）通常更详细，长度更长
  2. 如果用 sum，chosen 天然处于劣势，DPO 可能反向优化
  3. mean 让模型关注 per-token 质量，而不是输出长度
```

### 面试话术

> "sequence logprob 用 sum 还是 mean 取决于数据中 chosen/rejected 的长度分布。在医学场景，好的回答通常更详细——包含更多诊断推理和检查建议——所以 chosen 系统性地比 rejected 长。如果我用 sum logprob，DPO 会认为 chosen 的 reward 更低（因为 sum logprob 更负），这会导致优化方向反了。所以我用 mean logprob——在 per-token 层面比较质量，消除长度偏差。"

---

## Prompt 部分是否计入 Logprob

### 答案：不计入

DPO loss 只计算 **response 部分的 logprob**，prompt 部分只作为条件输入。

```
DPO 优化的是 π_θ(response | prompt)，而不是 π_θ(prompt, response)。

数学上：
  log π_θ(y | x) 中，x 是 prompt，y 是 response
  条件概率已经天然地把 prompt 看作给定条件

代码上：
  只 gather 出 response_ids 对应位置的 logprob
  prompt 部分的 logprob 不参与 loss 计算
```

### 为什么不计入 Prompt

1. **Prompt 是用户/系统给的，不是模型生成的**：不需要优化"生成 prompt"
2. **Prompt 的 logprob 是常数**：对于同一个 prompt，不管模型怎么变，prompt 部分的 logprob 在 chosen 和 rejected 中相消
3. **如果计入 prompt 部分**：会把 prompt 中的噪声也学进去，而且 chosen 和 rejected 的 prompt 相同，实际效果为 0（相消）

### 面试话术

> "Prompt 部分不计入 logprob 计算。DPO 优化的是条件分布 π_θ(response|prompt)，即给定 prompt 后生成优质 response 的能力。prompt 的 logprob 对 chosen 和 rejected 是相同的，在 DPO loss 的差值中自然消去。如果计入 prompt 反而可能引入不必要的噪声。"

---

## DPO 训练指标：Accuracy / Reward Margin / Chosen & Rejected Reward

### 四大核心指标

#### 1. DPO Accuracy

```python
# 定义：chosen 的隐式 reward > rejected 的隐式 reward 的比例
accuracy = (reward_chosen > reward_rejected).float().mean()
```

**含义**：在 batch 中，有多少比例的数据 model 正确地把 chosen 排在了 rejected 前面。

**正常值**：训练开始时约 0.5（随机），逐渐上升到 0.7~0.85。如果超过 0.95 意味着可能过拟合。

#### 2. Reward Margin

```python
# 定义：chosen 和 rejected 隐式 reward 的差距
margin = reward_chosen - reward_rejected
```

**含义**：模型对 chosen 和 rejected 的区分能力。margin 越大，区分越清晰。

**正常值**：应随着训练逐渐增大，从接近 0 到 ~0.5~2.0。如果 margin 爆炸（> 10），说明模型可能过拟合。

#### 3. Chosen Reward

```python
# 定义：chosen 回答的隐式 reward
reward_chosen = beta * (logp_policy_chosen - logp_ref_chosen)
```

**含义**：policy model 相对于 reference model 对 chosen 回答的"偏好提升"。

**正常值**：应该为正（policy 比 reference 更喜欢 chosen）。如果为负意味着 DPO 在反向优化。

#### 4. Rejected Reward

```python
# 定义：rejected 回答的隐式 reward
reward_rejected = beta * (logp_policy_rejected - logp_ref_rejected)
```

**含义**：policy model 相对于 reference model 对 rejected 回答的"偏好程度"。

**正常值**：应该为负（policy 比 reference 更不喜欢 rejected）。如果持续为正意味着 DPO 方向反了。

### 理想的训练曲线

```
         ↑
  Reward │  reward_chosen ────────────────
         │                              ／
         │                          ／
         │                      ／
    0    │  ─────────────────────────────── reward_rejected
         │
         │  ──── margin (递增)
         │
         └──────────────────────────────────→ Steps
              accuracy: 0.5 → 0.75 → 0.82
```

### 面试话术

> "DPO 训练我主要监控四个指标：accuracy（chosen 的 reward 是否高于 rejected 的比例）、reward margin（两者的差距）、以及 chosen reward 和 rejected reward 各自的绝对值。正常训练中，chosen reward 应该为正并逐渐增大，rejected reward 为负并逐渐减小，margin 逐步扩大。如果 rejected reward 也在增大（变正），说明模型在同时提升对所有输出的概率——这是无效训练。如果 margin 爆炸增长（比如从 0.5 跳到 10），说明模型在过拟合，需要加大 beta 或者减小 lr。"

---

## DPO Loss 不下降 / Reward 异常的诊断与修复

### 问题 1：DPO Loss 一直震荡不下降

```
可能原因：
  1. 学习率太大 → 梯度来回震荡
  2. chosen 和 rejected 质量太接近 → 模型无法可靠区分
  3. beta 太小 → KL 约束太弱，模型乱跑
  4. 数据噪声大 → 标注 inconsistent

修复：
  - 降低学习率（2e-4 → 5e-5）
  - 检查 chosen/rejected 的 logprob 差距（如果 < 0.5，数据区分度太差）
  - 增大 beta（0.01 → 0.1）
  - 人工抽查数据质量
```

### 问题 2：Chosen Reward 和 Rejected Reward 同时增大

```
现象：
  reward_chosen: -0.2 → +0.5 → +1.2 ✓
  reward_rejected: -0.1 → +0.3 → +0.8 ✗（应该为负！）

原因：
  模型学到了"对所有回答都提概率"而不是"区分好坏"
  → 可能在优化 π_θ 的整体语言模型能力，而非偏好

修复：
  - 增大 beta（强化 KL 约束）
  - 检查 reference model 是否正确加载（没有被更新）
  - 检查 loss 实现中是否有 sign 错误
```

### 问题 3：DPO Accuracy 很高但生成质量下降

```
现象：
  accuracy ≈ 0.95 但生成结果变差 → 过拟合

原因：
  模型学会了"在训练集上区分 chosen 和 rejected"
  但泛化能力下降

修复：
  - 早停（early stopping）
  - 增加 warmup steps
  - 增大 dropout
  - 增大数据集或增加数据增强
```

### 问题 4：Loss 迅速降到 0 附近就不再变化

```
现象：
  loss 在几百步内降到接近 0 → 停滞

原因：
  chosen 和 rejected 差距太大（如 rejected 是随机文本）
  → 模型轻松达到 margin >> 0 → loss ≈ 0

修复：
  - 替换 rejected 为更难的样本
  - 降低学习率进一步微调
```

### 面试话术

> "如果 DPO loss 不下降，我会按这个优先级排查：第一，看 lr 是否太大——DPO 对 lr 比 SFT 敏感得多，通常用 5e-6 到 5e-5；第二，看 chosen 和 rejected 的区分度——如果它们来自同一个模型采样且温度太高，可能几乎一样，模型学不到什么；第三，看 reference model 是否被错误地更新了——我曾经不小心把 reference 的 grad 打开了，导致两个模型同步移动，loss 一直在震但不收敛；第四，看 beta 是否合适——太大压制了学习，太小导致不稳定。"

---

## DPO 训练显存分析与 Reference Model 省显存技巧

### DPO 为什么比 SFT 更吃显存

```
SFT 训练显存需求：
  - Base Model (4bit, 冻结):          ~4 GB
  - LoRA Adapter (BF16):              ~0.02 GB
  - Activations:                      ~5-8 GB
  - Optimizer States (FP32, LoRA):    ~0.06 GB
  ─────────────────────────────────────────
  总计:                               ~12 GB

DPO 训练显存需求（naive 方式）：
  - Policy Model  (4bit + LoRA):      ~4 GB
  - Reference Model (BF16 全精度):     ~16 GB  ← 额外开销！
  - Activations (×2, 两个 model):      ~10-16 GB
  - Optimizer States:                 ~0.06 GB
  ─────────────────────────────────────────
  总计:                               ~35 GB ← 可能超出 RTX 5090 32GB！
```

### Reference Model 如何省显存

#### 方法一：提前合并 + 卸载到 CPU

```python
# 在训练前：将 SFT LoRA adapter 合并到 base model，卸载到 CPU
ref_model = AutoModelForCausalLM.from_pretrained(base_path)
ref_model = PeftModel.from_pretrained(ref_model, sft_lora_path)
ref_model = ref_model.merge_and_unload()   # 合并为 BF16 模型

# 转移到 CPU，需要时再移回来
ref_model = ref_model.to("cpu")
ref_model.eval()  # 推理模式（不计算梯度，节省激活值显存）
for p in ref_model.parameters():
    p.requires_grad = False
```

#### 方法二：4bit 量化 + 不 merge

```python
# Reference Model 也使用 QLoRA 加载（但冻结）
ref_model = AutoModelForCausalLM.from_pretrained(
    base_path,
    quantization_config=bnb_config,  # 4bit
)
ref_model = PeftModel.from_pretrained(ref_model, sft_lora_path)
# 冻结所有参数
for p in ref_model.parameters():
    p.requires_grad = False
# 显存：约 4-5 GB（4bit base + LoRA，而非 16 GB BF16）
```

#### 方法三：顺序计算（Trade-off 速度换显存）

```python
# 先计算 policy model 的 logprob + backward
# 再计算 reference model 的 logprob（不 backward）
# 两个 model 不需要同时驻留在显存中

with torch.no_grad():
    ref_chosen_logp = compute_logp(ref_model, chosen_ids)
    ref_rejected_logp = compute_logp(ref_model, rejected_ids)

# 将 ref_model 卸载到 CPU
ref_model.to("cpu")
torch.cuda.empty_cache()

# 然后正常训练 policy model
policy_chosen_logp = compute_logp(policy_model, chosen_ids)
policy_rejected_logp = compute_logp(policy_model, rejected_ids)
loss = dpo_loss(policy_chosen_logp, policy_rejected_logp,
                ref_chosen_logp, ref_rejected_logp, beta)
loss.backward()
```

#### 方法四：使用 gradient_checkpointing

```python
# policy model 开启 gradient checkpointing，用计算换显存
policy_model.gradient_checkpointing_enable()
# 激活值显存减少约 30-50%
```

### 面试话术

> "DPO 比 SFT 更吃显存，因为需要同时持有 policy model 和 reference model。reference model 在 naive 实现下以 BF16 全精度存储，一个 8B 模型就 16GB，加上 policy model 的量化权重和激活值，很容易超出 32GB 上限。我用了三个技巧：第一，reference model 也用 QLoRA 加载、合并、转 CPU 推理；第二，训练时先计算 reference logprob（no_grad），卸载到 CPU 后再正常训练 policy；第三，policy model 开启 gradient checkpointing 压缩激活值。这样 8B + DPO 在 RTX 5090 32GB 上可以在 batch_size=4、seq_len=2048 下稳定运行。"

---

## DPO vs IPO vs KTO vs ORPO vs SimPO vs CPO

DPO 之后涌现了大量变体，面试中可能被问到"你知道 DPO 有哪些改进版本吗"。

### 方法对比表

| 方法 | 全称 | 核心改动 | 是否需要 reference | 论文年份 |
|------|------|---------|-------------------|---------|
| **DPO** | Direct Preference Optimization | 直接将 RLHF 转化为二分类 loss | 需要 | 2023 |
| **IPO** | Identity Preference Optimization | 用平方 loss 替代 sigmoid，margin 更严格 | 需要 | 2023 |
| **KTO** | Kahneman-Tversky Optimization | 不要求成对数据，单条数据即可（好/坏分别处理） | 需要 | 2024 |
| **ORPO** | Odds Ratio Preference Optimization | 将 DPO loss 和 SFT loss 联合优化，无需 reference | **不需要** | 2024 |
| **SimPO** | Simple Preference Optimization | 用序列平均 logprob 的差作为 reward，加长度惩罚 | **不需要** | 2024 |
| **CPO** | Contrastive Preference Optimization | 直接用 chosen logprob - rejected logprob 的 contrastive loss | **不需要** | 2024 |

### 方法详解

#### IPO (Identity Preference Optimization)

```
DPO loss:  -log σ(β·Δ)
IPO loss:  (β·Δ - 1/(2τ))²   # 平方 loss

区别：
  - DPO 用 sigmoid + 交叉熵 → 当 margin 很大时梯度消失（saturation）
  - IPO 用平方 loss → 只要 margin ≠ target，就有梯度
  - IPO 不会 early saturation，理论上能学得更深

适用场景：
  偏好信号非常强（chosen 远好于 rejected）时，IPO 比 DPO 更能持续优化
```

#### KTO (Kahneman-Tversky Optimization)

```
核心创新：不需要成对数据！

DPO 需要 (prompt, chosen, rejected) 三元组
KTO 只需要 (prompt, answer, label)，label ∈ {good, bad}

好处：
  - 数据收集成本大幅降低（不需要人工对比两个回答）
  - 可以利用"点赞/点踩"式的众包数据
  - 单条标注的效率远高于成对标注

缺点：
  - 信号可能不如成对比较强
```

#### ORPO (Odds Ratio Preference Optimization)

```
核心创新：不需要 reference model！

将 SFT loss 和 DPO-like loss 联合优化：
  L_ORPO = L_SFT + λ · L_Odds

L_Odds 使用 odds ratio（几率比）:
  odds_θ(y|x) = π_θ(y|x) / (1 - π_θ(y|x))
  L_Odds = -log σ( log[odds_θ(chosen)] - log[odds_θ(rejected)] )

好处：
  - 不需要 reference model → 显存减半
  - SFT + 偏好联合优化 → 可能更稳定
```

#### SimPO (Simple Preference Optimization)

```
核心创新：不需要 reference model + 内置长度惩罚

reward = (1/|y|) · log π_θ(y|x)   # 平均 logprob（mean, 不是 sum）
         ↑ 长度归一化

L_SimPO = -log σ( β·(reward_chosen - reward_rejected) - γ )
                                                ↑ 额外的 margin 参数

好处：
  - 不需要 reference model
  - 长度惩罚天然内建（用 mean logprob）
  - 额外的 margin γ 确保 chosen 比 rejected 好一个最小差距
```

#### CPO (Contrastive Preference Optimization)

```
核心创新：最简单的 contrastive 框架

L_CPO = -log σ( log π_θ(chosen|x) - log π_θ(rejected|x) )

本质就是 DPO 中让 β=1、π_ref=uniform（或去掉 reference）

好处：极简实现
缺点：没有 KL 约束 → 容易跑偏
```

### 选型建议

```
如果你有：
  - 成对偏好数据 → DPO / IPO
  - 单条评分数据 → KTO
  - 显存紧张，没法同时跑两个模型 → SimPO / ORPO
  - 数据质量极高，不需要 KL 约束 → CPO
```

---

## DPO 为什么会放大幻觉

这是 DPO 一个关键缺陷，面试中极可能被问到。

### 核心机制

```
DPO 优化的是"人类偏好"，而非"事实正确性"

人类偏好什么？
  ✓ 自信、肯定的语气
  ✓ 详细、有条理的回答
  ✓ 流畅、专业的表达
  ✗ 不一定偏好事实正确！

如果 chosen 因为"看起来更专业"而被偏好：
  → DPO 学到的是"说得很肯定" = 好
  → 模型学会对不准确的内容也说得很肯定
  → 幻觉被强化——从"不太确定的错误"变成"非常自信的错误"
```

### 数学直觉

```
DPO 的隐式 reward：
  r(x, y) = β · log[π_θ(y|x) / π_ref(y|x)]

reward 不包含任何事实性校验！
模型只需要让 logprob 变高就能获得更高的 reward。
如果 chosen 中包含事实错误，但回答的语气自信、条理清晰，
DPO 仍然会奖励这种模式 → 模型学会"自信地犯错"。
```

### 为什么"长尾问题"更容易中招

```
高频问题（如高血压、糖尿病）：
  - 训练数据中 chosen 包含正确答案
  - rejected 包含明显错误
  - DPO 学到正确方向 ✓

长尾问题（如罕见病、偏门药物）：
  - 训练数据中 chosen 可能也有不准确之处（teacher 也不是完美的）
  - 但 chosen 的格式/语气/结构更好
  - DPO 学到的是"格式好 = 好"，而非"事实对 = 好"
  - 测试时模型对不准确的回答也说得头头是道 ✗
```

### 面试话术

> "DPO 有一个根本性的局限：它优化的是'人类偏好'而非'事实正确性'。如果训练数据中 chosen 的好表现在于语气自信、格式清晰、推理详细，DPO 会奖励模型输出这种风格的答案——即使内容可能是错的。在常见问题上这个风险较小，因为 chosen 通常是正确的；但在长尾问题上，chosen 本身可能也不完美，DPO 就可能强化模型的幻觉倾向。这在我们医学项目中得到了验证——DPO 后长尾罕见病问题上的幻觉率反而上升了。"

---

## 医疗场景 DPO 的特有风险

### 风险一："自信的误诊"

```
SFT 模型的错误：犹豫的、不确定的错误
  "可能是感冒，建议观察"（错误但低风险）

DPO 后的错误：确定性的、看起来专业的错误
  "根据你的症状描述，可以确诊为病毒性上呼吸道感染，
   建议口服复方氨酚烷胺，配合清热解毒口服液……"
  （如果有更严重的潜在病因，这种自信的误诊可能危及生命）
```

### 风险二：偏好与医学伦理冲突

```
医学回答的正确标准：
  - 准确性 > 流畅性
  - 安全性 > 肯定性
  - 免责声明 > 打包票

人类偏好（可能被 DPO 学到的）：
  - 肯定的答案感觉更"有用"
  - 详细的建议感觉更"专业"
  - 确定的诊断感觉更"靠谱"
```

### 风险三：数据 Bias 放大

```
如果标注者/Teacher 模型对某种风格的医学回答有偏好：
  → DPO 会放大这种偏好
  → 可能覆盖掉一些医学上必要的"不确定性"

例如：
  "需要进一步检查" vs "可以确诊为"
  前者更符合医学实践，但如果数据中后者被标注为 chosen，
  DPO 会让模型倾向于过早下结论。
```

### 面试话术

> "医疗是 DPO 的高风险场景。因为医学容错率极低——在其他领域，DPO 让模型更自信是好事；在医疗领域，DPO 让模型对错误答案更自信可能是致命的。我们的应对策略是 DPO 后接入 Safety-RAG——不依赖 DPO 来保证安全性，而是用外部知识库在推理阶段做事实校验。DPO 负责让回答格式和质量更好，RAG 负责让事实正确。"

---

## 项目复盘：DPO 为何在长尾医学问题上产生幻觉

### 问题描述

```
现象：
  DPO 训练后，常见病问答（高血压、糖尿病）质量提升明显
  但长尾问题（罕见病、少见药物交互作用）出现更严重的幻觉
  模型会编造不存在的症状、药物、剂量信息

数据特征：
  ~10K DPO 对，覆盖约 2,000 种医学问题
  其中约 70% 是高频问题（常见病），30% 是长尾问题
```

### 根因分析

**1. 数据分布不均**

```
高频问题 (~7K 对)：
  chosen 质量高（common disease，teacher 模型表现好）
  rejected 有明显错误

长尾问题 (~3K 对)：
  chosen 质量参差不齐（teacher 模型对罕见病也可能不完美）
  rejected 和 chosen 的差距主要是"格式"而非"事实"
  → DPO 学到的是"输出更详细的长回答"，而非"纠正事实"
```

**2. Teacher 模型本身在长尾问题上并不完美**

```
MiMo-v2.5-pro 虽然很强，但对罕见病的了解可能也不全面
如果 chosen 中包含不准确信息：
  → DPO 合法地把"不准确但格式好的回答"当作偏好目标
  → 模型学会模仿这种模式
```

**3. DPO 的隐式奖励只依赖 logprob**

```
reward = β · log(π_θ / π_ref)

没有任何机制检查"chosen 是否事实正确"
在长尾问题上：
  - 模型可以通过"说得更详细、更肯定"来获得更高 reward
  - 无需确保内容是正确的
```

**4. Mean logprob 加剧了"长度问题"**

```
虽然使用 mean logprob 避免了长度偏差，但：
  - 长回答中每个 token 的预测难度更低（有更多 context）
  - 更长 → 更大比例的 token 是容易预测的功能词
  - model 可能学会"生成长回答"来获得更高的 mean logprob
```

### 面试话术

> "我们在医学项目的 DPO 训练中发现了一个有趣的失败案例：长尾罕见病问题上的幻觉率不降反升。复盘发现三个根因：一是数据分布不均，长尾问题的 chosen 本身质量就不如高频问题；二是 Teacher 模型在罕见病领域也不是完全可靠，chosen 中杂糅了事实错误和良好格式——DPO 分不清哪个是'好'；三是 DPO 的隐式奖励只看 logprob，没有任何事实性校验机制。这让我意识到 DPO 不是银弹——它对数据质量的依赖比 SFT 更高。最终我们引入 Safety-RAG 在推理阶段做知识检索来弥补。"

---

## 为什么 DPO 后还需要 Safety-RAG

### RAG 解决了 DPO 解决不了的问题

```
DPO 能做的：
  ✓ 让回答风格更好（更专业、更有条理）
  ✓ 修正已知的常见错误模式
  ✓ 让模型更"符合人类偏好"

DPO 做不到的：
  ✗ 保证事实正确性（特别是长尾/新的知识）
  ✗ 纠正在 chosen 中就已存在的错误
  ✗ 防止模型对幻觉"过于自信"
  ✗ 覆盖训练数据中从未出现的新疾病/新药物
```

### Safety-RAG 的作用

```
Safety-RAG 在推理阶段做的事：
  1. 对用户问题检索外部知识库（PubMed、医学教材、药品说明书等）
  2. 将检索到的证据作为额外上下文注入 prompt
  3. 模型生成时基于检索到的证据，而非仅依赖训练时记住的知识
  4. 对模型的输出做事实性校验（与知识库比对）
```

### 为什么 Safety-RAG + DPO > 单独的 DPO

```
                    仅 DPO           DPO + Safety-RAG
高频问题准确性       高              高
长尾问题准确性       低（幻觉）      中高（RAG 弥补）
回答格式/风格        好              好
推理阶段可控性       低（纯生成）    高（知识锚定）
对新知识的适应       差（需重训）    好（更新知识库即可）
```

### 面试话术

> "DPO 和 Safety-RAG 解决的是不同层次的问题。DPO 优化的是'模型表达知识的方式'——让回答更专业、更有条理；Safety-RAG 解决的是'模型知道什么'——确保输出的内容是事实正确的。在医学场景，这两者缺一不可。DPO 让回答好看，RAG 让回答正确。没有 RAG 的 DPO 可能在长尾问题上把错误的回答包装得更加逼真，这是医学场景最危险的 failure mode。"

---

## 面试必答："DPO 是不是一定提升安全性"

### 标准回答

> "DPO 不一定提升安全性，甚至可能降低安全性。DPO 优化的是'偏好'而非'正确性'或'安全性'。安全性能不能提升完全取决于偏好数据中是否包含了安全相关的偏好信号。如果训练数据中的 chosen 在安全性上确实优于 rejected（比如 chosen 包含了免责声明而 rejected 没有），DPO 可以提升安全性。但如果数据中 chosen 被选中的原因与安全无关（比如只是因为格式好、更流畅），DPO 反而可能让模型在错误方向上变得更自信——这在医疗场景中极其危险。简而言之，DPO 能学到你展示给它看的'好'，但如果你的'好'定义里没有包含'安全'，DPO 就不会主动提升安全性。"

### 展开论述

```
安全性提升的前提：
  1. 训练数据中的 chosen 在安全维度上确实优于 rejected
  2. 安全性相关的信号强度足以被 DPO 捕捉到
  3. 数据分布覆盖了安全风险的高发场景

安全性不提升的情况：
  1. 数据只关注了回答的"有用性"和"流畅性"
  2. 长尾安全风险（rare adverse effects）在数据中占比太小
  3. chosen 本身包含不安全内容（如过度肯定的诊断）

安全性下降的情况（最危险）：
  1. DPO 让模型对所有输出更自信，包括不安全的输出
  2. 模型学会了"打包票"式回答，丢失了必要的谨慎
  3. 对未见过的安全场景，模型可能"自信地胡说"
```

### 如果需要追问"那怎么保证安全性"

> "我的做法是分层防护：第一层，在数据构造阶段确保 chosen 符合安全标准（如必须包含免责声明、不给出明确的药物剂量建议）；第二层，在 DPO 训练中监控安全相关的指标（如拒绝回答的比例、免责声明出现的频率）；第三层，DPO 训练后接入 Safety-RAG 在推理阶段做事实性校验——这是最可靠的一层，因为它不依赖模型参数化的'记忆'。"

---

## 面试必答："DPO 数据质量怎么保证"

### 标准回答

> "DPO 数据质量保证我分三步走。第一步是 chosen 质量控制——我们用 Teacher 模型（MiMo-v2.5-pro）生成 chosen，然后人工抽查 5-10% 确保没有明显的事实错误、伦理问题或糟糕格式。第二步是 rejected 质量控制——rejected 由 SFT 模型采样生成，我们用规则过滤掉太短、太长、重复或格式崩溃的样本。第三步是对比质量检查——确保 chosen 和 rejected 有实质性差距而非仅格式差异，我会随机抽查 chosen/rejected 对，验证 rejected 确实存在实质性问题而 just 是表述方式不同。此外还需要做数据分布分析——确保 chosen/rejected 的长度分布接近，各主题领域平衡覆盖。"

### 展开论述

**第一步：Chosen 质量控制**

```
1. Teacher 模型生成
2. 规则过滤：
   - 长度 ≥ 50 tokens（排除敷衍回答）
   - 包含专业术语（至少 N 个医学关键词）
   - 无明显格式错误
3. 人工抽查（5-10%）：
   - 事实准确性
   - 医学伦理性
   - 输出完整性
4. 自动过滤：
   - 重复度（与 prompt 的 n-gram 重叠率）
   - 是否有幻觉特征（如编造不存在的药物名）
```

**第二步：Rejected 质量控制**

```
1. SFT 模型采样（temperature=0.7~0.8）
2. 规则过滤：
   - 长度 20-1000 tokens（不要太短也不要太长）
   - 不包含特殊控制 token
   - 格式完整性（完整的句子）
3. 质量过滤：
   - 不能是纯通用、万能回答（如"请咨询医生"仅此一句）
   - 不能是纯格式错误（这太容易区分）
4. 去重：与 chosen 的 BLEU/ROUGE 相似度不能太低（差异太小）也不能太高（可能是复制）
```

**第三步：对比质量检查**

```python
# 检验 chosen/rejected 对的质量
def check_pair_quality(chosen, rejected):
    issues = []

    # 1. 长度差异检查（差距不应 > 5 倍）
    if len(rejected) == 0 or len(chosen) / len(rejected) > 5:
        issues.append("长度差异过大")

    # 2. 格式差异检查（不应仅是格式差异）
    if chosen.replace("\n", "") == rejected.replace("\n", ""):
        issues.append("差异仅为换行")

    # 3. 关键词重叠检查
    chosen_keywords = set(extract_medical_terms(chosen))
    rejected_keywords = set(extract_medical_terms(rejected))
    if chosen_keywords == rejected_keywords:
        issues.append("医学内容无实质差异")

    # 4. 基础质量检查
    if not rejected_has_any_error:
        issues.append("rejected 无明显错误，区分度过低")

    return issues
```

---

## 面试必答："你手写过 DPO loss 吗"

### 标准回答

> "手写过的。DPO loss 本质是一个二分类交叉熵，核心就是计算 policy model 和 reference model 在 chosen 和 rejected 上的 logprob，然后套公式。我在项目中实现了完整的 DPO loss 函数，包括 logprob 计算、隐式 reward 计算、loss 和 metrics 的封装。下面是我手写的核心实现。"

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
    # log_probs[:, prompt_len-1 : prompt_len+R-1, :] 是预测 R 个 token 的分布
    # 对每个位置，取实际 token 的 logprob
    response_log_probs_full = log_probs[:, prompt_len - 1 : prompt_len + response_len - 1, :]
    # shape: [batch_size, response_len, vocab_size]

    # Step 4: Gather 出实际 token 的 logprob
    # label_ids: [batch_size, response_len]，每个值 in [0, vocab_size-1]
    per_token_logp = response_log_probs_full.gather(
        dim=-1,
        index=label_ids.unsqueeze(-1)   # [batch_size, response_len, 1]
    ).squeeze(-1)  # [batch_size, response_len]

    # Step 5: 应用 label_mask（屏蔽 padding token）
    per_token_logp = per_token_logp * label_mask  # padding 位置变为 0

    # Step 6: 聚合 — mean（推荐）或 sum
    # mean: 消除长度影响
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

    # 模拟输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # label_ids 和 label_mask 只覆盖 response 部分
    label_ids = input_ids[:, prompt_len:]  # [batch_size, response_len]
    label_mask = torch.ones(batch_size, response_len)

    # 模拟模型前向输出
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
    print(f"sequence_logp values:   {seq_logp}")                 # 每个样本的 mean logprob


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

    参数说明：
        policy_chosen_logp:  当前训练的 policy model 对 chosen 回答的 logprob
        policy_rejected_logp: 同上，对 rejected 回答
        ref_chosen_logp:     reference model 对 chosen 回答的 logprob
        ref_rejected_logp:   同上，对 rejected 回答
        beta:                控制 KL 约束强度，β=0.1 为常用默认值
        loss_type:           "sigmoid" (DPO 标准), "hinge" (hinge loss), "ipo" (平方 loss)

    返回：
        loss:   标量，用于 backward
        metrics: dict，包含 accuracy / margin / rewards 等监控指标
    """
    # ===== Step 1: 计算隐式 reward =====
    # reward = β · (log π_θ - log π_ref)
    if reference_free:
        # 无 reference 模式（类 CPO）：直接用 logprob 差
        chosen_reward = beta * policy_chosen_logp
        rejected_reward = beta * policy_rejected_logp
        # 此时实际上 penalty = β · log π_θ
    else:
        chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
        rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)

    # chosen_reward:   [batch_size] — 希望它尽可能大
    # rejected_reward: [batch_size] — 希望它尽可能小

    # ===== Step 2: 计算 reward margin =====
    reward_margin = chosen_reward - rejected_reward  # [batch_size]

    # ===== Step 3: 根据 loss_type 计算 loss =====
    if loss_type == "sigmoid":
        # 标准 DPO loss：-log σ(margin)
        # margin 越大 → σ(margin) 越接近 1 → log 越接近 0 → loss 越小
        loss = -F.logsigmoid(reward_margin).mean()  # scalar

    elif loss_type == "hinge":
        # Hinge loss 变体：max(0, 1 - margin)
        # 更严格——必须 margin ≥ 1 才零 loss
        loss = torch.clamp(1.0 - reward_margin, min=0.0).mean()

    elif loss_type == "ipo":
        # IPO loss：(margin - 1/(2β))²
        # 优点：不会像 sigmoid 那样在 margin 很大时梯度消失
        target = 1.0 / (2.0 * beta)
        loss = ((reward_margin - target) ** 2).mean()

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # ===== Step 4: 计算监控指标（不参与 backward） =====
    with torch.no_grad():
        # 4a. Accuracy：chosen reward > rejected reward 的比例
        accuracy = (reward_margin > 0).float().mean().item()

        # 4b. 平均 reward margin
        avg_margin = reward_margin.mean().item()

        # 4c. Chosen 和 rejected 的平均 reward
        avg_chosen_reward = chosen_reward.mean().item()
        avg_rejected_reward = rejected_reward.mean().item()

        # 4d. Reward gap（chosen - rejected 的均值）
        reward_gap = avg_chosen_reward - avg_rejected_reward

        # 4e. Logprob gap（原始 logprob 差值，不考虑 β）
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

    # 模拟 logprob 值（在实际训练中，这些来自模型前向计算）
    # policy model 会尝试让 chosen 的 logprob 更高、rejected 更低
    policy_chosen_logp = torch.tensor([-2.1, -1.8, -3.0, -2.5],
                                       dtype=torch.float32)
    policy_rejected_logp = torch.tensor([-3.5, -2.2, -2.9, -4.0],
                                         dtype=torch.float32)

    # reference model 的 logprob（冻结，提供一个基线）
    ref_chosen_logp = torch.tensor([-2.5, -2.0, -3.2, -2.8],
                                    dtype=torch.float32)
    ref_rejected_logp = torch.tensor([-3.0, -2.1, -3.0, -3.5],
                                      dtype=torch.float32)

    # 使用 sigmoid loss_type
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
    print(f"Accuracy:      {metrics['accuracy']:.2%}  "
          f"(chosen_reward > rejected_reward 的比例)")
    print(f"Reward Margin: {metrics['reward_margin']:.4f}")
    print(f"Chosen Reward: {metrics['chosen_reward']:.4f}")
    print(f"Rejected Reward: {metrics['rejected_reward']:.4f}")
    print(f"Reward Gap:    {metrics['reward_gap']:.4f}")

    # 逐样本分析
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

    这个函数展示了在 DPO training step 中，
    如何将原始 batch 数据转化为 DPO loss 所需的 logprob 值。

    batch 结构（假设已经 padding 和 concatenate）：
        {
            "input_ids_chosen":   [batch_size, max_seq_len_chosen],   # prompt + chosen
            "attention_mask_chosen": [batch_size, max_seq_len_chosen],
            "label_ids_chosen":   [batch_size, max_response_len_chosen],  # chosen response tokens
            "label_mask_chosen":  [batch_size, max_response_len_chosen],  # 1=有效, 0=padding
            "prompt_len_chosen":  [batch_size],  # 每个样本的 prompt 长度

            "input_ids_rejected": [batch_size, max_seq_len_rejected],
            "attention_mask_rejected": [batch_size, max_seq_len_rejected],
            "label_ids_rejected": [batch_size, max_response_len_rejected],
            "label_mask_rejected": [batch_size, max_response_len_rejected],
            "prompt_len_rejected": [batch_size],
        }

    注意：chosen 和 rejected 的 seq_len 通常不同（因为 answer 长度不同），
    所以不能 stack 在一起，需要分别 forward。
    """
    # ===== 计算 chosen logprob =====
    outputs_chosen = model(
        input_ids=batch["input_ids_chosen"],
        attention_mask=batch["attention_mask_chosen"],
    )
    logits_chosen = outputs_chosen.logits  # [B, S_c, V]

    log_probs_chosen = F.log_softmax(logits_chosen, dim=-1)  # [B, S_c, V]

    # Gather: 对每个样本，取 label_ids 对应位置的 logprob
    per_token_logp_chosen = gather_response_logprob(
        log_probs=log_probs_chosen,
        label_ids=batch["label_ids_chosen"],
        prompt_lens=batch["prompt_len_chosen"],
    )  # [B, R_c]

    # 应用 mask → mean
    valid_tokens_chosen = batch["label_mask_chosen"].sum(dim=-1).clamp(min=1)  # [B]
    chosen_logp = (per_token_logp_chosen * batch["label_mask_chosen"]).sum(dim=-1)
    chosen_logp = chosen_logp / valid_tokens_chosen  # [B] — mean logprob

    # ===== 计算 rejected logprob =====
    outputs_rejected = model(
        input_ids=batch["input_ids_rejected"],
        attention_mask=batch["attention_mask_rejected"],
    )
    logits_rejected = outputs_rejected.logits  # [B, S_r, V]

    log_probs_rejected = F.log_softmax(logits_rejected, dim=-1)  # [B, S_r, V]

    per_token_logp_rejected = gather_response_logprob(
        log_probs=log_probs_rejected,
        label_ids=batch["label_ids_rejected"],
        prompt_lens=batch["prompt_len_rejected"],
    )  # [B, R_r]

    valid_tokens_rejected = batch["label_mask_rejected"].sum(dim=-1).clamp(min=1)
    rejected_logp = (per_token_logp_rejected * batch["label_mask_rejected"]).sum(dim=-1)
    rejected_logp = rejected_logp / valid_tokens_rejected  # [B]

    return {
        "chosen_logp": chosen_logp,       # [batch_size]
        "rejected_logp": rejected_logp,   # [batch_size]
        "per_token_chosen": per_token_logp_chosen,   # [batch_size, R_c]
        "per_token_rejected": per_token_logp_rejected,  # [batch_size, R_r]
    }


def gather_response_logprob(
    log_probs: torch.Tensor,      # [batch_size, seq_len, vocab_size]
    label_ids: torch.Tensor,      # [batch_size, response_len]
    prompt_lens: torch.Tensor,    # [batch_size]
) -> torch.Tensor:
    """
    从完整的 log_probs 中 gather 出 response 部分每个目标 token 的 logprob。

    维度：
        log_probs:   [B, S, V]
        label_ids:   [B, R]
        prompt_lens: [B]

    返回：
        per_token_logp: [B, R]
    """
    batch_size, seq_len, vocab_size = log_probs.shape
    device = log_probs.device
    response_len = label_ids.size(1)

    # 构建索引：对每个样本，我们需要 log_probs 中
    # [prompt_len-1, prompt_len, ..., prompt_len+R-2] 这 R 个位置
    # 即每个 position 预测下一个 token

    # 创建一个 batch 索引张量: [0, 0, ..., 1, 1, ..., B-1, B-1, ...]
    batch_indices = torch.arange(batch_size, device=device) \
        .unsqueeze(1).expand(-1, response_len)  # [B, R]

    # 创建位置索引: prompt_len-1 到 prompt_len+R-2
    start_positions = prompt_lens - 1  # [B]
    # [B, R]: 每个样本的 response token 对应 log_probs 的哪些行
    seq_indices = start_positions.unsqueeze(1) + \
                  torch.arange(response_len, device=device).unsqueeze(0)  # [B, R]

    # Gather：从 log_probs[b, s, :] 中取 label_ids[b, r] 位置的值
    # log_probs[batch_indices, seq_indices, label_ids]
    per_token_logp = log_probs[batch_indices, seq_indices, label_ids]  # [B, R]

    return per_token_logp


# ============================================================
# 维度追踪
# ============================================================
if __name__ == "__main__":
    print("DPO Batch Logprob 计算维度追踪:")
    B, P, R, V = 2, 10, 5, 100  # batch=2, prompt=10, response=5, vocab=100

    log_probs = torch.randn(B, P + R, V)
    label_ids = torch.randint(0, V, (B, R))
    prompt_lens = torch.tensor([P, P])

    result = gather_response_logprob(log_probs, label_ids, prompt_lens)
    print(f"  输入 log_probs:  {log_probs.shape}")    # [2, 15, 100]
    print(f"  输入 label_ids:  {label_ids.shape}")     # [2, 5]
    print(f"  输出 per_token:  {result.shape}")        # [2, 5]
    print(f"  → 对 2 个样本，各 gather 出 5 个 response token 的 logprob")
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
    """
    计算 DPO 训练的监控指标。

    在 DPO training step 中，每 N 步打印这些指标。
    """
    metrics = {}

    # ===== 1. 隐式 Rewards =====
    chosen_rewards = beta * (policy_chosen_logp - ref_chosen_logp)      # [B]
    rejected_rewards = beta * (policy_rejected_logp - ref_rejected_logp) # [B]

    # ===== 2. Reward Margin =====
    margins = chosen_rewards - rejected_rewards  # [B]

    # ===== 3. Accuracy =====
    # 定义：chosen_reward > rejected_reward 的比例
    accuracy = (margins > 0).float().mean().item()
    metrics["accuracy"] = accuracy

    # ===== 4. Reward Statistics =====
    metrics["chosen_reward_mean"] = chosen_rewards.mean().item()
    metrics["chosen_reward_std"] = chosen_rewards.std().item()
    metrics["rejected_reward_mean"] = rejected_rewards.mean().item()
    metrics["rejected_reward_std"] = rejected_rewards.std().item()
    metrics["margin_mean"] = margins.mean().item()
    metrics["margin_std"] = margins.std().item()

    # ===== 5. Logprob Statistics（原始 logprob 变化） =====
    metrics["chosen_logp_diff_mean"] = (policy_chosen_logp - ref_chosen_logp).mean().item()
    metrics["rejected_logp_diff_mean"] = (policy_rejected_logp - ref_rejected_logp).mean().item()

    # ===== 6. 异常检测指标 =====
    # 6a. Reward 是否同时增大（异常信号）
    both_positive = ((chosen_rewards > 0) & (rejected_rewards > 0)).float().mean().item()
    metrics["both_rewards_positive_ratio"] = both_positive

    # 6b. 有多少样本 margin < 0（training failed for this sample）
    negative_margin_ratio = (margins < 0).float().mean().item()
    metrics["negative_margin_ratio"] = negative_margin_ratio

    # 6c. Reward explosion detection
    max_abs_reward = max(
        chosen_rewards.abs().max().item(),
        rejected_rewards.abs().max().item()
    )
    metrics["max_abs_reward"] = max_abs_reward
    metrics["reward_explosion_warning"] = max_abs_reward > 10.0

    return metrics


# ============================================================
# 使用示例
# ============================================================
if __name__ == "__main__":
    B = 8  # batch_size
    torch.manual_seed(42)

    # 模拟正常的训练数据
    policy_chosen_logp = torch.randn(B) * 0.5 - 1.5   # mean ~ -1.5
    policy_rejected_logp = torch.randn(B) * 0.5 - 2.5  # mean ~ -2.5
    ref_chosen_logp = torch.randn(B) * 0.3 - 2.0       # mean ~ -2.0
    ref_rejected_logp = torch.randn(B) * 0.3 - 2.2     # mean ~ -2.2

    metrics = compute_dpo_metrics(
        policy_chosen_logp, policy_rejected_logp,
        ref_chosen_logp, ref_rejected_logp,
        beta=0.1,
    )

    print("=== DPO Training Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, bool):
            print(f"  {key}: {'⚠ YES' if value else '✓ OK'}")
        else:
            print(f"  {key}: {value:.4f}")

    # 理想指标的判断标准：
    #   accuracy > 0.6（大部分样本 margin 为正）
    #   chosen_reward_mean > 0（policy 比 reference 更喜欢 chosen）
    #   rejected_reward_mean < 0（policy 比 reference 更不喜欢 rejected）
    #   both_rewards_positive_ratio < 0.3（没有同时增大）
    #   max_abs_reward < 5（没有 reward 爆炸）
    #   negative_margin_ratio < 0.4（大多数样本正确区分）
```

### 示例 5：简化 DPO Training Step

```python
import torch
from transformers import Trainer
from typing import Dict


class SimpleDPOTrainingStep:
    """
    简化的 DPO 训练单步流程。
    展示从 batch → logprob → loss → backward → log 的完整过程。
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        ref_model: torch.nn.Module,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
    ):
        self.policy_model = policy_model  # 可训练的 policy model
        self.ref_model = ref_model        # 冻结的 reference model
        self.beta = beta
        self.loss_type = loss_type

        # 确保 reference model 在 eval 模式且不计算梯度
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def training_step(self, batch: Dict[str, torch.Tensor]):
        """
        单步 DPO 训练。

        batch 包含：
            input_ids_chosen, attention_mask_chosen, label_ids_chosen,
            input_ids_rejected, attention_mask_rejected, label_ids_rejected,
            prompt_lens, label_masks 等
        """
        # ===== Step 1: 计算 reference model 的 logprob（no_grad） =====
        with torch.no_grad():
            ref_outputs = compute_dpo_batch_logprobs(
                self.ref_model, batch
            )
            ref_chosen_logp = ref_outputs["chosen_logp"]      # [B]
            ref_rejected_logp = ref_outputs["rejected_logp"]  # [B]

        # ===== Step 2: 计算 policy model 的 logprob =====
        policy_outputs = compute_dpo_batch_logprobs(
            self.policy_model, batch
        )
        policy_chosen_logp = policy_outputs["chosen_logp"]      # [B]
        policy_rejected_logp = policy_outputs["rejected_logp"]  # [B]

        # ===== Step 3: 计算 DPO loss =====
        loss, metrics = dpo_loss(
            policy_chosen_logp=policy_chosen_logp,
            policy_rejected_logp=policy_rejected_logp,
            ref_chosen_logp=ref_chosen_logp,
            ref_rejected_logp=ref_rejected_logp,
            beta=self.beta,
            loss_type=self.loss_type,
        )

        # ===== Step 4: 反向传播 =====
        loss.backward()

        # ===== Step 5: 返回 loss 和 metrics 给 Trainer =====
        return loss, metrics


# ============================================================
# 集成到 HuggingFace Trainer 的示例
# ============================================================
class DPOTrainer(Trainer):
    """
    将 DPO training step 集成到 HuggingFace Trainer 中的简化示例。

    注意：这不是一个可直接使用的 Trainer，仅展示核心思路。
    实际项目中推荐使用 trl 库的 DPOTrainer。
    """

    def __init__(self, ref_model, beta=0.1, loss_type="sigmoid", **kwargs):
        super().__init__(**kwargs)
        self.ref_model = ref_model
        self.beta = beta
        self.loss_type = loss_type
        # 冻结 reference model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Trainer 会在每个 training step 调用此方法。

        参数：
            model:    policy model（可训练）
            inputs:   来自 dataloader 的 batch dict
        """
        # Step 1: reference model logprob（no_grad）
        with torch.no_grad():
            ref_chosen_logp = compute_sequence_logprob(
                self.ref_model,
                inputs["input_ids_chosen"],
                inputs["attention_mask_chosen"],
                inputs["label_ids_chosen"],
                inputs["label_mask_chosen"],
                inputs["prompt_lens"],
            )[0]  # 只取聚合值
            ref_rejected_logp = compute_sequence_logprob(
                self.ref_model,
                inputs["input_ids_rejected"],
                inputs["attention_mask_rejected"],
                inputs["label_ids_rejected"],
                inputs["label_mask_rejected"],
                inputs["prompt_lens"],
            )[0]

        # Step 2: policy model logprob
        policy_chosen_logp = compute_sequence_logprob(
            model,
            inputs["input_ids_chosen"],
            inputs["attention_mask_chosen"],
            inputs["label_ids_chosen"],
            inputs["label_mask_chosen"],
            inputs["prompt_lens"],
        )[0]
        policy_rejected_logp = compute_sequence_logprob(
            model,
            inputs["input_ids_rejected"],
            inputs["attention_mask_rejected"],
            inputs["label_ids_rejected"],
            inputs["label_mask_rejected"],
            inputs["prompt_lens"],
        )[0]

        # Step 3: DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logp=policy_chosen_logp,
            policy_rejected_logp=policy_rejected_logp,
            ref_chosen_logp=ref_chosen_logp,
            ref_rejected_logp=ref_rejected_logp,
            beta=self.beta,
            loss_type=self.loss_type,
        )

        # 将 metrics 记录到 log（Trainer 会自动处理）
        # self.log(metrics)

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
│   chosen: MiMo-v2.5-pro 生成                                           │
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
