# 09 Reward Model / PRM / ORM / Verifier

> **项目背景声明**：本项目（医学 LLM Teacher）中**没有训练 Reward Model（RM）**。项目使用 LLM-as-Judge（MiMo-v2.5-pro）进行质量评估和过滤，而非训练独立的 RM。但面试中需要理解 RM 的原理、PRM/ORM 的区别以及 verifier 的概念。以下内容基于论文阅读和面试准备整理。

---

### Q: Reward Model 是什么？Pairwise vs Pointwise RM 有什么区别？⭐⭐⭐⭐⭐

Reward Model（RM，奖励模型）是 RLHF 管线中的核心组件。它是一个独立的模型，输入为 (prompt, response) 对，输出一个标量分数，代表该 response 有多好。训练方式是在人工标注的偏好数据上训练，人类标注者对同一 prompt 的多个 response 进行排序，RM 学习预测这种排序。

**为什么需要 RM**：强化学习需要 reward 信号来指导 policy 更新。在没有明确 reward 函数的环境中（如对话质量、文本安全性），RM 充当 reward 函数的代理。

**Pairwise RM（InstructGPT 经典做法）**：
- 输入：一个 prompt + 一个 chosen response + 一个 rejected response
- 目标：chosen 的 reward 高于 rejected
- Loss：pairwise ranking loss（Bradley-Terry 模型）

**Pointwise RM**：
- 输入：一个 prompt + 一个 response
- 目标：直接预测标量分数（如 1-5 分）
- Loss：MSE 回归或交叉熵分类

**区别**：

| 维度 | Pairwise RM | Pointwise RM |
|------|-------------|--------------|
| 标注成本 | 需要成对比较 | 需要绝对评分 |
| 训练难度 | 较容易（相对信号） | 较难（需要校准） |
| 泛化能力 | 好（相对偏好更稳定） | 差（绝对分数难一致） |
| 应用 | RLHF 常用 | 评估/过滤场景 |

实际中多数 RLHF 系统使用 pairwise RM，因为人类更擅长做相对比较而不是打绝对分。

---

### Q: Pairwise RM Loss 怎么算？Reward Model 的架构是怎样的？⭐⭐⭐⭐

**Pairwise RM Loss（Bradley-Terry 模型）**：

```python
import torch
import torch.nn.functional as F

def pairwise_rm_loss(reward_model, prompts, chosen_responses, rejected_responses):
    """
    Pairwise Reward Model Loss (Bradley-Terry Model)
    """
    r_chosen = reward_model(prompts, chosen_responses)  # shape: (batch,)
    r_rejected = reward_model(prompts, rejected_responses)  # shape: (batch,)
    
    # Bradley-Terry 模型：chosen 比 rejected 好的概率
    # P(chosen > rejected) = σ(r_chosen - r_rejected)
    logits = r_chosen - r_rejected  # shape: (batch,)
    
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**Reward Margin 计算**：

```python
def compute_reward_margin(reward_model, prompts, chosen_responses, rejected_responses):
    with torch.no_grad():
        r_chosen = reward_model(prompts, chosen_responses)
        r_rejected = reward_model(prompts, rejected_responses)
        margin = r_chosen - r_rejected
    return {
        'margin_mean': margin.mean().item(),
        'margin_std': margin.std().item(),
        'accuracy': (margin > 0).float().mean().item(),
    }
```

**典型 RM 架构**：基于预训练 LLM，在 transformer 最后一层之上加一个线性 head 输出标量。取最后一个 token 的 hidden state，通过 reward_head（nn.Linear(hidden_size, 1)）输出 reward。

```python
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=4096):
        super().__init__()
        self.transformer = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        hidden_states = self.transformer(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        ).last_hidden_state  # (batch, seq_len, hidden_size)
        last_token_hidden = hidden_states[:, -1, :]  # (batch, hidden_size)
        reward = self.reward_head(last_token_hidden)  # (batch, 1)
        return reward.squeeze(-1)  # (batch,)
```

---

### Q: RM 与 LLM-as-Judge 有什么区别？⭐⭐⭐⭐⭐

这是一个面试高频对比题。

| 维度 | Reward Model (RM) | LLM-as-Judge |
|------|-------------------|--------------|
| 本质 | 专门训练的标量打分模型 | 通用 LLM 做评分/比较 |
| 输出 | 标量 reward 分数 | 分数/排名/评语 |
| 训练方式 | 在偏好数据上 fine-tune | 无需训练（prompt engineering） |
| 是否需要训练 | 是（需要标注 + 训练） | 否（开箱即用） |
| 推理速度 | 快（一次 forward） | 较慢（需要生成） |
| 可微分 | 是（可作为 RL 的 reward） | 否（文本输出，不可微分） |
| 成本 | 高（训练+维护） | 低（API 调用或本地推理） |
| 一致性 | 较好（固定参数） | 取决于 prompt 和模型版本 |
| 可解释性 | 低（一个数字） | 高（可以给出理由） |
| RLHF 集成 | 天生适配 PPO/GRPO | 需要额外转换 |

**核心区分**：RM 是**可微分的打分器**，能直接输出标量参与梯度回传，是 PPO/GRPO 的必需品；LLM-as-Judge 是**不可微分的评估器**，输出文本而非标量，不能直接用于 RL 训练。

---

### Q: ORM / PRM / Verifier / Critic / Value Model 的定义和区别是什么？⭐⭐⭐⭐⭐

这是面试中最容易被问的容易混淆概念。

**ORM (Outcome Reward Model)**：对完整 response 给出一个总体的 reward 分数，只看结局不看过程。输入完整 prompt + 完整 response，输出一个标量。优点：简单、标注成本低、训练容易。缺点：无法区分推理过程中的好坏步骤，稀疏 reward。InstructGPT 的 RM 就是 ORM。

**PRM (Process Reward Model)**：对推理过程的每一步给出 reward，每个中间步骤都有分数。输入 prompt + 部分推理步骤，输出该步骤的好坏分数。优点：稠密 reward 信号、能定位推理错误、助力 search/planning。缺点：标注成本极高（需要 per-step 标注）、训练困难。标注方式：人工标注每一步是否正确，或用最终答案正确性做自动标注（outcome-to-process mapping，Math-Shepherd 的做法）。典型使用：OpenAI 的 Let's Verify Step by Step (PRM800K)、Math-Shepherd。

**Verifier（验证器）**：判断一个 solution 是否正确/可接受，比 RM 更聚焦于"对错"而非"好坏"。二分类或多分类，而非连续打分。常用于 best-of-n sampling、rejection sampling。可以是神经网络（learned verifier）或规则（rule-based verifier）。所有 Verifier 都可以视为特殊形式的 RM（二值 reward），但不是所有 RM 都是 Verifier（RM 可以输出连续分数）。

**Critic / Value Model**：在 RL 框架中估计状态价值函数 V(s) 的模型。输入当前状态（prompt + 已生成的 token），输出对未来累积 reward 的估计。属于 RL 组件，不是独立的可评估工具。通常和 policy 共享 backbone 或独立训练。在 PPO 中用于计算 advantage，在 GRPO 中被 group-based normalization 替代。

**概念关系总结**：

```
Reward Model (打分模型)
├── ORM (Outcome RM): 对整个 response 打分
├── PRM (Process RM): 对每个步骤打分
└── Verifier (验证器): 对错判断（是 ORM/PRM 的特例）

RL 组件
├── Critic / Value Model: 估计 V(s)，用于 RL 训练
└── Reward Model: 提供 reward 信号

LLM-as-Judge
└── 不可微分的评估器，不是 RL 组件
```

---

### Q: PRM 为什么适合 Reasoning？ORM 有什么局限？⭐⭐⭐

**ORM 的局限**：Reward 稀疏（一整段推理只有一个 reward 信号）；Search 效率低（best-of-n 中只能对完整解答排序，不能中途剪枝）；Credit Assignment 困难（不能区分"思路对计算错"和"思路全错"）。

**PRM 的优势**：过程监督强（每一步都有信号，训练更高效）；支持 step-level search（可以在推理中途对不良步骤剪枝，类似 AlphaGo 的 MCTS）；可解释性好（能定位推理在哪一步出了问题）；能产生 Emergent 行为（PRM 训练模型更倾向于自我验证，如 R1 的"wait, let me check"行为）。

**PRM 标注成本**：标注 PRM 数据的难度远超 ORM。数学推理需要标注者理解每一步推导是否正确，OpenAI 的 PRM800K 雇佣了数学背景的标注者。医学推理需要医生级别的专业知识，成本更高。替代方案：利用最终答案正确性做自动标注（Math-Shepherd 的"如果最终答案正确，所有中间步骤标记为正确"的近似标注）。

---

### Q: Best-of-N / Rejection Sampling / Verifier-Guided Decoding 怎么实现？⭐⭐⭐

**Best-of-N Sampling**：对每个 prompt 采样 N 个 response，用 RM 打分，选分数最高的那个返回。典型 N 值：4（轻量）、16（常用）、64（高质量但推理成本高）。

```python
def best_of_n_sampling(policy_model, reward_model, tokenizer, 
                        prompts, n=16, temperature=0.7, max_new_tokens=512):
    results = []
    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
        best_response = None
        best_reward = float('-inf')
        for _ in range(n):
            response_ids = policy_model.generate(
                prompt_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True
            )
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            reward = reward_model(prompt, response_text)
            if reward > best_reward:
                best_reward = reward
                best_response = response_text
        results.append({'prompt': prompt, 'best_response': best_response, 'reward': best_reward})
    return results
```

**Rejection Sampling**：对每个 prompt 采样 N 个 response，只保留 verifier 分数超过阈值的 response。可用于构造高质量 SFT 数据或构造 DPO 的 chosen。

**Verifier-Guided Decoding**：每一步生成时，用 verifier 对候选路径打分和剪枝。可结合 beam search 实现，每步生成 top-k 候选 token，verifier 评估当前路径，保留 top-k 继续扩展。

---

### Q: RM 视角下的 Reward Hacking 有哪些常见方式？怎么缓解？⭐⭐⭐

RM 被 exploit 的常见方式：
1. **长度偏见**：RM 无意中学到了"长 response = 好"，模型就输出超长内容
2. **格式偏见**：RM 喜欢列表格式、markdown 格式，模型就堆砌格式
3. **关键词偏见**：RM 喜欢"当然"、"让我帮你"之类的礼貌用语，模型就大量使用
4. **分布外不可靠**：RM 只在训练分布的 response 上准确，模型 generate 出分布外内容时 RM 打分失准

缓解方法：
1. **KL penalty**：限制 policy 不偏离 reference 太远
2. **RM ensemble**：用多个 RM 的均值/最小值作为 reward
3. **对抗训练**：在训练中定期加入 policy 生成的新 response 重训 RM
4. **长度归一化**：reward 除以 response 长度
5. **Pretraining 混合**：PPO 目标中混入预训练 loss，防止语言能力退化

---

### Q: 医疗场景 Reward Model 有什么特殊风险和挑战？⭐⭐⭐

**医学 RM 的特殊挑战**：
1. **标注质量无法保证**：即使是医生，对同一病例的判断也不总是一致，标注者间一致性较低
2. **安全敏感**：RM 的任何缺陷都会直接放大到最终模型的输出中
3. **领域知识深度**：医学 RM 需要理解复杂的病理、药理、诊断逻辑，通用 RM 在医学上不可靠
4. **安全 vs 帮助的权衡**：如何让 RM 同时考虑安全性（不导致伤害）和帮助性（提供有用信息）？

**医疗安全 Reward 应该包含的维度**：
1. 医学正确性 (Medical Correctness)：诊断、治疗建议、药物信息是否准确
2. 安全性 (Safety)：是否包含危险建议、是否识别紧急情况、是否提示就医
3. 完整性 (Completeness)：是否覆盖了重要的鉴别诊断、必要的检查项目
4. 幻觉检测 (Hallucination)：是否编造医学事实、虚构文献、错误引用
5. 合规性 (Compliance)：是否符合医疗法规、隐私保护
6. 适当性 (Appropriateness)：回答是否适合患者的知识水平和需求
7. 免责声明 (Disclaimer)：是否包含必要的 medical disclaimer

---

### Q: 如何把 LLM-as-Judge 数据蒸馏成 RM？⭐⭐⭐

**蒸馏流程**：
1. 用 LLM-as-Judge 对一批 (prompt, response) 打分
2. 从同一 prompt 的多个 response 中选得分最高和最低的配对，构造 pairwise preference 数据
3. 用 pairwise RM loss 训练一个小 RM

**优势**：LLM-as-Judge 成本高，RM 推理快 10-100 倍；RM 可微分，能用于 PPO/GRPO。

**为什么本项目用 LLM-as-Judge 而不是训练 RM**：
1. 标注成本：训练 RM 需要大量 pairwise 偏好标注，本项目用 LLM-as-Judge 自动生成偏好信号节省人工成本
2. 医学专业知识：训练一个医学上可靠的 RM 需要大量医学标注，现有开源医学偏好数据不足
3. 工程优先级：项目当前重点在 Teacher 数据生成和 DPO 训练，训练 RM 的收益在现阶段有限
4. 可用性：LLM-as-Judge（MiMo-v2.5-pro）本身已经是强模型，做 quality filtering 和 data selection 足够好

---

## Reward Hacking 与 Reward Design

### Q: Reward hacking 是什么？模型可能通过哪些方式钻 reward 漏洞？star:4

Reward hacking 指模型通过"优化 reward 但非真实质量"的方式获得高分，即找到了 reward function 的漏洞。

**常见 hacking 模式**：
1. **长度贿赂**：生成冗长回答骗过偏好长度偏置的 reward model
2. **模板复读**：所有回答都用固定安全模板（"建议咨询医生..."），安全分高但无用
3. **过度拒答**：对任何涉及医疗的问题都说"不确定，请就医"
4. **伪推理**：生成看起来很长的链式推理但逻辑不通
5. **风格模仿**：学会 reward model 喜欢的词藻（"首先...其次...综上所述..."），内容没变好

**发现方法**：人工抽检 + reward 分布异常（分数过高且方差为零）+ 回答长度/模板率统计 + bad case 分析。

### Q: 如何设计多维 reward？安全性和有用性冲突怎么办？star:4

**reward 维度设计**（以医学为例）：

| 维度 | 权重 | 信号来源 | 备注 |
|------|------|---------|------|
| correctness | 0.30 | Judge/规则验证 | 事实准确 |
| safety | 0.25 | 安全分类器 | 无危险建议 |
| completeness | 0.15 | Judge | 信息完整 |
| helpfulness | 0.10 | Judge | 实用可操作 |
| caution | 0.10 | 规则 | 谨慎边界 |
| hallucination | 0.10 | Judge/引用校验 | 无幻觉 |

**安全 vs 有用冲突时的策略**：安全优先。设计"安全一票否决"——如果安全分低于阈值（如 <3/5），总 reward 直接置零或大幅降低，不管其他维度多高。

**但注意**：不能把"任何涉及药物的问题就打低分"——这会导致模型连"感冒药不能和酒一起喝"这种安全提醒都不说。需要精细标注哪些是"危险建议"，哪些是"安全用药科普"。

### Q: 负向 reward / penalty 应该如何设计？star:3

需要惩罚的行为：
- **幻觉惩罚** (-0.5)：编造疾病机制、药物剂量、检查结果
- **危险建议惩罚** (-1.0，一票否决级)：建议自行用药、替代正规治疗
- **格式错误惩罚** (-0.1)：输出不符合指定格式

惩罚权重不宜过大，否则模型会过度保守。惩罚应有"上限"——不能无限制扣分，防止训练崩溃。

---

## 背诵版总结

```
【RM 是什么】独立的打分模型，输入(prompt, response)，输出标量 reward
【Pairwise RM Loss】L = -log(σ(r_chosen - r_rejected)), Bradley-Terry 模型
【RM vs LLM-as-Judge】RM 可微分能用于 RL，Judge 不可微分只能评估
本项目的核心权衡

【ORM】Outcome RM, 对完整 response 打分，一个标量，简单但稀疏
【PRM】Process RM, 对每一步打分，稠密信号，标注成本极高
【Verifier】判断对错的特殊 RM，二分类而非连续分数
【Critic/Value】RL 组件，估计 V(s)，PPO 需要，GRPO 不需要

【Best-of-N】采样 N 个，RM 选最好的一个
【Rejection Sampling】采样 N 个，只保留超过阈值的
【Verifier-Guided Decoding】每步用 verifier 剪枝

【RM 蒸馏】用 LLM-as-Judge 标注 → 构造 pairwise 数据 → 训小 RM
    好处：RM 推理快、可微分、能用于 RL

【医疗 RM 风险】标注不一致、安全敏感、domain knowledge 需求高
【医疗安全 Reward 维度】正确性、安全性、完整性、幻觉检测、合规性、适当性、免责声明

【本项目为什么没有 RM】不是不能做，是资源和优先级选择：
    - LLM-as-Judge 自动过滤已经够用
    - 训练医学 RM 成本高、风险大
    - DPO 不需要 RM
```

---

## 面试可能的追问及回答要点

**Q: PRM 和 ORM 的 reward 信号有什么本质区别？**
ORM 是 sparse reward——整段推理只有一个分数，模型需要自己弄清楚哪个 token 对最终结果有贡献。PRM 是 dense reward——每一步都有 feedback，credit assignment 更精确。在数学题上，PRM 训练出的模型倾向于自我验证和纠错。

**Q: 如果让你设计一个医学 PRM，你会怎么做？**
先做 outcome-to-process 的自动标注：用选择题 ground truth 判断最终答案对不对，如果对了就假设推理步骤都对（Math-Shepherd 思路）。但医学推理的步骤"部分正确"很常见，纯自动标注有噪声。更好的方式是用医生标注少量数据 + LLM-as-Judge 自动扩展。

**Q: 你们的 Judge 和 RM 有什么区别？**
面试中要明确说——我们用的是 LLM-as-Judge（MiMo-v2.5-pro），是一个不可微分的基于 prompt 的评估器，输出文本评分和理由。它不是 RM（可微分的神经网络打分器），不能用于 RL 训练。但它的评分可以用来过滤 SFT 数据、构造 DPO preference、评估模型质量。如果未来要做 PPO/GRPO，可以考虑从 Judge 的标注中蒸馏出一个 RM。

**Q: Best-of-N 在推理时有什么工程上的优化？**
vLLM 支持 best-of-n 的并行采样（use_beam_search + best_of 参数）。也可以做 speculative best-of-n：先用小 RM 快速过滤，top-k 再用大 RM 精排。连续批处理可以大幅降低 latency。
