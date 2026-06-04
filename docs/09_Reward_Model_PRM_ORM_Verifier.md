# 09 Reward Model / PRM / ORM / Verifier

> **项目背景声明**：本项目（医学 LLM Teacher）中**没有训练 Reward Model（RM）**。项目使用 LLM-as-Judge（MiMo-v2.5-pro）进行质量评估和过滤，而非训练独立的 RM。但面试中需要理解 RM 的原理、PRM/ORM 的区别以及 verifier 的概念。以下内容基于论文阅读和面试准备整理。

---

## 一、Reward Model 核心概念

### 1.1 Reward Model 是什么

Reward Model（RM，奖励模型）是 RLHF 管线中的核心组件。它是一个独立的模型，输入为 (prompt, response) 对，输出一个标量分数，代表该 response 有多好。

**训练方式**：在人工标注的偏好数据上训练。人类标注者对同一 prompt 的多个 response 进行排序，RM 学习预测这种排序。

**为什么需要 RM**：强化学习需要 reward 信号来指导 policy 更新。在没有明确 reward 函数的环境中（如对话质量、文本安全性），RM 充当 reward 函数的代理。

### 1.2 Pairwise vs Pointwise RM

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

## 二、Reward Model 的核心计算

### 2.1 Pairwise RM Loss（Bradley-Terry 模型）

RLHF 中的经典 RM 训练 loss：

```python
import torch
import torch.nn.functional as F

def pairwise_rm_loss(reward_model, prompts, chosen_responses, rejected_responses):
    """
    Pairwise Reward Model Loss (Bradley-Terry Model)
    
    Args:
        reward_model: 输入 (prompt, response) → 输出标量 reward
        prompts: List[str], batch of prompts
        chosen_responses: List[str], 人类偏好的 response
        rejected_responses: List[str], 人类不偏好的 response
    
    Returns:
        loss: scalar tensor
    """
    # 分别计算 chosen 和 rejected 的 reward
    r_chosen = reward_model(prompts, chosen_responses)  # shape: (batch,)
    r_rejected = reward_model(prompts, rejected_responses)  # shape: (batch,)
    
    # Bradley-Terry 模型：chosen 比 rejected 好的概率
    # P(chosen > rejected) = σ(r_chosen - r_rejected)
    logits = r_chosen - r_rejected  # shape: (batch,)
    labels = torch.ones_like(logits)  # label=1 表示 chosen 更好
    # ← 注意：这里要求 chosen 的 reward > rejected
    
    loss = -F.logsigmoid(logits).mean()
    # 等价于: -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    
    return loss

# 或者使用 BCE loss 的形式
def pairwise_rm_loss_bce(reward_model, prompts, chosen_responses, rejected_responses):
    r_chosen = reward_model(prompts, chosen_responses)
    r_rejected = reward_model(prompts, rejected_responses)
    
    logits = r_chosen - r_rejected
    # BCE: 预测 chosen > rejected (label=1)
    loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    
    return loss
```

### 2.2 Reward Margin 计算

```python
def compute_reward_margin(reward_model, prompts, chosen_responses, rejected_responses):
    """
    计算 reward margin — chosen 和 rejected 之间的 reward 差距。
    margin 越大表示偏好区分越清晰。
    """
    with torch.no_grad():
        r_chosen = reward_model(prompts, chosen_responses)
        r_rejected = reward_model(prompts, rejected_responses)
        
        margin = r_chosen - r_rejected  # positive means correct
    
    return {
        'margin_mean': margin.mean().item(),
        'margin_std': margin.std().item(),
        'accuracy': (margin > 0).float().mean().item(),  # 准确率
        'margin_per_sample': margin.tolist()
    }

# 示例使用
# margins = compute_reward_margin(rm, test_prompts, test_chosen, test_rejected)
# print(f"Mean margin: {margins['margin_mean']:.3f}")
# print(f"Accuracy (margin > 0): {margins['accuracy']:.2%}")
```

### 2.3 Reward Model 架构

典型的 RM 架构（以 LLaMA 为例）：

```python
class RewardModel(nn.Module):
    """
    基于预训练 LLM 的 Reward Model
    在 transformer 最后一层之上加一个线性 head 输出标量
    """
    def __init__(self, base_model, hidden_size=4096):
        super().__init__()
        self.transformer = base_model  # 预训练 transformer
        self.reward_head = nn.Linear(hidden_size, 1)  # 标量输出
    
    def forward(self, input_ids, attention_mask):
        # 取最后一个 token 的 hidden state
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

## 三、RM 与 LLM-as-Judge 的区别

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

**核心区分**：
- RM 是**可微分的打分器**，能直接输出标量参与梯度回传，是 PPO/GRPO 的必需品
- LLM-as-Judge 是**不可微分的评估器**，输出文本而非标量，不能直接用于 RL 训练

---

## 四、ORM / PRM / Verifier / Critic / Value Model 的定义和区别

这是面试中最容易被问的容易混淆概念。

### 4.1 ORM (Outcome Reward Model)

**定义**：对完整 response 给出一个总体的 reward 分数。只看结局，不看过程。

**特点**：
- 输入：完整 prompt + 完整 response
- 输出：一个标量（如 0-1 之间）
- 优点：简单、标注成本低、训练容易
- 缺点：无法区分推理过程中的好坏步骤，稀疏 reward

**典型使用**：InstructGPT 的 RM 就是 ORM。

### 4.2 PRM (Process Reward Model)

**定义**：对推理过程的每一步给出 reward。每个中间步骤都有分数。

**特点**：
- 输入：prompt + 部分推理步骤
- 输出：该步骤的好坏分数
- 优点：稠密 reward 信号、能定位推理错误、助力 search/planning
- 缺点：标注成本极高（需要 per-step 标注）、训练困难

**标注方式**：
- 人工标注每一步是否正确
- 自动标注：如果最终答案正确，假设所有步骤都正确（有噪声）
- 用更强的模型自动判断每一步

**典型使用**：OpenAI 的 Let's Verify Step by Step (PRM800K)、Math-Shepherd

### 4.3 Verifier（验证器）

**定义**：判断一个 solution 是否正确/可接受。比 RM 更聚焦于"对错"而非"好坏"。

**特点**：
- 二分类或多分类，而非连续打分
- 常用于 best-of-n sampling、rejection sampling
- 可以是神经网络（learned verifier），也可以是规则（rule-based verifier）

**与 RM 的关系**：
- 所有 Verifier 都可以视为特殊形式的 RM（二值 reward）
- 但不是所有 RM 都是 Verifier（RM 可以输出连续分数）

### 4.4 Critic / Value Model

**定义**：在 RL 框架中估计状态价值函数 V(s) 的模型。输入当前状态（prompt + 已生成的 token），输出对未来累积 reward 的估计。

**特点**：
- 属于 RL 组件，不是独立的可评估工具
- 通常和 policy 共享 backbone 或独立训练
- 在 PPO 中用于计算 advantage
- 在 GRPO 中被 group-based normalization 替代

### 4.5 概念关系总结

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

## 五、PRM 为什么适合 Reasoning / ORM 的局限

### 5.1 ORM 的局限

1. **Reward 稀疏**：一整段推理只有一个 reward 信号，模型难以知道哪步错了
2. **Search 效率低**：在 best-of-n 中，ORM 只能对完整解答排序，不能中途剪枝
3. **Credit Assignment 困难**：不能区分"思路对计算错"和"思路全错"

### 5.2 PRM 的优势

1. **过程监督强**：每一步都有信号，训练更高效
2. **支持 step-level search**：可以在推理中途对不良步骤剪枝，类似 AlphaGo 的 MCTS
3. **可解释性**：能定位推理在哪一步出了问题
4. **Emergent 行为**：PRM 训练模型更倾向于自我验证（R1 的"wait, let me check"行为）

### 5.3 PRM 标注成本

标注 PRM 数据的难度远超 ORM：

- **数学推理**：需要标注者理解每一步推导是否正确。OpenAI 的 PRM800K 雇佣了数学背景的标注者。
- **医学推理**：需要医生级别的专业知识，成本更高。
- **替代方案**：利用最终答案正确性做自动标注（outcome-to-process mapping）。Math-Shepherd 用"如果最终答案正确，所有中间步骤标记为正确"的近似标注。

---

## 六、Best-of-N / Rejection Sampling / Verifier-Guided Decoding

### 6.1 Best-of-N Sampling（带代码）

```python
import torch

def best_of_n_sampling(policy_model, reward_model, tokenizer, 
                        prompts, n=16, temperature=0.7, max_new_tokens=512):
    """
    Best-of-N 采样：
    对每个 prompt 采样 N 个 response，
    用 RM 打分，选分数最高的那个返回。
    """
    results = []
    
    for prompt in prompts:
        # 编码 prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors='pt')
        
        best_response = None
        best_reward = float('-inf')
        
        # 采样 N 个 response
        for _ in range(n):
            response_ids = policy_model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            
            # RM 打分
            reward = reward_model(prompt, response_text)
            
            if reward > best_reward:
                best_reward = reward
                best_response = response_text
        
        results.append({
            'prompt': prompt,
            'best_response': best_response,
            'reward': best_reward
        })
    
    return results

# 典型 N 值选择
# n=4:  轻量，开销小
# n=16: 常用，开销可接受
# n=64: 高质量，推理成本高
```

### 6.2 Rejection Sampling

```python
def rejection_sampling(policy_model, verifier, tokenizer,
                       prompts, n=16, threshold=0.8):
    """
    Rejection Sampling：
    对每个 prompt 采样 N 个 response，
    只保留 verifier 分数超过阈值的 response。
    
    可以用于：
    1. 构造高质量 SFT 数据
    2. 构造 DPO 的 chosen
    """
    accepted = []
    rejected = []
    
    for prompt in prompts:
        for _ in range(n):
            response = policy_model.generate(prompt)
            score = verifier(prompt, response)
            
            if score >= threshold:
                accepted.append({'prompt': prompt, 'response': response, 'score': score})
                break  # 找到一个就停（或继续全部采样）
        else:
            # N 个都没过阈值
            # 保留分数最高的那个
            pass
    
    return accepted, rejected
```

### 6.3 Verifier-Guided Decoding

```python
def verifier_guided_beam_search(policy_model, verifier, tokenizer, 
                                 prompt, beam_size=4, max_steps=10):
    """
    Verifier-Guided Decoding（概念性伪代码）：
    每一步生成时，用 verifier 对候选路径打分和剪枝。
    
    注意：这是一个概念示意，实际实现需要深入到 generate 内部。
    """
    beams = [(prompt_token_ids, score=0.0)]
    
    for step in range(max_steps):
        candidates = []
        
        for beam_ids, beam_score in beams:
            # 生成下一步候选 token
            logits = policy_model(beam_ids).logits[:, -1, :]
            top_k_ids = logits.topk(beam_size).indices[0]
            
            for token_id in top_k_ids:
                candidate_ids = beam_ids + [token_id]
                candidate_text = tokenizer.decode(candidate_ids)
                
                # Verifier 评估当前路径
                step_score = verifier.score_step(prompt, candidate_text)
                new_score = beam_score + step_score
                candidates.append((candidate_ids, new_score))
        
        # 保留 top-k
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_size]
    
    return tokenizer.decode(beams[0][0])
```

---

## 七、Reward Hacking 和 Reward Overoptimization

（此部分在 08 号文件中已有详细说明，此处补充 RM 特定的视角。）

### 7.1 RM 视角下的 Reward Hacking

RM 被 exploit 的常见方式：
1. **长度偏见**：RM 无意中学到了"长 response = 好"，模型就输出超长内容
2. **格式偏见**：RM 喜欢列表格式、markdown 格式，模型就堆砌格式
3. **关键词偏见**：RM 喜欢"当然"、"让我帮你"之类的礼貌用语，模型就大量使用
4. **分布外不可靠**：RM 只在训练分布的 response 上准确，模型 generate 出分布外内容时 RM 打分失准

### 7.2 缓解方法

1. **KL penalty**：限制 policy 不偏离 reference 太远
2. **RM ensemble**：用多个 RM 的均值/最小值作为 reward
3. **对抗训练**：在训练中定期加入 policy 生成的新 response 重训 RM
4. **长度归一化**：reward 除以 response 长度
5. **Pretraining 混合**：PPO 目标中混入预训练 loss，防止语言能力退化

---

## 八、医疗场景 Reward Model 的风险

### 8.1 医学 RM 的特殊挑战

1. **标注质量无法保证**：即使是医生，对同一病例的判断也不总是一致。标注者间一致性（inter-annotator agreement）较低。
2. **安全敏感**：RM 的任何缺陷都会直接放大到最终模型的输出中。一个偏好的 RM 可能导致模型偏好危险的医学建议。
3. **领域知识深度**：医学 RM 需要理解复杂的病理、药理、诊断逻辑，通用 RM 在医学上不可靠。
4. **安全 vs 帮助 的权衡**：如何让 RM 同时考虑安全性（不导致伤害）和帮助性（提供有用信息）？

### 8.2 医疗安全 Reward 应该包含的维度

如果要训练医学 RM，reward 应该分解为多个子维度：

1. **医学正确性 (Medical Correctness)**：诊断、治疗建议、药物信息是否准确
2. **安全性 (Safety)**：是否包含危险建议、是否识别紧急情况、是否提示就医
3. **完整性 (Completeness)**：是否覆盖了重要的鉴别诊断、必要的检查项目
4. **幻觉检测 (Hallucination)**：是否编造医学事实、虚构文献、错误引用
5. **合规性 (Compliance)**：是否符合医疗法规、隐私保护
6. **适当性 (Appropriateness)**：回答是否适合患者的知识水平和需求
7. **免责声明 (Disclaimer)**：是否包含必要的 medical disclaimer

---

## 九、如何把 LLM-as-Judge 数据蒸馏成 RM

### 9.1 蒸馏流程

```python
def distill_rm_from_judge(llm_judge, prompts, responses, base_model):
    """
    从 LLM-as-Judge 蒸馏出 Reward Model。
    
    流程：
    1. 用 LLM-as-Judge 对一批 (prompt, response) 打分
    2. 构造 pairwise preference 数据
    3. 用 pairwise RM loss 训练一个小 RM
    
    优势：
    - LLM-as-Judge 成本高，RM 推理快 10-100 倍
    - RM 可微分，能用于 PPO/GRPO
    """
    # Step 1: LLM-as-Judge 批量打分
    scores = []
    for prompt, response in zip(prompts, responses):
        score = llm_judge.score(prompt, response)  # 返回 1-5 分
        scores.append(score)
    
    # Step 2: 构造 pairwise 数据
    # 从同一个 prompt 的多个 response 中选得分最高和最低的配对
    pairs = []
    for i, prompt in enumerate(prompts):
        # 对该 prompt 的不同 response 按分数排序
        prompt_indices = [j for j, p in enumerate(prompts) if p == prompt]
        prompt_scores = [(j, scores[j]) for j in prompt_indices]
        prompt_scores.sort(key=lambda x: x[1], reverse=True)
        
        if len(prompt_scores) >= 2:
            chosen_idx = prompt_scores[0][0]
            rejected_idx = prompt_scores[-1][0]
            pairs.append({
                'prompt': prompt,
                'chosen': responses[chosen_idx],
                'rejected': responses[rejected_idx]
            })
    
    # Step 3: 训练 RM
    rm = RewardModel(base_model)  # 基于小模型初始化
    optimizer = torch.optim.Adam(rm.parameters(), lr=1e-5)
    
    for epoch in range(3):
        for batch in dataloader(pairs, batch_size=32):
            loss = pairwise_rm_loss(
                rm, 
                batch['prompts'], 
                batch['chosen'], 
                batch['rejected']
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return rm
```

### 9.2 为什么本项目用 LLM-as-Judge 而不是训练 RM

1. **标注成本**：训练 RM 需要大量 pairwise 偏好标注。本项目用 LLM-as-Judge 自动生成偏好信号，节省人工成本。
2. **医学专业知识**：训练一个在医学上可靠的 RM 需要大量医学标注，现有开源的医学偏好数据不足以训练高质量 RM。
3. **工程优先级**：项目当前重点在 Teacher 数据生成和 DPO 训练，训练 RM 的收益在现阶段有限。
4. **可用性**：LLM-as-Judge（MiMo-v2.5-pro）本身已经是强模型，做 quality filtering 和 data selection 足够好。

---

## 十、背诵版总结

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

## 十一、面试可能的追问及回答要点

**Q: PRM 和 ORM 的 reward 信号有什么本质区别？**
A: ORM 是 sparse reward——整段推理只有一个分数，模型需要自己弄清楚哪个 token 对最终结果有贡献。PRM 是 dense reward——每一步都有 feedback，credit assignment 更精确。在数学题上，PRM 训练出的模型倾向于自我验证和纠错。

**Q: 如果让你设计一个医学 PRM，你会怎么做？**
A: 先做 outcome-to-process 的自动标注：用选择题 ground truth 判断最终答案对不对，如果对了就假设推理步骤都对（Math-Shepherd 思路）。但医学推理的步骤"部分正确"很常见，纯自动标注有噪声。更好的方式是用医生标注少量数据 + LLM-as-Judge 自动扩展。

**Q: 你们的 Judge 和 RM 有什么区别？面试官可能会挖这个坑。**
A: 面试中要明确说——我们用的是 LLM-as-Judge（MiMo-v2.5-pro），是一个不可微分的基于 prompt 的评估器，输出文本评分和理由。它不是 RM（可微分的神经网络打分器），不能用于 RL 训练。但它的评分可以用来过滤 SFT 数据、构造 DPO preference、评估模型质量。如果未来要做 PPO/GRPO，可以考虑从 Judge 的标注中蒸馏出一个 RM。

**Q: Best-of-N 在推理时有什么工程上的优化？**
A: vLLM 支持 best-of-n 的并行采样（use_beam_search + best_of 参数）。也可以做 speculative best-of-n：先用小 RM 快速过滤，top-k 再用大 RM 精排。连续批处理可以大幅降低 latency。
