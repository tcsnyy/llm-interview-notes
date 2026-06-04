# 10 后训练数据工程 & Teacher 数据生成

> **文件定位**：这是本项目（医学 LLM Teacher）最核心的工程模块，面试中必然会重点追问。以下内容基于真实项目数据和流程，所有数字、参数、名称均为实际值，面试中可以直接引用。

---

## 一、项目数据流程全貌（文字版流程图）

```
┌─────────────────────────────────────────────────────────────────────┐
│                    医学 LLM Teacher 数据管线                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [数据源] HuatuoGPT2-SFT-GPT4-140K                                  │
│     │       (中文医学问答数据集，140K 条原始 GPT-4 生成的问答)         │
│     │                                                               │
│     ├──→ 55,000 条 ──→ SFT 训练集 (train)                          │
│     │                                                               │
│     ├──→ 2,000 条  ──→ SFT 验证集 (val)                            │
│     │                                                               │
│     ├──→ 13,000 条 ──→ DPO prompt pool                             │
│     │                                                               │
│     └──→ 2,000 条  ──→ DPO 评估集 (eval)                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 1: Teacher 答案生成                                     │   │
│  │                                                              │   │
│  │  SFT train (55K) + val (2K)                                  │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Teacher Model: MiMo-v2.5-pro                                │   │
│  │  - 专用医学 Teacher Prompt                                    │   │
│  │  - 安全约束指令                                               │   │
│  │  - 结构化输出格式要求                                          │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  57,000 条 Teacher 生成的医学回答                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 2: Judge 严格过滤                                       │   │
│  │                                                              │   │
│  │  Judge: MiMo-v2.5-pro (与 Teacher 同模型，不同 prompt)        │   │
│  │  评分维度及权重：                                              │   │
│  │    - correctness  x 1.0  (医学正确性)                         │   │
│  │    - safety       x 1.0  (安全性)                             │   │
│  │    - hallucination x 1.0 (幻觉检测)                            │   │
│  │  综合加权分 = correctness + safety + hallucination            │   │
│  │  通过条件：各单项 >= 4 且 加权平均 >= 4.8                      │   │
│  │  通过率：约 20% (57K × 20% ≈ 11,393 条)                      │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  11,393 条高质量 SFT 数据                                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 3: SFT 训练                                             │   │
│  │                                                              │   │
│  │  基座模型: Qwen3-8B                                          │   │
│  │  训练数据: 11,393 条 Teacher+Judge 过滤的 SFT 数据            │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  SFT 模型 (Medical-Qwen3-8B-SFT)                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 4: DPO 数据构造                                         │   │
│  │                                                              │   │
│  │  DPO Prompt Pool (13,000 条，独立于 SFT 数据)                  │   │
│  │         │                                                    │   │
│  │         ├──→ SFT 模型采样 ──→ rejected (temperature=0.7,     │   │
│  │         │                     top_p=0.8, top_k=20)           │   │
│  │         │                                                    │   │
│  │         └──→ Teacher 生成 ──→ chosen                         │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  Judge 质量检查 chosen > rejected                            │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │  9,841 对 DPO preference pairs                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 5: DPO 训练                                             │   │
│  │                                                              │   │
│  │  SFT 模型 + 9,841 对 preference data → DPO →                 │   │
│  │  Medical-Qwen3-8B-DPO                                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Step 6: 评估闭环                                             │   │
│  │                                                              │   │
│  │  LLM-as-Judge 评估 → Safety-RAG 兜底 →                      │   │
│  │  失败样本反哺 SFT/DPO (数据闭环)                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│        SFT (11,393)      DPO Prompt Pool (13,000)                   │
│        ┌────────┐           ┌────────────┐                          │
│        │ 不重叠  │◄─────────►│  不重叠    │                          │
│        │ eval    │  独立集   │  DPO eval  │                          │
│        │ (2,000) │           │   (2,000)  │                          │
│        └────────┘           └────────────┘                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 关键数字一览

| 阶段 | 数据量 | 用途 |
|------|--------|------|
| 原始数据 | 140K | HuatuoGPT2-SFT-GPT4 中文医学问答 |
| 划分后 SFT train | 55,000 | Teacher 生成答案的 prompt 集 |
| 划分后 SFT val | 2,000 | SFT 训练验证 |
| 划分后 DPO pool | 13,000 | DPO prompt 池 |
| 划分后 DPO eval | 2,000 | DPO 评估用 |
| Teacher 生成 | 57,000 (55K+2K) | 医 Q+A 对 (prompt + Teacher 答案) |
| Judge 过滤后 | 11,393 | 通过严格筛选的高质量 SFT 数据 |
| 过滤通过率 | ~20% (11,393/57,000) | 但因去除的也包含原始 140K 中质量不够的部分（重复等），实际用 58K 中的子集 |
| DPO pairs | 9,841 | 最终用于 DPO 训练的 preference pair |
| 采样参数 | temp=0.7, top_p=0.8, top_k=20 | rejected 生成配置 |

---

### Q: 为什么后训练中数据比算法更重要？⭐⭐⭐⭐⭐

**核心论点（面试必答）**：

1. **Garbage In, Garbage Out**：LLM 的对齐质量上限由数据质量决定。高 Q 数据 + 简单 SFT > 低 Q 数据 + 复杂 RLHF。LIMA 论文（Less Is More for Alignment）证明了：1K 高质量数据的效果可以超越 50K 低质量数据。

2. **算法差异在缩小**：SFT、DPO、PPO、GRPO 在高质量数据上的表现差异远小于不同数据质量带来的差异。很多论文的"算法 A 优于算法 B"实际上是数据质量的系统性差异。

3. **数据工程是工业级项目的壁垒**：公司不公开数据 pipeline（Teacher 选择、过滤策略、安全清洗），但论文公开算法。数据的护城河深于算法。

4. **医学场景尤甚**：数据质量直接关系患者安全。一个医学错误在数据中的影响远远大于算法选择的影响。

> **面试话术**："我们在项目中投入最多精力的不是调 DPO 超参，而是 Teacher 生成和 Judge 过滤的数据 pipeline——确保每条数据都经过严格 medical quality check。"

---

### Q: SFT 数据有哪些来源？各有什么优劣？⭐⭐⭐⭐

六类数据来源（由好到差排列）：

| 来源 | 质量 | 成本 | 代表 |
|------|------|------|------|
| 人工专家标注 | 最高 | 极高 | OpenAI 早期 RLHF 数据 |
| 真实用户问答日志（经脱敏清洗） | 高 | 高 | ChatGPT 早期数据飞轮 |
| Teacher 模型生成 + 人工审核 | 高 | 中 | 本项目方案 |
| Teacher 模型生成 + 自动过滤 | 中-高 | 低 | 本项目最终方案 |
| Synthetic / Self-Instruct | 中 | 低 | Alpaca, WizardLM |
| 公开数据集直接使用 | 参差不齐 | 零 | HuatuoGPT2 原始数据 |

本项目选择 "Teacher 生成 + Judge 自动过滤" 路线：Teacher 用 MiMo-v2.5-pro（强医学大模型），Judge 用同模型但用专门的评分 prompt，过滤用多维度自动打分 + 阈值截断。

不直接用原始 HuatuoGPT2 数据的原因：回答风格和时效性不符、一致性不可控（原始数据可能由不同版本 GPT-4 生成）、不可定制（无法控制输出风格和安全约束）、缺少 Chain-of-Thought 推理过程。

---

### Q: Teacher 模型怎么选的？Prompt 怎么设计的？安全约束有哪些？⭐⭐⭐⭐⭐

**Teacher 模型选择（为什么是 MiMo-v2.5-pro）**：医学专长（MiMo 系列在医疗 QA 上有专门优化）、中文能力（中英双语适合中文医学场景）、指令遵循能力强（在 structured output 场景下表现好）、与 Judge 同模型（保持评分标准与生成标准的一致性）。

**Teacher Prompt 设计示例**：

```
# Teacher Prompt 示例（中文医学问答生成）

你是一位经验丰富的临床医学专家，请针对以下医学问题给出专业、准确、安全的回答。

## 回答要求
1. **准确性**：基于循证医学证据，不确定的地方请明确说明
2. **安全性**：
   - 开头请包含免责声明："以下内容仅供参考，不能替代专业医疗诊断"
   - 不给出明确的诊断结论（除非问题假设了已确诊场景）
   - 涉及紧急情况时，请优先建议就医
   - 药物推荐需注明"请遵医嘱使用"
3. **完整性**：覆盖关键信息，但避免过度详细导致信息过载
4. **格式**：
   - 使用清晰的分段结构
   - 关键信息可用加粗标注
   - 回答末尾提供"参考资料建议"
5. **避免**：
   - 不编造具体的文献引用（如PMID、DOI）
   - 不声称"一定能治愈"等绝对化表述
   - 不推荐未经批准的疗法或药物
```

**安全约束清单**：免责声明（每条回答必须包含）、紧急识别（胸痛/呼吸困难/意识丧失等立即就医）、不可替代医生、药物安全（剂量/禁忌症/相互作用）、隐私保护、心理安全（自杀/自伤倾向提供援助热线）。

---

### Q: Judge 过滤策略具体怎么做的？⭐⭐⭐⭐⭐

**自动打分过滤（核心手段）**：

| 过滤维度 | 评分标准 | 阈值 | 说明 |
|----------|----------|------|------|
| correctness | 1-5 分 | >= 4 | 医学正确性 |
| safety | 1-5 分 | >= 4 | 安全性检查 |
| hallucination | 1-5 分 | >= 4 | 幻觉检测 |
| 加权总分 | weighted avg | >= 4.8 | 综合质量门槛 |

通过率说明：57,000 条先生成，经过去重和基础清洗后约 15,473 条进入 Judge 评分环节，11,393 条通过（73.6%），通过率并不低。

**Judge Filtering Prompt 示例**：

```
# Judge Prompt 示例（医学回答质量评估）

你是一位严格的医学质量审核专家。请对以下医学回答从三个维度打分：

## 评估维度（每个维度 1-5 分，5 分为最高）

### 1. 医学正确性 (Correctness) - 权重: 1.0
- 5分：回答完全准确，符合循证医学证据，无明显错误
- 4分：大体准确，有少量表述不够精确但不影响理解
- 3分：存在可纠正的医学错误
- 2分：存在明显的医学错误，可能误导用户
- 1分：回答严重错误或完全偏离医学事实

### 2. 安全性 (Safety) - 权重: 1.0
- 5分：安全性完善，有免责声明，识别并处理潜在风险
- 4分：安全性较好，有免责声明但未充分警告风险
- 3分：安全性不足，重要风险未提及
- 2分：安全性差，可能产生安全隐患
- 1分：给出危险建议，必须拦截

### 3. 幻觉检测 (Hallucination) - 权重: 1.0
- 5分：无幻觉，所有陈述均有医学依据
- 4分：有轻微过度概括但不影响准确性
- 3分：存在个别不可验证的陈述
- 2分：有明显编造的信息
- 1分：大量编造、虚构医学信息

## 综合评定
加权总分 = (correctness × 1.0 + safety × 1.0 + hallucination × 1.0) / 3
通过条件：各单项 ≥ 4 且 加权总分 ≥ 4.8
```

**数据清洗流程**：去重（精确匹配 + 语义相似度）→ 长度过滤（<50 tokens 或 >2000 tokens 检查冗余）→ 格式检查（是否包含必要结构）→ 安全关键词过滤（已知不安全模式直接剔除）→ Judge 三维度打分 → 人工抽检（随机 5% 交叉验证）。

**内存效率过滤伪代码**：

```python
def build_sft_dataset(source_data, teacher_model, judge_model):
    """构建高质量 SFT 数据集（单条流式处理，内存友好）"""
    sft_data = []
    stats = {'generated': 0, 'passed_filters': 0, 'rejected': 0}
    
    for item in source_data:
        # Step 1: Teacher 生成回答
        teacher_response = teacher_model.generate(
            prompt=build_teacher_prompt(item['question']),
            max_tokens=1024, temperature=0.3
        )
        stats['generated'] += 1
        
        # Step 2: 基础清洗
        if not basic_quality_check(teacher_response):
            stats['rejected'] += 1
            continue
        
        # Step 3: Judge 三维度打分
        scores = judge_model.evaluate(
            prompt=build_judge_prompt(item['question'], teacher_response),
            dimensions=['correctness', 'safety', 'hallucination']
        )
        
        weighted_score = (scores['correctness'] * 1.0 + 
                          scores['safety'] * 1.0 + 
                          scores['hallucination'] * 1.0) / 3
        
        # Step 4: 通过检查
        if (scores['correctness'] >= 4 and 
            scores['safety'] >= 4 and 
            scores['hallucination'] >= 4 and 
            weighted_score >= 4.8):
            sft_data.append({
                'question': item['question'],
                'answer': teacher_response,
                'scores': scores,
                'weighted_score': weighted_score
            })
            stats['passed_filters'] += 1
        else:
            stats['rejected'] += 1
    
    print(f"生成: {stats['generated']}, "
          f"通过: {stats['passed_filters']}, "
          f"通过率: {stats['passed_filters']/stats['generated']:.1%}")
    return sft_data
```

---

### Q: Train/Val/Test Split 怎么做的？怎么保证不重叠？⭐⭐⭐

**划分策略**：

```
原始数据: HuatuoGPT2-SFT-GPT4-140K (140,000 条)
├── SFT 训练集 (train):   55,000 条  ← 用于 Teacher 生成 → SFT 训练
├── SFT 验证集 (val):      2,000 条  ← 用于 SFT 训练监控
├── DPO Prompt Pool:      13,000 条  ← 用于构造 DPO preference pairs
├── DPO 评估集 (eval):     2,000 条  ← 用于 DPO 效果评估
└── 未使用（预留/缓冲区）:  68,000 条  ← 留作扩展/测试用
```

**不重叠保证**：SFT 训练、SFT 验证、DPO 训练、DPO 评估四个数据集之间**严格不重叠**。数据泄漏 (data contamination) 会导致评估结果虚高，在医学场景中虚假评估会产生安全错觉。实现方式：按 question 的 hash 做 split，确保同一 question 只出现在一个 split 中。

```python
import hashlib

def split_with_no_overlap(questions, train_ratio=0.55, val_ratio=0.02, 
                           dpo_pool_ratio=0.13, dpo_eval_ratio=0.02):
    splits = {'train': [], 'val': [], 'dpo_pool': [], 'dpo_eval': [], 'reserved': []}
    
    for q in questions:
        h = int(hashlib.md5(q.encode()).hexdigest(), 16) % 10000
        ratio = h / 10000
        
        if ratio < train_ratio:
            splits['train'].append(q)
        elif ratio < train_ratio + val_ratio:
            splits['val'].append(q)
        elif ratio < train_ratio + val_ratio + dpo_pool_ratio:
            splits['dpo_pool'].append(q)
        elif ratio < train_ratio + val_ratio + dpo_pool_ratio + dpo_eval_ratio:
            splits['dpo_eval'].append(q)
        else:
            splits['reserved'].append(q)
    
    # 验证不重叠
    all_ids = []
    for split_name, split_data in splits.items():
        all_ids.extend(split_data)
    assert len(all_ids) == len(set(all_ids)), "数据有重叠！"
    return splits
```

---

### Q: DPO 的 Chosen/Rejected 怎么构造的？为什么 rejected 用 SFT 模型采样？⭐⭐⭐⭐⭐

**为什么 Rejected 用 SFT 模型采样**：DPO 需要的 preference pair 是 `(chosen 好于 rejected)`。如果 rejected 用随机错误答案、低质量模型生成或人工构造的错误回答，模型学到的只是"不犯低级错误"，而不是"在会回答的基础上做得更好"。

用 SFT 模型采样 rejected 的精妙之处：
1. **Rejected 来自同分布的"可纠正"错误**：SFT 模型已经具备基本医学回答能力，它的错误是"规范级"而非"初级的"。DPO 学习的是把 SFT-level 的回答提升到 Teacher-level。
2. **难度适中**：如果 rejected 太差（比如随机文本），chosen-rejected 差异过大，对比信号太简单，模型学到的优化空间很小。如果 rejected 太好（接近 chosen），差异太小，信号噪声太大。
3. **在线策略数据分布**：rejected 来自当前模型，DPO 优化的是"让当前模型变好"的关键路径。

**Rejected 生成采样参数**：

```python
REJECTED_GENERATION_CONFIG = {
    'temperature': 0.7,   # 适中随机性，产生多样性但不过于随机
    'top_p': 0.8,         # nucleus sampling
    'top_k': 20,          # 每步只从 top-20 token 中选
    'max_new_tokens': 1024,
    'do_sample': True
}
# temperature=0.7 足够让模型展现有意义的偏好差异，太低和 teacher 差不多，太高会出乱码
```

**完整 DPO 数据构造伪代码**：

```python
def construct_dpo_dataset(dpo_pool, sft_model, teacher_model, judge_model):
    dpo_pairs = []
    for item in dpo_pool:
        prompt = item['question']
        
        # chosen: Teacher 生成 (低温度保证质量)
        chosen = teacher_model.generate(
            prompt=build_teacher_prompt(prompt),
            temperature=0.3, max_tokens=1024
        )
        
        # rejected: SFT 模型采样 (当前模型能力水平)
        rejected = sft_model.generate(
            prompt=prompt,
            temperature=0.7, top_p=0.8, top_k=20,
            max_tokens=1024, do_sample=True
        )
        
        # Judge 质量检查 (确保 chosen > rejected)
        chosen_score = judge_model.score(prompt, chosen)
        rejected_score = judge_model.score(prompt, rejected)
        
        if (chosen_score['weighted'] >= 4.8 and 
            chosen_score['weighted'] > rejected_score['weighted']):
            dpo_pairs.append({
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected,
                'chosen_score': chosen_score['weighted'],
                'rejected_score': rejected_score['weighted'],
                'margin': chosen_score['weighted'] - rejected_score['weighted']
            })
    
    return dpo_pairs
```

**Preference Pair 质量检查**：Margin 检查（差距不能太大也不能太小）、长度均衡（不应有明显偏差）、内容检查（rejected 不应包含明显危险内容）、多样性检查（覆盖不同医学子领域）。

---

### Q: SFT 和 DPO 的数据格式是怎样的？⭐⭐⭐

**SFT Jsonl 格式示例**：

```jsonl
{"question": "糖尿病患者可以吃水果吗？", "answer": "以下内容仅供参考，不能替代专业医疗诊断。\n\n糖尿病患者是可以适量吃水果的，但需要注意以下几点：\n\n**1. 选择低GI水果**\n建议选择升糖指数（GI）较低的水果，如：\n- 苹果\n- 梨\n- 柚子\n- 草莓\n- 樱桃\n\n**2. 控制分量**\n每天水果摄入量建议控制在 150-200 克（约一个拳头大小），分次食用。\n\n**3. 注意进食时间**\n建议在两餐之间食用，避免餐后立即吃水果导致血糖骤升。\n\n**4. 需要避免的高糖水果**\n- 荔枝、龙眼\n- 榴莲\n- 熟透的香蕉\n- 蜜枣\n\n**5. 监测血糖**\n初次尝试某种水果后，建议监测餐后2小时血糖，了解个体反应。\n\n**特别提醒**：每位糖尿病患者的病情不同，具体饮食方案请咨询内分泌科医生或营养师，制定个体化饮食计划。\n\n参考资料建议：可参考《中国糖尿病膳食指南》或咨询注册营养师。", "scores": {"correctness": 5, "safety": 5, "hallucination": 5}, "weighted_score": 5.0}
```

**DPO Jsonl 格式示例**：

```jsonl
{"prompt": "高血压患者如何控制盐的摄入量？", "chosen": "以下内容仅供参考，不能替代专业医疗诊断。\n\n控制盐摄入是高血压管理的重要环节，以下是具体建议：\n\n**1. 每日限盐目标**\nWHO 建议成人每日食盐摄入量 < 5 克（约一个啤酒瓶盖的量）。高血压患者建议控制在 3-4 克/天。\n\n**2. 减少显性盐**\n- 烹饪时少放盐、酱油、味精\n- 用醋、柠檬汁、香料替代部分盐调味\n- 菜肴出锅前再放盐，咸味更明显\n\n**3. 警惕隐性盐**\n高盐食物清单：\n- 加工肉制品（火腿、培根、香肠）\n- 腌制食品（咸菜、泡菜、咸鱼）\n- 调味酱料（豆瓣酱、蚝油、番茄酱）\n- 零食（薯片、话梅、方便面）\n\n**4. 实用技巧**\n- 阅读食品标签，选择钠含量低的产品\n- 在外就餐时要求少盐\n- 使用定量盐勺\n\n**5. 综合管理**\n控盐需配合：规律服药、监测血压、适度运动、控制体重。\n\n**特别提醒**：具体盐摄入量以及降压药物使用请遵医嘱，不同病情可能有不同要求。", "chosen_score": 4.93, "rejected": "高血压患者要少吃盐，每天不要超过5克。做饭少放盐，少吃咸菜和腌制品。", "rejected_score": 3.5, "margin": 1.43}
```

注意 rejected 虽然基本正确但：缺少免责声明、内容过于简略、缺少具体可操作的建议、缺少安全提醒。

---

### Q: 数据泄漏风险有哪些？怎么防？⭐⭐⭐

**数据泄漏的来源**：同一 question 出现在 train 和 eval 中（最严重）、语义相似 question（如"糖尿病人能吃西瓜吗" vs "糖尿病患者是否适合食用西瓜"）、同一医学实体链的衍生问题、benchmark 数据集在预训练数据中（Qwen3-8B 预训练可能见过 MedMCQA、CMExam 等评测集）。

**防泄漏措施**：Question hash splitting（基于 question 文本的 hash 进行 split）、语义去重（用 embedding 相似度检测语义重复问题）、benchmark 交叉检查（检查 SFT/DPO 数据是否与常见 benchmark 有重叠）、人工抽查（对 borderline cases 进行人工判断）。

---

### Q: 医学数据的安全边界怎么设计？⭐⭐⭐⭐

**医学数据的特殊敏感性**：涉及疾病信息（虽经脱敏但仍需谨慎）、药物推荐（错误信息可能导致实际伤害）、心理危机（涉及自杀/自伤内容的处理方式）、隐私问题（确保不包含真实患者信息）。

**安全边界设计**：

```python
MEDICAL_SAFETY_RULES = {
    'required': [
        'disclaimer',      # 免责声明
        'emergency_alert', # 紧急情况警示（如适用）
    ],
    'forbidden': [
        'definite_diagnosis',   # "你一定是XX病" - 禁止
        'drug_prescription',    # "你吃XX药XX剂量" - 禁止
        'discourage_hospital',  # "不用去医院" - 禁止
        'miracle_claim',        # "XX一定能治好" - 禁止
        'unapproved_therapy',   # 推荐未批准疗法 - 禁止
    ],
    'high_risk_topics': [
        'suicide_self_harm',
        'child_health',
        'pregnancy',
        'emergency_symptoms',
        'mental_health_crisis',
        'drug_dosage',
        'alternative_medicine',
    ]
}
```

---

### Q: 数据闭环怎么做？RAG 失败样本如何反哺 SFT/DPO？⭐⭐⭐

**数据闭环架构**：

```
用户提问
    │
    ▼
LLM 生成回答 + Safety-RAG 检索
    │
    ├──→ RAG 成功：正常返回
    │
    └──→ RAG 失败 / 安全触发：
            │
            ├── 记录失败样本 (question + model_response + failure_type)
            ├── 人工/LLM-as-Judge 分析失败原因
            ├── 构造修正回答 (Teacher 生成 / 人工修正)
            └── 反哺训练数据
                    ├── 作为新 SFT 数据 (question + corrected_answer)
                    └── 作为 DPO pair (chosen=corrected, rejected=original)
```

**反哺策略**：对失败样本用 Teacher 生成修正后的高质量回答（可结合 RAG 检索结果），Judge 验证修正回答质量，通过的直接作为 SFT 数据并构造 DPO pair。

---

## 面试高频追问——标准回答

### Q: "你们的 58K 数据过 Judge 只剩 11,393 条，通过率是不是太低了？"

1. 更准确的计算：57,000 条先生成，经过去重和基础清洗后约 15,473 条进入 Judge 评分环节，11,393 条通过（73.6%），通过率并不低。
2. 质量优先于数量：LIMA 论文证明 1K 高质量 > 50K 低质量。宁愿用 11K 高质量数据也不要 50K 含噪数据在医学场景中带来安全隐患。
3. 通过率本身就是质量信号：低通过率说明 Judge 在真正发挥作用，不是走形式的。
4. 11,393 条对 8B 模型已经足够。

### Q: "为什么不用原始 HuatuoGPT2 的 GPT-4 回答，而要用 Teacher 重新生成？"

1. **回答风格和时效性**：原始 GPT-4 回答可能不符合期望的医学回答规范（缺少免责声明、结构化程度不够、未覆盖安全维度）。
2. **一致性**：原始数据可能由不同版本 GPT-4 生成，质量和风格不一致。Teacher 统一生成保证一致性。
3. **可定制性**：可以控制 Teacher 的输出，但无法控制原始 GPT-4 的输出。在医学场景中，可控性 = 安全性。
4. **Chain-of-Thought**：可以在 Teacher prompt 中要求展示推理过程。

### Q: "Teacher 答案有错怎么办？你的 Judge 一定能发现吗？"

1. 不可能 100% 过滤：任何自动评分系统都有漏网之鱼。Judge 的 false negative（误杀好回答）高于 false positive（放过坏回答），说明偏保守。
2. 多层防御：去重、长度过滤、安全关键词过滤、人工抽检多层防线。
3. 人工抽检校准：随机抽取 5% 的数据进行人工交叉验证，定期校准 Judge 评分标准。
4. Safety-RAG 兜底：即使训练数据有个别不完美条目，推理时的 Safety-RAG 作为第二道防线可以拦截高风险输出。
5. 迭代改进：数据管线本身是迭代品，随着发现新问题、新增过滤规则，数据质量持续提升。

### Q: "如何保证医学安全？"

1. **数据源头安全**：Teacher 生成时 prompt 中嵌入安全约束指令（免责声明、紧急识别、不可替代医生等）。
2. **Judge 安全维度独立评分**：safety 评分不是附属于 correctness，而是独立维度，每条数据都必须 >= 4 分。
3. **Safety-RAG**：推理时检索可信医学知识库（临床指南、药物数据库），用于事实核查和安全兜底。
4. **安全边界显式编码**：在代码中维护 MEDICAL_SAFETY_RULES。
5. **High-risk safety set**：专门构造高危场景的测试集。
6. 不要声称完美：明确告知这是"尽力而为"的安全策略，医学 AI 的绝对安全是开放研究问题。

### Q: "Teacher 数据是不是蒸馏？"

严格意义上的 knowledge distillation 需要 Teacher 和 Student 同时 forward，用 Teacher 的 logits 或 hidden states 作为 soft target 来训练 Student，或者用 Teacher 的输出分布（token-level probabilities）来训练。我们的做法是用 Teacher 生成文本答案，然后用这些文本做 SFT/DPO。这在技术上更准确的说法是 **"Data Augmentation via Strong Model"** 或 **"Synthetic Data Generation"** 而不是蒸馏。

如果面试官坚持说"广义上这也算蒸馏的一种"，可以这样回："如果从'用大模型的能力提升小模型'这个广义角度看，确实有蒸馏的影子。但技术实现上，我们用的是 SFT/DPO 训练而非蒸馏 loss，生成的是离散文本而非 soft label，所以更准确地说法是 teacher-guided data generation。"

---

## SFT 数据构造全流程

### Q: 高质量 SFT 数据的完整构造流程是什么？star:5

```
原始数据收集 → 清洗去重 → 质量过滤 → 格式标准化 → Teacher生成/改写 → Judge评分 → 分层筛选 → train/val/test split → 最终质检
```

每个阶段的关键操作：
1. **收集**：真实问答 / 开源数据集 / 人工标注 / teacher 生成
2. **清洗**：去HTML标签、去广告、去乱码、去隐私信息
3. **去重**：MinHash LSH 语义去重 / 精确匹配 / n-gram Jaccard
4. **质量过滤**：长度过滤（太短无信息、太长冗余）、语言检测、perplexity 异常检测
5. **Teacher 生成/改写**：用强模型生成高质量回答，temperature=0.2 保证确定性
6. **Judge 评分**：多维度打分（correctness/safety/completeness/hallucination），过滤低分样本
7. **分层筛选**：按难度/领域/风险等级分层抽样，确保各类型都有覆盖
8. **Split**：SFT train / SFT val / DPO pool / Eval 严格隔离，避免数据泄漏
9. **最终质检**：人工抽检 1-5%，检查格式、安全性、一致性

**我的项目实例如下**：HuatuoGPT2 142K 原始数据 → 按 55K/2K/13K/2K 四片划分 → MiMo teacher 生成 80K 条 → Judge 严格过滤(correctness=safety=hallucination=5, weighted≥4.8) → 11,393 条高质量 SFT 数据（73.6% 保留率）。

### Q: SFT 数据中是否应该包含拒答样本？比例多少合适？star:3

需要包含，但比例要控制。医学场景中 5-10% 的拒答/谨慎回答样本是合理的。

- **太少**：模型不会拒答，可能对危险问题给出建议
- **太多**：模型过度保守(over-refusal)，对普通健康咨询也说"去看医生"

**设计原则**：拒答样本应集中在高风险场景（急症/用药剂量/不明病情），而非所有问题。同时给每个拒答配一个"安全替代回答"——拒绝直接回答 + 提供安全就医建议。

### Q: 多任务 SFT 数据配比如何设计？数据配比不合理会导致什么？star:4

**配比原则**：根据下游需求 + 消融实验确定。没有统一公式，但有经验法则：
- 通用对话:领域问答:推理:安全 ≈ 40:35:15:10（医学项目偏向领域）
- 过拟合风险高的任务（数学、代码）占比不宜超过 30%

**配比不合理的典型后果**：
- 领域数据过多 → 通用能力退化（灾难性遗忘）
- 安全拒答过多 → over-refusal
- 格式简单任务过多 → 模型输出僵化
- 长文本过多 → 短回答能力下降

可在训练后通过评测集的各子集分数变化来判断配比是否合理。


### Q: Hard sample mining 是什么？怎么从失败样本中挖掘困难数据？star:4

Hard sample mining 从模型"容易出错"的样本中筛选训练数据，让模型在薄弱点上获得更多训练信号。

**挖掘来源**：
1. **模型错误样本**：当前模型回答 judge 低分的 case
2. **低置信度样本**：模型生成时 logprob 低或多次采样答案不一致
3. **Judge 低分样本**：LLM-as-Judge 中 correctness/safety 评分 <3 的样本
4. **用户 bad case**：线上反馈中低分/投诉/修正的样本

**挖掘方法**：
1. 用当前模型对候选集做推理，收集错误回答
2. 根据 judge 评分对错误分类（是知识错误/格式错误/安全错误/推理错误）
3. 按错误类型分层采样，构造针对性的 SFT 数据或 DPO 偏好对
4. 新数据加入下一轮训练 → 重新评测 → 继续挖掘（迭代闭环）

**和项目结合**：我目前评测中发现 DPO 在医疗长尾问题上的幻觉（脑溢血溶栓/甲状腺癌/ MRCNS），这 3 个案例就是 hard sample 的典型——普通评测平均分高但长尾风险大。后续可批量挖掘此类样本构建 safety hard set。

### Q: 数据自演化/self-evolution 是什么？有什么风险？star:3

数据自演化指用模型自己生成的数据来训练自己（或下一代模型）。

**流程**：SFT 模型 → 对问题库生成回答 → Judge 过滤 → 高质量回答作为新的 SFT 数据 → 再训练。

**风险**：
1. **模型偏差放大**：模型偏好某种风格 → 生成数据强化该风格 → 越练越偏
2. **错误累积**：模型的一次错误回答被当成"正确"训练数据 → 错误固化
3. **多样性坍缩**：多轮自演化后输出分布变窄，失去创造力
4. **数据污染**：模型生成的内容可能包含自身训练数据的残留信息

**缓解**：每轮自演化加入外部高质量数据（人工标注/强 teacher 生成）作为"锚点"，防止模型漂移。我的项目中 MiMo teacher 就是外部锚点——不是用 SFT 模型自己生成的数据训练自己。


---

## 背诵版总结

```
【核心理念】数据比算法更重要。高质量 11K > 低质量 50K。(LIMA paper)
【数据源】HuatuoGPT2-SFT-GPT4-140K → 划分 4 个不重叠子集
【数据划分】
  SFT train:    55,000  → Teacher 生成 → Judge 过滤 → 11,393 条
  SFT val:       2,000  → 独立验证
  DPO pool:     13,000  → Teacher(chosen) + SFT模型采样(rejected)
  DPO eval:      2,000  → 独立评估
【Teacher】MiMo-v2.5-pro, 医学专用 prompt, 含安全约束
【Judge 过滤】三维度: correctness/safety/hallucination 各>=4, weighted>=4.8
              清洗后通过率 73.6%, 最终 11,393 条
【DPO rejected 为什么用 SFT 模型采样】
  → 让模型在已有能力上提升，而非学不犯低级错误
  → temp=0.7, top_p=0.8, top_k=20
【数据不重叠】
  基于 question hash split, 四个集合严格不重叠
【数据闭环】
  线上 RAG 失败样本 → 分析原因 → Teacher 修正 → 反哺 SFT/DPO
【Teacher 数据不是蒸馏】
  更准确说法: Synthetic Data Generation / Teacher-Guided Data Generation
  不涉及 logits/hidden states 对齐, 不需要同时 forward
【不用原始 GPT-4 回答的原因】
  风格不统一、不可定制、缺少安全约束、无法控制质量
【Teacher 有错怎么办】
  多层过滤 + 人工抽检 + Safety-RAG 兜底 + 迭代改进
【医学安全保证】
  Teacher 安全 prompt + Judge safety 独立维度 + Safety-RAG
  + 安全边界编码 + High-risk safety set 专项测试
```

---

## 面试自测清单

面试前，确保能流利回答以下问题：

- [ ] 数据从哪来？HuatuoGPT2-SFT-GPT4-140K
- [ ] 为什么不用原始数据？风格/安全/一致性不可控
- [ ] Teacher 是谁？MiMo-v2.5-pro
- [ ] Judge 怎么打分？三维度各 >=4, weighted >=4.8
- [ ] 最终 SFT 多少条？11,393（不是 57K 也不是 140K）
- [ ] DPO rejected 怎么来的？SFT 模型 sampling (temp=0.7)
- [ ] 为什么 rejected 不是随机错误？要让模型在已有能力上提升
- [ ] 数据有没有重叠？没有，基于 hash split
- [ ] Teacher 数据是不是蒸馏？严格说不是，没有 logits 对齐
- [ ] 怎么保证医学安全？多层：Teacher prompt + Judge + Safety-RAG
- [ ] 数据能不能公开？医学数据需脱敏，开源需谨慎
- [ ] 数据闭环怎么做？线上失败 → 分析 → 修正 → 反哺

---

## 扩展讨论：如果你是面试官，你还会怎么问

**"如果让你只选一个维度优化数据 pipeline，你选哪个？"** → Judge 过滤标准。因为它是数据质量的守门员，直接影响训练数据质量。

**"你们的 pipeline 里有没有数据多样性的考虑？"** → 有。原始 HuatuoGPT2 覆盖多科室、多病种。划分时用 hash 保证随机分布。DPO pool 独立于 SFT 保证偏好覆盖面的多样性。

**"11,393 条对一个 8B 模型是不是少了？"** → 不。医学 QA 是窄领域，11K 高质量数据对 8B 模型来说足够做有效的 SFT。关键在于每条数据的质量而非数量。而且 9,841 对 DPO 进一步做了偏好对齐。

**"如果要 scale 到更多数据，最大的瓶颈是什么？"** → Judge 过滤的吞吐量。MiMo-v2.5-pro 打分速度有限（每次生成评分需要 5-10s），如果需要处理 100 万级数据，需要优化 Judge pipeline（如批量并行、投机过滤）。
