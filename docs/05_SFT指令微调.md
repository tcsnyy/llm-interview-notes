# 05 SFT 指令微调

---

## 目录
- [一、SFT 基础理论](#一sft-基础理论)
- [二、SFT 目标函数与 Loss 计算](#二sft-目标函数与-loss-计算)
- [三、数据构造与预处理](#三数据构造与预处理)
- [四、SFT 训练技巧与常见问题](#四sft-训练技巧与常见问题)
- [五、领域 SFT 与医学 SFT 专题](#五领域-sft-与医学-sft-专题)
- [六、Teacher 数据相关](#六teacher-数据相关)
- [七、代码实现](#七代码实现)
- [八、面试版本回答](#八面试版本回答)
- [九、背诵版总结](#九背诵版总结)

---

## 一、SFT 基础理论

### 1.1 Q: 解释一下 SFT 是什么，它和 CPT（Continue Pre-Training）有什么区别？你的 SFT 项目做了什么？star:5

**SFT（Supervised Fine-Tuning，监督微调）**，也叫 Instruction Tuning（指令微调），是在预训练模型的基础上，用**有监督的（问题，回答）对**进行训练，让模型学会"听懂人话并按要求回答"。

可以这样理解三个阶段：
- **Pre-Training（预训练）**：教模型"人类语言长什么样"。用海量无标注文本，学 next token prediction。模型学会语法、常识、推理——但不会对话。
- **CPT（Continue Pre-Training，继续预训练）**：在特定领域的无标注文本上继续做 next token prediction，往模型脑子里灌领域知识。比如用医学教科书、病历让模型学医学术语和知识。
- **SFT（指令微调）**：教模型"对话格式和任务行为"。用（指令/问题, 期望回答）对，让模型学会：看到 user 消息要先理解，再给出符合要求的回复。**SFT 的核心是行为对齐，不是知识注入。**

**SFT 和 CPT 的核心区别：**

| 维度     | CPT                                  | SFT                                            |
| -------- | ------------------------------------ | ---------------------------------------------- |
| 训练数据 | 无标注文本（raw text）               | 标注的（问题, 回答）对                          |
| 训练目标 | Next Token Prediction（同预训练）     | Next Token Prediction（但只在回答部分算 loss）   |
| 数据格式 | 普通文档/代码/网页                   | 结构化对话（system + user + assistant）          |
| 主要作用 | 注入领域知识                         | 学习对话格式和行为                               |
| 参数量级 | 几十B到几百B token                   | 几万到几十万条样本                               |
| 学习率   | 较高（1e-5 ~ 5e-5）                  | 较低（1e-6 ~ 1e-5）                              |
| 训练轮数 | 1-3 epoch                            | 1-3 epoch（非常容易过拟合）                      |

在医学 LLM 项目中，我做的是 SFT——用医学问答对（teacher 模型生成的答案 + 筛选后的高质量数据）让 Qwen2.5/Qwen3 学会医学问答的对话格式。在 SFT 之前如果做了 CPT，是用医学无标注文本灌知识；SFT 阶段则是让模型学会"以医学助手的身份、谨慎专业地回答问题"。

**注意**：
- 很多人说 SFT 是"知识注入"——这是错的，SFT 主要做行为对齐。知识注入靠 CPT。少量 SFT 数据想注入大量知识是不可能的。
- "SFT 就是 supervised fine-tuning 的字面意思"——这是面试的减分回答。要说清楚"指令微调"这个更准确的叫法。

---

### 1.2 Q: SFT 的 loss 是怎么计算的？什么是 Teacher Forcing？Causal LM loss 和普通交叉熵有什么区别？star:5

**SFT 的目标函数就是 Causal Language Modeling Loss（因果语言模型损失）**，本质上就是交叉熵损失（Cross-Entropy Loss），但在自回归的条件下计算。

公式：
```
Loss = -1/N * Sigma log P(y_t | y_<t, x)
```
- x 是输入（user message、system prompt）
- y 是输出（assistant answer）
- y_t 是第 t 个 token
- y_<t 是第 t 个 token 之前的所有 token
- 模型要最大化每个位置预测正确 token 的概率

**Teacher Forcing（强制教学）：**训练时，无论模型上一个位置预测得对不对，下一个位置的输入都使用**真实 label** 而不是模型的预测。这就像老师（正确答案）在每个步骤都在场纠正你。好处是训练稳定，loss 快速收敛，避免了错误累积（error propagation）。缺点是训练和推理存在 gap（训练时永远看正确答案，推理时可能看到自己之前的错误输出）——这叫 exposure bias。

**Causal LM Loss 的特殊之处：**每预测一个 token 时，模型只能看到当前位置**之前**的 token（因果注意力掩码）。因此 Causal LM loss = 对序列中每个位置的交叉熵求和取平均。但实际代码实现中，通常用一次前向传播并行计算所有位置的 loss（因为 causal mask 让每个位置只关注 prefix，所以可以一并算）。

在我的 SFT 训练中，loss 是 `torch.nn.CrossEntropyLoss` 或 HuggingFace Trainer 内置的 causal LM loss，但在算 loss 时做了过滤——**只对 assistant 部分的 token 计算 loss**（详见 2.3）。

**注意**：
- "Teacher forcing 就是让模型看答案"——这个说法不能说错，但不严谨。应该说"在训练时用真实 label 替代模型预测作为下一个时刻的输入"。
- 把 Causal LM loss 说成 "masked language model loss"——MLM 是 BERT 的目标，SFT 是 Causal LM 目标。

---

## 二、SFT 目标函数与 Loss 计算

### 2.1 Q: 你在实现 SFT loss 的时候，为什么要把 logits 和 labels 错开一位（shift）？能画出 token 的对齐关系吗？star:5

这涉及到自回归模型的"预测下一个 token"语义。

假设输入序列的 token IDs 是：[t1, t2, t3, t4, t5]（比如 "我/今天/很/开心/EOS"）

模型做前向传播时：
- 位置 1 看 t1，预测 t2
- 位置 2 看 t1, t2，预测 t3
- 位置 3 看 t1, t2, t3，预测 t4
- 位置 4 看 t1, t2, t3, t4，预测 t5
- 位置 5 看 t1~t5，预测 t6（虽然是 `[PAD]`）

所以 logits[0] 对应预测 label[1]，logits[1] 对应预测 label[2]，以此类推——logits 比 labels 少最后一个，labels 比 logits 少第一个。

**实际代码中的做法：**
```python
# 原始 labels: [t1, t2, t3, t4, t5]
# 前向传播得到 logits: shape = (batch, seq_len, vocab_size)
# 每个位置的 logits 预测 "下一个 token"

shift_logits = logits[..., :-1, :].contiguous()  # 去掉最后一个位置的预测
shift_labels = labels[..., 1:].contiguous()        # 去掉第一个位置的 label

# 这样 shift_logits[i] 和 shift_labels[i] 就是正确对齐的
# shift_logits[0] 预测的是 shift_labels[0]（即原始 labels[1]）
loss = CrossEntropyLoss(shift_logits.view(-1, vocab_size), 
                         shift_labels.view(-1))
```

**对齐关系图：**
```
Tokens:     [BOS]  我    今天   很    开心   [EOS]
Position:    0     1     2     3     4      5
logits:     l0    l1    l2    l3    l4     l5
predicts:   t1    t2    t3    t4    t5     t6
labels:    忽略   t1    t2    t3    t4     t5

对齐后：
shift_logits:  l0    l1    l2    l3    l4
shift_labels:  t1    t2    t3    t4    t5
```

**注意**：
- 忘了 shift 直接用 `loss_fn(logits, labels)`，结果把 t1 当成 t0 的 label 来算 loss，每个位置的 loss 都算错了一位，模型根本学不对
- shift 方向搞反：`labels[..., :-1]` 和 `logits[..., 1:]` 是另一种错法

---

### 2.2 Q: 在 SFT 训练中，为什么 user 和 system prompt 的部分不计算 loss？如果算了会怎样？star:5

**SFT 的目标是：给定一个输入（system + user），模型生成正确的输出（assistant）。**

如果对 user 部分也算 loss，后果是：
1. **模型会学习"复读 user 的话"**：因为 user 部分的 token 也是真实的（标签等于自己），模型看到 user 文本后会倾向于输出同样的内容，这就变成了一个"复读机"
2. **行为目标混淆**：模型分不清自己到底应该"复读输入"还是"生成新回答"
3. **训练浪费**：SFT 数据中 user 和 system prompt 通常比 assistant 回答长（尤其是医学场景中 system prompt 很长），大部分 loss 算在了无关的地方

**正确做法**：
- 构造 labels 时，把 user 和 system prompt 对应位置的 label 设为 -100（PyTorch CrossEntropy 的 ignore_index）
- 只把 assistant 回答部分的 token 作为真实 label
- 这样模型就只学"看到 user 说 X 后，我应该回答 Y"

**代码逻辑：**
```python
# messages = [
#    {"role": "system", ...},
#    {"role": "user", ...},
#    {"role": "assistant", ...}
# ]

# 1. 先用 apply_chat_template 得到完整的 input_ids
# 2. 在 assistant 部分的 token 位置放真实 token ID
# 3. 其他位置放 -100（ignore_index）
```

我的项目中就是这样做的——通过构造 loss mask，只在 assistant token 区域算 loss。这是一个 SFT 训练的"标配"。

**注意**：在 DPO 场景下，chosen/rejected 整个序列都要算 log prob（不仅仅是 assistant 部分），但这个 log prob 只是用来计算偏好差，不是用来梯度更新的。新手容易把 SFT 的 loss mask 思路直接套到 DPO 上。

---

### 2.3 Q: 你在训练中怎么处理 labels？-100 是干什么的？attention_mask 和 loss_mask 有什么区别？star:5

**`-100` 的含义：**在 PyTorch 的 `CrossEntropyLoss` 中，`ignore_index` 默认为 -100。所有 label 为 -100 的位置在计算 loss 时会被跳过——不贡献 loss 也不贡献梯度。

所以你的 labels 张量长这样：
```
labels: [-100, -100, -100, -100,  t5,  t6,  t7,  t8, -100]
         ^---- system prompt ----^  ^-- assistant --^  ^ PAD
```

**Attention Mask vs Loss Mask：**这是两个不同的概念，但很多人混淆。

|          | Attention Mask                                    | Loss Mask（通过 labels=-100 实现）               |
| -------- | ------------------------------------------------- | ------------------------------------------------ |
| 作用     | 控制哪些 token 参与 self-attention 计算           | 控制哪些位置的预测参与 loss 计算                  |
| 值       | 1（参与）/ 0（不参与，如 PAD）                    | token_id / -100（不参与 loss）                    |
| 在哪里用 | self-attention 中加到 softmax 之前（加 -inf）      | loss 函数的 `ignore_index`                        |
| 影响     | 不正确的 attention mask 会让 PAD 污染有用 token   | 不正确的 loss mask 会让模型学"复读"user 的话     |

**两者配合使用的场景：**

```
                     token1 token2 token3 PAD  PAD
input_ids:           [t1,   t2,    t3,    t_p, t_p]
attention_mask:      [1,    1,     1,     0,   0  ]   <- 告诉 attention 忽略 PAD
labels:              [-100, -100,  t3,   -100, -100]   <- 只有 assistant 的 t3 算 loss
role:                  system      user    asst
```

在项目中，attention_mask 由 tokenizer 自动生成（1 对应真实 token，0 对应 padding），loss mask 通过在构造 labels 时把非 assistant 部分设为 -100 来实现。

**注意**：
- 有人说 "labels=-100 等价于 attention_mask=0"——这是错的。Attention mask 影响所有 attention 层的计算（连 user 部分也要参加 attention，不然模型看不到 user 消息），但 labels=-100 仅影响 loss 计算。
- PAD 位置既要 attention_mask=0 也要 labels=-100，两者双保险。

---

## 三、数据构造与预处理

### 3.1 Q: 训练时 max_seq_length 设多大？为什么选这个值？怎么处理超长样本？star:4

**max_seq_length 怎么选：**
1. 看**模型的最大上下文长度**（通常叫 `max_position_embeddings`）。Qwen2 系列一般是 32768 或 131072，但你不需要用满
2. 看**数据的长度分布**。对训练数据做统计分析，取 95% 或 99% 分位数
3. 看**GPU 显存**。序列长度翻倍，显存占用约翻 4 倍（attention 是 O(n^2) 的）
4. 一个经验：SFT 阶段 2048~4096 基本够用；医学场景如果有很长的 system prompt 或病历，可能需要 4096~8192

**Truncation 策略：**
- **只截断 user/messages 部分**：这是最常见的做法，保留最近的几条对话，把最早的截掉
- **截断方式**：`truncation="left"` 或手动从左边截断。对于多轮对话，保留最近的几轮通常比保留最早的几轮更有意义
- **绝对不要截断 assistant 回答**：如果回答太长，这条数据本身可能质量不高（模型答了一大堆废话），可以直接过滤掉

**Padding 策略**：Batch 内要统一长度，通常用 `DataCollatorForSeq2Seq` 或自定义 collator。Right padding（训练用），pad_token 对应的 label 设为 -100。

在医学项目中，根据对话长度分布选择 max_seq_length。一般 system prompt ~200 tokens，问诊对话 ~500-800 tokens，回答 ~300-500 tokens，总计约 1000-1500 tokens。设置 max_seq_length=2048 通常够了。如果有长篇病历，可以提升到 4096。

**注意**：
- 把 max_seq_length 设成模型最大值（如 131072）但 GPU 根本不够——一个 batch 都放不下
- 截断时把 assistant 回答的尾巴截掉了——这个样本的 label 就不完整了
- 用 right truncation（截掉最近的对话）而不是 left truncation

---

### 3.2 Q: 多轮对话场景下，SFT 数据怎么构造？是把每一轮拆开还是一起训练？star:4

多轮对话有两种构造方式，各有优劣：

**方式一：每个（user, assistant）对独立训练（单轮拆开）**
```
样本1: system + user_1 -> assistant_1
样本2: system + user_2 -> assistant_2
...
```
- 优点：简单直接，不依赖上下文
- 缺点：丢失了多轮对话的上下文信息和连贯性

**方式二：完整多轮对话一起训练（标准做法）**
```
样本: system + user_1 + assistant_1 + user_2 + assistant_2 + user_3 + assistant_3
```
- labels 只对每个 assistant 部分计算 loss（user 部分都是 -100）
- 优点：模型学会在对话中保持上下文，理解"追问"、"确认"等交互模式
- 这是更接近真实聊天场景的做法

**项目中推荐做法**：把整个多轮对话当一条样本，但做以下检查：
1. 过滤掉过长对话（超过 max_seq_length 有效部分的）
2. 确保每轮 assistant 回答都有质量（有一个差的就整条去掉）
3. 最后一轮的 assistant 回答通常是最重要的（代表最终解决问题），可以考虑用最后的回答做质量加权

在医学 LLM 项目中，多轮对话很常见（患者追问、澄清症状、用药指导等），用方式二更合适。

**注意**：
- 拆开训练时忘了保留历史 context，导致模型不知道"这个追问是基于什么回答的"
- 多轮对话中有一轮 assistant 回答质量差，整体保留了，模型学到了坏的回答模式

---

### 3.3 Q: 你了解 packing 吗？SFT 训练时做样本 packing 有什么好处和风险？star:3

**Packing** 是把多条短样本拼接成一条长样本（到 max_seq_length），提高 GPU 利用率。

**优点：**
1. **减少 PAD 浪费**：如果 max_seq_length=4096，但大多数样本只有 500 tokens，不 packing 的话 87% 的算力都浪费在 PAD 上了
2. **提高吞吐量**：同样的 GPU 时间，实际训练的有效 token 数更多
3. **等价于增大 batch size**：一次前向传播处理更多有效 token

**风险（非常重要）：**
1. **样本间互相 attention 的问题**：如果把 A 样本和 B 样本拼在一起，模型可能在 attention 时把 A 的信息"偷看"到 B 的生成中，这就不是标准的 causal LM 训练了。解决方案：**在样本边界处重置 attention mask**，让不同样本之间互相看不到
2. **EOS 被埋没**：如果样本 B 紧跟在样本 A 后面，样本之间的 EOS 就不再是"序列结束"的语义了。解决方案：在样本之间插入一个特殊的分隔符，或者确保 attention mask 隔开
3. **实现复杂度**：需要在 collator 中做 packing 逻辑，同时修改 attention mask 为 block-diagonal 结构

**是否需要 packing**：
- 如果你的数据长度分布均匀且都接近 max_seq_length，packing 收益很小
- 如果大量样本远短于 max_seq_length，packing 收益显著
- 在 SFT 阶段，样本通常不会太短（500-2000 tokens），packing 收益中等

在 MiniMind 和医学项目中，如果对话长度接近 max_seq_length，可以不 packing；如果大批量短对话，可以考虑 packing。如果项目中没有实现 packing，面试时可以直接说"我没有做 packing，原因是我的数据长度分布比较均匀"——这也是一个合理的回答。

**注意**：做 packing 时忘了改 attention mask —— 这是最常见的 bug。必须用 block-diagonal attention mask，确保样本间看不到彼此。

> **注意：本节描述的标准 packing 做法和 block-diagonal attention mask 的知识点，在 MiniMind / 医学 LLM 两个项目中均未找到专门的 packing 实现。但面试中可能被问及，需要了解。**

---

### 3.4 Q: SFT 数据有哪几种常见格式？你在项目中是怎么转换数据格式的？star:4

三种主流 SFT 数据格式：

**1. Alpaca 格式（最简单）**
```json
{
  "instruction": "翻译下面的句子：Hello World",
  "input": "",
  "output": "你好，世界"
}
```
- 来自 Stanford Alpaca 项目
- 单一 instruction，不适合多轮对话
- `input` 可以为空，表示没有额外上下文

**2. ShareGPT / Chat 格式（常用）**
```json
{
  "conversations": [
    {"from": "system", "value": "你是一个有帮助的助手"},
    {"from": "human", "value": "你好"},
    {"from": "gpt", "value": "你好！有什么可以帮助你的？"}
  ]
}
```
- 天然支持多轮对话
- `from` 字段区分角色
- 是 LLaMA-Factory 等主流框架的标准输入格式

**3. Messages 格式（HuggingFace 标准）**
```python
messages = [
    {"role": "system", "content": "你是医学助手"},
    {"role": "user", "content": "我头痛怎么办"},
    {"role": "assistant", "content": "头痛原因很多..."}
]
```
- 直接喂给 `tokenizer.apply_chat_template()`
- 是 HuggingFace 生态的通用格式
- 支持任意多轮和任意 role

**在项目中的处理流程：**
```
原始数据（json/csv） -> 清洗 -> 转成 messages 格式 -> 
tokenizer.apply_chat_template() -> input_ids + labels -> 训练
```

在医学项目中，数据可能来自多个来源（医学问答网站、教师模型生成等），需要写一个统一的转换脚本，把所有来源的数据都转成 messages 格式，保证最终的 chat template 是一致的。

**注意**：
- 不同来源的数据 role 名称可能不一致（human/user、gpt/assistant），需要统一映射
- system prompt 是否应该放在每条数据中？——最好放在数据预处理阶段统一加上，不要依赖训练代码中后加

---

## 四、SFT 训练技巧与常见问题

### 4.1 Q: 训练过程中 loss 一直下降，但评估效果不提升，甚至变差了，你觉得可能是什么原因？star:5

**Loss 下降但效果不提升（或反而变差）的可能原因：**

1. **过拟合 SFT 数据**：SFT 数据量少（几万条），但模型参数大（7B+），非常容易过拟合。模型背下了训练集但泛化差。表现：训练 loss 很低，验证 loss 不降甚至上升。

2. **数据多样性不足**：所有数据都是同一风格/主题，模型丧失了处理其他话题的能力。典型的医学场景：模型只会回答医学问题，问"今天天气怎么样"就乱说。

3. **灾难性遗忘**：SFT 训练让模型"忘记"了预训练学到的通用能力。表现：模型在通用 benchmark（如 MMLU、C-Eval）上的分数大幅下降。

4. **数据质量差**：数据中 assistant 回答本身就有问题（错误信息、矛盾），loss 低只是因为模型学会了"复读这些错误答案"。

5. **没有正确的评估体系**：只用 loss 看训练进展是不够的。loss 只能说明模型学会了训练数据的模式，不能说明回答质量好。

**如何缓解：**

- **Early Stopping**：在验证集上监控 loss/perplexity，不只看训练 loss。通常 1-3 个 epoch 就够，多了反而过拟合
- **降低学习率**：SFT 学习率建议 1e-6 到 5e-6（比 CPT 低一个数量级），过高的学习率容易破坏预训练权重
- **加正则化**：weight decay、dropout 等（但 LLM 中不太常用）
- **数据混合**：SFT 数据中混入少量通用数据（如 10-20%），帮助模型保持通用能力
- **Multi-task SFT**：不只是医学问答，混入摘要、翻译等通用任务，保持模型的指令遵循能力
- **使用 LoRA 而不是全参数微调**：LoRA 只调整少量参数，天然对原始权重有保护作用，灾难性遗忘更小

在医学项目中，可能会观察到：模型在医学测试集上好了，但在通用对话上变差了。这是典型的部分灾难性遗忘。可以通过（1）混合 10-20% 通用 SFT 数据（2）用 LoRA 微调（3）降低学习率 来缓解。

**注意**：
- 只看 training loss 就觉得训练好了——要同时看验证集和人工评估
- 数据量不小就狂训 10 个 epoch——SFT 阶段 2-3 个 epoch 已经足够

---

### 4.2 Q: 你在医学领域做 SFT，和通用 SFT 相比有什么特殊考虑？领域 SFT 有什么优缺点？star:4

**领域 SFT 的优点：**
1. **垂直领域效果好**：在特定领域的知识和表达方式上显著优于通用模型
2. **可控性强**：可以通过 system prompt 和数据设计精准控制回答风格
3. **数据利用率高**：几万条高质量领域数据就能产生明显效果

**领域 SFT 的缺点/风险：**
1. **灾难性遗忘**：模型在非目标领域的通用能力下降
2. **领域偏见**：模型过度依赖训练数据中的特定观点，缺乏多样性
3. **数据偏差放大**：如果训练数据有某种倾向（如过度诊断、过度开药），模型学得比数据更"极端"
4. **覆盖不全**：医学领域浩如烟海，几万条数据不可能覆盖所有情况
5. **安全风险**（见下一节）

**缓解策略：**
- 数据混合（领域:通用 = 8:2 或 7:3）
- LoRA 微调代替全参数微调
- 在 system prompt 中增加安全声明
- 保留基础模型的"拒绝回答"能力（不要把所有拒绝回答的样本都过滤掉）

我的医学 LLM 项目中，本质上做的就是领域 SFT。面试时可以说："我做的医学 SFT 针对医学问答场景进行了专门优化，同时通过数据混合和 LoRA 微调缓解了通用能力的退化。"

**注意**：把领域 SFT 和领域预训练（CPT）混为一谈——前者是行为对齐，后者是知识注入。

---

## 五、领域 SFT 与医学 SFT 专题

### 5.1 Q: 医疗场景下做 SFT 有什么特别需要注意的安全风险？你是怎么处理的？star:5

医疗 SFT 的安全风险比通用 SFT 高得多，因为错误回答可能直接危害用户健康。主要风险：

**1. 错误诊断风险**：模型可能因训练数据偏差，把常见症状误判为罕见重病。比如：用户说"头痛"，模型回答"可能是脑瘤"，造成不必要的恐慌。

**2. 危险用药建议**：模型可能建议两种有相互作用的药物同时服用，可能给出错误的剂量建议（如成人剂量建议给儿童），可能建议禁用药物给特定人群（如孕妇）。

**3. 过度自信**：模型可能用确定性的口吻给出不确定的判断，误导用户。如："你的症状一定是XXX"——实际还需要进一步检查。

**4. 隐私泄露**：训练数据中可能包含真实病历信息，模型在推理时可能"回忆"出训练数据中的真实患者信息。

**5. 法律风险**：在中国，提供医疗建议需要执业医师资格，AI 模型的建议只能作为参考，必须有免责声明。

**应对措施（面试时重点说）：**
1. **训练阶段**：在 system prompt 中明确"我是一个 AI 助手，不能替代医生诊断"；在训练答案中加入"建议就医"、"请咨询专业医生"等医学谨慎性措辞；过滤掉明确诊断结论、用药剂量、手术建议等高风险内容
2. **数据安全**：不使用真实病历作为训练数据；对训练数据做脱敏处理
3. **部署阶段**：在输出中添加免责声明；设置敏感词过滤（如具体药品剂量）；对于紧急症状（如胸痛、意识丧失），强制输出"请立即就医"

在医学项目中，在构造训练数据时就设计了带有医学谨慎性的回答风格，在 system prompt 和回答模板中都包含了免责声明和安全提示。这是一个非常加分的面试回答点。

**注意**：
- 认为"模型输出准确就好，不需要免责声明"——在法律和伦理上，免责声明是必须的，不管你模型多准
- 把安全责任完全推给 system prompt——system prompt 只是辅助，训练数据的质量和安全性才是根本

---

### 5.2 Q: 你的 SFT 数据是用 teacher 模型生成的，这种方式有什么优缺点？直接用原始答案不行吗？star:5

**优点：**
1. **质量可控**：Teacher 模型（如 GPT-4、Claude）生成的答案格式规范、结构清晰、语言流畅
2. **风格统一**：所有答案遵循同样的风格模板，模型学习到的回答风格一致
3. **可大规模扩充**：只要有 prompt，就能用 teacher 大量生成，打破了人工标注的瓶颈
4. **便于加入安全约束**：在 teacher prompt 中加入医学谨慎性、免责声明等要求，确保所有答案都合规

**风险（面试官最喜欢追问）：**
1. **模型同质化（Model Collapse）**：如果 student 只用 teacher 的输出做 SFT，student 学到的只是 teacher 的"风格和偏好"，而不是真正的医学知识。久而久之，整个生态的模型趋向于同一个 teacher 的行为模式
2. **Teacher 的错误被放大**：Teacher 模型也会有幻觉和错误，student 不仅学会这些错误，还可能把它们"放大"（因为 student 能力弱于 teacher）
3. **缺乏多样性**：Teacher 的答案通常过于工整，缺乏真实医患对话的多样性（如口语化、追问、不确定表达等）
4. **分布偏移**：Teacher 的输出风格（如总是分条列举、总以"总的来说"结尾）可能不符合真实使用场景
5. **缺乏真实的"不确定"表达**：Teacher 倾向于给出"确定感"很强的答案，即使信息不足也硬着头皮回答——这在医学场景很危险

在说完优点后**主动提到风险和你的应对措施**，会显得你有深度思考："我意识到了 teacher 数据的问题，所以我做了以下处理：1）在 teacher prompt 中要求加医学谨慎性表述；2）对生成的答案做了质量过滤和去重；3）混入少量真实医学问答数据增加多样性。"

**注意**：
- 只说 teacher 数据的优点不主动说风险——面试官会觉得你"只会用，不会想"
- 以为 teacher 模型越强越好——GPT-4 的回答对 7B 模型可能"太难学"，目标输出和模型能力不匹配反而是问题

---

### 5.3 Q: 你有原始医学问答数据，为什么不直接用原始答案做 SFT，而是要让 teacher 重新生成一遍？star:4

这是项目中一个非常好的设计决策，值得在面试中展开讲：

**1. 原始答案质量参差不齐**：医学问答网站上（如丁香园、好大夫）的医生回答质量差异巨大。有的医生回答过于简短（"没事，多喝水"），缺乏医学依据；有的医生倾向于开特定药厂的药，有商业偏见；有的回答拼写错误、格式混乱。

**2. 风格不统一**：不同医生的回答风格差异很大（口语化/专业化、确定/谨慎、简短/详细）。不做统一会导致模型学习到不稳定的回答风格，用户在每次对话中感受到"不同医生在回答"。

**3. 安全和合规性**：原始回答可能包含具体用药剂量、明确诊断结论。Teacher 模型生成的答案可以加入免责声明和安全约束。这相当于对原始数据做了一层"安全过滤"。

**4. 格式规范**：Teacher 生成的答案可以统一格式（结构化的分点回答、先分析后建议等），这对用户阅读体验有很大提升。

**5. 原始信息保留**：Teacher 可以根据原始答案重写，保留其中的医学核心信息。相当于做了一次"润色 + 安全审查 + 格式统一"。

面试话术："我选择用 teacher 生成答案而不是直接使用原始答案，主要是从质量统一、安全保障、格式规范三个角度考虑的。原始医学问答数据质量差异太大，直接训练会让模型学到不一致的回答模式。我的做法是用 teacher 参考原始答案生成一份格式统一、谨慎合规的回答，相当于做了一次智能数据清洗。"

**注意**：
- "原始答案质量太差所以扔了"——这不准确，应该说"用 teacher 对原始答案做了重写和质量提升"
- 不要暗示 teacher 重写时会丢失信息，重点是"保留核心医学知识 + 优化表达形式"

---

### 5.4 Q: Teacher 生成了答案之后，你怎么做质量过滤的？怎么避免训练数据污染测试集？star:5

**质量过滤：**
1. **长度过滤**：太短（<50 tokens）的答案可能没有实质内容，太长（>2000 tokens）的答案可能废话多或 teacher 在瞎编
2. **关键词过滤**：检查答案中是否包含免责声明、就医建议等必要安全语句，不合格的剔除
3. **格式检查**：检查答案是否完整（比如是否以完整句子结尾，而不是截断的），被截断的过滤
4. **语义质量**：用一个小模型或规则判断答案是否回答到了用户的问题。拒绝回答类的答案也要保留（因为"拒绝回答"也是一种重要的能力）
5. **安全过滤**：检测是否包含危险用药建议、明确诊断等，有则标记为"需人工复核"或直接过滤

**语义去重：**
1. **相似度去重**：用 embedding 模型（如 text2vec 或 BGE）将所有答案做向量化，计算 cosine similarity，去除相似度 > 0.95 的重复项
2. **n-gram 去重**：检查是否有大段重复文本（可能是 teacher 的模板化表达），去重或回退到内容更丰富的版本
3. **question-level 去重**：同样的用户问题出现在测试集中时要剔除对应训练数据

**避免 Eval Leakage（测试集污染）：**这是非常重要但很容易被忽略的问题。

1. **n-gram 重叠检测**：用模糊匹配检测训练集和测试集之间的 overlap。如果有 13-gram 重复，就可以认为是污染
2. **语义相似度检测**：不仅仅做 n-gram 匹配，还要用 embedding 做语义相似度检测，避免"换了说法但其实同一个问题"的污染
3. **数据来源隔离**：训练集和测试集的 user questions 必须来自不同来源/不同时间段
4. **在数据处理流程最前端做**：在 teacher 生成答案之前，就要先检查 question 是否与测试集重叠，避免被污染

面试话术："在 teacher 生成答案后，我做了多轮过滤：长度过滤、关键词检查、格式检查。去重方面用了 embedding 相似度做语义去重。最重要的，在数据处理最开始，我就把和测试集可能重叠的数据隔离出去，防止 eval leakage。"

**注意**：
- 先做 teacher 生成再做去重和防泄漏——如果 question 和测试集重叠了，teacher 生成的答案就浪费了，应该在数据准备的**最开始**就隔离测试集
- 只用 n-gram 去重——"怎么治疗头痛"和"头痛应该怎样治疗"在 n-gram 层面不同但语义相同

> **注意：本节中描述的 embedding 语义去重、LLM-as-judge 质量打分等高级过滤策略，可能在你当前项目代码中未完整实现。面试时可以说"这是我的过滤 pipeline 设计，部分模块已在项目中落地"——这样既展示了系统思维，又诚实。**

---

### 5.5 Q: 你怎么控制模型回答的长度和风格？医学场景下如何让模型回答更'人文'、更谨慎？star:4

**控制回答长度：**
1. **通过训练数据控制（最根本）**：训练数据中 assistant 回答的长度决定了模型学到的"默认回答长度"。所以在 teacher 生成时就设定目标长度
2. **通过 system prompt 控制**：在 system prompt 中加入"请用简洁的语言回答"或"请详细解释"，但仅靠 prompt 不够稳定
3. **通过采样参数控制**：max_new_tokens、repetition_penalty 等——但这是部署层的控制，不是训练层的
4. **训练时加入长度多样性**：不要让所有回答都是 200 字，混入不同长度（50 字/200 字/500 字）的样本，模型才能在不同场景下灵活调整

**加入人文关怀：**
- 在 teacher 生成 prompt 中加入："回答开头要对患者表达关心和共情"、"使用温和的语气"
- 训练数据中嵌入模板化的关怀语句，但注意不要变成千篇一律的"您好，我很理解您的心情"
- 更好的方式：用多样化的关怀表达，让模型学会自然地表达共情

**加入医学谨慎性：**
这是医学 SFT 最核心的要求。具体做法：
1. **免责声明嵌入**：每个涉及诊断/用药的回答末尾加上"以上仅为参考，请咨询专业医生"
2. **不确定性的表达**：训练数据中用"可能"、"建议进一步检查"、"不排除"等措辞
3. **紧急情况引导**：涉及胸痛、意识丧失等症状时，回答中强调"请立即就医"
4. **不越界**：不做明确诊断、不给具体药量、不建议手术

在项目中，很可能在 teacher 生成阶段就通过 prompt 控制了回答的"长度 + 风格 + 安全性"。这是一个非常好的做法——在数据源头就保证了质量。

**注意**：
- 只用 prompt 控制长度而不在训练数据中体现——模型会"不理睬"prompt 的长度要求
- 关怀语句过度模板化——"我很理解您的心情"这种万金油句子用多了反而令人反感

---

### 5.6 Q: 医学 LLM 最大的安全问题是给出错误的诊断和用药建议，你怎么从训练层面避免？star:5

这是一个非常有深度的问题，体现了安全意识和工程落地能力。

**避免过度诊断（Over-diagnosis）：**

1. **训练数据中平衡严重性和常见性**：不要让 90% 的头痛样本对应的回答都是"可能是脑瘤"或"可能是高血压危象"。常见症状 -> 常见病因的回答比例要合理：比如 100 个头痛样本中，70 个对应"压力/睡眠不足"，25 个对应"颈椎问题/偏头痛"，5 个对应"需排除严重病因"

2. **在回答中给出概率感**：错误示例："你头痛是因为脑瘤"（确定性诊断）；正确示例："头痛最常见的原因是压力和睡眠不足，但也可能与偏头痛、颈椎问题等有关。如果头痛持续加重或伴有呕吐等症状，建议就医检查"（不确定性 + 分层建议）

3. **加入"打回"训练**：训练数据中不仅要有"正常回答"，还要有对一些危险问题的"拒绝/打回"样本。比如用户问"我胸口疼了两天，吃什么药好"，assistant 回答不应是推荐药，而是"请立即就医，不要自行用药"

**避免危险用药建议：**

1. **系统性过滤**：在 teacher 生成阶段，明确要求 teacher "不要给出具体药品名称和剂量"
2. **药品名单过滤**：训练数据产出后，用正则匹配常见药品名，有具体剂量的一律打回重修
3. **训练拒绝能力**：保留一些这样的训练样本——User: "我怀孕了，肚子疼可以吃止痛药吗"，Assistant: "孕期用药需要特别谨慎，请务必咨询产科医生，不要自行服用任何药物。如果您腹痛持续或加重，请立即就医"
4. **不是完全不说药**：偶尔可以说"临床常用的降血压药包括ACEI类、CCB类等，具体用药需医生根据您的情况开具"——说了大类但没说具体药品，既展示了医学知识又不危险

面试话术："我在训练数据的设计阶段就做了安全控制。teacher 生成的 prompt 中明确要求：不做确定性诊断、不推荐具体药品和剂量、对紧急症状引导就医。同时，我保留了基础模型拒绝回答的能力——在训练数据中保留了约 5% 的'安全打回'样本。"

**注意**：
- 过度过滤导致模型什么都不敢说——医学助手也需要给有用信息，关键是"有用 + 安全"的平衡
- 以为靠 system prompt 就能解决安全问题——系统 prompt 可以被越狱攻击绕过，真正的安全得靠训练数据

---

### 5.7 Q: 介绍一下你在医学 LLM 项目中做的 SFT 工作。（STAR 法则，3-5 分钟）star:5

这是面试中的"大问题"，要在 3-5 分钟内完整地讲出你的工作，展示项目经验和技术深度。

**S（Situation - 背景）：**
"我在做的是一个医学中文问答的 LLM 项目。基础模型是 Qwen2.5/Qwen3（7B），目标是在医学问答场景下给出安全、专业、有人文关怀的回答。原始数据来自医学问答社区（如丁香园），但质量参差不齐。"

**T（Task - 任务）：**
"我的任务是：1）构造高质量的 SFT 训练数据；2）设计训练方案；3）解决医学场景下的安全和格式问题；4）确保训练-部署的一致性。"

**A（Action - 行动）：**
"我做了这几件事：
1. **数据清洗和 Teacher 生成**：用 teacher 模型（如 GPT-4 或 Qwen-Max）参考原始答案重新生成高质量、风格统一的回答。在 teacher prompt 中加入安全约束（免责声明、不确定性表达、就医引导）。
2. **数据过滤**：做了长度过滤、安全关键词检查、embedding 语义去重、测试集隔离防泄漏。
3. **Chat Template 统一**：全流程用 `tokenizer.apply_chat_template()`，保证从数据构造到训练到部署格式完全一致。对于 Qwen3，设置 `enable_thinking=False` 避免思考泄漏。
4. **SFT 训练**：用 Qwen2.5 作为基座模型，构造 messages 格式训练数据，assistant-only loss mask，学习率 2e-6，1-2 个 epoch，用 LoRA 微调以缓解灾难性遗忘。混入了 15% 的通用 SFT 数据保持通用能力。
5. **部署对齐**：vLLM 部署时用相同的 chat template，left padding 做 batch 推理。"

**R（Result - 结果）：**
"最终模型在医学问答测试集上取得了显著的 improvement，同时保持了良好的通用对话能力。安全测试中，模型能在 90%+ 的危险场景下给出安全建议而非危险回答。"

**加分细节：**
- 可以提到"我意识到 SFT 主要是行为对齐，所以在 SFT 之前做了 CPT 来注入领域知识"
- "训练过程中发现 loss 一直降但评估集效果不动，分析发现是 teacher 数据的风格太单一，加入了一些真实人类回答数据后改善明显"

**注意**：
- 讲得太技术细节化（如一直在说学习率、batch size）而不说项目为什么要这么做
- 讲得太宏观（"我做了一个医学AI助手"）而没有技术细节
- 不提安全考虑——医学项目不提安全性是大减分

---

## 六、Teacher 数据相关

### 6.1 Teacher 数据总结

**核心观点提炼（面试时一句话总结）：**

"我的 SFT 数据主要采用 teacher 模型生成的答案。用 teacher 的好处是质量可控、风格统一、便于加入安全约束；风险是模型同质化和 teacher 错误传播。我的应对是：在 teacher prompt 层面加强约束、对生成结果做多层过滤去重、混入少量真实数据增加多样性、数据最开始就隔离测试集防止 eval leakage。"

---

## 七、代码实现

> 以下代码为 SFT 训练中的核心模块实现，带有详细注释和输入/输出维度标注。这些代码体现了你对 SFT 底层实现的真正理解，面试中可能被要求手写或讲解。

### 7.1 SFT Loss（Causal LM Shift Loss）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def sft_loss(logits, labels, ignore_index=-100):
    """
    SFT 的因果语言模型损失（Causal LM Loss）
    
    核心思路：
    1. 每个位置的 logits 预测的是"下一个" token
    2. 因此要把 logits 和 labels 错开一位对齐
    
    Args:
        logits: 模型输出, shape = (batch_size, seq_len, vocab_size)
                例如: (4, 2048, 151936)
        labels: 标签, shape = (batch_size, seq_len)
                例如: (4, 2048)
                其中 system/user/pad 部分为 -100
        ignore_index: 忽略的标签值，默认为 -100
    
    Returns:
        loss: 标量，SFT 损失值
    
    对齐示意:
        Tokens:    [BOS]  我    今天   很    开心   [EOS]
        logits:    l0     l1    l2    l3    l4     l5
        predicts:  t1     t2    t3    t4    t5     (t6 丢弃)
        labels:   (-100)  t1    t2    t3    t4     t5
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Step 1: 错位对齐
    # logits 去掉最后一个位置（最后一个位置预测的 token 没有对应 label）
    # labels 去掉第一个位置（第一个位置的 label 没有对应的 logits 预测它）
    shift_logits = logits[..., :-1, :].contiguous()
    # shape: (batch_size, seq_len-1, vocab_size)
    # 例如: (4, 2047, 151936)
    
    shift_labels = labels[..., 1:].contiguous()
    # shape: (batch_size, seq_len-1)
    # 例如: (4, 2047)
    
    # Step 2: 展平后计算交叉熵
    # CrossEntropyLoss 需要 (N, C) 的 logits 和 (N,) 的 labels
    shift_logits_flat = shift_logits.view(-1, vocab_size)
    # shape: (batch_size * (seq_len-1), vocab_size)
    # 例如: (4 * 2047 = 8188, 151936)
    
    shift_labels_flat = shift_labels.view(-1)
    # shape: (batch_size * (seq_len-1),)
    # 例如: (8188,)
    
    # Step 3: 计算 loss
    # ignore_index=-100 使 loss 自动忽略 system/user/pad 部分的 token
    loss = F.cross_entropy(
        shift_logits_flat,
        shift_labels_flat,
        ignore_index=ignore_index,
        reduction='mean'  # 只对有效 token 取平均
    )
    # loss: 标量，例如 tensor(2.34)
    
    return loss


# ==================== 使用示例 ====================
# 假设:
# - batch_size = 2
# - seq_len = 6
# - vocab_size = 1000 (简化)
# - labels 中 -100 表示忽略, 其他为 token ID

# 模拟数据
logits_demo = torch.randn(2, 6, 1000)  # (2, 6, 1000)
labels_demo = torch.tensor([
    [-100, -100, -100, 42, 87, 99],     # 样本1: system/system/user/asst1/asst2/asst3
    [-100, -100, 55,   66, 77, -100]     # 样本2: system/user/asst1/asst2/asst3/(PAD无label但位置被EOS占了)
])

loss = sft_loss(logits_demo, labels_demo)
print(f"SFT Loss: {loss.item():.4f}")
# 输出示例: SFT Loss: 7.8234
```

---

### 7.2 Assistant-Only Loss Mask 构造

```python
def create_assistant_loss_mask(input_ids, tokenizer):
    """
    构造 SFT 训练的 labels，只对 assistant 部分计算 loss
    
    核心思路：
    1. 先找到 assistant 回复在序列中的起止位置
    2. 把非 assistant 部分的 label 设为 -100
    3. assistant 部分的 label 保持对应的 token ID
    
    Args:
        input_ids: 完整的 token 序列, shape = (seq_len,)
                   由 tokenizer.apply_chat_template() 生成
        tokenizer: HuggingFace tokenizer，需要包含 chat_template 信息
    
    Returns:
        labels: shape = (seq_len,), dtype = long
                非 assistant 部分为 -100, assistant 部分为对应 token ID
    
    示例输入:
        input_ids = [bos, sys1, sys2, user1, user2, asst1, asst2, eos]
        system prompt 占 [1:3], user message 占 [3:5], assistant 占 [5:8]
    
    示例输出:
        labels = [-100, -100, -100, -100, -100, asst1, asst2, eos]
    """
    seq_len = input_ids.shape[0]
    
    # Step 1: 初始化 labels 为全 -100
    labels = torch.full_like(input_ids, fill_value=-100)
    # shape: (seq_len,)
    # 例如: [-100, -100, -100, -100, -100, -100, -100, -100]
    
    # Step 2: 找到 assistant 回复的起止位置
    # 方法: 通过查找特殊 token（如 <|im_start|>assistant）来定位
    # 
    # 具体实现取决于你的数据构造方式。以下是两种常见方法：
    
    # === 方法 A: 在构造数据时就记录位置（推荐）===
    # 在 apply_chat_template 之前/同时，记录每个 role 在序列中的起止位置
    # 例如:
    # messages = [
    #     {"role": "system", "content": "...", "start": 0, "end": 50},
    #     {"role": "user", "content": "...", "start": 50, "end": 120},
    #     {"role": "assistant", "content": "...", "start": 120, "end": 300}
    # ]
    # 然后 labels[120:300] = input_ids[120:300]
    # 
    # 优点: 精确、高效
    
    # === 方法 B: 通过特殊 token ID 定位（通用但较慢）===
    # 找到 <|im_start|>assistant 的 token ID
    # Qwen2 的 assistant 开始标记通常由多个 token 组成
    # im_start_token = tokenizer.encode("<|im_start|>")[0]
    # assistant_token = tokenizer.encode("assistant")[0]
    # 
    # 在 input_ids 中查找 "<|im_start|>assistant" 模式
    # 然后从这个位置的下一个 token 开始，直到遇到 <|im_end|> 或序列结束
    
    # === 方法 C: 最简化版本（如果数据构造方式支持）===
    # 如果已知 assistant 内容始终在序列的最后一段
    # 可以通过找到最后一个 <|im_start|> 来确定 assistant 的起始位置
    # 
    # 以下用简化示例示意:
    
    # 假设在构造数据时已经知道 assistant 部分的起始位置
    # assistant_start_idx = ...  # 由数据构造流程给出
    # assistant_end_idx = ...    # 由数据构造流程给出
    # labels[assistant_start_idx:assistant_end_idx] = input_ids[assistant_start_idx:assistant_end_idx]
    
    # 返回构造好的 labels
    return labels


# ==================== 更实用的实现（基于模板匹配）====================
def create_assistant_loss_mask_from_template(input_ids, tokenizer):
    """
    基于模板匹配构造 assistant-only labels
    
    适用于 Qwen2/Qwen3 的 chat template:
    <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>
    
    Args:
        input_ids: shape (seq_len,), 完整的 token 序列
        tokenizer: HuggingFace tokenizer
    
    Returns:
        labels: shape (seq_len,), dtype = long
    """
    seq_len = input_ids.shape[0]
    labels = torch.full_like(input_ids, fill_value=-100)
    
    # 编码特殊标记
    im_start_id = tokenizer.encode("<|im_start|>")[0]
    # 注意：Qwen 的 tokenizer 中 "<|im_start|>" 可能是一个 token 也可能是多个
    # 实际情况中需要检查 tokenizer 的 tokenization 结果
    
    # 找到所有的 <|im_start|> 位置
    # 最后一个 <|im_start|> 之后的就是 assistant 回复
    im_start_positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    
    if len(im_start_positions) >= 3:
        # 有 system, user, assistant 三个角色
        # 最后一个 <|im_start|> 之后是 assistant
        assistant_start = im_start_positions[-1].item()
        
        # assistant 内容从 <|im_start|>assistant\n 之后开始
        # 实际起始位置需要跳过 "<|im_start|>assistant\n" 这部分 token
        # 简化处理：从下一个位置开始就是 assistant 内容 token
        # 更精确的做法是找到 assistant\n 之后的第一个非特殊 token
        assistant_content_start = assistant_start + 3  # 跳过 <|im_start|>assistant\n
        
        # assistant 结束位置到 <|im_end|>（包含）
        # 从后往前找第一个非 -100 的位置作为结束
        assistant_content_end = seq_len
        
        # 设置 labels
        labels[assistant_content_start:assistant_content_end] = \
            input_ids[assistant_content_start:assistant_content_end]
    
    return labels
```

---

### 7.3 Batch Collator（Padding + Labels 构造）

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch

@dataclass
class SFTDataCollator:
    """
    SFT 训练的 Data Collator
    
    功能:
    1. 对 batch 内的样本做 right padding（训练用）或 left padding（推理用）
    2. 构造 labels（非 assistant 部分设为 -100）
    3. 构造 attention_mask
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度
        padding_side: "right" 用于训练, "left" 用于推理
    """
    tokenizer: object
    max_length: int = 2048
    padding_side: str = "right"  # 训练用 right padding
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: list of dict, 每个 dict 包含:
                - input_ids: List[int], token IDs
                - labels: List[int], labels（含 -100）
        
        Returns:
            batch: dict 包含:
                - input_ids: (batch_size, max_seq_len)
                - attention_mask: (batch_size, max_seq_len)
                - labels: (batch_size, max_seq_len)
        
        示例输入 (batch_size=2):
            features = [
                {"input_ids": [1, 2, 3], "labels": [-100, -100, 4]},
                {"input_ids": [5, 6, 7, 8, 9], "labels": [-100, -100, 10, 11, 12]}
            ]
        
        示例输出 (max_length=6, padding_side="right"):
            batch = {
                "input_ids": [[1, 2, 3, PAD, PAD, PAD],
                              [5, 6, 7, 8, 9,  PAD]],
                "attention_mask": [[1, 1, 1, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 0]],
                "labels": [[-100, -100, 4, -100, -100, -100],
                           [-100, -100, 10, 11,  12,   -100]]
            }
        """
        batch_input_ids = []
        batch_labels = []
        
        # Step 1: 收集 input_ids 和 labels
        for feature in features:
            batch_input_ids.append(feature["input_ids"])
            batch_labels.append(feature["labels"])
        
        # Step 2: Padding（根据 padding_side 决定 left 还是 right）
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        padded_input_ids = self._pad_sequences(
            batch_input_ids, 
            pad_value=pad_token_id,
            max_length=self.max_length,
            padding_side=self.padding_side
        )
        # shape: (batch_size, max_seq_len)
        # 例如: (4, 2048)
        
        # Step 3: Padding labels（-100 填充）
        padded_labels = self._pad_sequences(
            batch_labels,
            pad_value=-100,  # label 的 pad 值必须是 -100（ignore_index）
            max_length=self.max_length,
            padding_side=self.padding_side
        )
        # shape: (batch_size, max_seq_len)
        # 例如: (4, 2048)
        
        # Step 4: 构造 attention_mask
        # attention_mask = 1 对应有效 token, 0 对应 PAD token
        attention_mask = (padded_input_ids != pad_token_id).long()
        # shape: (batch_size, max_seq_len)
        # 例如: (4, 2048)
        # 
        # 注意: 如果 pad_token_id 和某些真实 token 重合（不太可能但理论上）
        # 需要用更严格的方式（如基于实际 PAD 位置）
        
        # Step 5: 截断处理（如果序列超过 max_length）
        # 训练时通常在数据预处理阶段就截断好了，collator 只做 padding
        # 但为了安全，这里也做一次检查
        if padded_input_ids.shape[1] > self.max_length:
            padded_input_ids = padded_input_ids[:, :self.max_length]
            padded_labels = padded_labels[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]
        
        return {
            "input_ids": padded_input_ids,       # (batch_size, max_seq_len)
            "attention_mask": attention_mask,    # (batch_size, max_seq_len)
            "labels": padded_labels,             # (batch_size, max_seq_len)
        }
    
    def _pad_sequences(
        self, 
        sequences: List[List[int]], 
        pad_value: int,
        max_length: int,
        padding_side: str = "right"
    ) -> torch.Tensor:
        """
        对不等长序列做 padding
        
        Args:
            sequences: List of List[int], 不等长的序列列表
            pad_value: 填充值
            max_length: 目标长度
            padding_side: "right" 或 "left"
        
        Returns:
            padded: shape = (batch_size, max_length), dtype = long
        """
        batch_size = len(sequences)
        padded = torch.full(
            (batch_size, max_length), 
            fill_value=pad_value, 
            dtype=torch.long
        )
        
        for i, seq in enumerate(sequences):
            seq = seq[:max_length]  # 截断
            seq_len = len(seq)
            
            if padding_side == "right":
                # Right padding: [token1, token2, ..., tokenN, PAD, PAD, ...]
                padded[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
            elif padding_side == "left":
                # Left padding: [PAD, PAD, ..., token1, token2, ..., tokenN]
                padded[i, -seq_len:] = torch.tensor(seq, dtype=torch.long)
        
        return padded


# ==================== 使用示例 ====================
"""
# 在 HuggingFace Trainer 中使用:

from transformers import Trainer

collator = SFTDataCollator(
    tokenizer=tokenizer,
    max_length=2048,
    padding_side="right"  # SFT 训练用 right padding
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,  # 传入自定义 collator
    tokenizer=tokenizer,
)
"""
```

---

### 7.4 Labels 构造流程（完整的端到端示例）

```python
import torch
from transformers import AutoTokenizer

def construct_sft_sample(
    tokenizer,
    system_prompt: str,
    user_query: str,
    assistant_answer: str,
    max_length: int = 2048
):
    """
    端到端的 SFT 样本构造流程
    
    从原始对话到可以直接训练的 (input_ids, labels) 对
    
    Args:
        tokenizer: HuggingFace AutoTokenizer
        system_prompt: 系统提示词
        user_query: 用户问题
        assistant_answer: 助手回答
        max_length: 最大序列长度（用于截断）
    
    Returns:
        input_ids: torch.Tensor, shape = (seq_len,)
        labels: torch.Tensor, shape = (seq_len,)
        attention_mask: torch.Tensor, shape = (seq_len,)
    
    完整流程:
    Step 1: 构造 messages 列表
    Step 2: apply_chat_template 得到完整 token 序列
    Step 3: 识别 assistant 部分位置
    Step 4: 构造 labels（非 assistant 设为 -100）
    Step 5: 截断处理
    """
    
    # ============ Step 1: 构造 messages 列表 ============
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant_answer}
    ]
    
    # ============ Step 2: Apply Chat Template ============
    # 先把 messages 渲染成格式化文本（不 tokenize，用于找到 assistant 位置）
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 先看文本，方便 debug
        add_generation_prompt=False
    )
    # formatted_text 例如:
    # "<|im_start|>system\n你是医学助手<|im_end|>\n<|im_start|>user\n头痛怎么办<|im_end|>\n<|im_start|>assistant\n建议就医...<|im_end|>"
    
    # 现在 tokenize
    full_input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False
    ).squeeze(0)
    # shape: (seq_len,)
    # 例如: (285,) 表示总共 285 个 token
    
    seq_len = full_input_ids.shape[0]
    
    # ============ Step 3: 构造只有 assistant 的 token 序列用于定位 ============
    # 这个小技巧: 用只有 assistant 角色的 message 来获取 assistant 回答的 token IDs
    assistant_only = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": assistant_answer, "only_content": True}],
        # 注意这只是一个示意——实际上需要自己写逻辑来获取 assistant 的纯内容 token
        # 常用方法: 直接对比完整序列和 "system+user" 序列来找到 assistant 的起止位置
        tokenize=False,
        add_generation_prompt=False
    )
    
    # 更实际的方法: 用不含 assistant 的 messages 来定位
    messages_without_assistant = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    prompt_only_ids = tokenizer.apply_chat_template(
        messages_without_assistant,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True  # 注意: 加了这个后会产生一个引导 assistant 开始的标记
    ).squeeze(0)
    # shape: (prompt_len,)
    # 这个序列的长度就是 assistant 回答在完整序列中的起始位置
    
    assistant_start_idx = prompt_only_ids.shape[0]
    
    # ============ Step 4: 构造 labels ============
    labels = torch.full_like(full_input_ids, fill_value=-100, dtype=torch.long)
    # shape: (seq_len,)
    # 初始全部为 -100
    
    # assistant 部分用真实的 token IDs
    labels[assistant_start_idx:] = full_input_ids[assistant_start_idx:].clone()
    # labels 例如: [-100, -100, ..., -100, t150, t151, ..., t285]
    #                       ^ assistant_start_idx
    
    # ============ Step 5: Truncation & Attention Mask 构造 ============
    # 截断
    if seq_len > max_length:
        # 从左边截断（保留最近的对话）
        # 注意: 截断后需要同步调整 labels 中 assistant 的位置
        full_input_ids = full_input_ids[-max_length:]
        labels = labels[-max_length:]
        seq_len = max_length
    
    # Attention mask（如果后面在 collator 中处理，这里可以先不设）
    attention_mask = torch.ones_like(full_input_ids, dtype=torch.long)
    # shape: (seq_len,)
    
    return full_input_ids, labels, attention_mask


# ==================== 使用示例 ====================
"""
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

system_prompt = "你是一个专业的医学助手，请谨慎回答，不要给出确定的诊断。"
user_query = "我最近总是头痛，应该怎么办？"
assistant_answer = "头痛可能由多种原因引起，包括压力、睡眠不足、颈椎问题等。建议您保持良好作息，如果症状持续或加重，请及时就医检查。以上建议仅供参考，请咨询专业医生。"

input_ids, labels, attention_mask = construct_sft_sample(
    tokenizer=tokenizer,
    system_prompt=system_prompt,
    user_query=user_query,
    assistant_answer=assistant_answer,
    max_length=2048
)

print(f"input_ids shape: {input_ids.shape}")
print(f"labels shape: {labels.shape}")
print(f"Non -100 labels count: {(labels != -100).sum().item()}")
print(f"-100 labels count: {(labels == -100).sum().item()}")
"""
```

---

## 八、面试版本回答

### 8.1 面试 1 分钟回答版本

> **"请介绍一下你的 SFT 项目经验和理解"**

"SFT 叫指令微调，核心是在预训练模型上，用有监督的（问题，回答）对训练，让模型学会对话格式和任务行为。我在医学 LLM 项目中做了完整的 SFT pipeline：用 teacher 模型生成高质量医学问答答案，通过 `tokenizer.apply_chat_template()` 统一格式化，构造 assistant-only loss mask 进行训练。为了保证安全，我在 teacher prompt 中加入了医学谨慎性要求，对生成答案做了安全过滤。训练-部署全链路用同一个 chat template，用 Qwen2.5 作为基座、LoRA 微调来缓解灾难性遗忘。最终模型在医学问答上表现良好，同时保持了安全合规的回答风格。"

### 8.2 面试 3 分钟回答版本

> **"详细介绍一下你的 SFT 工作"**

"SFT 是让预训练模型学会'对话行为'的关键阶段。它和 CPT 不同——CPT 用无标注文本灌知识，SFT 用问答对教行为模式——一个是知识注入，一个是行为对齐。

我的医学 LLM 项目里，SFT pipeline 包含五个关键环节。

第一是数据准备。原始数据来自医学问答社区，但质量参差不齐。我用 teacher 模型参考原始答案重新生成统一的答案，在生成 prompt 中嵌入了医学谨慎性约束——不说确定诊断、不推荐具体药物、加免责声明。生成后做了多层过滤：长度过滤、关键安全词检查、embedding 语义去重，并且在流程最开始就隔离了测试集防止 eval leakage。

第二是 Chat Template 一致性。从数据构造到训练到部署，我全链路使用 `tokenizer.apply_chat_template()`，避免手写格式导致的分布外问题。对于 Qwen3 模型我设置了 `enable_thinking=False`，防止 `<think>` 标签泄漏到用户端。

第三是 Loss 设计。我实现了因果语言模型 loss，关键点是 logits 和 labels 的 shift 对齐——每个位置的 logits 预测下一个 token。然后构造 assistant-only loss mask，把 system 和 user 部分的 label 设为 -100。这样做确保模型只学'看到问题如何回答'，而不是复读问题。Padding 位置也要设为 -100。

第四是训练策略。我用 Qwen2.5 作为基座，LoRA 微调（因为全参数微调对 7B 模型来说灾难性遗忘风险大），学习率 2e-6，1-2 个 epoch。混入了约 15% 的通用 SFT 数据来维持通用能力。训练中监控的不只是 loss，还有验证集的医学指标和通用能力变化。

第五是部署对齐。vLLM 部署时使用同样的 chat template，推理用 left padding 保证 batch 推理的正确性。

最终结果上，模型在医学问答上表现专业且安全，能在 90%+ 的危险场景下给出安全建议。我印象比较深的一个教训是，有一版模型 loss 一直降但评估效果不动，后来发现是 teacher 数据风格过于单一，混入一些真实人类回答数据后明显改善。"

---

## 九、背诵版总结

### 核心概念一句话速记

1. **SFT**：用（问题, 回答）对训练模型学对话行为，不是知识注入，是行为对齐
2. **CPT vs SFT**：CPT 用无标注文本灌知识，SFT 用标注数据教行为
3. **Teacher Forcing**：训练时用正确答案作为下一步输入，不等模型自己预测的结果
4. **Causal LM Loss**：每个位置预测下一个 token 的交叉熵之和，shift 一位对齐
5. **Shift**：`shift_logits = logits[:, :-1, :]`, `shift_labels = labels[:, 1:]`
6. **Assistant-Only Loss**：user/system 部分 label=-100，assistant 部分 label=真实 token ID
7. **-100**：PyTorch `CrossEntropyLoss` 的 `ignore_index`，这些位置不参与 loss 计算
8. **Attention Mask vs Loss Mask**：Attention Mask 控制谁参与 attention 计算；Loss Mask（labels=-100）控制谁算 loss。是两回事
9. **Right Padding 训练 / Left Padding 推理**：推理用 left 是确保生成位置永远在序列最后
10. **Packing**：拼短样本提高 GPU 利用率，但需要用 block-diagonal attention mask 防止串扰
11. **数据格式**：Alpaca（单轮）、ShareGPT（多轮）、Messages（HuggingFace 通用）
12. **Loss 降但效果差**：过拟合、数据单一、灾难性遗忘、评估不合理——至少这四种可能性
13. **灾难性遗忘**：SFT 导致通用能力下降，缓解：LoRA、数据混合、低学习率、early stopping
14. **Teacher 数据**：质量统一但风险是模型同质化和错误传播
15. **Eval Leakage**：训练前就隔离测试集，不是训完后才检查

### 项目亮点一句话

"我用 Qwen2.5 做医学 SFT，核心亮点是：全链路 chat template 一致、assistant-only loss 设计、teacher 数据 + 安全约束 + 多层过滤、LoRA 微调缓解遗忘、训练部署一致。"

### 经典踩坑速记

- 手写 chat template -> 格式偏差 -> 部署回答以空格开头
- 忘了 shift logits -> loss 算错一位 -> 模型输出一直是 `[UNK]`
- user 部分也算了 loss -> 模型变成复读机
- 用 right padding 做 batch 推理 -> 最后一个 token 是 PAD -> 乱输出
- SFT 训 10 个 epoch -> 严重过拟合 -> 只会背数据
- Teacher 数据不过滤 -> 模型学会 teacher 的幻觉
- 测试集被训练数据污染 -> 自欺欺人的"高分"

### 面试前一页 PPT

```
SFT = 预训练模型 + (问题,回答)对 + assistant-only loss

目标: 行为对齐（非知识注入）

关键实现:
+-- Loss: Causal LM + shift + assistant-only mask
+-- 数据: Teacher 生成 + 安全过滤 + chat template 统一
+-- 训练: LoRA + 数据混合 + 低学习率 + 1-3 epoch
+-- 部署: 同 chat template + left padding + enable_thinking=False

安全（医学特有）:
+-- 不明确诊断、不推荐具体药品/剂量
+-- 加免责声明、紧急情况引导就医
+-- 保留拒绝回答能力（5% 安全打回样本）

常见问题:
+-- Loss 降效果不提升 -> 过拟合/数据单一
+-- 灾难性遗忘 -> LoRA + 混合数据
+-- Teacher 同质化 -> 混合真实数据
+-- 格式不一致 -> 全程 apply_chat_template
```
