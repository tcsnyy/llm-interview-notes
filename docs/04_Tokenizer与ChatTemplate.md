# 04 Tokenizer 与 Chat Template

---

## 一、Tokenizer 基础

### 1.1 Tokenizer 是什么

**面试官怎么问：**
> "解释一下 tokenizer 在 LLM 中的作用，你了解哪些常见的 tokenization 算法？"

**考察什么：**
你是否真正理解"模型不认识文字、只认识数字"这个基本事实，以及你是否知道不同分词算法的优劣和使用场景。

**标准回答：**

Tokenizer 就是把自然语言文本转换成模型能理解的数字序列（token IDs）的组件，同时也是模型输出 token IDs 之后反向映射回文字的解码器。

可以理解成一个"翻译官"：用户写的是中文/英文，模型内部运算全是向量和数字，tokenizer 负责在二者之间做翻译。

核心流程：
- 训练前：用大量语料训练一个 tokenizer，得到词表（vocabulary）
- 编码（encode）：文本 → token IDs
- 解码（decode）：token IDs → 文本

LLM 领域主流的 tokenization 算法有四种：
1. **BPE（Byte-Pair Encoding）**：从字符级开始，反复合并频率最高的 token pair，直到达到目标词表大小。GPT 系列用的就是这个。
2. **WordPiece**：和 BPE 类似，但合并策略不是按频率而是按"互信息最大"——选能让训练语料似然提升最多的 pair 合并。BERT 用这个。
3. **SentencePiece**：一个框架式的工具，底层可以用 BPE 或 Unigram。关键在于它把空格也当成普通字符处理，不依赖英文天然的空格分词，所以对中文、日文等多语言更友好。LLaMA/Qwen 用的都是 SentencePiece + BPE。
4. **Byte-Level BPE**：BPE 的一种变体，不是从 Unicode 字符开始合并，而是从 byte（0-255）开始。好处是：绝对不会出现 UNK（未知字符），因为任何字符都能用 byte 序列表示。GPT-2 之后的 OpenAI 模型都用这个。

**可能追问：**
> "BPE 的训练过程具体是怎样的？"

**追问回答：**

BPE 的训练过程：
1. 把所有文本拆成字符序列，比如 "hello" → "h", "e", "l", "l", "o"，末尾加一个特殊的结束符（如 `</w>` 表示词末尾）
2. 统计所有相邻 pair 的频率
3. 把频率最高的 pair 合并成一个新的 token，比如 "l" + "l" → "ll"
4. 更新文本中所有出现这个 pair 的地方，用新 token 替换
5. 重复步骤 2-4，直到词表达到预设大小（例如 32000、50000、100000 等）

**项目结合：**
在 MiniMind 项目和医学 LLM 项目中，都使用了 Qwen2 系列的 tokenizer（SentencePiece + BPE），词表大小约 151936。这意味着你的项目实际在用工业级的分词方案，面试时可以强调这个体量。

**易错点：**
- 把 tokenizer 当成简单的"按词切分"——BPE 是子词级的，一个字可能被切成多个 token
- 认为中文一个字就是一个 token——在 BPE 下，"我吃苹果"可能变成 ["我", "吃", "苹", "果"] 或 ["我吃", "苹果"]，取决于词表
- 混淆 tokenizer 和 embedding——tokenizer 输出整数 ID，embedding 把这个 ID 映射成向量，是两步

---

### 1.2 BPE 深入：Byte-Level BPE vs 普通 BPE

**面试官怎么问：**
> "普通 BPE 和 byte-level BPE 有什么区别？为什么 GPT-2 之后要改用 byte-level？"

**标准回答：**

普通 BPE 是在 Unicode 字符级别上合并的，也就是说，如果训练语料里没见过某个字符（比如罕见的中文字、特殊符号），它就没办法编码，只能输出 UNK。

Byte-Level BPE 是从 byte（0-255）粒度开始的——任何 Unicode 字符在底层都是一串 bytes（UTF-8 编码）。这意味着：
- **永远不会出现 UNK**：哪怕是个火星文符号，它也有对应的 UTF-8 bytes 序列，就能编码
- 词表里有 256 个基础 token（对应所有 byte），然后在这之上用 BPE 合并
- 缺点：同一个字符可能被切成多个 byte-level token，导致 token 序列变长

**举例：**
- "A" 的 UTF-8 编码是 `[65]`，刚好 1 个 byte
- "中" 的 UTF-8 编码是 `[228, 184, 173]`，3 个 bytes
- 在 byte-level BPE 早期阶段，"中" 会被切成 3 个基础 token，训练多了可能合并成一个

**易错点：**
容易说"byte-level BPE 就是 SentencePiece"，这是两个概念：SentencePiece 是一个 tokenizer 训练框架，既可以实现 BPE 也可以实现 Unigram；而 byte-level BPE 是一种具体的算法策略。

---

### 1.3 Special Tokens

**面试官怎么问：**
> "你项目里用了哪些 special tokens？BOS/EOS/PAD/UNK 分别是什么意思？为什么要加这些？"

**标准回答：**

| Token    | 全称                   | 作用                                                                 |
| -------- | ---------------------- | -------------------------------------------------------------------- |
| BOS      | Beginning of Sequence  | 标记序列开头。GPT 类因果模型通常不需要，因为自回归生成不需要开始信号  |
| EOS      | End of Sequence        | 标记序列结束。**非常重要**：模型遇到 EOS 才停止生成，没有 EOS 会无限生成下去 |
| PAD      | Padding                | 填充用，把不同长度的句子补到同一长度方便 batch 训练。PAD 对应的 label 必须设为 -100 忽略 |
| UNK      | Unknown                | 未知 token。byte-level BPE 下理论上不会出现 UNK，但词表中通常仍然保留 |
| SEP      | Separator              | BERT 时代的分隔符，GPT 类模型用 role/format 标记替代                 |

**项目结合：**
在医学 LLM 和 MiniMind 项目中，Qwen2 tokenizer 的 EOS token 是 `<|im_end|>`，PAD token 通常没有专用 token，而是直接用 EOS 或设置 `pad_token = eos_token`。

**易错点：**
- 忘记设 `pad_token`：很多 tokenizer（尤其是 LLaMA 系列）默认没有 pad_token，如果不设，tokenizer 调用时会报错。常见的做法是 `tokenizer.pad_token = tokenizer.eos_token`
- 混淆 BOS 和 PAD：PAD 只在 batch 训练时出现，BOS 在序列开头；GPT 类模型中通常没有 BOS，但 Qwen 有 `<|im_start|>`

---

### 1.4 中文 Tokenizer 的问题和医学文本 Tokenization

**面试官怎么问：**
> "中文场景下 tokenizer 有什么特殊问题？你做医学 LLM 的时候遇到 tokenization 相关的坑吗？"

**考察什么：**
你是否知道 token 效率的概念，以及是否在实践中遇到过且解决过 tokenization 问题。

**标准回答：**

中文 tokenizer 的几个核心问题：

**1. Token 效率低（Token Fertility）：**
英文天然用空格分词，一个英文单词通常 1-2 个 token；中文没有空格，在 BPE 体系下一个词可能被切成多个 token。比如 "高血压" 可能被切成 ["高血", "压"] 或 ["高", "血压"]，取决于词表。

Token 效率的衡量指标叫 **Fertility**（每个语义单位被切成的平均 token 数）。中文文本的 fertility 显著高于英文，意味着同样的语义信息，中文需要更多 tokens，推理成本更高。

**2. 医学/专业术语被过度切分：**
这是做医学 LLM 的核心痛点。

举例（假设用的是 Qwen2 tokenizer）：
- "冠心病" → 可能切成 ["冠", "心病"]（不合理）
- "阿司匹林" → 可能切成 ["阿", "司", "匹", "林"]（四个 token！）
- "新型冠状病毒" → 在疫情的语料多了之后可能是一个 token，但 "甲型H1N1流感" 仍然会被切碎

**影响：**
- 推理成本增加：一个医学术语本来用 1 个 token 就能表达，结果用了 5 个，回答同样内容的成本就翻了几倍
- 语义碎片化：模型看到的是碎片化的 token 序列，"阿"/"司"/"匹"/"林" vs "阿司匹林"，模型需要自行拼凑语义，增加了学习难度
- 知识检索困难：如果做 RAG，被切碎的术语在向量检索时也面临同样问题

**3. 如何缓解：**
- 方案一（推荐但工作量大）：扩充词表（vocab expansion），在基础 tokenizer 上用医学语料继续训练 BPE，让医学高频词成为独立 token
- 方案二（工业界常用）：直接用，但是用更大的上下文长度弥补 token 效率低的问题
- 方案三（训练时优化）：在 prompt 中明确给出医学术语的全称和俗称，帮助模型理解

**项目结合：**
在你的医学 LLM 项目中，如果没有做过 vocab expansion（标注即可），可以直接说："我项目中使用了 Qwen2 的原始 tokenizer，没有做 vocab expansion。面试官如果要追问，可以说这是一个可以优化的方向——如果能收集足够多的医学领域文本，用 SentencePiece 继续训练 BPE，扩充词表，可以显著提升 token 效率和医学实体理解。"

**可能追问：**
> "如果让你做 vocab expansion，你会怎么实现？"

**追问回答：**
1. 收集医学领域语料（中文医学教科书、临床指南、医学论文等）
2. 用 SentencePiece 在原 tokenizer 的模型上做增量训练，新增 5000-10000 个医学专用 token
3. 注意：新增 token 意味着 embedding 矩阵和 LM head 矩阵都要扩展——embedding 新增行、LM head 新增列
4. 初始化：新增 embedding 可以用原有相关 token 的 embedding 的平均值初始化，或者随机初始化后用少量数据 fine-tune
5. 风险：vocab expansion 后必须重新做一轮 CPT（continue pre-training），否则新增 token 的 embedding 是未训练的，模型表现会很差

**易错点：**
- 以为中文一个字就是一个 token——实际上 BPE 下中文一个字经常被切成多个 token
- 认为 vocab expansion 只是简单加几个 token——embedding 矩阵需要扩展并且重新训练

---

## 二、Chat Template

### 2.1 Chat Template 是什么

**面试官怎么问：**
> "你了解 chat template 吗？SFT/DPO 训练为什么必须用 chat template？"

**标准回答：**

**Chat Template 本质上是一个格式化函数**，把对话结构（system prompt、user message、assistant response 的列表）转换成模型在预训练阶段见过的格式的纯文本字符串。

为什么需要它：
- 模型在预训练阶段没见过 "你好啊" 这种原始对话格式
- 模型见过的是带特定标记（如 `<|im_start|>user\n你好啊<|im_end|>`）的格式化文本
- Chat template 就是把对话转成模型"认识"的格式

**统一格式的重要性：**
不同模型的 chat template 完全不同：

| 模型家族         | Chat Template 风格                                           |
| ---------------- | ------------------------------------------------------------ |
| Qwen2/Qwen3     | `<\|im_start\|>system\n...<\|im_end\|>\n<\|im_start\|>user\n...<\|im_end\|>` |
| LLaMA 3          | `<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>\n\n...<\|eot_id\|>` |
| ChatML            | `<\|im_start\|>system\n...<\|im_end\|>\n<\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n...<\|im_end\|>` |
| DeepSeek          | `User: ...\n\nAssistant: ...`                                |

**为什么 SFT/DPO 必须用一致的 chat template：**

一句话总结：**你的训练数据长什么样，模型就学什么样。如果你训练时候用格式 A，上线部署时用格式 B，模型看到不认识的格式就会产生分布外（OOD）行为。**

具体来说：
1. **格式对齐**：SFT 数据里的 `assistant` 部分在 chat template 里是被特定 token（如 `<|im_start|>assistant`）引导的，模型学到的就是"看到这些 token 就开始生成回答"
2. **停止条件**：模型学到的 EOS 是 `<|im_end|>`，部署时如果 chat template 不对，模型可能不知道什么时候停
3. **角色区分**：system prompt 和 user message 在 chat template 中通过不同标记区分，如果不统一，模型可能混淆角色

**项目结合：**
在医学 LLM 项目中，你的训练数据通过 Qwen2 的 `tokenizer.apply_chat_template()` 完成格式化，你自定义了 chat template 的开关（如设置 `enable_thinking=False`）。部署时（vLLM/Ollama）用同样的 chat template，保证了训练-部署一致性。

**可能追问：**
> "如果我自己写一个字符串拼接来构造 chat format 而不是用 apply_chat_template，会有什么问题？"

**追问回答：**

这就是很多新手踩过的坑。手写拼接的问题是：

1. **Special token 不会被正确转义**：如果你手写 `<|im_start|>` 但 tokenizer 不认识这串字符组合（因为 tokenizer 需要看到的是特殊 token 的 ID，而不是字符串），就会导致格式错误
2. **字符串和 token 不是一一对应的**：`apply_chat_template` 返回的是 token IDs，而手写字符串再 tokenize，可能会因为 tokenization 的边界问题导致格式标记被错误切分
3. **版本不一致**：模型日后的 tokenizer 更新可能会改变某些 token 的 ID，你手写的字符串拼接在版本升级后就失效了
4. **role 的顺序、system prompt 的位置、是否带 EOS** 等细节极其容易出错

---

### 2.2 tokenizer.apply_chat_template 的作用

**面试官怎么问：**
> "你在训练数据构造中是怎么用 apply_chat_template 的？它到底做了什么？"

**标准回答：**

`tokenizer.apply_chat_template(messages, tokenize=True)` 做的事：

1. **接收结构化输入**：一个 messages 列表，如：
```python
[
    {"role": "system", "content": "你是一个医学助手"},
    {"role": "user", "content": "我头痛怎么办"},
    {"role": "assistant", "content": "头痛原因很多，建议就医..."}
]
```

2. **调用 tokenizer 内部的 jinja 模板**：把上述结构化消息渲染成模型训练时见过的格式，例如 Qwen2 会渲染成：
```
<|im_start|>system
你是一个医学助手<|im_end|>
<|im_start|>user
我头痛怎么办<|im_end|>
<|im_start|>assistant
头痛原因很多，建议就医...<|im_end|>
```

3. **返回 token IDs + attention_mask**（如果 `tokenize=True`）

**为什么不能手写：**
见上一节（2.1 追问回答）。

**项目结合：**
你在构造训练数据时的标准流程：
```python
# 伪代码
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer}
]
tokenized = tokenizer.apply_chat_template(
    messages, 
    tokenize=True, 
    return_tensors="pt"
)
```

**易错点：**
- 有些模型（如 LLaMA 3）的 `apply_chat_template` 需要手动设置 `add_generation_prompt=True` 才能在 assistant 部分生成时留出位置
- `tokenize=True` 和 `tokenize=False` 的区别：前者返回 token IDs，后者返回格式化后的文本字符串（用于 debug）

---

### 2.3 Qwen3 Chat Template 特殊机制

**面试官怎么问：**
> "你做 Qwen3 项目的时候，chat template 里 enable_thinking 是用来干什么的？为什么要设为 False？"

**标准回答：**

Qwen3 是支持 thinking（思考链）的模型，它的 chat template 里有一个 `enable_thinking` 开关：

- **`enable_thinking=True`**：模型会先生成 `<think>...</think>` 内部的思考内容，然后再生成正式回答
- **`enable_thinking=False`**：模型直接生成回答，不产生思考过程

**为什么在医学 SFT 项目里设为 False：**

1. **训练目标明确**：你做的是医学问答 SFT，不是推理型任务。如果启用 thinking，模型在训练时会学到先写思考再写回答，但你的训练数据可能没有思考内容，造成格式不匹配
2. **推理效率**：thinking 过程会生成额外 token（可能几百到几千个），在医学问答这种"直接给答案"的场景下，这些思考 token 浪费推理时间和成本
3. **部署简化**：不启用 thinking，answers 更干净，不需要做 `<think>...</think>` 的后处理
4. **避免思考泄漏**：如果 thinking 里有患者隐私信息或医学推理错误，直接输出到前端是安全隐患

**具体实现：**
在 vLLM 部署时，chat template 里设置 `"enable_thinking": False`：
```python
# vLLM 部署时的参数
--chat-template /path/to/chat_template.jinja
# chat_template 中 enable_thinking=False
```

**易错点：**
- 有人在训练时忘了设 `enable_thinking=False`，结果模型随机生成 `<think>` 标签，导致格式混乱
- 有人在训练时用了 thinking，但部署时没配对应参数，结果 thinking 标签泄漏到用户端

---

### 2.4 `<think>` 泄漏是什么

**面试官怎么问：**
> "你听说过 think 泄漏吗？是怎么产生的，怎么避免？"

**标准回答：**

**`<think>` 泄漏**指的是推理模型（如 Qwen3、DeepSeek-R1）在训练时学会的"先思考再回答"格式，在部署时没有被正确过滤或隐藏，导致用户看到了模型内部思考过程的现象。

**泄漏场景举例：**
```
用户：我头痛怎么办？
模型输出：
<think>用户可能描述的是一个常见的头痛症状，需要排除脑出血等急症...
我应该在回答中建议就医，同时询问伴随症状...</think>
头痛可能由多种原因引起，建议您...
```

**为什么是问题：**
1. **用户体验差**：用户看不懂模型内部推理过程
2. **安全风险**：思考过程中可能包含模型对用户的"私下评价"、对敏感信息的分析等
3. **隐私风险**：思考过程中可能包含对用户输入中敏感信息的进一步推断

**如何避免：**
1. **训练时**：SFT 阶段直接用 `enable_thinking=False` 的配置，不让模型学习 thinking 行为
2. **部署时**：在应用层做后处理，用正则匹配 `<think>...</think>` 并过滤（但不推荐，不可靠）
3. **Prompt 引导**：如果是推理模型，可以在 system prompt 中要求"不要输出思考过程"
4. **模型层面**：Qwen3 等模型支持在 API 调用时传 `enable_thinking` 参数

**项目结合：**
在医学 LLM 项目中，通过在训练时统一使用 `enable_thinking=False` 的 chat template，从根源上避免了思考泄漏问题——模型根本就没学过在医学问答场景下思考。

**易错点：**
- 以为 think 标签只是一个普通字符串——它本质上是模型学到的"行为模式"，不算严格的安全漏洞，但影响体验和隐私
- 只做后处理过滤而不在训练时修正——类似"头痛医头"，没解决根因

---

### 2.5 训练和部署 Chat Template 不一致的后果

**面试官怎么问：**
> "如果训练时用的 chat template 和上线部署的不一样，会出什么问题？你为什么这么重视一致性？"

**标准回答：**

这是 SFT 部署阶段最容易出问题的地方之一，后果分几个等级：

**Level 1：轻微格式差异（用户可能察觉不到）**
- 训练时 assistant 回复后不带 EOS，部署时带了 → 模型可能多输出一个空 token
- system prompt 位置不同（有些模型 system 在最后，有些在最前）

**Level 2：中等差异（回答质量显著下降）**
- 训练时的 user/assistant 标记是 `<|im_start|>` 但部署时用的是 `<user> </user>` → 模型看到了分布外的格式标记，生成质量大幅下降
- 训练时有 `enable_thinking=True`，部署时没配，模型生成的 `<think>` 标签被当普通文本展示

**Level 3：严重差异（完全不可用）**
- 部署时少了一个 role 标记，模型不知道什么时候应该生成 assistant 回答 → **无限生成空格或乱码**
- 训练时用的是 Qwen 的 chat template，部署时用的是 ChatML → **角色混淆，模型把 system prompt 当 user message 回复，把 user message 当 assistant 补全**

**为什么你这么重视：**
一个血泪教训：有人在 4 卡 A100 上跑了 3 天 SFT，结果因为部署时 chat template 多了一个空格，所有回答都以一个空格开头，用户反馈"感觉模型很卡"。

**根本原因：**
LLM 本质上是**模式匹配器**——它在训练时学到的模式是"看到 A 标记就产生 B 行为"。部署时如果标记变了（哪怕一个空格），模式匹配就打断了。

**项目结合：**
你在医学 LLM 项目中，从数据构造（`tokenizer.apply_chat_template`）到训练（trainer 调用）到部署（vLLM 的 `--chat-template`），全程用同一个 chat template 文件，保证端到端的一致性。这是一个做得很好的工程实践。

**易错点：**
- 不要被"就多了一个空格"的说法迷惑——LLM 对空格极其敏感，因为 tokenizer 可能把一个额外空格当成一个新 token 的开始，引发连锁反应
- 训练时用的是 `apply_chat_template(tokenize=True)` 的自动行为，部署时也应该用同样的方式，而不是手动字符串拼接

---

### 2.6 DPO 中 Chosen/Rejected 的 Prompt Template 必须一致

**面试官怎么问：**
> "做 DPO 的时候，chosen 和 rejected 的 prompt 部分为什么必须完全一致？如果我给了两个不同的 system prompt 会怎样？"

**标准回答：**

DPO 的核心逻辑是：模型在**同一个 prompt 下**，学习偏好 chosen 回答而非 rejected 回答。

**DPO loss 公式背后的直觉：**
奖励差 = 模型对 chosen 的 log prob - 模型对 rejected 的 log prob
模型要最大化这个差值。

如果 prompt 部分不一样，模型学到的就不是"哪种回答更好"，而是"哪个 prompt 更顺眼"——完全偏离了 DPO 的训练目标。

**具体来说：**
1. **输入必须在 token 级别完全一致**：chosen 和 rejected 的 prompt 部分（从 BOS 到 assistant 回答开始之前的所有 token）必须逐 token 相同
2. **system prompt 必须一致**：不能让 chosen 用 A 版 system prompt、rejected 用 B 版
3. **truncation 必须一致**：不能因为 chosen 回答短就留更多 context，rejected 回答长就截断更多

**实验后果（如果违反了）：**
- 模型可能学习"喜欢左边的 prompt 格式"（trivial preference）而不是真正的回答质量偏好
- DPO loss 可能不收敛或收敛到奇怪的方向
- 评估时发现模型对某些 prompt 格式有"偏见"

**项目结合：**
在你的医学项目中构造 DPO 数据时（如果做了 DPO），标准做法是：用同一个 question + system prompt 构造两个数据条目，一个带 chosen answer，一个带 rejected answer，确保 prompt 部分在 token 化后完全一样。

**易错点：**
- 新手容易在构造数据时，为 chosen 和 rejected 写了不同的 system prompt（比如想"让被拒绝的那个看起来语气不好"），这其实污染了 DPO 信号
- 手动控制 truncation 时，可能因为 chosen/rejected 长度不同导致 prompt 部分被截断长度不同

---

### 2.7 Padding Side：Left vs Right

**面试官怎么问：**
> "batch 训练的时候为什么通常用 right padding，而推理的时候用 left padding？反过来行不行？"

**标准回答：**

**训练时用 Right Padding：**

```
[token1, token2, token3, PAD, PAD]    ← 短序列
[token1, token2, token3, token4, token5] ← 长序列
```

- 把 PAD 放在右边，模型的 attention 从左到右，真实的 context 在左边连续排列，PAD 不会影响前缀的 attention 计算
- 但如果用了 causal mask，PAD 在右边就不会有 token attend 到 PAD（因为 causal attention 只看左边），所以 PAD 本质上是"不干扰"的

**推理时用 Left Padding（尤其在 batch inference 中）：**

```
[PAD, PAD, token1, token2, token3]    ← 短序列
[PAD, token4, token5, token6, token7]  ← 中序列
[token8, token9, token10, token11, token12] ← 长序列
```

为什么推理要用 left padding？**因为要保证最后生成的 token 始终在序列的最后一个位置。**

自回归生成每次预测下一个 token，模型关注的是最后一个有效 token 的位置。如果 right padding：
```
[token1, token2, PAD, PAD]
```
最后一个位置是 PAD，模型无法正确预测下一个 token。

Left padding 则保证了：
```
[PAD, PAD, token1, token2]  ← 最后一个位置是有效的 token2
```
模型基于 token2 预测下一个 token 是正常的。

**训练时 left padding 行不行？**
可以，但不推荐。Left padding 会让 causal attention 的计算中出现"先看到 PAD 再看到真实 token"的情况，虽然 loss 计算时 mask 掉了 PAD，但 attention 模式下 PAD 可能对有用的 token 产生干扰。

**项目结合：**
在 MiniMind 和医学项目中，训练时（SFT）使用 right padding，推理时（vLLM/Ollama）框架会自动处理为 left padding。

**易错点：**
- 用 HuggingFace Trainer 时 `DataCollatorForSeq2Seq` 默认用的是 right padding（用于训练），修改 padding side 要用 `tokenizer.padding_side = "left"` 设置
- 推理时自己写 batch 逻辑但忘了设 left padding——结果最后一个 token 是 PAD，模型输出全乱

---

## 三、项目实现总结

### 3.1 MiniMind 项目中的 Tokenizer 相关实现

在 MiniMind 项目中，tokenizer 相关的核心实现要点：

- **Tokenizer 选择**：使用 Qwen2 系列的 tokenizer（SentencePiece + BPE），词表大小约 151936
- **pad_token 设置**：`tokenizer.pad_token = tokenizer.eos_token`
- **SFT 数据构造**：通过 `tokenizer.apply_chat_template()` 将多轮对话（messages 格式）转换为 token IDs
- **Labels 构造**：assistant 部分的 token 设为真实 label，user 和 system 部分设为 -100 忽略
- **Chat Template 管理**：使用 Qwen2 的默认 chat template（jinja 模板），保证训练和推理一致
- **max_seq_length**：根据 GPU 显存设置，一般为 2048 或 4096

**面试时可以强调的点：**
- MiniMind 项目虽然没有做 vocab expansion，但用了一个很好的工程实践：全程通过 `apply_chat_template` 保证格式统一
- 如果面试官问"你觉得还有什么优化空间"，可以说"收集中文医疗语料做 incremental BPE 训练扩充词表"

---

### 3.2 医学 LLM 项目中的 Chat Template 相关实现

在医学 LLM 项目中，chat template 的实现要点：

- **基础模型**：基于 Qwen2.5 或 Qwen3，使用其自带 chat template
- **思考控制**：对于 Qwen3，在 chat template 中设置 `enable_thinking=False`，确保医学回复直接输出、不含思考过程
- **训练数据构造**：
  - 每条样本用 `tokenizer.apply_chat_template(messages, tokenize=True)` 转为 token IDs
  - 严格控制 user 部分、assistant 部分、system prompt 的格式一致
- **部署对齐**：
  - vLLM 部署时加载相同的 chat template，确保训练-部署无格式偏移
  - 通过配置文件管理 chat template 版本，避免部署时用错模板
- **安全考虑**：
  - 训练数据中 assistant 回答末尾加 EOS（`<|im_end|>`），model 学会在回答完成后自动停止
  - 确保部署时不以 `enable_thinking=True` 上线，防止 `<think>` 泄漏

**面试时可以强调的亮点：**
- "我在项目中特别注重了训练-部署的一致性，从数据构造到部署，全程通过 chat template 配置管理，避免了分布外问题"
- "考虑到医学场景的安全需求，我主动关闭了推理模型的 thinking 功能，确保回答格式干净可控"

---

## 背诵版总结（面试前 5 分钟速记）

### 核心概念一句话
1. **Tokenizer**：文本和 token ID 之间的翻译官，BPE 是主流算法
2. **BPE**：从字符配对开始，反复合并频率最高的 pair
3. **Byte-Level BPE**：从 byte 开始合并，永不出现 UNK
4. **SentencePiece**：训练框架，把空格当普通字符，对中文友好
5. **Special Tokens**：BOS(开始)、EOS(结束/停止)、PAD(填充)、UNK(未知)
6. **Chat Template**：结构化对话 → 格式化文本的转换函数，保证训练/部署格式一致
7. **apply_chat_template**：Tokenizers 库提供的格式化方法，自动处理特殊 token，不要手写
8. **enable_thinking=False**：Qwen3 中关闭思考链，避免 `<think>` 泄漏
9. **Right Padding 训练 / Left Padding 推理**：保证生成时最后位置是有效 token
10. **DPO prompt 一致**：chosen 和 rejected 的 prompt 必须 token 级完全相同

### 项目亮点（面试直接说）
- 用 Qwen2/3 商业级 tokenizer（SentencePiece + BPE，151936 词表）
- 全程 `apply_chat_template` 保证训练-部署一致
- 关闭 Qwen3 thinking 防止思考泄漏
- 未来可做：医学语料扩充词表提升 token 效率

### 经典踩坑（面试官最喜欢问）
- 手写 chat template → 多一个空格、少一个 role、token 边界错误
- 推理忘了设 left padding → batch 推理最后位置是 PAD
- 训练时 pad_token 忘了设 → tokenizer 报错
- DPO chosen/rejected prompt 不一致 → 学到假偏好
- enable_thinking 忘关 → `<think>` 泄漏到用户端
