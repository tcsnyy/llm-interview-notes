# 12. RAG 与 Safety-RAG 详细八股

---

## 文档总目录

- [A. RAG 基础](#a-rag-基础)
- [B. 文档解析与清洗](#b-文档解析与清洗)
- [C. Chunking](#c-chunking)
- [D. Embedding](#d-embedding)
- [E. 向量库与索引](#e-向量库与索引)
- [F. 稀疏检索](#f-稀疏检索)
- [G. Hybrid Retrieval](#g-hybrid-retrieval)
- [H. Rerank](#h-rerank)
- [I. Query 处理](#i-query-处理)
- [J. Context 组织](#j-context-组织)
- [K. 生成阶段](#k-生成阶段)
- [L. RAG 评测](#l-rag-评测)
- [M. RAG 与后训练结合](#m-rag-与后训练结合)
- [N. 我的 Safety-RAG 项目讲法](#n-我的-safety-rag-项目讲法)
- [背诵版总结](#背诵版总结)

---

## A. RAG 基础

### Q: 什么是 RAG？为什么需要它？

RAG（Retrieval-Augmented Generation）= 检索增强生成。在 LLM 生成回答之前，先从外部知识库中检索相关文档片段，将其作为上下文注入 prompt，让模型基于"找到的证据"来回答。核心思想：**让 LLM 从"闭卷考试"变成"开卷考试"。**

**为什么需要 RAG：**

> "LLM 有三大固有问题：知识截止日期（训练数据有时间窗口）、幻觉（模型会编造不存在的事实）、知识更新成本高（重新训练/微调代价大）。RAG 通过外挂知识库解决了这三个问题——知识库可以实时更新、检索到的内容为生成提供 grounding、不需要重新训练模型。"

**医学场景的特殊性：**

> "医学是高风险领域，幻觉的代价可能是患者生命。比如我们的 DPO 模型在脑溢血案例中推荐了溶栓药物——这是溶栓的绝对禁忌症。这种错误靠训练很难消除，因为训练数据中可能缺少足够的负面样本。RAG 通过在 prompt 中直接注入参考指南，让模型'有据可查'，从机制上降低幻觉。"

**医学场景特别需要 RAG 的原因：**
- 医学知识更新快：新的临床指南、药物、诊疗方案不断发布
- 错误代价极高：一次幻觉可能 = 患者生命危险
- 知识粒度要求细：同一个疾病的不同亚型可能治疗方案完全不同
- 可溯源要求：医学回答需要能追溯到来源（guideline/paper/专家共识）

---

### Q: RAG 的基本流程是什么？

```
用户Query
    │
    ▼
┌──────────────┐
│ Query处理     │ → 改写/扩展/分类
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ 检索 Retrieval │ → 向量检索 + 关键词检索
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Rerank        │ → 精排 top-k
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Context组装   │ → 拼接/去重/截断
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ LLM 生成      │ → 基于context生成回答
└──────────────┘
```

---

### Q: Naive RAG vs Advanced RAG 有什么区别？

| | Naive RAG | Advanced RAG（我们的方案） |
|---|---|---|
| 检索 | 单一向量检索 | Hybrid（向量+关键词） |
| 路由 | 无 | 规则路由（8条规则） |
| Rerank | 无 | bge-reranker-v2-m3 |
| Query处理 | 原样送入 | 同义词扩展（synonyms.yaml） |
| Context | 简单拼接 | 元数据过滤 + 安全规则注入 |
| 评测 | 无 | grounding score + hallucination tracking |

---

### Q: RAG 有哪些典型失败模式？

1. **检索失败（Retrieval Failure）**：知识库有相关内容但没检索到
2. **知识覆盖缺失（Knowledge Gap）**：知识库本身缺少相关信息
3. **上下文忽略（Context Ignoring）**：检索到了正确信息但模型没有使用
4. **上下文冲突（Context Conflict）**：检索到的信息与模型内部知识冲突，模型选择了错误的内部知识
5. **信息过载（Information Overload）**：检索到太多内容，关键信息被稀释

---

### Q: RAG vs Fine-tuning 怎么取舍？

| 维度 | Fine-tuning | RAG |
|------|------------|-----|
| 知识注入方式 | 参数化记忆 | 外部检索 |
| 更新成本 | 需重新训练 | 更新文档即可 |
| 可解释性 | 黑盒 | 可展示检索来源 |
| 领域精度 | 中等（有遗忘风险） | 高（直接引用） |
| 推理能力 | 强（端到端学习） | 依赖prompt engineering |
| 长尾覆盖 | 弱（缺少样本） | 强（有文档即可） |
| 我们的选择 | SFT + DPO 打底 | RAG 兜底 |

---

### Q: DPO 与 RAG 是什么关系？

> "DPO 让模型学会'好好说话'——有共情、有结构、有逻辑。RAG 让模型学会'说对话'——有事实依据、有安全底线。两者不是替代关系而是互补关系。在我们的项目中，DPO 阶段优化的是回答质量（SFT 学格式、DPO 学偏好），RAG 阶段解决的是知识准确性问题。实验数据也证明了这一点：DPO+RAG 比纯 DPO 在 grounding 上提高 0.23，在 hallucination 上改善 0.15。"

---

### Q: Guardrails 是什么概念？

Guardrails = 护栏，是在 RAG 管线中的安全保护层。包括：
- **输入护栏**：query 分类、敏感内容过滤、路由到安全分支
- **检索护栏**：metadata 过滤、来源可信度检查
- **输出护栏**：回答中的危险内容检测、安全声明强制注入

在我们的项目中，8 条路由规则和 safety_rules 注入就是 Guardrails 的具体实现。

---

### Q: 知识库构建的冰山模型是什么？

冰山之上（用户能看到）：检索到的 chunk → 生成的回答

冰山之下（大量工程工作）：PDF 解析（格式混乱、表格、图片）、文本清洗（页眉页脚、参考文献、乱码）、Chunking 策略（多级、重叠、语义边界）、元数据标注（来源、章节、可信度）、索引优化（IVF/HNSW 参数）

---

### Q: Embedding 模型怎么选？RAG pipeline 的核心权衡是什么？

**Embedding 选择标准：** MTEB benchmark 排名（但不是唯一标准）、对目标领域的适配性（通用 embedding 在医学上可能效果一般）、支持的序列长度（医学文本可能很长）、多语言支持（如果涉及中文+英文文献）。我们选择 BGE-M3 的原因：支持多语言、8192 token 长度、在医学相关 benchmark 上表现好。

**核心权衡：**

> "RAG 做的是 precision-recall 权衡。recall 高了意味着返回更多候选，但噪声也多了。precision 高了意味着前几个结果精准，但可能漏掉相关内容。我们的策略是'粗召回 + 精排序'：粗召回阶段用 hybrid retrieval 保证 recall（top-20），精排阶段用 reranker 保证 precision（top-5）。"

---

## B. 文档解析与清洗

### Q: 为什么文档解析是 RAG 的最大瓶颈之一？

> "在 RAG 项目中，80% 的坑在数据处理阶段。PDF 是展示格式，不是数据格式——它的内部结构是'在这个坐标画这段文字'，而不是'这段文字是一个段落'。把 PDF 变成干净的、有结构的文本，远比很多人想象的要困难。"

**PDF 解析的常见坑：**
1. **双栏排版**：解析器可能按行从左到右读，导致跨栏串行
2. **表格**：表格可能被解析为混乱的文本流
3. **图片中的文字**：需要 OCR 才能识别
4. **页眉页脚**：每页重复的无意义内容被当作正文
5. **参考文献**：参考文献格式特殊，容易打乱 chunk 边界
6. **特殊字符/公式**：Unicode 乱码
7. **分页符断句**：一个句子被分页切断
8. **字体编码问题**：某些医学符号（如 μg、μmol/L）可能无法正确解析

**解析工具选择：**

| 工具 | 优点 | 缺点 | 适用 |
|------|------|------|------|
| PyMuPDF (fitz) | 快、保留布局信息 | 表格处理弱 | 纯文本文档 |
| pdfplumber | 表格提取好 | 慢 | 表格多的文档 |
| Unstructured.io | 全流程、支持多种格式 | 重、有时不稳定 | 混合格式 |
| Marker (开源) | 转 Markdown 效果好 | 较新，生态不成熟 | 需要结构化输出 |
| 我们使用的 | PyMuPDF + 自定义清洗 | 可控 | 医学 PDF |

---

### Q: 医学文档清洗 Pipeline 怎么设计？

```python
def clean_medical_text(raw_text: str) -> str:
    """
    医学PDF文本清洗Pipeline
    """
    # 1. 去除页眉页脚（通过重复模式识别）
    text = remove_headers_footers(raw_text)
    
    # 2. 统一Unicode（全角转半角、特殊医学符号标准化）
    text = normalize_unicode(text)
    
    # 3. 合并被换行切断的句子
    text = merge_broken_sentences(text)
    
    # 4. 修复PDF解析产生的单词间多余空格
    text = fix_spacing(text)
    
    # 5. 移除独立页码
    text = remove_page_numbers(text)
    
    # 6. 规范化参考文献区域
    text = normalize_references(text)
    
    # 7. 去除过多的空行
    text = collapse_newlines(text)
    
    return text
```

**医学文档特有的清洗需求：**
1. **药物名称标准化**：阿司匹林 vs 乙酰水杨酸 vs Aspirin 统一
2. **剂量单位规范化**：mg vs 毫克，μg vs mcg
3. **医学术语缩写展开**：CABG → 冠状动脉旁路移植术
4. **检查值单位归一化**：mmol/L vs mg/dL（需要换算）
5. **ICD编码识别**：保留但不依赖编码做 chunking

---

### Q: 表格、元数据、多语言混合怎么处理？

**表格处理策略：**
- **简单表格**（<10行）：转为 Markdown 表格格式保留在 text 中
- **复杂表格**：提取为结构化 JSON + 生成自然语言描述
- **表格标题**：保留作为 metadata，便于检索时识别

```python
def process_medical_table(table):
    """处理医学表格"""
    if is_simple_dosage_table(table):
        return table_to_markdown(table)
    elif is_diagnostic_criteria_table(table):
        return table_to_natural_language(table)
    else:
        json_repr = table_to_json(table)
        nl_repr = table_to_description(table)
        return f"{json_repr}\n{nl_repr}"
```

**元数据提取（每个 chunk 携带）：**

```python
chunk_metadata = {
    "source": "内科学_第9版.pdf",
    "chapter": "第十章_脑血管疾病",
    "section": "10.3_缺血性脑卒中",
    "page": 234,
    "chunk_id": "internal_medicine_ch10_sec3_chunk12",
    "doc_type": "guideline",  # guideline/textbook/paper/consensus
    "credibility": "high",     # high/medium/low
    "publication_year": 2018,
    "medical_domain": "neurology",
    "contains_drug_info": True,
    "contains_dosage": True,
}
```

**多语言混合处理：** 我们的 15 本医学 PDF 中包含中英文混排的内容。处理策略：检测每个段落的语言（langdetect/fasttext）、对中文段落用中文分词器、对英文段落保留英文分词、使用支持多语言的 BGE-M3 embedding。

---

### Q: 文档去重和版本管理怎么做？

**文档去重：**
- **完全去重**：MD5 hash 比对 chunk 内容，完全相同则只保留一份（保留 metadata 更全的那份）
- **近义去重**：对内容相似度 > 0.95（embedding cosine）的 chunk，保留来源更权威的
- **互补保留**：内容相似但角度不同的 chunk 都保留（如不同教材对同一疾病的描述）

**文档版本管理：** 医学指南会更新（如高血压指南从 2018 到 2023）。每个文档标注版本号和发布年份、旧版本不删除但降低检索权重、在 metadata 中标记 "superseded_by: xxx"、检索时优先返回最新版本。

**为什么不用现成的文档解析 SaaS：**

> "第一是数据安全——医学指南虽然公开但处理 pipeline 涉及内部数据流。第二是可控性——SaaS 的解析策略是黑盒的，出现问题无法调试。第三是医学领域的定制需求——通用的解析工具不认识 μmol/L vs mg/dL 的差异，不会做药物名称标准化。自己写虽然工作量大，但每一个环节都是可控可调的。"

---

### Q: 15 本 PDF 是怎么处理的？

> "我用 15 本医学教材和指南 PDF 构建了知识库。原始 PDF 约 8000 页，经过解析和清洗后得到约 11707 个 chunks。解析阶段最大的挑战是医学 PDF 的排版多样性——有些是双栏、有些有大量表格、有些包含英文摘要。我用了 PyMuPDF 做基础解析，然后写了一个专门的清洗 pipeline 处理页眉页脚、断行修复、药物名称标准化。最终每个 chunk 都携带了包括来源章节、医学领域、可信度等在内的 10+ 个元数据字段。"

---

## C. Chunking

### Q: 什么是 Chunking？常见策略有哪些？

将长文档切分为固定或可变长度的文本块（chunks），每个 chunk 作为独立的检索单元。

**为什么 Chunking 策略至关重要：**
- Chunk 太小：语义不完整，无法独立回答问题
- Chunk 太大：检索精度下降，包含太多无关信息，稀释关键内容
- 切分不当：关键信息被切在两个 chunk 中，谁也检索不到

**常见 Chunking 策略对比：**

| 策略 | 方法 | 优点 | 缺点 |
|------|------|------|------|
| Fixed-size | 固定 N tokens | 简单、可控 | 破坏语义边界 |
| Recursive | 按 \n\n, \n, 。, .  递归切分 | 保持语义 | 仍需调参 |
| Sentence-based | 按句子切分+合并到目标size | 语义完整 | 对无标点文本失效 |
| Semantic | 用 embedding 相似度判断切分点 | 语义最优 | 计算成本高 |
| Document-structure | 按标题/章节结构切分 | 与文档结构一致 | 依赖文档格式 |

---

### Q: 我们项目中的 Chunking 策略是什么？为什么选 512 token？

采用 **RecursiveCharacterTextSplitter + 医学语义边界保护**：

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

medical_separators = [
    "\n\n## ",      # Markdown二级标题
    "\n\n### ",     # Markdown三级标题  
    "\n\n#### ",    # Markdown四级标题
    "\n\n",         # 段落分隔
    "\n",           # 换行
    "。",           # 中文句号
    ". ",           # 英文句号
    "；",           # 中文分号
    "; ",           # 英文分号
    " ",            # 空格（最后手段）
]

text_splitter = RecursiveCharacterTextSplitter(
    separators=medical_separators,
    chunk_size=512,        # token 数
    chunk_overlap=64,      # 重叠 token 数
    keep_separator=True,   # 保留分隔符
)
```

**为什么选择 512 token chunk_size：**

> "512 是一个经验值，但我们在医学场景下做了验证。太小（256）会导致很多 chunk 只有半句话，嵌入后语义不完整；太大（1024）会导致检索精度下降——一篇 textbook chunk 可能包含 3 个不同疾病的信息，检索时 noise 高。512 + 64 overlap 在 chunk 语义完整性和检索精度之间取得了平衡。另外，512 token 对于 BGE-M3 的 embedding 来说正好在其最优性能区间内。"

**Chunk Overlap 的设计：** Overlap 的作用是防止关键信息落在两个相邻 chunk 的边界上。64 token overlap（512 的 12.5%）保证至少跨越 1-2 个完整句子。过大的 overlap 会显著增加 embedding 计算量和存储。

---

### Q: 医学场景有什么特殊的 Chunking 规则？

1. **药物信息保护**：药物的适应症、禁忌症、剂量必须在同一个 chunk 内（不能切割）
2. **诊断标准完整性**：将"诊断标准"段落强制作为独立 chunk 单元
3. **章节边界尊重**：不在两个不同的疾病主题之间出现 overlap
4. **表格完整性**：表格不跨 chunk 切分

```python
def should_not_split(text: str) -> bool:
    """判断当前位置是否不应该切分"""
    medical_protection_patterns = [
        r'禁忌症[：:]',    # 禁忌症列表
        r'诊断标准[：:]',  # 诊断标准
        r'用法用量[：:]',  # 用药剂量
        r'不良反应[：:]',  # 不良反应列表
    ]
    return any(re.search(p, text) for p in medical_protection_patterns)
```

---

### Q: 什么是 Small-to-Big Retrieval（两层 Chunk 结构）？

我们构建了两层 chunk 结构：

```
Parent Chunk (1024 tokens)
    ├── Child Chunk 1 (512 tokens, overlap 64 with child 2)
    ├── Child Chunk 2 (512 tokens, overlap 64 with child 1 and 3)
    └── Child Chunk 3 (512 tokens, ...)
```

检索策略：
- 粗召回阶段：用 child chunks（512 token）做向量检索，保证精度
- 返回结果时：如果 child chunk 被选中，返回其 parent chunk（1024 token），保证语义完整性

这叫 **Small-to-Big Retrieval**。

**从 8000 页到 11707 chunks 的计算：**

```
8000 pages × ~400 words/page = ~3,200,000 words
/ ~300 words per 512-token chunk ≈ 10,667 chunks (理论值)
实际: 11,707 chunks (差异: ~10% overhead from overlap + 不完整chunk)
```

**面试追问：为什么不是 Dynamic Chunking？**

> "Dynamic chunking（基于语义相似度动态切分）理论上更好，但在我们的场景下有工程权衡。15 本 PDF 约 8000 页，做 semantic chunking 需要先对全文做 embedding——计算量相当于将整个语料 embedding 两遍。而 recursive split 可以一次扫描解决。我们没有观察到 recursive split 带来的语义破坏严重影响最终效果——因为有 64 token overlap 和 parent chunk fallback 兜底。"

---

## D. Embedding

### Q: Embedding 在 RAG 中的角色是什么？为什么选 BGE-M3？

Embedding 模型将文本映射到高维向量空间，语义相近的文本在空间中距离近。这是向量检索的基础。

```
text → embedding model → vector (1024-dim for BGE-M3)
```

**我们选择 BGE-M3 的原因：**

| 考量因素 | BGE-M3 的表现 |
|---------|--------------|
| 多语言支持 | 中英文 + 100+ 语言 |
| 序列长度 | 8192 tokens（远超医学 chunk 的 512） |
| 检索精度 | MTEB Retrieval 榜单前列 |
| 稠密+稀疏双能力 | 支持 dense + sparse 双向量输出 |
| 开源可部署 | 可本地部署，数据不出域 |
| 中文医学能力 | 在我们的评测集上 recall@5 = 0.87 |

**为什么不用 OpenAI Embedding：**

> "第一，数据安全——医学数据不能经过第三方 API。第二，成本——11707 chunks 即使不大，但每次迭代评测都会有大量 API 调用。第三，BGE-M3 的双向量能力（dense + sparse）天然支持 hybrid retrieval，不需要额外部署 BM25。第四，中文医学术语的 embedding 质量——我们做过对比，BGE-M3 在中文医学短 query 的 recall 上优于 text-embedding-3-large。"

---

### Q: Dense vs Sparse Embedding 有什么区别？

| | Dense Embedding | Sparse Embedding |
|---|---|---|
| 原理 | 每个维度都有非零值 | 大部分维度为零（词汇维度） |
| 优势 | 语义泛化、同义词 | 精确关键词匹配 |
| 劣势 | 对稀有术语覆盖弱 | 对同义词改写不敏感 |
| 例子 | BGE-M3 (dense) | BM25, BGE-M3 (sparse) |
| 适用 | 语义相似检索 | 精确术语检索 |

**BGE-M3 的优势：一个模型同时输出 dense 和 sparse 两种向量。**

---

### Q: Embedding 的维度选择、Normalization 和批处理怎么做？

BGE-M3 默认 1024 维。为什么不用更低维度：
- 768 维：信息压缩更多，可能丢失医学术语的细微差异
- 1024 维：BGE-M3 的默认最优维度
- 1536 维：存储和计算成本高，边际收益小

**Normalization 至关重要：** 对 embedding 做 L2 normalization 后，cosine similarity 等于 inner product，检索速度可以大幅提升。

```
cos(A, B) = A·B / (||A|| * ||B||)
如果 ||A|| = ||B|| = 1 (L2 normalized)，则 cos(A, B) = A·B
```

**批处理：**

```python
def batch_embed_chunks(chunks, model, batch_size=32):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
            show_progress_bar=True
        )
        embeddings.extend(batch_embeddings)
    return embeddings
```

---

### Q: Embedding 的医学知识适配和 Cache 策略怎么做？

**通用 embedding 在医学领域的局限性：** 通用训练数据中医学文本比例低；"MI"在通用语料中可能是"Mission Impossible"，在医学中应该是"Myocardial Infarction"；医学术语的同义词关系可能没有被充分学习。

**我们的应对：** 使用 BGE-M3 的多语言能力 + 中文医学术语的自然覆盖；通过 synonyms.yaml 对 query 做同义词扩展，弥补 embedding 对医学同义词的覆盖不足。

**Cache 策略：** 对 chunk 内容做 MD5 hash；如果 MD5 没变，直接从缓存加载 embedding；只对新/修改的 chunk 重新 embedding。

**面试追问：有没有对比过不同 embedding 模型？**

> "我们在项目早期做了一个小规模的对比实验。用 20 个医学 query 测试了 BGE-M3、text-embedding-3-large（via API）、m3e-base 三个模型。BGE-M3 的 recall@5 是 0.87，text-embedding-3-large 是 0.84，m3e-base 是 0.79。BGE-M3 在中文医学术语的检索上表现最好，而且支持本地部署。不过 n=20 的对比只是初步验证，不是严格的 benchmark。"

---

## E. 向量库与索引

### Q: 向量数据库怎么选型？为什么用 Milvus？

| 方案 | 适用规模 | 优点 | 缺点 |
|------|---------|------|------|
| Faiss (内存) | <100K | 极快、无依赖 | 无持久化、无metadata |
| Chroma | <500K | 易用、轻量 | 性能弱 |
| Milvus Lite | <1M | 持久化、metadata filter | 需要部署 |
| Qdrant | <10M | 性能好 | 需要维护 |
| Pinecone (云) | 任意 | 免运维 | 数据上云、费用 |

> "11707 chunks + 1024 维 embedding = 约 48MB 向量数据。这个规模用 Faiss 放内存都跑得动，但我们选择 Milvus 的原因有三：一是 metadata filter——需要按章节、医学领域、可信度过滤；二是持久化——索引不用每次启动重新构建；三是为将来扩展留空间——如果知识库从 15 本扩展到 500 本，Milvus 也能撑住。"

---

### Q: Milvus 的索引类型和参数怎么配置？

**IVF_FLAT（我们使用的）：**

```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # Inner Product (因为向量已L2归一化 = Cosine)
    "params": {"nlist": 128}
}
```

- 原理：K-means 聚类 → 检索时只搜索最近的 N 个聚类中心
- 参数：nlist=128（11707 个向量分 128 个聚类，每个约 91 个向量）
- 优点：精度高（接近 brute-force）
- 缺点：需要存储原始向量

**为什么用 Inner Product 而非 Cosine：** 因为 embedding 在入库前已经 L2 normalized，IP = Cosine。而 IP 的计算比 Cosine 快（少一步求模运算）。

**检索参数调优：**

```
nprobe=1   → 只搜最近 1 个聚类 → 快（~1ms）但 recall 低
nprobe=16  → 搜最近 16 个聚类 → 中等（~5ms）recall 高
nprobe=128 → 搜所有聚类 → 等于 brute-force → 慢但 recall=100%

我们的选择: nprobe=16
11707 chunks / 128 clusters ≈ 91 per cluster
搜索 16 clusters ≈ 1456 个向量 → 质量与速度的最佳平衡
```

---

### Q: Metadata Filter 怎么实现？相似度阈值怎么设？

```python
# 检索时附加 metadata filter
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param=search_params,
    limit=20,
    expr=f'medical_domain == "neurology" and credibility == "high"'
)
```

这就是我们的"metadata filter"层——在向量检索的同时，用标量过滤剪枝不相关或不信任的来源。

**相似度阈值：** 不是所有检索结果都有用。如果 top-1 的相似度只有 0.3（cosine），说明知识库中没有相关信息。我们设置阈值 threshold = 0.5：如果 max similarity < 0.5，标记为 "可能无相关知识"，触发 fallback 逻辑——不强制 RAG，用 DPO 模型的固有知识 + 强安全声明。

**向量库的量级估算：**

```
存储估算 (11707 chunks):
- Vector data: 11707 × 1024 × 4 bytes (float32) ≈ 48 MB
- Metadata: 11707 × ~500 bytes ≈ 6 MB
- Index (IVF): ~5 MB overhead
- Total: ~60 MB

这基本可以放内存里，检索速度极快。
```

**Query 延迟分析：**

```
Typical retrieval latency breakdown:
1. Query embedding: ~20ms (BGE-M3 on GPU)
2. Vector search (IVF_FLAT, nprobe=16): ~5ms
3. Sparse retrieval (BGE-M3 sparse): ~10ms
4. Rerank (bge-reranker-v2-m3, top-20 → top-5): ~50ms
5. Total: ~85ms

这远低于 LLM 生成延迟（通常 2-10 秒），不构成瓶颈。
```

---

## F. 稀疏检索

### Q: 什么是稀疏检索？为什么需要它？

稀疏检索基于关键词匹配，使用 TF-IDF 或 BM25 算法。每个文档表示为一个稀疏向量，维度 = 词表大小，大部分维度为 0。

> "向量检索擅长语义相似但可能遗漏精确术语匹配。举个例子：用户问'华法林的 INR 目标值'，向量检索可能返回'抗凝药物的监测指标'相关内容，但不一定精确返回'INR 2.0-3.0'这句话。稀疏检索通过关键词匹配，恰好在精确术语查询上有优势。两种检索方式互补。"

**Dense vs Sparse 的核心差异（举例）：**

```
Query: "阿司匹林过敏患者可以用氯吡格雷吗"

Dense Retrieval 可能返回:
  → "抗血小板药物的选择与禁忌"（语义相关，但可能不精确命中）

Sparse Retrieval 可能返回:
  → "对阿司匹林过敏或不能耐受的患者，可选用氯吡格雷（75mg/d）作为替代"
    （精确命中关键词"阿司匹林""过敏""氯吡格雷"）
```

**稀疏检索在医学场景的优势：** 医学查询中大量使用精确的术语（药物名称、疾病名称、检验指标、剂量）。这些术语的同义词/改写很少，因此**精确关键词匹配对医学 RAG 至关重要**。

---

### Q: BGE-M3 的 Sparse 能力怎么用？为什么比 BM25 好？

BGE-M3 的独特优势：一个 forward pass 同时输出 dense 和 sparse 两种向量。

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

# 一次编码，两种输出
output = model.encode(
    ["阿司匹林过敏患者可以用氯吡格雷吗"],
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=False,
)

dense_vector = output['dense_vecs'][0]      # (1024,) float32
sparse_vector = output['lexical_weights'][0] # dict: {token_id: weight}
```

**为什么 BGE-M3 Sparse > BM25：**
1. **深度学习分词**：BGE-M3 的 tokenizer 基于 SentencePiece，对中文医学文本的分词效果远好于 jieba 等传统分词器
2. **Learned weight**：BM25 的权重是启发式（TFxIDF），BGE-M3 的权重是学习出来的——哪些词对检索重要是模型学到的
3. **多语言统一**：中英文在同一个 token 空间中，不需要分别处理

**面试追问：SPLADE 比 BGE-M3 sparse 更好吗？**

> "SPLADE 是专门的 learned sparse retrieval 模型，在学术 benchmark 上确实略优于 BGE-M3 的 sparse 输出。但我们选择 BGE-M3 的是因为它的'unified'特性——一个模型同时输出 dense 和 sparse，减少了系统复杂度和部署成本。"

---

## G. Hybrid Retrieval

### Q: 什么是 Hybrid Retrieval？为什么 Dense + Sparse 组合最经典？

将多种检索方式的结果融合，取各自之长。

```
Query → dense embedding → vector search → dense_results (top-20)
     → sparse embedding → keyword search → sparse_results (top-20)
                                            ↓
                                     Result Fusion
                                            ↓
                                   merged_results (top-20)
```

**完美互补：**

| | Dense | Sparse |
|---|---|---|
| 语义泛化 | 强（同义词、改写） | 弱 |
| 精确匹配 | 弱 | 强 |
| 稀有术语 | 弱（训练数据少） | 强（IDF高） |
| 模糊查询 | 强 | 弱 |
| 医学场景 | 症状描述→疾病 | 药物名→药品信息 |

---

### Q: Hybrid Retrieval 怎么实现？RRF 融合怎么用？

```python
def hybrid_retrieve(query, collection, top_k=20):
    """Hybrid retrieval: dense + sparse"""
    output = model.encode([query], return_dense=True, return_sparse=True)
    dense_vec = output['dense_vecs'][0]
    sparse_vec = output['lexical_weights'][0]
    
    dense_results = collection.search(
        data=[dense_vec], anns_field="dense_embedding",
        param={"metric_type": "IP", "params": {"nprobe": 16}}, limit=top_k,
    )
    sparse_results = collection.search(
        data=[sparse_vec], anns_field="sparse_embedding",
        param={"metric_type": "IP"}, limit=top_k,
    )
    
    merged = reciprocal_rank_fusion(dense_results, sparse_results, k=60)
    return merged[:top_k]
```

**Reciprocal Rank Fusion (RRF)：**

```python
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    RRF_score(d) = Σ 1/(k + rank_i(d))
    """
    scores = {}
    for rank, result in enumerate(dense_results):
        scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)
    for rank, result in enumerate(sparse_results):
        scores[result.id] = scores.get(result.id, 0) + 1 / (k + rank + 1)
    
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]
```

**为什么用 RRF 而不是分数归一化：**

> "Dense 的 cosine similarity 和 sparse 的 BM25/learned weight 分数分布完全不同——dense 分数通常在 0.5-1.0 之间，sparse 分数分布更广。直接比较或加权是 Apples to Oranges。RRF 只用排名信息，规避了分数分布不一致的问题。k=60 是一个经验参数，控制排名靠后的文档被降权的程度。"

---

### Q: Dense-only vs Hybrid 的对比实验结论是什么？

在我们的 53 题评测集上：

```
检索策略           Recall@5    Recall@20    MRR
Dense-only         0.78        0.87         0.65
Sparse-only        0.62        0.72         0.48
Hybrid (RRF)       0.85        0.93         0.73
Hybrid + Rerank    0.92        0.97         0.81
```

"Hybrid + Rerank"把 Recall@5 从 0.78 提到了 0.92——**这意味着 top-5 中几乎总有一个正确的 chunk。**

**为什么除了 dense+sparse，还要规则路由：** Hybrid retrieval 解决的是"怎么搜"，规则路由解决的是"在哪搜"。规则路由根据 query 类型缩小候选空间（紧急情况 query → 只搜急救指南相关的 chunks、药物 query → 只搜 drug_info=True 的 chunks 等）。这相当于在 11707 chunks 的数据库中预先做了水平分表。

---

## H. Rerank

### Q: 什么是 Rerank？为什么需要它？

Rerank（重排序）是在粗召回（retrieval）之后，用更强的模型对候选文档做精细排序。

```
粗召回 (Retrieval): 高 recall，低 precision
精排序 (Rerank):    高 precision，从 top-K 中选出 top-k
```

粗召回用的 embedding 模型做的是"单塔"比对——query 和 document 分别 embedding，然后算相似度。这种架构的优势是快（document embedding 可以预先计算），但精度有限。

Reranker 做的是"双塔"或"交叉"比对——query 和 document 一起送入模型，做深度的 token-level 交互。代价是慢（每个 pair 都要 inference），但精度高得多。

```
单塔 (Bi-Encoder):  
  Query → [Encoder] → v_q
  Doc   → [Encoder] → v_d
  Score = cos(v_q, v_d)
  ✓ 快 ✗ 精度有限

交叉 (Cross-Encoder):
  [Query, Doc] → [Encoder] → Score
  ✓ 精度高（深度交互）✗ 慢（每个 query-doc pair 都要计算）
```

---

### Q: 我们用的 Reranker 是什么？效果如何？

**bge-reranker-v2-m3：** 与 BGE-M3 embedding 同系列，兼容性好；支持多语言（中英文均可）；支持 8192 token 输入；在中文 benchmark 上表现优异。

```python
from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)

def rerank(query, candidates, top_k=5):
    pairs = [[query, cand] for cand in candidates]
    scores = reranker.compute_score(pairs, normalize=True)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
```

**Rerank 的计算量分析：**

```
粗召回返回 top-20 candidates
每个 candidate 与 query 组成一个 pair
20 pairs × ~50ms (cross-encoder inference) ≈ 1 second
但 20 个 pairs 可以 batch 推理 → ~200ms（batch_size=20）
```

**效果提升：**

```
Without Rerank:  Recall@5: 0.85, Precision@5: 0.62
With Rerank:     Recall@5: 0.92 (+8%), Precision@5: 0.78 (+26%)

结论: Rerank 在 precision 上的提升远大于 recall——因为它主要淘汰的是"看起来相关但不准确的"结果。
```

**面试追问：为什么不直接用 Reranker 搜全量？**

> "成本不允许。如果用 Cross-Encoder 跑全量 11707 个 chunks，每个 query 需要 11707 次 inference，每次 ~50ms，总计约 10 分钟，无法实时。而 embedding retrieval 只需要做一次 query embedding + 一次向量相似度计算（11707 次内积 = <1ms），把候选压缩到 top-20，再用 reranker 细排——这是精度和速度的最优平衡点。"

---

## I. Query 处理

### Q: 为什么 Query 处理在医学 RAG 中尤其重要？8 条路由规则是什么？

> "用户不会用医学术语提问。他们可能说'我头很痛，左边，一跳一跳的'而不是'左侧搏动性头痛'。query 处理的作用就是把用户的自然语言转化为检索系统能理解的表示。在医学场景下，这还涉及安全紧急度判断——'我胸口痛'和'我昨天胸口有点不舒服'的处理优先级完全不同。"

**我们设计的 8 条路由规则：**

```python
ROUTING_RULES = [
    {
        "name": "emergency",
        "pattern": r"胸痛|呼吸困难|大出血|意识丧失|突然.*不能|猝死|心脏停跳",
        "domain_filter": "emergency_medicine",
        "priority": 1,  # 最高优先级
        "safety_level": "high",
    },
    {
        "name": "medication",
        "pattern": r"药|吃.*片|剂量|副作用|禁忌|怎么服用",
        "domain_filter": "pharmacology",
        "metadata_filter": {"contains_drug_info": True},
        "priority": 2,
    },
    {
        "name": "rare_disease",
        "pattern": r"罕见病|少见|发病率.*低|特殊类型|不典型",
        "domain_filter": "rare_diseases", "priority": 3,
    },
    {
        "name": "oncology",
        "pattern": r"癌|肿瘤|恶性|化疗|放疗|转移|靶向",
        "domain_filter": "oncology", "priority": 3,
    },
    {
        "name": "cardiovascular",
        "pattern": r"心脏|血管|血压|心电|血脂|冠心病|心梗",
        "domain_filter": "cardiology", "priority": 3,
    },
    {
        "name": "neurology",
        "pattern": r"脑|神经|头痛|癫痫|帕金森|中风|卒中",
        "domain_filter": "neurology", "priority": 3,
    },
    {
        "name": "endocrine",
        "pattern": r"血糖|糖尿病|甲状腺|激素|内分泌",
        "domain_filter": "endocrinology", "priority": 3,
    },
    {
        "name": "general",
        "pattern": r".*",
        "domain_filter": None, "priority": 99,  # 最低优先级
    },
]
```

**Emergency Route 的特殊处理：** 如果 query 被路由到 emergency，不依赖 RAG 检索的延迟，直接注入预置的急救安全规则，生成的回答中强制包含就医建议。

---

### Q: Synonyms 扩展和 Query 改写怎么做？

**synonyms.yaml：**

```yaml
# synonyms.yaml - 医学同义词扩展
高血压:
  - 血压高
  - 高血压病
  - HTN
  - hypertension
脑出血:
  - 脑溢血
  - 颅内出血
  - 出血性脑卒中
  - ICH
心肌梗死:
  - 心梗
  - 心脏病发作
  - 急性心梗
  - AMI
  - MI
```

**实现：**

```python
def expand_query_with_synonyms(query):
    expanded_terms = []
    for term, synonyms in SYNONYMS.items():
        if term in query:
            expanded_terms.extend(synonyms[:2])  # 最多加2个同义词
    if expanded_terms:
        return query + " " + " ".join(expanded_terms)
    return query
```

**医学 Query 的特殊处理——否定检测：**

```
"是不是不能吃华法林？" 
→ 检测到否定 → 检索"华法林 禁忌症 注意事项" 而非仅仅检索"华法林 用法"

"没有高血压需要吃降压药吗"
→ 检测到"没有" → 检索"降压药 适应症 正常血压 用药"
```

---

## J. Context 组织

### Q: Context 组织怎么做？为什么重要？

> "Context 组织是 RAG 中被严重低估的环节。假设你检索到了 5 个高质量 chunks，但如果排列顺序混乱、包含冗余、格式不一致，LLM 可能被 confused——甚至忽略正确的 chunk 而采信了错误的内容。在医学场景中，这可能导致严重的安全问题。一个好的 context 组织应该让最权威、最相关的信息出现在最显眼的位置。"

**Context 组织的 Pipeline：**

```python
def organize_context(retrieved_chunks, query, safety_rules=None):
    # Step 1: 去重
    chunks = deduplicate_chunks(retrieved_chunks)
    # Step 2: 按相关性 + 权威性排序
    chunks = sort_by_relevance_and_credibility(chunks)
    # Step 3: 截断到 token 预算
    chunks = truncate_to_budget(chunks, max_tokens=2048)
    # Step 4: 格式化（添加来源标注）
    formatted = format_chunks_with_citation(chunks)
    # Step 5: 注入安全规则（如果有）
    if safety_rules:
        formatted = inject_safety_rules(formatted, safety_rules)
    return formatted
```

---

### Q: 排序策略、Token 预算和 Lost in the Middle 怎么处理？

**排序策略——Relevance + Credibility：**

```python
def sort_by_relevance_and_credibility(chunks):
    """
    排序权重: rerank_score: 60%, credibility: 25%, recency: 10%, source_match: 5%
    """
    credibility_map = {"high": 1.0, "medium": 0.6, "low": 0.3}
    def sort_key(chunk):
        relevance = chunk.rerank_score
        credibility = credibility_map.get(chunk.credibility, 0.5)
        recency = 1.0 if chunk.year >= 2020 else 0.7
        return 0.6 * relevance + 0.25 * credibility + 0.1 * recency
    return sorted(chunks, key=sort_key, reverse=True)
```

**Lost in the Middle 现象与对策：** LLM 倾向于关注 prompt 开头和结尾的信息，中间的内容容易被忽略。我们的对策：最重要的 chunk 放在开头（primacy effect）、第二重要的 chunk 放在结尾（recency effect）、中等重要的放在中间。

**Token 预算管理：** LLM 的 context window 是有限的（Qwen3-8B = 32K tokens）。Context 不是越大越好——信息过载会导致 LLM 忽略关键信息（Lost in the Middle 现象）。我们严格控制 context 在 2000 token 以内（约 5-6 个 chunk），优先保证利用率而非覆盖率。

---

### Q: Safety Rules 注入和最终 Prompt 结构是什么？

在 context 的末尾（显眼位置）注入硬性安全规则：

```python
SAFETY_RULES = """
## ⚠️ 回答安全规则（必须严格遵守）

1. **绝对禁止**推荐任何处方药的具体用法用量，必须注明"请遵医嘱"
2. **绝对禁止**在紧急症状下建议"观察""休息"，必须建议立即就医
3. **必须**在回答末尾添加免责声明："以上信息仅供参考，不能替代专业医疗诊断"
4. **必须**在不确定时明确表达"我不能确定，建议咨询医生"
5. **禁止**对任何疾病的预后做出确定性断言
6. **禁止**编造医学研究、数据或引用不存在的文献
"""
```

**最终的 Prompt 结构：**

```
┌─────────────────────────────────┐
│ SYSTEM PROMPT                   │
│ "你是一个医学知识助手..."       │
├─────────────────────────────────┤
│ RETRIEVED CONTEXT               │
│ [来源1] ... [来源2] ...         │
├─────────────────────────────────┤
│ SAFETY RULES（安全规则）         │
│ 1. 禁止推荐处方药用法用量       │
├─────────────────────────────────┤
│ USER QUERY                      │
│ "用户的问题是：..."             │
├─────────────────────────────────┤
│ 留给 LLM 生成的回答             │
└─────────────────────────────────┘
```

---

## K. 生成阶段

### Q: RAG 的生成 Prompt 和参数怎么设置？

```python
RAG_SYSTEM_PROMPT = """你是一个基于医学知识库的AI助手。你的回答必须严格基于提供的参考来源。
如果参考来源中没有足够信息，请明确说明，并建议用户咨询专业医生。
绝对不要编造参考来源中没有的医学信息。

## 回答规则
1. 优先使用参考来源中的信息
2. 引用信息时标注来源编号，如[来源1]
3. 如有不确定，请诚实表达
4. 涉及用药、诊断、治疗时，必须包含就医建议
5. 紧急症状场景必须优先建议立即就医
"""
```

**生成参数设置：**

```python
generation_config = {
    "temperature": 0.3,        # 低温度 → 减少随机性 → 减少幻觉
    "top_p": 0.9,
    "max_tokens": 2048,
    "do_sample": True,
    "repetition_penalty": 1.05,
}
```

**为什么 temperature=0.3 而不是 0：**

> "在医学场景中，temperature=0（greedy decoding）可能更安全（确定性高），但实际测试发现 temperature=0.3 在保持医学准确性的同时，回答更自然，共情更好。我们做了小规模对比，0.3 和 0 的安全分数差异不显著，但 helpfulness 和 empathy 有明显提升。"

---

### Q: RAG 生成 vs 无 RAG 生成的对比？生成后安全检查怎么做？

**实际案例对比：**

```
Case: "脑溢血患者在家应该怎么处理"

无RAG（DPO-only）回答:
"脑溢血患者在家中应保持平卧，可考虑服用降压药和溶栓药物..."
→ 致命错误：脑溢血是溶栓的绝对禁忌症

有RAG（DPO+RAG）回答:
"根据《中国脑出血诊治指南》[来源1]，脑出血属于急症，患者应立即就医。
不建议自行服用任何药物。根据指南，溶栓治疗是脑出血的禁忌症[来源3]。
在等待救护车期间，应让患者平卧、头部偏向一侧、保持呼吸道通畅[来源2]。"
→ 正确且安全
```

**生成后安全检查：**

```python
def post_generation_safety_check(response, query):
    checks = []
    # 检查1: 是否包含免责声明
    if "仅供参考" not in response and "不能替代" not in response:
        checks.append({"type": "missing_disclaimer", "severity": "medium"})
    # 检查2: 是否在紧急query中推荐就医
    if is_emergency_query(query):
        if "就医" not in response and "120" not in response:
            checks.append({"type": "missing_er_advice", "severity": "high"})
    # 检查3: 是否提到了特定的处方药和剂量
    drug_dose_pattern = r'\d+\s*(mg|mg/d|g|μg|mcg|ml)'
    if re.search(drug_dose_pattern, response) and "遵医嘱" not in response:
        checks.append({"type": "drug_dosage_without_disclaimer", "severity": "high"})
    # 检查4: 幻觉检测（回答中的实体是否出现在context中）
    entities_in_response = extract_medical_entities(response)
    entities_in_context = extract_medical_entities(context)
    novel_entities = entities_in_response - entities_in_context
    if novel_entities:
        checks.append({"type": "potential_hallucination", "entities": list(novel_entities)})
    return checks
```

---

### Q: 面试中关于 RAG 生成阶段的追问怎么回答？

**Q: RAG 会不会限制模型的推理能力？**

> "存在这种风险。如果 prompt 中 context 占比过大，模型会退化成一个'摘要器'而非'推理器'。我们在 prompt 中会加入'请综合参考来源的信息，结合你的医学知识进行推理分析'这样的指令，同时保持 context 在 2000 token 左右，避免过度约束。另外，context 提供的是事实依据，推理仍然由模型完成。"

**生成温度对医学准确性的影响（实测）：**

```
Temperature 0.0:  准确性最高，但回答生硬
Temperature 0.3:  准确性几乎无下降，自然度明显提升
Temperature 0.7:  开始出现轻微的事实偏差
Temperature 1.0+: 幻觉率显著上升

结论: 医学RAG场景 temperature 不应超过0.5
```

---

## L. RAG 评测

### Q: RAG 评测的三层框架是什么？核心指标有哪些？

```
Layer 1: Retrieval Quality（检索质量）
  - Recall@K, Precision@K, MRR, NDCG
  
Layer 2: Generation Quality（生成质量）
  - Faithfulness（是否忠于context）
  - Answer Relevance（是否回答了问题）
  - Context Relevance（context是否相关）
  
Layer 3: End-to-End Impact（端到端效果）
  - Overall score improvement
  - Hallucination rate reduction
  - Safety score improvement
```

**RAGAS 评测框架三个核心指标：**
1. **Faithfulness（忠实度）**：生成的内容是否都能在 context 中找到支撑
2. **Answer Relevancy（答案相关性）**：回答是否针对问题
3. **Context Relevancy（上下文相关性）**：检索到的 context 是否与问题相关

---

### Q: Grounding Score 是什么？和 Answer Correctness 有什么区别？

grounding score 衡量回答中有多少事实断言可以在检索到的 context 中找到支撑。

```python
def compute_grounding_score(response, context_chunks, judge_model):
    prompt = f"""
    评估以下回答的 "grounding"程度。
    【参考来源】{context_chunks}
    【AI回答】{response}
    请判断回答中的每个医学事实断言是否能在参考来源中找到支撑。给出 1-5 分。
    """
    return judge_model(prompt)
```

在我们的 53 题评测中，DPO+RAG 的 grounding score = 3.68，比 DPO-only（3.45）提升了 0.23。

**Grounding 高 ≠ Correctness 高：**
- Context 本身就有错误 → grounding 高但 correctness 低
- 模型正确使用了自己的知识但 context 中没有 → grounding 低但 correctness 高
- Context 正确且模型用了 → grounding 高且 correctness 高（理想情况）

这要求我们在构建知识库时必须保证 context 的质量——**Garbage Context In, Garbage Answer Out.**

---

### Q: 我们项目中的实际评测数据是什么？

```
DPO+RAG vs DPO-only (53 questions, 8-dim LLM-as-Judge):

Overall Score:      3.78 → 3.99 (+0.21, +5.6%)
Grounding Score:    3.45 → 3.68 (+0.23, +6.7%)
Hallucination:      2.80 → 2.95 (+0.15, +5.4%)
Correctness:        3.85 → 3.95 (+0.10, +2.6%)
Safety:             3.60 → 3.85 (+0.25, +6.9%)

分层（按风险等级）:
LOW risk (n=42):    4.10 → 4.15 (+0.05)
MEDIUM risk (n=8):  3.20 → 3.55 (+0.35)
HIGH risk (n=3):    2.10 → 3.40 (+1.30)
```

**Retrieval 消融实验：**

```
配置                              Recall@5    Final Score
Full (Dense+Sparse+Rerank)        0.92        3.99
- Rerank                          0.85        3.82
- Sparse                          0.78        3.71
- Rerank - Sparse (Dense only)    0.78        3.64
No Retrieval (DPO-only)           -           3.78
```

每个组件都有正向贡献，Rerank 的贡献最大。

**面试追问：评测怎样服务于迭代？**

> "评测不是一次性的。我们在 SFT→DPO→RAG 每一轮迭代中都跑完整的评测 pipeline。SFT 后评测发现 answer quality 提升了但 safety 无明显改进。DPO 后评测发现 overall 提升了但高风险 case 出现了退化。正是这个发现驱动了 Safety-RAG 的开发。评测数据驱动了每一次技术决策，而非拍脑袋。"

---

## M. RAG 与后训练结合

### Q: RAG 与 SFT、DPO 分别是什么关系？

**RAG 与 SFT 的关系：** RAG 和 SFT 不是互斥的。SFT 让模型学会回答的格式、风格、医学思维框架；RAG 在推理时提供实时的知识注入。我们的策略：**先 SFT 打好基础（学会格式和共情），再 RAG 注入知识（保证准确性）。**

**RAG 与 DPO 的关系（核心发现）：**

> "DPO 让模型变得更'自信'——在所有问题上都倾向于给出详细回答。这在大多数 case 上是好事（更 helpful），但在高风险医学长尾 case 上是危险的（更可能出现致命幻觉）。RAG 的任务不是反转 DPO 的效果，而是在 DPO 的基础上增加一个安全阀——当检索到的高质量 context 与模型内部知识冲突时，引导模型信任外部知识而非内部可能错误的记忆。"

**RAG 作为 Alignment Tax 的补救：**

```
DPO-only:      整体好, 高风险差   (Alignment Tax)
DPO+RAG:       整体更好, 高风险好  (Tax 被 RAG 抵消)
```

---

### Q: RAFT 是什么？RAG 生成的数据能反哺训练吗？

**RAFT（Retrieval Augmented Fine-Tuning）：** 在 SFT 阶段就混入带 context 的训练样本，让模型学会"如何利用检索到的文档"。

```python
raft_sample = {
    "system": "你是一个医学助手。请基于提供的参考文档回答问题。",
    "context": "【参考文档1】华法林的主要不良反应是出血...",
    "question": "华法林最常见的副作用是什么",
    "answer": "根据参考文档，华法林最常见的不良反应是出血[来源1]..."  # 带引用
}
```

我们在项目中没有使用 RAFT，但 RAFT 是未来可以尝试的方向——如果模型经常忽略 context，RAFT 可以改善这个问题。

**RAG 生成的数据反哺训练：** RAG 生成的高质量、带引用的回答可以作为下一轮 SFT/DPO 的训练数据，让模型内化"回答时要给出引用"的行为模式。

**为什么不能只用 RAG 不用训练：**

> "纯 RAG 的问题在于模型没有学会'如何使用 context'。一个只做过 pre-training 的模型面对 context 时可能：1）忽略 context 继续用内部知识，2）被 context 中的噪声误导，3）回答过于生硬像在复读。SFT + DPO 让模型理解了医学回答的格式、共情表达、安全声明的重要性——这些 RAG 本身提供不了。"

**面试追问：你觉得后训练和 RAG 的最终形态是什么？**

> "我认为最终的形态是'RAG-aware Fine-tuning'——在训练阶段就让模型学会如何与外部知识库协作。不是简单地'把训练和 RAG 分开做然后再合起来'，而是训练时就考虑了 RAG 的存在。比如：训练数据中混合有 context 和无 context 的样本，让模型学会判断什么时候需要检索、什么时候依赖自己的知识、什么时候应该承认不知道。"

---

## N. 我的 Safety-RAG 项目讲法

### Q: Safety-RAG 项目 1 分钟怎么介绍？

> "我在医学 LLM Teacher 项目中设计并实现了一套 Safety-RAG 系统，用于解决 DPO 训练后模型在部分高风险医学长尾问题上幻觉加剧的问题。我用 15 本医学教材和指南构建了包含 11707 个 chunks 的知识库，搭建了五层 RAG 架构：规则路由（8 条规则，区分紧急/药物/罕见病等场景）、metadata 过滤、BGE-M3 的 hybrid retrieval（dense+sparse 双路召回）、bge-reranker-v2-m3 精排、以及 safety rules 硬注入。在 53 题评测中，DPO+RAG 相比 DPO-only 在 overall 上提升 0.21，grounding 提升 0.23，hallucination 改善 0.15。3 个高风险挑战案例中，致命幻觉被修正了 2 个。"

### Q: Safety-RAG 项目 3 分钟怎么介绍？

> "这个项目的背景是：我们用 Qwen3-8B 做医学 Teacher，经历了 Teacher 数据生成→SFT→DPO→LLM-as-Judge 的完整管线。在 LLM-as-Judge 的 medical safety review 中，我们发现了一个关键问题：DPO 训练后的模型在大多数问题上表现很好，但在 3 个高风险医学长尾问题上出现了严重幻觉——最典型的是脑溢血案例，DPO-only 竟然推荐了溶栓药物，这在临床上是绝对禁忌症，可能导致患者死亡。
>
> 我分析了根因：DPO 优化的是偏好对齐，而不是事实对齐。偏好数据中详细但错误的回答可能得分高于简短但诚实的拒绝。加上 DPO 训练中长尾安全样本被高频模式稀释，导致模型在不确定时变得'过度自信'。
>
> 解决方案是 Safety-RAG——不是在训练阶段消除幻觉（这很难），而是在推理阶段为模型提供外部知识支撑。我设计了五层架构：
>
> 第一层是规则路由，8 条正则规则将用户 query 分类到不同的医学领域，缩小检索空间并触发不同的安全策略。
>
> 第二层是 metadata filter，利用每个 chunk 的元数据在向量检索前做剪枝。
>
> 第三层是 hybrid retrieval，用 BGE-M3 同时输出 dense 和 sparse 两种向量，dense 负责语义泛化、sparse 负责精确医学术语匹配，两者通过 RRF 融合，recall@20 达到 0.93。
>
> 第四层是 reranker，用 bge-reranker-v2-m3 把 top-20 精排到 top-5，recall@5 从 0.85 提升到 0.92。
>
> 第五层是 safety rules 注入，在 prompt 中硬性植入 6 条安全规则，作为 LLM 生成时的硬约束。
>
> 效果方面，53 题的端到端评测显示：overall +0.21，grounding +0.23，hallucination +0.15。特别关键的是高风险 case 的安全分从 2.1 提升到 3.8。3 个挑战案例中，脑溢血和甲状腺癌的被完全修正，MRCNS 的被部分修正。我也诚实地说，hallucination 下降 66.7% 是案例层面的发现，不是统计结论。"

### Q: Safety-RAG 项目 5 分钟怎么介绍（含细节）？

> [包含 3 分钟版本的所有内容，再加上以下细节]
>
> **知识库构建细节：** 15 本 PDF 涵盖了内科学、外科学、药理学、急诊医学、神经病学等核心领域。原始约 8000 页，PyMuPDF 解析后用自定义的清洗 pipeline 处理了页眉页脚、断行修复、药物名标准化、Unicode 规范化。Chunking 用的是 recursive character split（chunk_size=512, overlap=64），保护医学语义边界不切割药物信息和诊断标准。最终 11707 chunks，每个携带 10+ 个元数据字段。
>
> **Synonyms 同义词扩展：** 建立了一个 synonyms.yaml，覆盖了 200+ 组医学同义词。比如"脑溢血→脑出血→颅内出血→出血性脑卒中→ICH"让稀疏检索可以召回更多相关的 chunk。
>
> **工程踩坑：** 最大的坑是 judge prompt 的 JSON 输出不稳定——MiMo-v2.5-pro 有时候会在 JSON 外面套 markdown 代码块，有时候多输出一个逗号。我加了 JSON 格式校验+重试机制，两次重试失败就走人工标记。另一个坑是 Milvus 的 sparse vector 插入——BGE-M3 的 sparse vector 输出是 dict 格式，需要转换成 Milvus 支持的稀疏向量格式。
>
> **下一步改进方向：** 1）构建更大规模的安全 benchmark（500+ 题）；2）用 RAG 生成的高质量带引用回答反哺 DPO 训练；3）Agentic RAG——让模型在不确定时主动触发二次检索或自我反思；4）引入图表解析能力。

### Q: 面试官追问精选（30 个典型问题）

**Q1: 为什么选择 BGE-M3 而不是其他 embedding 模型？**

> "BGE-M3 有三个核心优势：一是 dense + sparse 双向量能力，一个模型支持 hybrid retrieval。二是多语言支持。三是实测效果——在我们的小规模对比中（20 个医学 query），BGE-M3 的 recall@5 是 0.87，优于 text-embedding-3-large 的 0.84。"

**Q2: 你的 Hallucination Rate 是怎么计算的？**

> "我们用 LLM-as-Judge 在 hallucination 维度上打分，分数 ≤3 分（5 分制）视为存在幻觉。关键是分层统计——DPO-only 在 HIGH risk case 上的幻觉率是 100%（3/3），DPO+RAG 降到了 33%（1/3）。如果不分层看 overall rate，会被大量低风险 case 稀释掉关键信号。"

**Q3: 规则路由的正则是硬编码的？会不会太脆弱？**

> "我承认这是一个 trade-off。8 条正则规则是工程师直觉 + 医学常识的产物，不是最优方案。优点是快（<1ms）、解释性强、可以精确控制紧急场景的触发。缺点是覆盖不全、需要人工维护。下一步可以考虑用训练好的轻量级分类器（BERT-based）替代正则，同时保留正则作为高优先级的安全 override。"

**Q4: RAG 在什么情况下会降低性能？**

> "在我们的 53 题评测中，有 2 个 case RAG 得分低于 DPO-only。原因是检索到了'部分相关但关键细节有误'的 context，模型被误导了。这提醒我们：RAG 是双刃剑——context 质量决定了一切。"

**Q5: 你的模型是 8B，如果用更大的模型（如 70B），RAG 还需要吗？**

> "需要。大模型可能更少出现幻觉，但不是零幻觉。而且医学知识是动态更新的——即使是 GPT-4，如果它的训练数据截止到 2023 年，它就不会知道 2024 年更新的临床指南。RAG 解决的是知识时效性和可溯源性问题，这与模型大小不完全相关。"

**Q6: 你对 RAG 在生产环境中的延迟有信心吗？**

> "100-150ms 的 RAG 开销在 2-10 秒的总延迟中几乎不可感。如果未来需要极致优化，可以用模型量化（BGE-M3 INT8）、更小的 reranker、cache query embedding 等手段压缩到 50ms 以内。"

**Q7: 你做的过程中有没有什么出人意料的发现？**

> "最让我意外的是 DPO 加剧幻觉这件事。我原本以为 preference alignment 应该是全方面提升，但实际数据告诉我 alignment 是 trade-off——提升了 helpfulness 就可能在 safety 上付出代价。这件事让我深刻理解了 alignment tax 的概念。"

**Q8: 这个项目的最大创新点是什么？**

> "我认为有两个。一是将 Safety-RAG 定位为 DPO 后训练 pipeline 中的安全补丁——它不是独立的 RAG 系统，而是为 DPO 的 Alignment Tax 做补救。二是五层架构的 defense-in-depth 设计——从路由到检索到精排到规则注入到检查，多层次保障医学安全。"

**Q9: 如何向非技术人员解释 RAG？**

> "可以把 LLM 想象成一个考试中的学生。没有 RAG 时，它是闭卷考试——全靠记忆。RAG 就是给这个学生一本参考书——在回答每个问题之前，先翻到相关的章节，然后基于找到的信息来回答。书的目录就是我们的路由规则，翻书的动作是检索，书的索引就是 embedding。"

**Q10: 如何解释 Hybrid Retrieval + Rerank？**

> "Hybrid Retrieval 是在图书馆中同时用两种方式找书。一种是根据书的主题分类（dense——语义相似），一种是按书名关键词精确查找（sparse——关键词匹配）。两种结果用 RRF 算法融合，找到 20 本最相关的书。Rerank 是让一个专家浏览这 20 本书的目录和简介，选出最相关、最权威的 5 本。"

**Q11-Q30:** 涵盖文档去重、RAG 延迟、fallback 机制、表格处理、embedding fine-tuning、安全边界、下一步改进、面试自我介绍等——完整回答见原文档项目讲法章节。

---

## 背诵版总结

### 核心数字速记

```
知识库: 15 本 PDF → 8000 页 → 11707 chunks
Embedding: BGE-M3, 1024-dim
Chunk: 512 tokens, overlap 64
路由规则: 8 条 (emergency/medication/rare_disease/oncology等)
Retrieval: Dense + Sparse → RRF 融合 → Rerank (20→5)
Reranker: bge-reranker-v2-m3
安全规则: 6 条硬约束注入 prompt

评测结果 (53题):
  Overall:     DPO-only 3.78 → DPO+RAG 3.99 (+0.21)
  Grounding:   3.45 → 3.68 (+0.23)
  Hallucination: 2.80 → 2.95 (+0.15)
  HIGH风险:    2.10 → 3.40 (+1.30) ← 关键提升!

3个挑战案例: 脑溢血溶栓错误→修正, 甲状腺癌分类错误→修正, MRCNS→部分修正
幻觉: 3→1, 下降66.7% (案例层面, 非统计结论)
```

### 核心架构速记

```
五层 Safety-RAG 架构:
1. 规则路由 (8 rules) → 缩小候选空间
2. Metadata Filter → 来源可信度过滤
3. Hybrid Retrieval (D+S) → 粗召回 top-20
4. Reranker → 精排 top-5
5. Safety Rules → 硬约束注入

检索指标: Recall@5: 0.92, Recall@20: 0.93, Hit Rate@5: 0.94, MRR: 0.81
RAG 延迟: ~100-150ms
```

### 面试必背概念

```
1. RAG = 开卷考试 (检索 + 生成)
2. Chunk Strategy = 512 + 64 overlap + 医学语义保护
3. Dense + Sparse: 语义泛化 + 精确匹配
4. RRF: 1/(k+rank) 融合, 不依赖原始分数分布
5. Rerank: Cross-Encoder 精排, 提升 precision
6. Lost in the Middle: 中间信息被忽略, 重要的放两端
7. Alignment Tax: DPO 提升 helpfulness 但代价是 safety
8. Guardrails: 输入-检索-输出多层安全护栏
9. Grounding Score: 回答中的事实能否追溯到 context
10. 幻觉检测: 回答中出现了 context 中没有的医学实体
```

### 面试高频 10 问答

**Q: 为什么需要 RAG?**
A: 知识截止日期、幻觉、更新成本。医学场景 error cost 极高，需要 external knowledge grounding。

**Q: 为什么选 BGE-M3?**
A: Dense+Sparse 双向量、多语言、8192 token、开源可部署、实测 recall 最优。

**Q: Hybrid Retrieval 怎么做?**
A: BGE-M3 同时出 dense+sparse 向量 → 分别检索 top-20 → RRF 融合 → 保留 top-20。

**Q: 为什么需要 Rerank?**
A: 粗召回高 recall 低 precision，Rerank 用 Cross-Encoder 做精细比对，提升 precision。

**Q: Chunk size 怎么定的?**
A: Grid search 256/512/768/1024，512 在 recall 和 precision 间最优。

**Q: 如何处理检索失败?**
A: Reranker score < 0.3 → 扩大检索 → 仍失败 → 回退 DPO 模式 + 强安全声明。

**Q: RAG 和 Fine-tuning 的关系?**
A: 互补。FT(SFT+DPO) 学回答格式和风格，RAG 提供事实依据。不是替代而是互补。

**Q: 为什么 DPO 后幻觉加剧?**
A: DPO 优化偏好非事实，偏好数据中详细错误 > 简短正确拒绝，模型学会过度自信。

**Q: 幻觉下降 66.7% 可信吗?**
A: 案例层面定性验证（3→1），非统计结论。Must add disclaimer every time.

**Q: 下一步做什么?**
A: 500+ safety benchmark、RAG 数据反哺 DPO、Agentic RAG、embedding fine-tuning、多模态。
