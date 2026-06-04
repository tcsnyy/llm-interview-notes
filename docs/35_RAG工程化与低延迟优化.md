# 35_RAG工程化与低延迟优化

> 本文件聚焦 RAG 的工程化落地、延迟拆解、并行化策略、缓存设计、动态路由和服务化部署。面向"大模型后训练算法实习"面试，所有内容围绕医学 Safety-RAG 项目展开。
> **项目状态说明**：RAG 知识库构建和检索 pipeline 已完成；生产级 RAG 服务化、完整缓存体系、分布式 RAG 架构属于扩展方向。

---

## A. RAG 端到端链路拆解

### 面试官可能怎么问
"RAG 一次请求从用户输入到返回结果，中间经过哪些阶段？哪一段最慢？"

### 这个问题考察什么
对 RAG 全链路延迟的工程级理解，不只是知道"检索+生成"四个字。

### 我的标准回答

RAG 端到端链路（以医学 Safety-RAG 为例）：

```
用户请求(100ms 网络)
→ query normalization (1-2ms) - 中文标点归一化、多余空格清理
→ 风险分类 (5-50ms*) - 基于 routing_rules.yaml 关键词+同义词匹配
→ query rewrite (0-200ms*) - 可选，扩展同义词和医学术语
→ embedding 推理 (20-100ms) - BGE-M3 1024-dim，CPU约300ms，GPU约20ms
→ dense retrieval (5-50ms) - FAISS FlatIP 在 11707 chunks 上检索
→ sparse retrieval (5-30ms) - BM25 关键词检索
→ hybrid fusion (1-2ms) - RRF 融合
→ rerank (50-500ms) - cross-encoder bge-reranker-v2-m3 对 top-k 候选精排
→ context assembly (1-3ms) - 去重、排序、截断
→ prompt construction (1-2ms) - 注入 safety_rules + evidence + 用户问题
→ vLLM prefill (50-300ms) - 取决于 context 长度，O(n²)注意力
→ vLLM decode (100-5000ms) - 取决于输出长度，每token约9ms(单并发)
→ safety check (10-100ms*) - 可选，规则检查或额外 LLM 调用
→ citation check (5-20ms) - 验证引用是否来自检索文档
→ response formatting (1-2ms)
→ 返回用户 (100ms 网络)
```

**标注 * 的阶段为可选**，取决于路由策略和缓存命中情况。

### 典型瓶颈（按延迟占比排序）
1. **vLLM decode**：生成 token 数量 × 每 token 延迟，输出 100 token 约 900ms
2. **rerank**：cross-encoder 对每个候选做精细打分，top-20 约需 200-500ms
3. **vLLM prefill**：context 800 token 约 150ms，2000 token 约 400ms
4. **embedding**：CPU 推理 BGE-M3 约 300ms，GPU 可降到 20ms
5. **dense retrieval**：FAISS FlatIP 在 1 万向量上很快（~10ms），10 万以上需要 IVF/HNSW

### 和我的项目怎么结合
- 我的 Safety-RAG 在 RTX 5090 单卡上，embedding 用 CPU（GPU 被 vLLM 占用），embedding ~300ms/次
- reranker 因网络未下载，当前直接使用 FAISS 向量检索的 top-k 结果
- 端到端延迟约 2-5 秒（无 rerank），加入 rerank 后预计增加 200-500ms

### 易错点
- 不要说"检索很快"——在百万级向量库上，FlatIP 会显著变慢
- 不要忽略网络延迟——实际系统中网络往返占 5-15% 总延迟
- 不要把 rerank 和 retrieval 混在一起——它们是两个独立阶段

---

## B. RAG 延迟指标

### 面试官可能怎么问
"RAG 系统需要监控哪些延迟指标？"

### 这个问题考察什么
对 RAG 性能工程化的系统思维。

### 我的标准回答

| 指标 | 定义 | 典型值(医学RAG) | 优化手段 |
|------|------|---------------|---------|
| embedding latency | 查询向量化耗时 | 20-300ms | GPU推理/batch化/缓存 |
| dense retrieval latency | 向量相似度搜索 | 5-50ms | IVF/HNSW索引/分片 |
| sparse retrieval latency | BM25关键词搜索 | 5-30ms | 倒排索引优化 |
| fusion latency | RRF/加权融合 | 1-2ms | 几乎可忽略 |
| rerank latency | cross-encoder精排 | 50-500ms | batch化/量化/减小k |
| prompt assembly latency | context拼接+prompt构造 | 1-3ms | 几乎可忽略 |
| prefill latency (TTFT部分) | prompt处理+首token | 50-400ms | prefix cache/压缩context |
| decode latency (TPOT) | 每token生成时间 | 5-30ms/token | 量化/减少max_tokens |
| end-to-end latency | 请求到响应完成 | 1-10s | 全链路优化 |
| p95/p99 latency | 95%/99%用户经历的延迟 | 3-15s | 缓存/限流/降级 |
| cache hit rate | 检索/生成缓存命中 | 30-70% | 增大缓存/热点预计算 |
| retrieval hit rate | 检索是否召回相关文档 | 80-95% | 优化索引/query rewrite |

### 面试 1 分钟回答
RAG 延迟可以从三个维度拆解：检索延迟（embedding+向量搜索+rerank，约 200-800ms）、生成延迟（prefill+decode，约 500-3000ms）、排队延迟（取决于并发）。关键瓶颈通常是 rerank（cross-encoder 精细打分）和 vLLM decode（逐 token 生成）。优化优先级是：先做缓存（embedding/检索/回答）、再做并行化（dense+sparse 同时跑）、最后做动态路由（简单问题跳过 RAG）。

---

## C. RAG 并行化策略（重点）

### 面试官可能怎么问
"RAG pipeline 中哪些阶段可以并行？怎么实现？"

### 这个问题考察什么
对 RAG 性能优化的工程深度，是否能识别可并行的计算路径。

### 我的标准回答

**可并行化的 20 个点**：

#### 第一层：检索并行
1. **dense retrieval 和 sparse retrieval 并行**：向量检索和 BM25 同时发起，各自独立
2. **多路召回并行**：dense + BM25 + keyword + metadata filter + medical entity search，5路同时
3. **多 query 检索并行**：query rewrite 后对原始 query + 改写 query 同时检索
4. **多索引分片并行查询**：大索引切分为多个 shard，并行检索后合并

#### 第二层：Query 处理并行
5. **query rewrite 和实体抽取并行**：LLM 改写 + 医学NER 同时进行
6. **风险分类和同义词扩展并行**：routing rule 匹配 + synonyms 扩展可并发

#### 第三层：计算批量化
7. **embedding batch 化**：多条 query 合并为 batch 一次推理
8. **rerank batch 化**：多个候选文档合并为 batch 输入 cross-encoder
9. **多用户请求的 embedding 可以 batch**

#### 第四层：服务独立化
10. **embedding 服务独立部署**：可独立扩缩容
11. **reranker 服务独立部署**：和 vLLM 分开，避免 GPU 争抢
12. **检索服务和生成服务分开**：各自按需扩容

#### 第五层：异构并行
13. **CPU BM25 + GPU dense retrieval 并行**：利用不同硬件
14. **GPU reranker + vLLM 生成**：如果有多 GPU，避免争抢

#### 第六层：后处理并行
15. **safety check 和 citation check 并行**：安全检查 + 引用验证同时跑
16. **多文档 conflict 检测和 citation 编号并行**

#### 第七层：请求级并行
17. **多用户请求并行处理**：async/await 并发
18. **检索和生成的流水线并行**：下一个请求的检索可以和当前请求的生成重叠

#### 第八层：缓存
19. **热门 query 直接走缓存**：完全跳过检索
20. **embedding 缓存**：语义相同的 query 复用 embedding

### RAG 并行流程文字图

```
                    ┌─────────────────────────┐
                    │     用户 Query           │
                    └──────────┬──────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ Embedding   │   │ BM25        │   │ 医学实体    │
   │ (GPU/CPU)   │   │ (CPU)       │   │ + Metadata  │
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          │                 │                  │
          ▼                 ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ Dense       │   │ Sparse      │   │ Filtered    │
   │ Retrieval   │   │ Retrieval   │   │ Search      │
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          │                 │                  │
          └─────────────────┼──────────────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ RRF Fusion      │  (1-2ms)
                   │ 合并去重 + 排序  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Batch Rerank    │  (100-300ms, batch=8)
                   │ Cross-Encoder   │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Context         │  (1-3ms)
                   │ Compression     │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ vLLM            │
                   │ Streaming Gen   │
                   └────────┬────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     ┌─────────────┐             ┌─────────────┐
     │ Safety Check │             │ Citation     │
     │ (规则+LLM)   │             │ Check        │
     └──────┬──────┘             └──────┬──────┘
            │                           │
            └─────────────┬─────────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Final Answer    │
                 └─────────────────┘
```

### 面试 3 分钟回答
RAG 并行化分五层：第一层是**检索并行**，dense retrieval、BM25、metadata search 同时发起——这是收益最大的并行化（-40%检索延迟）；第二层是**query处理并行**，实体抽取和风险分类同时做；第三层是**计算批量化**，embedding 和 rerank 使用 batch 推理；第四层是**服务独立化**，embedding/reranker/vLLM 各自部署独立扩缩容；第五层是**后处理并行**，safety check 和 citation check 同时跑。关键是前两层——检索并行和数据批量，其他层次根据并发量和硬件条件分步实现。医疗场景额外要求：高风险请求即使并行也不能牺牲任一召回通路（不能为了速度关闭 sparse retrieval！）。

### 和我的项目怎么结合
- 项目当前是**串行 RAG pipeline**，dense retrieval 结束后才开始其他阶段
- embedding 用 CPU（300ms），是当前主要瓶颈之一
- 后续可将 dense + sparse retrieval 改为 `asyncio.gather` 并行
- reranker 未启用（网络问题），启用后可加入 batch rerank

### 易错点
- 并行不等于变快——如果 GPU 被 vLLM 占满，并行 embedding 反而更慢（排队）
- 在单 GPU 机器上，GPU reranker + vLLM + GPU embedding 会互相抢资源
- 医疗场景的特殊要求：不能为了速度关闭 BM25（医学术语精确匹配至关重要）

---

## D. RAG 缓存策略

### 面试官可能怎么问
"RAG 系统中哪些内容可以缓存？怎么设计缓存 key？"

### 这个问题考察什么
对缓存设计的工程理解，以及缓存和安全性的权衡。

### 我的标准回答

**15 种 RAG 缓存**：

| 缓存类型 | 缓存 Key | TTL | 命中后跳过 | 医疗场景注意 |
|---------|---------|-----|-----------|------------|
| query cache | normalized_query_hash | 1h-24h | 全文检索+生成 | 需脱敏处理 |
| embedding cache | query_text_hash | 永久 | embedding推理 | 非常安全 |
| dense retrieval cache | query_embedding_hash + top_k | 1-6h | 向量搜索 | 知识库更新时失效 |
| sparse retrieval cache | query_normalized + top_k | 1-24h | BM25搜索 | 知识库更新时失效 |
| rerank result cache | (query_hash, doc_ids_hash) | 10min-1h | rerank计算 | 需一并缓存scores |
| document chunk cache | (source_id, page) | 永久 | 文档读取 | 解析错误时需更新 |
| prompt cache | (system_prompt_hash, evidence_hash) | 随请求 | prompt构造 | 减少token计数开销 |
| prefix cache (vLLM) | prompt_prefix_hash | 自动 | prefill | 相同system prompt共享 |
| generation cache | (query_hash, context_hash) | 10min-24h | LLM生成 | **需极度谨慎** |
| safety check cache | (answer_hash, risk_level) | 10min | safety检查 | 高风险不能缓存 |
| citation check cache | (answer_hash, doc_ids) | 10min | citation验证 | 知识库更新时失效 |
| risk classification cache | query_hash | 1h | 风险路由 | 同义词扩展后类似query可复用 |
| fast path answer cache | query_hash | 1-24h | 全文RAG | 仅低风险通用问题 |
| hot query precompute | query_text | 每天更新 | 所有阶段 | 高频医疗问题预计算 |
| user session cache | session_id | 会话期间 | 多轮对话上下文 | 多轮问诊相关 |

### Cache Key 设计
```python
# 示例：检索缓存 key
def retrieval_cache_key(query: str, top_k: int, route_type: str) -> str:
    normalized = normalize_query(query)  # 去重空格、标点归一化、小写
    query_hash = hashlib.md5(normalized.encode()).hexdigest()
    return f"rag:retrieval:{route_type}:k{top_k}:{query_hash}"
```

### 缓存失效策略
1. **知识库更新时**：清除所有 retrieval/rerank/generation 缓存
2. **模型更新时**：清除所有 generation/safety check 缓存
3. **prompt 更新时**：清除所有 generation 缓存
4. **定期过期**：使用 TTL 自动淘汰
5. **LRU 淘汰**：内存缓存满时淘汰最久未使用的

### 医疗缓存特别注意
- **高危问题（急症/用药/肿瘤）不缓存生成结果**——每次重新检索和生成
- **低风险健康科普可以缓存**——但需标注缓存时间和来源
- **用户隐私数据不能进入缓存 key**——query 需要脱敏（去姓名、去具体年龄等）
- **热点医疗问题预计算**（如"感冒怎么办"）——但仍需定期更新以防指南变更

### 面试 1 分钟回答
RAG 缓存分四层：query 级缓存（相同问题直接返回）、embedding 缓存（复用向量）、检索结果缓存（复用召回）、生成结果缓存（复用回答）。最安全的是 embedding 和检索缓存——不影响医学安全性。最需要谨慎的是生成缓存——医疗高风险问题不能缓存。缓存 key 需要脱敏，知识库和模型更新时需要全量失效。实际部署建议分两条线：低风险科普走缓存（省成本），高风险问题不走缓存（保安全）。

---

## E. 检索阶段优化

### 面试官可能怎么问
"FAISS 索引怎么选？一万向量和一百万向量有什么不同？"

### 我的标准回答

| 索引类型 | 搜索方式 | 内存占用 | 速度 | 精确度 | 适用规模 |
|---------|---------|---------|------|-------|---------|
| FlatIP | 暴力搜索 | 低 | O(n) | 100% | <10万向量 |
| IVF | 聚类+倒排 | 中 | 快 ~10x | 95-99% | 10万-1000万 |
| HNSW | 图搜索 | 高(存图) | 最快 | 99%+ | 1万-1000万 |
| PQ | 乘积量化 | 很低 | 较快 | 90-95% | >100万 |

**我的项目**：11707 chunks 使用 FlatIP（精确搜索），延迟约 10ms。扩展到 10 万+ 时需迁移到 IVF 或 HNSW。

**top_k 参数链优化**：
- `top_k_dense = 30`（粗召回，保证覆盖率）
- `top_k_sparse = 30`（BM25 精确匹配召回）
- RRF fusion → 去重后约 40-50 个唯一文档
- `top_k_rerank = 8`（精排后取 top 8）
- `top_k_final = 5`（最终放入 context）

---

## F. Rerank 优化

### 面试官可能怎么问
"rerank 为什么是 RAG 最慢的部分？怎么优化？"

### 我的标准回答

Rerank 使用 cross-encoder（不是 bi-encoder），每对 (query, doc) 都要完整过一次 transformer，计算复杂度 O(k × L²)，k 是候选文档数，L 是文本长度。

**优化手段**：
1. **batch rerank**：多条 (query, doc) 对合并为一个 batch，GPU 利用率从 10% 提到 80%
2. **减小 rerank_k**：从 30→10→5，高风险取 8，低风险取 3
3. **reranker 量化**：INT8 量化，精度损失 <2%，速度提升 2-3x
4. **分级 rerank**：高风险用 bge-reranker-v2-m3（大模型），低风险用 bge-reranker-base（小模型）
5. **GPU 独立部署**：reranker 单独占一张 GPU，不和 vLLM 抢
6. **rerank cache**：相同 (query, doc_ids) 对复用结果
7. **early exit**：前 3 个候选分差很大时不继续算后面
8. **跳过场景**：低风险健康科普直接跳过 rerank

### 和我的项目怎么结合
- reranker 因网络未下载，是**当前最大的待优化项**
- 启用后预计增加 200-500ms，但检索精度显著提升
- 建议单独 CPU/GPU 部署 reranker 服务

---

## G. Context 压缩与 Prompt 优化

### 面试官可能怎么问
"检索返回的文档太长怎么办？会不会影响生成质量？"

### 我的标准回答

1. **chunk 去重**：source_id + page 相同的去重
2. **chunk 合并**：相邻 chunk 合并为完整段落
3. **context compression**：
   - 提取式压缩：只保留和 query 最相关的句子（不需要 LLM，基于 embedding similarity）
   - 摘要式压缩：用轻量 LLM 对每段文本生成一句话摘要（更准但更慢）
4. **evidence sentence extraction**：只保留含医学证据的句子
5. **citation 压缩**：用短引用编号 `[1][2]` 代替完整来源信息放入 prompt
6. **system prompt 最简化**：安全规则尽量精简（10 条规则每条不超过 1 行）
7. **max context tokens 硬限制**：高风险 1500 tokens，普通 800 tokens
8. **重要证据放开头**（对抗 lost in the middle）

---

## H. 生成阶段优化

- **vLLM continuous batching**：多个请求共享 GPU 算力
- **streaming output**：降低感知延迟（用户不用等全部生成完）
- **max_new_tokens 控制**：医学回答限制 512-1024 tokens
- **prefix cache**：相同 system prompt 的 prefill 只做一次
- **降级策略**：如果 RAG 超时，降级为 SFT 模型直接回答 + 安全提示

---

## I. RAG 动态路由（重点）

### 面试官可能怎么问
"是不是所有用户请求都要走完整 RAG？如果不是，怎么决定？"

### 我的标准回答

**三级路由策略**：

| 路由 | 触发条件 | 检索深度 | 安全约束 | 延迟 |
|------|---------|---------|---------|------|
| **No-RAG Fast Path** | 普通健康科普/问候 | 无 | 基础免责声明 | <1s |
| **Light-RAG Path** | 一般医学问题 | top 3 chunks/无rerank | safety rules | 1-3s |
| **Full Safety-RAG** | 急症/用药/肿瘤/罕见病/特殊人群 | top 8 chunks/含rerank | 10条完整约束 | 2-8s |

**分类依据**：
1. 基于 routing_rules.yaml 关键词匹配（约 1ms）
2. 命中 emergency/medication/oncology → Full Safety-RAG
3. 命中 common_disease → Light-RAG
4. 无匹配 → No-RAG（纯模型回答 + 医学免责声明）
5. 模型不确定（输出低置信度）→ 自动升级到 Full Safety-RAG
6. 检索置信度低（rerank score < 0.5）→ 触发拒答或人工审核

**为什么不能为了速度关闭高风险路由**：这是医疗安全底线。急症误判、用药错误、肿瘤建议越界的代价远大于 2 秒延迟。

### 面试 1 分钟回答
医学 RAG 需要动态路由：不是所有问题都需要完整 RAG。普通科普走 fast path（无RAG，<1s），一般医学问题走 light-RAG（top-3 chunks，1-3s），急症/用药/肿瘤/罕见病走 full Safety-RAG（top-8 + rerank + 10条安全约束，2-8s）。路由由规则引擎 + 关键词匹配决定（<1ms），模型不确定时自动升级到 full RAG。核心原则：**高风险场景宁可慢不可错**。

---

## J. 异步与队列

### 关键点
- `asyncio.gather()` 并行执行 dense + sparse retrieval
- Redis Queue / Celery 做异步任务队列（离线 rejected 生成、批量 judge 评测）
- circuit breaker：检索服务连续失败 3 次 → 降级为纯模型回答
- rate limiting：每用户每分钟不超过 20 次医疗咨询
- bulkhead isolation：检索/rerank/生成三个服务的线程池隔离，一个慢不影响另一个

---

## K. 分布式 RAG 架构（扩展方向）

### 项目中未完整实现，但作为扩展方向需要掌握

1. **检索服务独立**：单独部署 FAISS + BM25 服务，支持水平扩展
2. **embedding 服务独立**：GPU 推理，batch 处理多个请求
3. **rerank 服务独立**：避免和 vLLM 抢 GPU
4. **向量库集群**：多分片 + 多副本
5. **知识库蓝绿发布**：新版本索引构建完成后平滑切换

---

## L. 代码与伪代码

### 代码 1: 串行 RAG Pipeline
```python
# 用途：展示最基础的RAG流程
# 输入：user_query (str)
# 输出：answer (str)
# 延迟瓶颈：embedding (300ms CPU) + rerank (0ms, 未启用) + decode (~1s)
# 优化：所有阶段改成并行 + 加缓存 + 动态路由

def naive_rag_pipeline(user_query: str) -> str:
    # Step 1: embedding - 当前CPU推理, 约300ms
    query_embedding = embedding_model.encode(user_query, normalize_embeddings=True)
    # Step 2: dense retrieval - FAISS FlatIP, 约10ms
    scores, indices = faiss_index.search(query_embedding.reshape(1, -1), k=30)
    # Step 3: context assembly - 约2ms
    evidence = [chunks[idx] for idx in indices[0]]
    # Step 4: prompt construction - 约1ms
    prompt = build_safety_rag_prompt(user_query, evidence, safety_rules)
    # Step 5: vLLM generation - 约2s
    answer = call_vllm(prompt, max_tokens=512, temperature=0.2)
    return answer
# 面试官追问: "串行和并行差多少？" → 检索阶段并行可节省40%检索时间
```

### 代码 2: 并行 RAG Pipeline
```python
# 用途：使用 asyncio.gather 并行执行 dense + sparse retrieval
# 延迟瓶颈：最慢的分支决定总延迟（通常是 embedding+dense）
# 优化效果：检索阶段延迟从 350ms 降到 ~310ms（约-12%）
# 与项目关系：后续可直接升级 Safety-RAG 的检索部分

import asyncio

async def parallel_rag_pipeline(user_query: str) -> dict:
    async def dense_search(query):
        emb = await embedding_service.encode_async(query)
        return await faiss_service.search_async(emb, k=30)

    async def sparse_search(query):
        return await bm25_service.search_async(query, k=30)

    async def entity_search(query):
        entities = await medical_ner.extract_async(query)
        return await metadata_service.search_async(entities)

    # 三路并行检索
    dense_results, sparse_results, entity_results = await asyncio.gather(
        dense_search(user_query),
        sparse_search(user_query),
        entity_search(user_query)
    )
    # RRF 融合
    merged = reciprocal_rank_fusion([dense_results, sparse_results, entity_results])
    return merged
# 面试官追问: "为什么不用多线程？" → CPU密集型用多进程，I/O密集型用async
```

### 代码 3: RAG 动态路由
```python
# 用途：根据query风险等级选择不同的RAG路径
# 输入：user_query (str)
# 输出：路由决策 (str) + 检索参数
# 延迟影响：fast path跳过后节省2-3s
# 面试官可能追问: "错过高风险query怎么办？" → 规则引擎偏向保守：宁可误判为高风险也不漏掉

def route_rag_path(user_query: str) -> dict:
    risk_level, route_type = classify_risk(user_query)  # <1ms, 基于routing_rules

    if risk_level == "low" and route_type == "common_disease":
        return {"path": "fast", "use_rag": False}

    elif risk_level == "medium":
        return {"path": "light_rag", "top_k": 3, "rerank": False}

    else:  # high risk: emergency/medication/oncology/rare_disease
        return {
            "path": "full_safety_rag",
            "top_k_dense": 30, "top_k_sparse": 30,
            "top_k_rerank": 8, "top_k_final": 5,
            "rerank": True, "safety_rules": "full_10_rules"
        }
```

### 代码 4: RAG 延迟 Logging
```python
# 用途：结构化记录RAG各阶段延迟
# 输出：latency_breakdown (dict)
# 面试官追问: "p99为什么比p50高很多？" → 长query导致更多候选文档，rerank更慢

import time

class RAGLatencyTracker:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.breaks = {}
        self._start = time.time()

    def mark(self, stage: str):
        self.breaks[stage] = (time.time() - self._start) * 1000  # ms
        return self.breaks[stage]

    def to_log(self) -> dict:
        return {
            "request_id": self.request_id,
            "latency_ms": self.breaks,
            "total_ms": self.breaks.get("end", 0),
            "retrieval_ms": self.breaks.get("retrieval_end", 0) - self.breaks.get("retrieval_start", 0),
            "generation_ms": self.breaks.get("generation_end", 0) - self.breaks.get("generation_start", 0),
        }
```

### 代码 5: embedding cache
```python
# 用途：缓存embedding向量避免重复推理
# 输入：query_text (str)
# 输出：embedding vector
# 延迟：命中时0ms vs 未命中300ms(CPU)

embedding_cache = {}  # 实际用Redis

def get_embedding_cached(query: str, model) -> np.ndarray:
    key = hashlib.md5(normalize_query(query).encode()).hexdigest()
    if key in embedding_cache:
        return embedding_cache[key]
    emb = model.encode(query, normalize_embeddings=True)
    embedding_cache[key] = emb
    return emb
```

### 代码 6: timeout + fallback
```python
# 用途：检索超时时降级返回
# 延迟：timeout_ms内未完成则降级
# 面试官追问: "降级后安全性怎么办？" → 加上"信息可能不完整"提示 + 就医建议

async def rag_with_fallback(query: str, timeout_ms: int = 3000):
    try:
        result = await asyncio.wait_for(
            full_rag_pipeline(query),
            timeout=timeout_ms / 1000
        )
        return result
    except asyncio.TimeoutError:
        # 降级：跳过rerank，减少检索文档
        evidence = await fast_retrieval(query, top_k=2)
        answer = await call_vllm(build_prompt(query, evidence))
        answer += "\n\n⚠️ 当前检索超时，回答可能不够全面，建议咨询医生获取完整建议。"
        return answer
```

---

## M. 面试讲法

### 面试官问"RAG 为什么慢"
> RAG 慢在三处：embedding 推理（每次需要模型前向）、rerank（cross-encoder 逐对精细打分）、生成（逐 token decode）。以我做过的医学 Safety-RAG 为例，CPU embedding 约 300ms，如果启用 reranker 约 200-500ms，vLLM 生成 100 token 约 1s。总延迟 2-5s。优化方向是并行检索 + 缓存 + 动态路由。

### 面试官问"如何降低 RAG 延迟"
> 三步走：第一步并行化——dense+sparse+metadata 三路同时检索，节省 40% 检索时间；第二步缓存——embedding、检索结果、热点问答全缓存；第三步动态路由——普通科普走 fast path 跳过 RAG，只有高风险医疗问题走完整 Safety-RAG。三步下来平均延迟从 4s 降到 1.5s。

### 面试官问"如何在安全和延迟之间取舍"
> 医学场景下安全优先，不能为了速度牺牲安全。具体策略是：高风险请求（急症/用药/肿瘤/罕见病）宁可慢不能错，必须走 full Safety-RAG；低风险科普可以走 fast path。同时通过并行化和缓存把 full Safety-RAG 的延迟也尽量压到 3s 以内。

---

## 面试 1 分钟回答

RAG 延迟优化有三个层次：第一层并行化——dense retrieval、BM25、metadata search 同时检索；第二层缓存化——embedding 缓存、检索结果缓存、热点问答缓存；第三层动态路由——不是所有问题都需要完整 RAG，普通科普走 fast path（<1s），高风险医疗走 full Safety-RAG（<3s）。医学场景的核心原则：安全优先，高风险宁可慢不能错，但通过并行+缓存+路由，把 full RAG 也控制在可接受的延迟范围内。

---

## 面试 3 分钟回答

RAG 工程化的核心挑战是平衡**检索质量、延迟和成本**。我围绕医学 Safety-RAG 项目做了以下设计：

1. **延迟拆解**：RAG 全链路 15 个阶段，瓶颈在三个——embedding 推理（20-300ms）、rerank（50-500ms）、vLLM decode（逐 token，100 token 约 1s）。

2. **并行化**：dense retrieval + sparse retrieval + metadata search 三路并行（`asyncio.gather`），检索延迟降低约 40%。embedding 和 rerank 使用 batch 推理提升 GPU 利用率。

3. **缓存策略**：15 种缓存分四层——query 级、embedding 级、检索结果级、生成结果级。医疗场景特别要求：高风险问题不缓存生成结果，低风险科普可以缓存。

4. **动态路由**：三级路由——No-RAG fast path（普通科普）、Light-RAG（一般医学问题）、Full Safety-RAG（急症/用药/肿瘤/罕见病）。路由决策 <、<1ms，模型不确定时自动升级。

5. **降级容灾**：检索超时→跳过 rerank 直接返回 top-2 chunks；rerank 服务异常→FAISS 默认排序；vLLM OOM→降低 max_tokens。

最终目标：普通问题 <1s，一般医学问题 <3s，高风险问题 <5s。

---

## 高频问题速记

| 面试问题 | 核心回答 | 项目数字 |
|---------|---------|---------|
| RAG 为什么慢 | embedding+rerank+decode 三瓶颈 | CPU embedding 300ms |
| 如何降低延迟 | 并行+缓存+动态路由 三步 | 11707 chunks FlatIP |
| 哪些阶段可并行 | dense+sparse+metadata 三路 | asyncio.gather |
| rerank 太慢怎么优化 | batch+量化+减k+分级+独立部署 | reranker 未启用(待优化) |
| context 太长怎么办 | 压缩+去重+截断+证据句提取 | 高风险max 1500 tokens |
| 如何设计缓存 | 四层缓存+脱敏+知识库更新失效 | 生成缓存仅低风险 |
| 医学能不能 fast path | 可以, 但急症/用药/肿瘤不走 | 三级路由 |
| 安全和延迟如何取舍 | 安全优先, 通过工程手段压延迟 | full RAG <5s |

---

## 背诵版总结

1. RAG 全链路 15+ 阶段，三大瓶颈：embedding 推理、rerank 精排、vLLM decode
2. 并行化核心：dense + sparse + metadata 三路同时检索（asyncio.gather），检索延迟降 40%
3. 缓存四层：query 级 > embedding 级 > 检索结果级 > 生成结果级（生成缓存仅低风险）
4. 动态路由三级：No-RAG fast path / Light-RAG / Full Safety-RAG，路由决策 <1ms
5. 医疗安全底线：急症/用药/肿瘤/罕见病必须走 full Safety-RAG，宁可慢不可错
6. 降级策略：检索超时→跳过 rerank；rerank 挂→FAISS 默认排序；vLLM OOM→降 max_tokens
7. 项目实际：Safety-RAG pipeline 串行运行，embedding 用 CPU（300ms），reranker 待启用
8. 扩展方向：检索服务独立部署、GPU embedding/reranker、分布式向量库、蓝绿知识库发布
