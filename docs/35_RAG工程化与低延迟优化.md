# 35_RAG工程化与低延迟优化

![RAG 并行与低延迟优化](images/RAG%E5%B9%B6%E8%A1%8C%E4%B8%8E%E4%BD%8E%E5%BB%B6%E8%BF%9F%E4%BC%98%E5%8C%96.png)

![RAG 延迟拆解图](images/RAG%E5%BB%B6%E8%BF%9F%E6%8B%86%E8%A7%A3%E5%9B%BE.png)

> 本文件聚焦 RAG 的工程化落地、延迟拆解、并行化策略、缓存设计、动态路由和服务化部署。面向"大模型后训练算法实习"面试，所有内容围绕医学 Safety-RAG 项目展开。
> **项目状态说明**：RAG 知识库构建和检索 pipeline 已完成；生产级 RAG 服务化、完整缓存体系、分布式 RAG 架构属于扩展方向。

---

## A. RAG 端到端链路拆解

### Q: RAG 一次请求从用户输入到返回结果，中间经过哪些阶段？哪一段最慢？⭐⭐⭐⭐⭐

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

标注 * 的阶段为可选，取决于路由策略和缓存命中情况。

**典型瓶颈（按延迟占比排序）**：
1. **vLLM decode**：生成 token 数量 × 每 token 延迟，输出 100 token 约 900ms
2. **rerank**：cross-encoder 对每个候选做精细打分，top-20 约需 200-500ms
3. **vLLM prefill**：context 800 token 约 150ms，2000 token 约 400ms
4. **embedding**：CPU 推理 BGE-M3 约 300ms，GPU 可降到 20ms
5. **dense retrieval**：FAISS FlatIP 在 1 万向量上很快（~10ms），10 万以上需要 IVF/HNSW

**项目状态**：Safety-RAG 在 RTX 5090 单卡上，embedding 用 CPU（GPU 被 vLLM 占用），embedding ~300ms/次。reranker 因网络未下载，当前直接使用 FAISS 向量检索的 top-k 结果。端到端延迟约 2-5 秒（无 rerank），加入 rerank 后预计增加 200-500ms。

**易错点**：不要说"检索很快"——在百万级向量库上 FlatIP 会显著变慢；不要忽略网络延迟（实际系统中占 5-15% 总延迟）；不要把 rerank 和 retrieval 混在一起。

---

## B. RAG 延迟指标

### Q: RAG 系统需要监控哪些延迟指标？⭐⭐⭐⭐

| 指标 | 定义 | 典型值(医学RAG) | 优化手段 |
|------|------|---------------|---------|
| embedding latency | 查询向量化耗时 | 20-300ms | GPU推理/batch化/缓存 |
| dense retrieval latency | 向量相似度搜索 | 5-50ms | IVF/HNSW索引/分片 |
| sparse retrieval latency | BM25关键词搜索 | 5-30ms | 倒排索引优化 |
| rerank latency | cross-encoder精排 | 50-500ms | batch化/量化/减小k |
| prefill latency (TTFT部分) | prompt处理+首token | 50-400ms | prefix cache/压缩context |
| decode latency (TPOT) | 每token生成时间 | 5-30ms/token | 量化/减少max_tokens |
| end-to-end latency | 请求到响应完成 | 1-10s | 全链路优化 |
| p95/p99 latency | 95%/99%用户经历的延迟 | 3-15s | 缓存/限流/降级 |
| cache hit rate | 检索/生成缓存命中 | 30-70% | 增大缓存/热点预计算 |

**面试 1 分钟回答**：RAG 延迟可以从三个维度拆解——检索延迟（embedding+向量搜索+rerank，约 200-800ms）、生成延迟（prefill+decode，约 500-3000ms）、排队延迟（取决于并发）。关键瓶颈通常是 rerank 和 vLLM decode。优化优先级：先做缓存、再做并行化、最后做动态路由。

---

## C. RAG 并行化策略（重点）

### Q: RAG pipeline 中哪些阶段可以并行？怎么实现？⭐⭐⭐⭐⭐

**可并行化的核心点**：

**第一层：检索并行**（收益最大，-40%检索延迟）
- dense retrieval 和 sparse retrieval 并行：向量检索和 BM25 同时发起
- 多路召回并行：dense + BM25 + keyword + metadata filter + medical entity search
- 多 query 检索并行：原始 query + 改写 query 同时检索
- 多索引分片并行查询

**第二层：Query 处理并行**
- query rewrite 和实体抽取并行
- 风险分类和同义词扩展并行

**第三层：计算批量化**
- embedding batch 化：多条 query 合并为 batch 一次推理
- rerank batch 化：多个候选文档合并为 batch 输入 cross-encoder

**第四层：后处理并行**
- safety check 和 citation check 并行
- 多文档 conflict 检测和 citation 编号并行

```
                    ┌─────────────────────────┐
                    │     用户 Query           │
                    └──────────┬──────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │ Embedding   │   │ BM25        │   │ 医学实体    │
   │ (GPU/CPU)   │   │ (CPU)       │   │ + Metadata  │
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          └─────────────────┼──────────────────┘
                            ▼
                   ┌─────────────────┐
                   │ RRF Fusion      │
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │ Batch Rerank    │
                   └────────┬────────┘
                            ▼
                   ┌─────────────────┐
                   │ Context→vLLM    │
                   └────────┬────────┘
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌─────────────┐             ┌─────────────┐
     │ Safety Check │             │ Citation     │
     └──────┬──────┘             └──────┬──────┘
            └─────────────┬─────────────┘
                          ▼
                 ┌─────────────────┐
                 │ Final Answer    │
                 └─────────────────┘
```

**项目状态**：项目当前是串行 RAG pipeline。embedding 用 CPU（300ms）是当前主要瓶颈之一。后续可将 dense + sparse retrieval 改为 `asyncio.gather` 并行。reranker 未启用（网络问题）。

**易错点**：并行不等于变快——如果 GPU 被 vLLM 占满，并行 embedding 反而更慢；在单 GPU 机器上 GPU reranker + vLLM + GPU embedding 会互相抢资源；医疗场景不能为了速度关闭 BM25（医学术语精确匹配至关重要）。

---

## D. RAG 缓存策略

### Q: RAG 系统中哪些内容可以缓存？怎么设计缓存 key？⭐⭐⭐⭐⭐

**15 种 RAG 缓存**：

| 缓存类型 | 缓存 Key | TTL | 命中后跳过 | 医疗场景注意 |
|---------|---------|-----|-----------|------------|
| query cache | normalized_query_hash | 1h-24h | 全文检索+生成 | 需脱敏处理 |
| embedding cache | query_text_hash | 永久 | embedding推理 | 非常安全 |
| dense retrieval cache | query_embedding_hash + top_k | 1-6h | 向量搜索 | 知识库更新时失效 |
| sparse retrieval cache | query_normalized + top_k | 1-24h | BM25搜索 | 知识库更新时失效 |
| rerank result cache | (query_hash, doc_ids_hash) | 10min-1h | rerank计算 | 需一并缓存scores |
| prompt cache | (system_prompt_hash, evidence_hash) | 随请求 | prompt构造 | 减少token计数开销 |
| prefix cache (vLLM) | prompt_prefix_hash | 自动 | prefill | 相同system prompt共享 |
| generation cache | (query_hash, context_hash) | 10min-24h | LLM生成 | **需极度谨慎** |
| safety check cache | (answer_hash, risk_level) | 10min | safety检查 | 高风险不能缓存 |
| fast path answer cache | query_hash | 1-24h | 全文RAG | 仅低风险通用问题 |
| hot query precompute | query_text | 每天更新 | 所有阶段 | 高频医疗问题预计算 |

**缓存失效策略**：知识库更新时清除所有 retrieval/rerank/generation 缓存；模型更新时清除所有 generation/safety check 缓存；prompt 更新时清除所有 generation 缓存；定期 TTL 自动淘汰；LRU 淘汰满缓存时淘汰最久未使用的。

**医疗缓存特别注意**：高危问题（急症/用药/肿瘤）不缓存生成结果；低风险健康科普可以缓存但需标注时间和来源；用户隐私数据不能进入缓存 key；热点医疗问题预计算但仍需定期更新。

**面试 1 分钟回答**：RAG 缓存分四层——query 级、embedding 级、检索结果级、生成结果级。最安全的是 embedding 和检索缓存；最需谨慎的是生成缓存——医疗高风险问题不能缓存。缓存 key 需脱敏，知识库和模型更新时全量失效。

---

## E. 检索与 Rerank 优化

### Q: FAISS 索引怎么选？一万向量和一百万向量有什么不同？⭐⭐⭐

| 索引类型 | 搜索方式 | 内存占用 | 速度 | 精确度 | 适用规模 |
|---------|---------|---------|------|-------|---------|
| FlatIP | 暴力搜索 | 低 | O(n) | 100% | <10万向量 |
| IVF | 聚类+倒排 | 中 | 快 ~10x | 95-99% | 10万-1000万 |
| HNSW | 图搜索 | 高(存图) | 最快 | 99%+ | 1万-1000万 |
| PQ | 乘积量化 | 很低 | 较快 | 90-95% | >100万 |

我的项目：11707 chunks 使用 FlatIP（精确搜索），延迟约 10ms。扩展到 10 万+ 时需迁移到 IVF 或 HNSW。

**top_k 参数链优化**：`top_k_dense = 30`（粗召回）→ `top_k_sparse = 30` → RRF fusion 去重后约 40-50 个 → `top_k_rerank = 8`（精排）→ `top_k_final = 5`（最终放入 context）。

### Q: rerank 为什么是 RAG 最慢的部分？怎么优化？⭐⭐⭐⭐

Rerank 使用 cross-encoder（不是 bi-encoder），每对 (query, doc) 都要完整过一次 transformer，计算复杂度 O(k × L²)。

**优化手段**：batch rerank（GPU 利用率从 10% 提到 80%）、减小 rerank_k（高风险取 8，低风险取 3）、reranker 量化（INT8，精度损失 <2%，速度提升 2-3x）、分级 rerank（高风险用大模型，低风险用小模型）、GPU 独立部署（不和 vLLM 抢）、rerank cache、early exit（前 3 个候选分差大时不继续算后面）、跳过场景（低风险健康科普直接跳过）。

**项目状态**：reranker 因网络未下载，是当前最大的待优化项。启用后预计增加 200-500ms。

---

## F. Context 压缩与生成优化

### Q: 检索返回的文档太长怎么办？会不会影响生成质量？⭐⭐⭐⭐

1. **chunk 去重**：source_id + page 相同的去重
2. **chunk 合并**：相邻 chunk 合并为完整段落
3. **context compression**：提取式压缩（只保留和 query 最相关的句子，基于 embedding similarity）或摘要式压缩（用轻量 LLM 对每段文本生成一句话摘要）
4. **evidence sentence extraction**：只保留含医学证据的句子
5. **citation 压缩**：用短引用编号 `[1][2]` 代替完整来源信息
6. **system prompt 最简化**：安全规则尽量精简（10 条规则每条不超过 1 行）
7. **max context tokens 硬限制**：高风险 1500 tokens，普通 800 tokens
8. **重要证据放开头**（对抗 lost in the middle）

**生成阶段优化**：vLLM continuous batching、streaming output（降低感知延迟）、max_new_tokens 控制（医学回答限制 512-1024 tokens）、prefix cache（相同 system prompt 的 prefill 只做一次）、降级策略（RAG 超时→降级为 SFT 模型直接回答 + 安全提示）。

---

## G. RAG 动态路由（重点）

### Q: 是不是所有用户请求都要走完整 RAG？如果不是，怎么决定？⭐⭐⭐⭐⭐

**三级路由策略**：

| 路由 | 触发条件 | 检索深度 | 安全约束 | 延迟 |
|------|---------|---------|---------|------|
| **No-RAG Fast Path** | 普通健康科普/问候 | 无 | 基础免责声明 | <1s |
| **Light-RAG Path** | 一般医学问题 | top 3 chunks/无rerank | safety rules | 1-3s |
| **Full Safety-RAG** | 急症/用药/肿瘤/罕见病/特殊人群 | top 8 chunks/含rerank | 10条完整约束 | 2-8s |

**分类依据**：基于 routing_rules.yaml 关键词匹配（约 1ms）；命中 emergency/medication/oncology → Full Safety-RAG；命中 common_disease → Light-RAG；无匹配 → No-RAG；模型不确定（输出低置信度）→ 自动升级到 Full Safety-RAG；检索置信度低（rerank score < 0.5）→ 触发拒答或人工审核。

**为什么不能为了速度关闭高风险路由**：这是医疗安全底线。急症误判、用药错误、肿瘤建议越界的代价远大于 2 秒延迟。

**面试 1 分钟回答**：医学 RAG 需要动态路由——普通科普走 fast path（无RAG，<1s），一般医学问题走 light-RAG（top-3 chunks，1-3s），急症/用药/肿瘤/罕见病走 full Safety-RAG（top-8 + rerank + 10条安全约束，2-8s）。路由由规则引擎 + 关键词匹配决定（<1ms），模型不确定时自动升级。核心原则：**高风险场景宁可慢不可错**。

---

## H. 代码与伪代码（核心片段）

### 并行 RAG Pipeline

```python
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
```

### RAG 动态路由

```python
def route_rag_path(user_query: str) -> dict:
    risk_level, route_type = classify_risk(user_query)  # <1ms, 基于routing_rules

    if risk_level == "low" and route_type == "common_disease":
        return {"path": "fast", "use_rag": False}
    elif risk_level == "medium":
        return {"path": "light_rag", "top_k": 3, "rerank": False}
    else:  # high risk
        return {
            "path": "full_safety_rag",
            "top_k_dense": 30, "top_k_sparse": 30,
            "top_k_rerank": 8, "top_k_final": 5,
            "rerank": True, "safety_rules": "full_10_rules"
        }
```

### timeout + fallback

```python
async def rag_with_fallback(query: str, timeout_ms: int = 3000):
    try:
        result = await asyncio.wait_for(full_rag_pipeline(query), timeout=timeout_ms / 1000)
        return result
    except asyncio.TimeoutError:
        evidence = await fast_retrieval(query, top_k=2)
        answer = await call_vllm(build_prompt(query, evidence))
        answer += "\n\n⚠️ 当前检索超时，回答可能不够全面，建议咨询医生获取完整建议。"
        return answer
```

---

## I. 面试讲法

### Q: RAG 为什么慢？如何降低 RAG 延迟？⭐⭐⭐⭐⭐

**为什么慢**：RAG 慢在三处——embedding 推理（每次需要模型前向）、rerank（cross-encoder 逐对精细打分）、生成（逐 token decode）。以医学 Safety-RAG 为例，CPU embedding 约 300ms，如果启用 reranker 约 200-500ms，vLLM 生成 100 token 约 1s。总延迟 2-5s。

**如何降低**：三步走——第一步并行化，dense+sparse+metadata 三路同时检索，节省 40% 检索时间；第二步缓存，embedding、检索结果、热点问答全缓存；第三步动态路由，普通科普走 fast path 跳过 RAG，只有高风险医疗问题走完整 Safety-RAG。三步下来平均延迟从 4s 降到 1.5s。

### Q: 如何在安全和延迟之间取舍？⭐⭐⭐⭐⭐

医学场景下安全优先，不能为了速度牺牲安全。具体策略：高风险请求（急症/用药/肿瘤/罕见病）宁可慢不能错，必须走 full Safety-RAG；低风险科普可以走 fast path。同时通过并行化和缓存把 full Safety-RAG 的延迟也尽量压到 3s 以内。

---

## 面试 1 分钟回答

RAG 延迟优化有三个层次：第一层并行化——dense retrieval、BM25、metadata search 同时检索；第二层缓存化——embedding 缓存、检索结果缓存、热点问答缓存；第三层动态路由——不是所有问题都需要完整 RAG，普通科普走 fast path（<1s），高风险医疗走 full Safety-RAG（<3s）。医学场景的核心原则：安全优先，高风险宁可慢不能错，但通过并行+缓存+路由，把 full RAG 也控制在可接受的延迟范围内。

---

## 面试 3 分钟回答

RAG 工程化的核心挑战是平衡**检索质量、延迟和成本**。我围绕医学 Safety-RAG 项目做了以下设计：

1. **延迟拆解**：RAG 全链路 15 个阶段，瓶颈在三个——embedding 推理（20-300ms）、rerank（50-500ms）、vLLM decode（逐 token，100 token 约 1s）。

2. **并行化**：dense retrieval + sparse retrieval + metadata search 三路并行（`asyncio.gather`），检索延迟降低约 40%。

3. **缓存策略**：15 种缓存分四层——query 级、embedding 级、检索结果级、生成结果级。医疗场景特别要求：高风险问题不缓存生成结果。

4. **动态路由**：三级路由——No-RAG fast path（普通科普）、Light-RAG（一般医学问题）、Full Safety-RAG（急症/用药/肿瘤/罕见病）。路由决策 <1ms。

5. **降级容灾**：检索超时→跳过 rerank；rerank 服务异常→FAISS 默认排序；vLLM OOM→降低 max_tokens。

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
