# 00_README_如何使用这套资料

## 一、这套资料的用途

这是一套**大模型后训练算法实习面试**的八股复习资料库，专为以下目标而写：

1. **岗位对口**：LLM Post-training Intern / Alignment Intern / SFT-DPO-RLHF Intern / 大模型算法实习
2. **项目驱动**：以你本地的**医学 LLM Teacher 项目（Qwen3-8B）**为主线，以 **MiniMind 小模型从零训练**项目为辅助
3. **可背诵**：所有知识点都按"面试官怎么问 → 考察什么 → 我的标准回答 → 可能追问 → 追问回答 → 项目结合 → 易错点 → 背诵版总结"组织
4. **可长期维护**：纯 Markdown，可以持续更新

---

## 二、适合哪些岗位

| 岗位名称 | 匹配度 | 说明 |
|---------|-------|------|
| 大模型后训练算法实习 | ⭐⭐⭐⭐⭐ | 最匹配，本资料就是为此设计的 |
| LLM Alignment Intern | ⭐⭐⭐⭐⭐ | 对齐、安全、DPO/RAG 覆盖全面 |
| SFT / DPO 方向实习 | ⭐⭐⭐⭐⭐ | SFT+DPO 八股非常详细 |
| RLHF / RLVR 方向实习 | ⭐⭐⭐⭐ | GRPO/RLVR/PPO 均有覆盖（但项目未实现） |
| 大模型评测实习 | ⭐⭐⭐⭐ | LLM-as-Judge 八股详细 |
| RAG 应用实习 | ⭐⭐⭐⭐ | Safety-RAG 八股非常详细，8000+ 字 |
| 推理部署实习 | ⭐⭐⭐ | vLLM 部分覆盖，但不是专攻方向 |
| 预训练算法实习 | ⭐⭐ | 有 Transformer 基础，但不是主线 |

---

## 三、每个文件对应什么模块

| 编号 | 文件名 | 模块 | 重要程度 |
|------|-------|------|---------|
| 00 | README_如何使用这套资料 | 导航 | ⭐⭐ |
| 01 | 岗位画像与复习路线 | 策略 | ⭐⭐⭐⭐⭐ |
| 02 | 大模型基础八股 | 基础 | ⭐⭐⭐⭐ |
| 03 | Transformer与Attention | 基础 | ⭐⭐⭐⭐ |
| 04 | Tokenizer与ChatTemplate | 基础 | ⭐⭐⭐⭐ |
| 05 | SFT指令微调 | 核心算法 | ⭐⭐⭐⭐⭐ |
| 06 | LoRA_QLoRA_PEFT | 核心算法 | ⭐⭐⭐⭐⭐ |
| 07 | DPO偏好优化 | 核心算法 | ⭐⭐⭐⭐⭐ |
| 08 | RLHF_PPO_GRPO_RLVR | 扩展算法 | ⭐⭐⭐ |
| 09 | Reward_Model_PRM_ORM_Verifier | 扩展算法 | ⭐⭐⭐ |
| 10 | 后训练数据工程_Teacher数据生成 | 数据工程 | ⭐⭐⭐⭐⭐ |
| 11 | 评测体系与LLM-as-Judge | 评测 | ⭐⭐⭐⭐⭐ |
| 12 | RAG与SafetyRAG详细八股 | RAG | ⭐⭐⭐⭐⭐ |
| 13 | 推理部署与vLLM | 部署 | ⭐⭐⭐ |
| 14 | 训练工程与显存优化 | 工程 | ⭐⭐⭐⭐ |
| 15 | 安全对齐与医疗场景 | 安全 | ⭐⭐⭐⭐ |
| 16 | 手撕代码合集 | 代码 | ⭐⭐⭐⭐⭐ |
| 17 | 算法题与PyTorch题 | 代码 | ⭐⭐⭐⭐ |
| 18 | 当前热点与趋势_2025_2026 | 热点 | ⭐⭐⭐ |
| 19 | 医学LLM_Teacher项目深挖问答 | 项目 | ⭐⭐⭐⭐⭐ |
| 20 | MiniMind项目可讲点 | 项目 | ⭐⭐ |
| 21 | 一周冲刺复习计划 | 策略 | ⭐⭐⭐⭐⭐ |
| 22 | 100问Checklist | 自测 | ⭐⭐⭐⭐⭐ |
| 23 | 150张Flashcards | 记忆 | ⭐⭐⭐⭐ |
| 24 | 面试自我介绍与项目讲述模板 | 表达 | ⭐⭐⭐⭐⭐ |
| 25 | 后训练算法对比总表 | 速查 | ⭐⭐⭐⭐ |
| 26 | 常见追问与避坑回答 | 策略 | ⭐⭐⭐⭐⭐ |

---

## 四、推荐复习顺序

### 如果你有 2 周以上：
```
Week 1: 基础 → 核心算法 → 项目
Week 2: 补充 → 代码 → 模拟
```

### 如果你只有 1 周：
按"七天冲刺复习计划"（文件 21）执行。

### 推荐阅读顺序：
```
01 岗位画像 → 02 基础八股 → 03 Transformer → 04 Tokenizer
→ 05 SFT → 06 LoRA → 10 数据工程 → 07 DPO
→ 11 评测 → 12 RAG → 15 安全对齐
→ 13 部署 → 14 工程 → 08 RLHF/GRPO → 09 RM/PRM
→ 18 热点 → 19 项目深挖 → 20 MiniMind → 24 自我介绍
→ 16 手撕代码 → 17 算法题 → 22 100问 → 23 Flashcards → 25 对比总表 → 26 避坑
```

---

## 五、7 天冲刺复习路线

详见文件 21，这里给出概要：

| 天 | 主题 | 核心任务 |
|----|------|---------|
| Day 1 | 大模型基础 + Transformer + Tokenizer | 理解+背诵基础概念 |
| Day 2 | SFT + LoRA/QLoRA | 理解SFT全流程，手撕loss |
| Day 3 | DPO + Reward Model | 理解DPO公式，手撕DPO loss |
| Day 4 | RAG + Safety-RAG | 背熟RAG五层架构，讲清楚项目中的RAG |
| Day 5 | LLM-as-Judge + 医疗安全 | 背熟评测维度和安全设计 |
| Day 6 | vLLM + 训练工程 + 手撕代码 | 手撕关键代码段 |
| Day 7 | 项目深挖 + 模拟面试 | 1分钟/3分钟/5分钟版本练熟 |

---

## 六、面试前一天复习路线

1. **上午**：背诵 24_面试自我介绍与项目讲述模板 的所有版本
2. **下午**：刷一遍 22_100问Checklist，标记不会的
3. **傍晚**：过一遍 26_常见追问与避坑回答
4. **晚上**：看一遍 19_医学LLM_Teacher项目深挖问答 的重点 20 问
5. **睡前**：过一遍 25_后训练算法对比总表

---

## 七、必背模块（第一优先级）

这些模块必须能**脱稿流利回答**：

1. **SFT**：目标函数、assistant-only loss、labels=-100、teacher forcing、chat template
2. **DPO**：loss 公式、beta 含义、chosen/rejected 构造、为什么 rejected 用 SFT 采样
3. **LoRA/QLoRA**：公式、rank/alpha/dropout、merge、NF4、Double Quantization
4. **RAG**：五层架构（路由+filter+向量+rerank+safety_rules）、hybrid retrieval、rerank
5. **LLM-as-Judge**：score-based、pairwise、safety review、长度偏置、位置偏置
6. **vLLM**：PagedAttention、continuous batching、TTFT、TPOT、显存估算
7. **医学 Teacher 项目**：1分钟/3分钟/5分钟版本、数据构造闭环、DPO 长尾幻觉发现、Safety-RAG 修复

---

## 八、哪些模块只需要了解（第二/三优先级）

这些模块**不需要手撕代码**，但要能说清楚"是什么、为什么火、和我的项目有什么关系"：

1. **OPD**：on-policy distillation 思想，和 teacher 项目的关系
2. **GRPO**：不需要 value model，group relative reward，适合 reasoning
3. **RLVR**：verifiable reward，rule-based reward
4. **DAPO / Dr. GRPO**：GRPO 变体
5. **PRM/ORM**：process reward vs outcome reward
6. **Agent post-training**：tool-use training 基本概念

---

## 九、如何结合项目回答

**核心原则**：每个技术概念都要能"回到项目"。

例如面试官问"什么是 DPO"：
- ❌ 只背公式
- ✅ 先说公式 → 再说我的项目中怎么构造 chosen/rejected（teacher 答案作为 chosen、SFT 模型采样作为 rejected）→ 再说发现 DPO 在医疗长尾上有幻觉 → 引出 Safety-RAG

---

## 十、如何避免夸大自己做过的内容

1. **明确区分**：每个文件都会标注"项目中已实现"和"项目中未实现，面试需要了解"
2. **保守表达**：
   - "我做过" → 确实在代码/日志中能找到
   - "我了解原理" → 没实现但读过论文/博客
   - "下一步计划" → 没做过但有想法
3. **不要说的**：
   - "我训练过 PPO"（如果没做过）
   - "我部署了线上系统"（如果有只有本地 vLLM）
   - "我做了 OPD"（如果只是 offline teacher SFT+DPO）

---

## 背诵版总结

1. 这套资料是为大模型后训练算法实习面试准备的
2. 主线是医学 LLM Teacher 项目（Qwen3-8B SFT → DPO → Safety-RAG）
3. 辅助是 MiniMind 从零训练小模型项目
4. 必背模块：SFT、DPO、LoRA/QLoRA、RAG、LLM-as-Judge、vLLM、医学项目
5. 了解即可：OPD、GRPO、RLVR、DAPO、PRM/ORM
6. 每个知识点都要能回到项目
7. 没做过的不要说自己做过
8. 面试前一天重点背诵项目讲述模板和避坑回答

---

## 十一、工程落地模块（33-39号文件）

针对"有后训练能力但缺乏线上落地经验"的情况，新增工程落地模块：

| 编号 | 文件名 | 模块 | 重要程度 |
|------|-------|------|---------|
| 33 | 项目落地与工程全景 | 工程架构思维 | ⭐⭐⭐⭐ |
| 34 | vLLM深入与性能优化 | 推理部署 | ⭐⭐⭐⭐⭐ |
| 35 | RAG低延迟与缓存设计 | RAG工程 | ⭐⭐⭐⭐⭐ |
| 36 | 分布式训练实战 | 训练工程 | ⭐⭐⭐⭐ |
| 37 | 分布式推理与多卡部署 | 推理工程 | ⭐⭐⭐⭐ |
| 38 | 监控灰度与A/B测试 | 运维 | ⭐⭐⭐ |
| 39 | 医学项目落地讲法 | 面试总结 | ⭐⭐⭐⭐⭐ |

---

## 十二、工程落地复习路线

按以下顺序复习工程落地内容：

1. **先看33项目落地**：建立"从Demo到生产"的全局思维，理解单机Demo和线上系统的差异
2. **再看34 vLLM**：背熟vLLM部署参数、压测指标、TTFT/TPOT/p95/p99、continuous batching原理、streaming/SSE
3. **再看35 RAG低延迟**：背熟RAG延迟拆解、并行化改造、多级缓存策略、fast path/慢path路由、rerank加速
4. **再看36 分布式训练**：背熟DDP/FSDP/DeepSpeed ZeRO对比、多卡训练命令、单卡扩展到多卡的完整回答
5. **再看37 分布式推理**：背熟Tensor Parallel/Pipeline Parallel/多副本负载均衡、单卡vs多卡决策树
6. **再看38 监控灰度**：了解K8s/灰度/A-B测试/回滚/隐私保护概念（不需要深入实现细节）
7. **最后看39 医学项目落地讲法**：这是面试核心——如何诚实地讲"我做了但没有上线"的项目

---

## 十三、必背模块（加入工程落地部分）

在原有基础上增加：

- **vLLM部署参数**：max-num-seqs、max-model-len、gpu-memory-utilization、enable-prefix-caching、dtype
- **性能指标**：TTFT/TPOT/p95/p99 latency/tokens/s/QPS 的含义和面试讲法
- **RAG延迟拆解**：embedding→向量检索→rerank→LLM生成 各阶段耗时占比
- **RAG并行化**：embedding并行、检索并行、rerank批处理
- **动态路由**：fast path（简单问题跳过RAG）vs 慢path（复杂/高风险问题完整RAG）
- **缓存策略**：embedding cache、向量检索结果cache、LLM回答cache、prefix cache
- **分布式概念**：DDP vs FSDP vs DeepSpeed ZeRO-1/2/3 的区别、Tensor Parallel vs Pipeline Parallel vs 多副本

---

## 十四、了解即可模块（加入工程落地部分）

在原有基础上增加：

- **K8s / 容器编排**：了解基本概念（Pod/Service/Deployment/HPA），不需要会写YAML
- **灰度发布 / 金丝雀发布**：了解"新模型先服务5%流量→观察→逐步扩大"的基本流程
- **A/B测试**：了解对照组/实验组的分流方法和指标对比
- **完整线上监控**：了解prometheus/grafana的基本概念，不需要具体搭建
- **多节点部署**：了解多副本+load balancer的基本架构，不需要具体的Nginx/Envoy配置

---

## 十五、7天冲刺复习路线（加入Day6工程落地）

| 天 | 主题 | 核心任务 |
|----|------|---------|
| Day 1 | 大模型基础 + Transformer + Tokenizer | 理解+背诵基础概念 |
| Day 2 | SFT + LoRA/QLoRA | 理解SFT全流程，手撕loss |
| Day 3 | DPO + Reward Model | 理解DPO公式，手撕DPO loss |
| Day 4 | RAG + Safety-RAG | 背熟RAG五层架构，讲清楚项目中的RAG |
| Day 5 | LLM-as-Judge + 医疗安全 | 背熟评测维度和安全设计 |
| Day 6 | vLLM + 训练工程 + 工程落地 | 手撕关键代码段 + 背熟vLLM参数表/压测结果讲法/RAG延迟优化/单卡vs多卡回答 |
| Day 7 | 项目深挖 + 模拟面试 + 工程落地讲法 | 1分钟/3分钟/5分钟版本练熟 + 背诵39号文件"项目没上线怎么讲落地能力" |

**Day 6 工程落地重点**：
1. vLLM参数速查表（max-num-seqs/max-model-len/gpu-memory-utilization/dtype）
2. 压测结果讲法（877 tok/s / TTFT 42ms / 饱和并发16 / p95延迟）
3. RAG延迟优化三步走（并行化→缓存→fast path路由）
4. 单卡vs多卡决策树（小于13B用单卡多副本，大于13B考虑TP/PP）
5. 背诵"单卡5090怎么训练8B"完整回答

---

## 十六、面试前一天复习路线（加入工程落地）

1. **上午**：背诵 24_面试自我介绍与项目讲述模板 的所有版本
2. **下午**：刷一遍 22_100问Checklist，标记不会的
3. **傍晚**：过一遍 26_常见追问与避坑回答
4. **晚上**：看一遍 19_医学LLM_Teacher项目深挖问答 的重点 20 问 + 工程落地Q/A
5. **睡前**：过一遍 25_后训练算法对比总表 + vLLM参数速查表 + RAG延迟拆解表
6. **加项**：背诵 39_医学项目落地讲法 中"项目没上线怎么体现落地能力"的核心回答（3分钟版本）

---

## 十七、工程文件背诵版总结

1. **vLLM核心参数**：max-num-seqs=16(饱和并发)、max-model-len=4096(控制KV Cache)、gpu-memory-utilization=0.90(留10%缓冲)、dtype=bfloat16
2. **性能指标速记**：877 tok/s、TTFT 42ms(1并发)/100ms(16并发)、TPOT ~24ms(16并发)、显存峰值28.86GB
3. **RAG延迟拆解**：embedding~50ms→向量检索~30ms→rerank~80ms→LLM生成~2000ms。优化重点：rerank与LLM生成并行、embedding/检索加缓存
4. **Fast path路由**：简单问题(感冒/发烧/常见药)跳过RAG直接生成；高风险问题(危重症/罕见病/复杂用药)走完整RAG。预期70%请求走fast path
5. **缓存三层**：L1-embedding缓存(完全相同query复用)、L2-向量检索结果缓存(相似query复用)、L3-LLM回答缓存(完全相同query+context复用)
6. **单卡5090训练8B**：QLoRA(NF4 4bit基础+LoRA adapter)+gradient checkpointing+paged_adamw_8bit+pre-compute ref logprobs(DPO)+max_seq_len=2048。SFT显存~18GB，DPO显存~22-24GB
7. **8卡扩展方案**：QLoRA→全参微调(ZeRO-3/FSDP)、更大batch size(128→1024)、更长序列(4096→8192)、更大模型(8B→14B/32B)
8. **多副本vs张量并行**：8B模型单卡够用→部署4个单卡副本+负载均衡(4x吞吐)；70B模型单卡放不下→张量并行切到4张卡(TP=4)
9. **项目没上线怎么讲落地能力**：强调"完整的工程思维"——训练→merge→vLLM→压测→性能优化→RAG延迟优化→缓存设计→监控方案设计。不是"做了demo"，是"完成了从模型到可部署服务的完整链路，性能经压测验证，关键工程问题有思考和解决方案"
10. **诚实边界**：明确区分"已实现"(vLLM部署+压测+LoRA merge)和"设计阶段"(K8s/灰度/完整监控/多节点)，不要模糊界限
