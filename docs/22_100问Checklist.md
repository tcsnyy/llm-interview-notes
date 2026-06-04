# 22 100问Checklist（222问）

> 使用方法：逐条检查掌握程度，标记 V1=未掌握 / V2=基本掌握 / V3=熟练，以及是否能结合项目回答 Y/N。面试前至少所有问题达到 V2。

---

## 一、大模型基础（10问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 1 | 大模型"大"在哪里？参数量、数据量、计算量的 scaling law 是什么？ | V_ | Y/N |
| 2 | GPT 系列从 GPT-1 到 GPT-4 的核心演进路径？ | V_ | Y/N |
| 3 | Decoder-only、Encoder-only、Encoder-Decoder 三种架构各适合什么任务？ | V_ | Y/N |
| 4 | 为什么现在主流都是 Decoder-only？ | V_ | Y/N |
| 5 | 什么是涌现能力（emergent abilities）？哪些能力被认为是涌现的？ | V_ | Y/N |
| 6 | 大模型的幻觉（hallucination）有哪几种类型？根源是什么？ | V_ | Y/N |
| 7 | Pre-training 和 Post-training 的区别和各自的子阶段？ | V_ | Y/N |
| 8 | 上下文学习（In-Context Learning）是什么？跟 fine-tuning 的区别？ | V_ | Y/N |
| 9 | 思维链（Chain-of-Thought）是什么？为什么能提升推理能力？ | V_ | Y/N |
| 10 | 训练 loss 持续下降但生成质量变差是怎么回事？（Goodhart's Law） | V_ | Y/N |

---

## 二、Transformer（10问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 11 | Transformer 的完整架构图？每层的输入输出？ | V_ | Y/N |
| 12 | Self-Attention 的计算公式？Q/K/V 是什么？ | V_ | Y/N |
| 13 | Multi-Head Attention 为什么多个头？每个头在学什么？ | V_ | Y/N |
| 14 | 为什么 Attention 要除以 sqrt(d_k)？ | V_ | Y/N |
| 15 | Pre-Norm vs Post-Norm 的区别？为什么 LLaMA/Qwen 用 Pre-Norm？ | V_ | Y/N |
| 16 | Cross-Attention 是什么？Decoder-only 为什么没有？ | V_ | Y/N |
| 17 | Causal Mask 是什么？为什么需要？怎么实现？ | V_ | Y/N |
| 18 | 位置编码三种方式（Absolute / Relative / RoPE）各有什么优缺点？ | V_ | Y/N |
| 19 | SwiGLU 激活函数是什么？相比 ReLU 的改进？ | V_ | Y/N |
| 20 | Flash Attention 的原理？为什么快？省了多少显存？ | V_ | Y/N |

---

## 三、Tokenizer（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 21 | BPE tokenizer 的训练流程？vocab_size 怎么选？ | V_ | Y/N |
| 22 | WordPiece 和 BPE 的区别？ | V_ | Y/N |
| 23 | SentencePiece 和 HuggingFace Tokenizers 的区别？ | V_ | Y/N |
| 24 | Tokenizer 的 special tokens 有哪些？各自的作用？ | V_ | Y/N |
| 25 | Tokenizer 的 vocab_size 太小/太大各有什么问题？ | V_ | Y/N |
| 26 | 中文 tokenizer 为什么效率比英文低？怎么改进？ | V_ | Y/N |
| 27 | chat template 是什么？有什么用？ | V_ | Y/N |
| 28 | 为什么不能随意换 tokenizer（不重新训练 embedding）？ | V_ | Y/N |

---

## 四、SFT（12问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 29 | SFT 的训练目标是什么？loss 公式？ | V_ | Y/N |
| 30 | SFT 数据格式是什么？和预训练数据有什么区别？ | V_ | Y/N |
| 31 | 为什么 SFT 只计算 response 部分的 loss？prompt 部分 mask 掉？ | V_ | Y/N |
| 32 | SFT 应该训练几个 epoch？为什么容易过拟合？ | V_ | Y/N |
| 33 | SFT 数据量和质量哪个更重要？为什么？ | V_ | Y/N |
| 34 | SFT 数据的多样性如何保证？ | V_ | Y/N |
| 35 | SFT 的学习率、batch size 怎么选？ | V_ | Y/N |
| 36 | SFT 后 loss 很低但回答很差是什么问题？（reward hacking / 过拟合） | V_ | Y/N |
| 37 | 多轮对话 SFT 怎么做？跟单轮有什么不同？ | V_ | Y/N |
| 38 | 怎么构造高质量的 SFT 数据？有哪些数据增强手段？ | V_ | Y/N |
| 39 | SFT 对模型能力的影响：学会的是什么？丧失的是什么？ | V_ | Y/N |
| 40 | 你项目 SFT 的具体参数？（数据量、lr、batch、epoch、max_length） | V_ | Y/N |

---

## 五、LoRA / QLoRA（10问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 41 | LoRA 的数学公式？A 和 B 的维度？初始化策略？ | V_ | Y/N |
| 42 | LoRA 的 rank r 怎么选？α 是什么？α/r 的实际意义？ | V_ | Y/N |
| 43 | LoRA 应该加到哪些层？为什么一般加 attention 和 FFN 的线性层？ | V_ | Y/N |
| 44 | QLoRA 的四个核心技术？每个省了多少显存？ | V_ | Y/N |
| 45 | NF4 量化的原理？为什么比 INT4 好？ | V_ | Y/N |
| 46 | 双重量化（Double Quantization）是什么？ | V_ | Y/N |
| 47 | LoRA vs Adapter vs Prefix Tuning vs Prompt Tuning 的对比？ | V_ | Y/N |
| 48 | LoRA 合并（merge）后和全量微调效果一样吗？差距在哪里？ | V_ | Y/N |
| 49 | 多个 LoRA 怎么切换？怎么做 MoE-style 的 LoRA？ | V_ | Y/N |
| 50 | 你项目 QLoRA 的具体配置？（r、α、target_modules、quant_config） | V_ | Y/N |

---

## 六、DPO（12问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 51 | DPO 的 loss 公式？每个符号的含义？ | V_ | Y/N |
| 52 | DPO 的 BT 模型假设是什么？怎么从 BT 模型推导出 DPO loss？ | V_ | Y/N |
| 53 | β（beta）的作用？β 太小/太大各有什么后果？ | V_ | Y/N |
| 54 | DPO 梯度分析的结论？加权因子是什么？ | V_ | Y/N |
| 55 | DPO 为什么容易过拟合？怎么检测和缓解？ | V_ | Y/N |
| 56 | DPO 的 chosen/rejected 数据应该怎么构造？什么算好的 preference pair？ | V_ | Y/N |
| 57 | DPO 训练中 reference model 的作用？为什么不能去掉？ | V_ | Y/N |
| 58 | DPO vs PPO 的优缺点对比？什么场景用哪个？ | V_ | Y/N |
| 59 | DPO 为什么可能导致幻觉增加？ | V_ | Y/N |
| 60 | Iterative DPO 是什么？和单轮 DPO 有什么区别？ | V_ | Y/N |
| 61 | 你项目 DPO 的具体参数？（数据量、β、epoch、lr、发现的问题） | V_ | Y/N |
| 62 | DPO 训练后 reward accuracy 很高但生成质量下降，怎么解释？ | V_ | Y/N |

---

## 七、RLHF / PPO / GRPO / RLVR（12问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 63 | RLHF 的完整流程？三个阶段的模型各是什么？ | V_ | Y/N |
| 64 | PPO 的四个模型（policy、reference、reward、value/critic）各自的作用？ | V_ | Y/N |
| 65 | PPO 的 clipped objective 是什么？为什么需要 clip？ | V_ | Y/N |
| 66 | KL 散度约束在 PPO 中怎么实现的？为什么需要？ | V_ | Y/N |
| 67 | PPO 的 GAE（Generalized Advantage Estimation）是什么？ | V_ | Y/N |
| 68 | GRPO 是什么？和 PPO 的区别？（去掉 value model，用 group 内 relative reward） | V_ | Y/N |
| 69 | RLOO 是什么？为什么不需 value model？ | V_ | Y/N |
| 70 | REINFORCE++ 相比 PPO 有什么改进？ | V_ | Y/N |
| 71 | DAPO 解决什么问题？（长尾 token 的 preference 问题） | V_ | Y/N |
| 72 | RLVR（RL with Verifiable Rewards）是什么？什么场景适用？ | V_ | Y/N |
| 73 | Reward Hacking 是什么？怎么缓解？ | V_ | Y/N |
| 74 | On-policy vs Off-policy RL for LLM 的区别？各自的挑战？ | V_ | Y/N |

---

## 八、OPD（5问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 75 | OPD 是什么？全称？核心思想？ | V_ | Y/N |
| 76 | OPD 和 DPO 的区别？on-policy 体现在哪里？ | V_ | Y/N |
| 77 | OPD 的数据怎么生成？为什么需要当前策略采样？ | V_ | Y/N |
| 78 | OPD 相比 DPO 的效果提升有多大？计算成本增加多少？ | V_ | Y/N |
| 79 | OPD 和 Iterative DPO 的关系？ | V_ | Y/N |

---

## 九、Reward Model / PRM / ORM（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 80 | Reward Model 的训练目标是什么？数据格式？loss？ | V_ | Y/N |
| 81 | Outcome Reward Model（ORM）和 Process Reward Model（PRM）的区别？ | V_ | Y/N |
| 82 | PRM 怎么标注？怎么训练？ | V_ | Y/N |
| 83 | Reward Model 为什么容易 reward hacking？ | V_ | Y/N |
| 84 | Reward Model 的泛化问题：训练分布外的打分不可靠怎么办？ | V_ | Y/N |
| 85 | 用什么指标评估 Reward Model 好坏？ | V_ | Y/N |
| 86 | 为什么 DPO 不需要显式训练 Reward Model？ | V_ | Y/N |
| 87 | Verifier 和 Reward Model 的区别？ | V_ | Y/N |

---

## 十、RAG（12问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 88 | RAG 的标准流程分几步？每步的关键技术？ | V_ | Y/N |
| 89 | Naive RAG / Advanced RAG / Modular RAG 的区别？ | V_ | Y/N |
| 90 | embedding model 怎么选？什么指标评估？ | V_ | Y/N |
| 91 | chunk size 怎么确定？overlap 的作用？ | V_ | Y/N |
| 92 | Hybrid Search（Dense + Sparse）是什么？怎么做融合？ | V_ | Y/N |
| 93 | Reranker 是什么？什么时候需要？常见的 reranker 模型？ | V_ | Y/N |
| 94 | RAG 的"lost in the middle"问题是什么？怎么缓解？ | V_ | Y/N |
| 95 | RAG 检索错误了怎么办？怎么评估检索质量？ | V_ | Y/N |
| 96 | RAG 和 Fine-tuning 分别适合什么场景？各有优劣？ | V_ | Y/N |
| 97 | RAG 评估指标有哪些？（检索端 + 生成端 + 端到端） | V_ | Y/N |
| 98 | 你项目的医疗知识库怎么构建的？数据来源？chunk 策略？ | V_ | Y/N |
| 99 | Safety-RAG 和标准 RAG 的核心区别是什么？为什么多一层验证？ | V_ | Y/N |

---

## 十一、LLM-as-Judge（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 100 | LLM-as-Judge 的设计原则是什么？常见 bias 有哪些？ | V_ | Y/N |
| 101 | Position bias 怎么缓解？Swap augmentation 怎么做？ | V_ | Y/N |
| 102 | 绝对打分 vs 成对比较（pairwise）各有什么优劣？ | V_ | Y/N |
| 103 | 怎么验证 LLM-as-Judge 的可靠性？用什么统计指标？ | V_ | Y/N |
| 104 | 你项目的评测体系设计？（维度、权重、judge 模型） | V_ | Y/N |
| 105 | 安全维度的 judge 不可靠怎么补偿？ | V_ | Y/N |
| 106 | MT-Bench / AlpacaEval / Chatbot Arena 的区别？ | V_ | Y/N |
| 107 | LLM-as-Judge 和 reward model 的关系？可以互相替代吗？ | V_ | Y/N |

---

## 十二、vLLM（6问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 108 | vLLM 的核心技术有哪些？每个解决了什么问题？ | V_ | Y/N |
| 109 | PagedAttention 的原理？为什么能提高吞吐？ | V_ | Y/N |
| 110 | Continuous batching 是什么？和 static batching 的区别？ | V_ | Y/N |
| 111 | vLLM vs TGI vs TensorRT-LLM 的对比？ | V_ | Y/N |
| 112 | vLLM 部署的关键参数？（max-model-len, gpu-memory-utilization, tensor-parallel-size） | V_ | Y/N |
| 113 | 你的项目 vLLM 部署的具体配置？并发能力？延迟？ | V_ | Y/N |

---

## 十三、训练工程（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 114 | 混合精度训练（fp16/bf16）的区别？为什么 bf16 不需要 loss scaling？ | V_ | Y/N |
| 115 | 梯度累积（Gradient Accumulation）的原理？有什么用？ | V_ | Y/N |
| 116 | Gradient Checkpointing 的原理？省多少显存？换什么代价？ | V_ | Y/N |
| 117 | ZeRO-1 / 2 / 3 的区别？各自的通信开销？ | V_ | Y/N |
| 118 | FSDP 和 DeepSpeed ZeRO-3 的区别？ | V_ | Y/N |
| 119 | 训练不收敛/震荡/Loss 为 NaN 怎么办？排查顺序？ | V_ | Y/N |
| 120 | 怎么估计训练需要多少显存？（参数 + 梯度 + 优化器状态 + 激活值） | V_ | Y/N |
| 121 | 你的训练使用的硬件和优化策略？（GPU 数量、显存、batch、GA） | V_ | Y/N |

---

## 十四、医疗安全（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 122 | 医疗 AI 安全的三个核心原则？ | V_ | Y/N |
| 123 | 什么是"过度拒答"问题？怎么检测和缓解？ | V_ | Y/N |
| 124 | 怎么在 prompt 层面避免模型给出处方建议？ | V_ | Y/N |
| 125 | 模型说了"我不是医生"但是之后又给了医疗建议怎么办？ | V_ | Y/N |
| 126 | 你们的安全 benchmark 是什么？用了什么指标？ | V_ | Y/N |
| 127 | 药品说明书 vs 临床指南冲突怎么处理？ | V_ | Y/N |
| 128 | 如何做安全的数据闭环？（用户反馈 → 安全标注 → 模型更新） | V_ | Y/N |
| 129 | 你们的安全防护和非医疗 LLM 应用的安全防护有什么不同？ | V_ | Y/N |

---

## 十五、项目深挖（15问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 130 | 你的项目完整 pipeline 是什么？每个阶段的输入输出？ | V_ | Y/N |
| 131 | 为什么选 Qwen3-8B 而不是其他模型？ | V_ | Y/N |
| 132 | 58K SFT 数据怎么来的？teacher 数据如何保证质量？ | V_ | Y/N |
| 133 | 10K DPO 数据怎么构造的？偏好信号从哪里来？ | V_ | Y/N |
| 134 | LLM-as-Judge 评测的维度？为什么选这三个？ | V_ | Y/N |
| 135 | DPO 后幻觉反而增加了，你怎么分析的？怎么解决的？ | V_ | Y/N |
| 136 | Safety-RAG 怎么设计的？为什么比简单拼接知识更安全？ | V_ | Y/N |
| 137 | "幻觉下降 66.7%"这个数字怎么来的？怎么正确表达？ | V_ | Y/N |
| 138 | 这个项目重新做的话，你会改进什么？ | V_ | Y/N |
| 139 | 项目的技术难点和创新点分别是什么？ | V_ | Y/N |
| 140 | 这个项目最能体现你算法能力的地方？ | V_ | Y/N |
| 141 | MiniMind 项目做了什么？学到了什么？ | V_ | Y/N |
| 142 | 除了你做的，你还了解哪些后训练方法？区别是什么？ | V_ | Y/N |
| 143 | 你对"后训练算法工程师"的理解是什么？ | V_ | Y/N |
| 144 | 你未来想深入学习哪个方向？（RLHF scaling / RAG / Agent / Safety） | V_ | Y/N |

---

## 十六、手撕代码（8问）

| # | 问题 | 掌握程度 |
|---|------|:---:|
| 145 | 手写 Scaled Dot-Product Attention（含 mask） | V_ |
| 146 | 手写 Multi-Head Attention 完整 forward | V_ |
| 147 | 手写 SFT loss（含 prompt mask） | V_ |
| 148 | 手写 DPO loss（PyTorch） | V_ |
| 149 | 手写 LoRA 层 forward | V_ |
| 150 | 手写完整的训练循环（含梯度累积） | V_ |
| 151 | 手写 BPE tokenizer 的合并逻辑 | V_ |
| 152 | 手写 RoPE 或 ALiBi 位置编码 | V_ |

---

## 十七、工程落地 -- 项目落地（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 153 | 模型如何从训练环境部署到生产环境？完整流程是怎样的？ | V_ | Y/N |
| 154 | 模型服务化（Model-as-a-Service）的 API 怎么设计？同步/异步/流式？ | V_ | Y/N |
| 155 | RESTful API vs gRPC vs WebSocket 在模型推理场景怎么选？ | V_ | Y/N |
| 156 | 模型版本管理怎么做？如何实现模型回滚？ | V_ | Y/N |
| 157 | 知识库版本管理和模型版本管理如何关联？两者独立更新的影响？ | V_ | Y/N |
| 158 | 你的项目 vLLM API 的服务化设计？（endpoint、请求/响应格式、错误码） | V_ | Y/N |
| 159 | 多个 LoRA adapter 如何在不重启服务的情况下热切换？ | V_ | Y/N |
| 160 | 模型部署的 CI/CD pipeline 怎么设计？自动化测试包含哪些？ | V_ | Y/N |

---

## 十八、工程落地 -- vLLM（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 161 | vLLM 启动的关键参数有哪些？每个参数的影响？ | V_ | Y/N |
| 162 | vLLM 压测需要关注哪些指标？（TTFT/TPOT/吞吐/延迟分布/错误率） | V_ | Y/N |
| 163 | LoRA merge 后部署 vs LoRA 动态加载 serving 的区别？各有什么优劣？ | V_ | Y/N |
| 164 | PagedAttention 的 page/block 大小如何选择？对显存和性能的影响？ | V_ | Y/N |
| 165 | 如何根据模型参数量和序列长度估算 vLLM 部署的显存占用？ | V_ | Y/N |
| 166 | Continuous batching 的调度策略详解？prefill 和 decode 如何混合调度？ | V_ | Y/N |
| 167 | vLLM 的 prefix caching 是什么？在 RAG 场景中如何利用？ | V_ | Y/N |
| 168 | vLLM vs SGLang vs TensorRT-LLM 在延迟/吞吐/生态上的对比？ | V_ | Y/N |

---

## 十九、工程落地 -- 推理性能（6问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 169 | TTFT (Time to First Token) 是什么？影响因素有哪些？ | V_ | Y/N |
| 170 | TPOT (Time per Output Token) 是什么？和 TTFT 的关系？ | V_ | Y/N |
| 171 | 推理吞吐（tokens/s）如何计算？如何最大化吞吐？ | V_ | Y/N |
| 172 | 推理延迟的 p50/p95/p99 怎么测量？各代表什么？ | V_ | Y/N |
| 173 | 并发用户数增加时，TTFT 和 TPOT 如何变化？饱和点怎么判断？ | V_ | Y/N |
| 174 | 流式输出（streaming）对 TTFT 和用户感知延迟的影响？ | V_ | Y/N |

---

## 二十、工程落地 -- RAG延迟（8问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 175 | RAG 端到端延迟怎么拆解？各阶段典型耗时比例？ | V_ | Y/N |
| 176 | embedding 检索的延迟瓶颈在哪里？如何优化？（FAISS IVF/HNSW） | V_ | Y/N |
| 177 | Reranker 会增加多少延迟？何时可以跳过 reranker？ | V_ | Y/N |
| 178 | LLM generation 阶段延迟在 RAG 总延迟中的占比？如何优化？ | V_ | Y/N |
| 179 | RAG 的 p99 延迟为什么容易飙高？如何优化 p99？ | V_ | Y/N |
| 180 | 检索结果数量（top-k）和生成质量/延迟的 trade-off？ | V_ | Y/N |
| 181 | 多路召回（dense+sparse）如何影响延迟？如何并行化？ | V_ | Y/N |
| 182 | RAG 延迟的端到端监控怎么做？如何定位瓶颈环节？ | V_ | Y/N |

---

## 二十一、工程落地 -- RAG并行（6问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 183 | dense 检索和 sparse 检索如何并行执行？（asyncio/线程池/多进程） | V_ | Y/N |
| 184 | RAG pipeline 的异步化设计？（检索和生成的重叠执行） | V_ | Y/N |
| 185 | asyncio.gather 在 RAG 多路召回中的应用？错误处理怎么做？ | V_ | Y/N |
| 186 | 检索结果的 batch 化处理？何时用 batch 何时用 streaming？ | V_ | Y/N |
| 187 | embedding 请求的批量处理（batching）如何减少网络往返？ | V_ | Y/N |
| 188 | RAG 各阶段的流水线化（pipelining）设计？如何减少空等时间？ | V_ | Y/N |

---

## 二十二、工程落地 -- RAG缓存（5问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 189 | RAG 系统中可以做哪些缓存？各自缓存什么内容？ | V_ | Y/N |
| 190 | Query cache（相似问题缓存）怎么做？用什么做 key？（query embedding/语义哈希） | V_ | Y/N |
| 191 | Embedding cache：相同文本片段避免重复 embedding 计算？ | V_ | Y/N |
| 192 | 检索结果缓存（retrieval cache）的设计？TTL 和失效策略？ | V_ | Y/N |
| 193 | 医疗场景的检索结果缓存有什么特殊注意事项？（隐私、时效性、个性化） | V_ | Y/N |

---

## 二十三、工程落地 -- 动态路由（5问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 194 | 什么是动态路由？在 RAG 场景中如何区分 fast path 和 full path？ | V_ | Y/N |
| 195 | 简单问题走 light-RAG（无 verifier），复杂问题走 full Safety-RAG，怎么判断？ | V_ | Y/N |
| 196 | 急症关键词（胸痛/呼吸困难/意识丧失）如何触发紧急路由？（直接建议就医） | V_ | Y/N |
| 197 | 用药咨询类请求如何路由到高安全等级 pipeline？ | V_ | Y/N |
| 198 | 动态路由的规则引擎设计？如何平衡规则准确率和系统复杂度？ | V_ | Y/N |

---

## 二十四、工程落地 -- 分布式训练（5问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 199 | DDP (DistributedDataParallel) 的工作原理？all-reduce 在什么时候执行？ | V_ | Y/N |
| 200 | FSDP (Fully Sharded Data Parallel) 和 DDP 的区别？参数分片粒度？ | V_ | Y/N |
| 201 | ZeRO-1/2/3 各自的通信开销和显存节省比例？ | V_ | Y/N |
| 202 | DeepSpeed 的配置文件（ds_config.json）关键参数有哪些？ | V_ | Y/N |
| 203 | torchrun 的启动方式？--nproc_per_node 和 --nnodes 的含义？ | V_ | Y/N |

---

## 二十五、工程落地 -- 分布式推理（4问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 204 | Tensor Parallelism (TP) 和多副本部署（Data Parallel serving）分别适用什么场景？ | V_ | Y/N |
| 205 | 多 GPU 推理服务如何启动？--tensor-parallel-size 参数怎么设？ | V_ | Y/N |
| 206 | 多副本部署时的负载均衡策略？（轮询/最少连接/一致性哈希） | V_ | Y/N |
| 207 | Nginx 反向代理 + 多 vLLM 实例的架构设计？健康检查怎么做？ | V_ | Y/N |

---

## 二十六、工程落地 -- 线上监控（5问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 208 | 线上模型服务需要监控哪些 metrics？（QPS/延迟/错误率/显存/队列长度） | V_ | Y/N |
| 209 | 请求日志和模型输出日志如何设计？哪些字段必须记录？ | V_ | Y/N |
| 210 | 模型的输出审计日志怎么做？（用户追问→模型回复→安全标签 全链路追踪） | V_ | Y/N |
| 211 | Prometheus + Grafana 如何集成到模型服务中？需要暴露哪些 endpoint？ | V_ | Y/N |
| 212 | 告警规则如何设计？什么情况触发 P0/P1/P2 告警？ | V_ | Y/N |

---

## 二十七、工程落地 -- 灰度发布（3问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 213 | Canary 发布（金丝雀发布）的流程？如何逐步切流量？ | V_ | Y/N |
| 214 | Blue-Green 部署 vs Canary 的优劣对比？什么场景用哪个？ | V_ | Y/N |
| 215 | 模型上线后发现问题如何回滚？回滚窗口期如何确定？ | V_ | Y/N |

---

## 二十八、工程落地 -- 安全审计（3问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 216 | 如何检测模型输出中的危险内容？（处方建议/诊断声明/紧急误导） | V_ | Y/N |
| 217 | 服务熔断（circuit breaker）在模型服务中如何设计？什么条件触发？ | V_ | Y/N |
| 218 | 服务降级策略？（vLLM 不可用时回退到简单规则引擎/静态 FAQ） | V_ | Y/N |

---

## 二十九、工程落地 -- 医学服务化（4问）

| # | 问题 | 掌握程度 | 结合项目 |
|---|------|:---:|:---:|
| 219 | 医疗对话的隐私脱敏如何实现？（姓名/年龄/医院名称等 PHI 识别和脱敏） | V_ | Y/N |
| 220 | 医疗知识库缓存的隐私安全问题？（患者查询记录不能用于其他用户） | V_ | Y/N |
| 221 | 急症检测如何在服务层实现优先级调度？（急症请求优先处理） | V_ | Y/N |
| 222 | 线上用户反馈如何形成数据闭环反哺模型训练？ | V_ | Y/N |

---

## 使用方法

1. **第一遍**：通读 222 问，标记掌握程度（V1/V2/V3）
2. **第二遍**：V1 的题目优先学习（看对应八股文件 + Cards）
3. **第三遍**：V2 的题目提升到 V3（能流畅口头表达）
4. **面试前一天**：随机抽取 50 题自测，目标 80% 以上能流畅回答
5. **"结合项目 Y/N"**：如果你选了 N，思考怎么把这个问题和你的项目关联起来

> 标记示例：V3/Y 表示熟练且能结合项目回答，V1/N 表示未掌握且不能结合项目——这种要优先补。

---

## 背诵版总结

1. 共222问，分29组，每组的掌握情况直接影响面试表现
2. **第一优先级（必V3）**：SFT(12问)、DPO(12问)、LoRA(10问)、RAG(12问)、项目深挖(15问)
3. **第二优先级（至少V2）**：大模型基础(10问)、Transformer(10问)、评测(8问)、安全(8问)、vLLM(8问)、推理性能(6问)、RAG延迟(8问)
4. **第三优先级（V2即可）**：RLHF/GRPO(12问)、OPD(5问)、RM/PRM(8问)、工程落地-项目落地(8问)、RAG并行(6问)、RAG缓存(5问)、动态路由(5问)、分布式训练(5问)、分布式推理(4问)、线上监控(5问)、灰度发布(3问)、安全审计(3问)、医学服务化(4问)
5. **工程落地组（第17-29组）**：即使项目未完全上线，也要掌握原理，面试时能结合"本地 vLLM 部署+压测"的实践经验回答
6. "结合项目=否"的需要重点补，面试中每个问题都尽量引回医学LLM项目的工程实践
7. 面试前一天随机抽取50题自测，目标80%以上能流畅回答
8. **新增工程落地 70 问重点**：vLLM 压测指标/延迟拆解/p99优化/缓存设计/动态路由/监控告警是面试高频考点
