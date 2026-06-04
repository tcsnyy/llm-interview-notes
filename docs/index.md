# 🏠 LLM 后训练算法面试八股

> **大模型后训练算法实习 / LLM Post-training Intern / Alignment Intern 面试系统化复习资料**

---

## 📖 这是什么？

这是一套为 **"大模型后训练算法实习"** 面试准备的**中文八股复习资料库**，共 34 篇 Markdown 文档，覆盖从基础概念到项目深挖的完整面试知识体系。

## 🎯 适合岗位

| 岗位 | 匹配度 |
|------|:---:|
| 大模型后训练算法实习 | ⭐⭐⭐⭐⭐ |
| LLM Alignment Intern | ⭐⭐⭐⭐⭐ |
| SFT / DPO 方向实习 | ⭐⭐⭐⭐⭐ |
| RLHF / RLVR 方向实习 | ⭐⭐⭐⭐ |
| 大模型评测实习 | ⭐⭐⭐⭐ |
| RAG 应用实习 | ⭐⭐⭐⭐ |

## 🗂️ 资料结构

### 基础知识
- 大模型基础八股（Decoder-only、Causal LM、Cross Entropy、采样参数……）
- Transformer 与 Attention（MHA/MQA/GQA、RoPE、KV Cache、FlashAttention……）
- Tokenizer 与 ChatTemplate（BPE、ChatML、Qwen3、`<think>` 泄漏、padding side……）

### 核心算法
- **SFT 指令微调**（目标函数、assistant-only loss、labels=-100、packing）
- **LoRA / QLoRA**（公式推导、rank/alpha/dropout、merge、NF4、显存估算）
- **DPO 偏好优化**（loss 公式、beta、chosen/rejected 构造、和 RLHF 的区别）
- **RLHF / GRPO / RLVR**（PPO、GRPO、verifiable reward、DeepSeek-R1 后训练）
- **Reward Model / PRM / ORM**（pairwise RM、best-of-n、process supervision）
- **后训练数据工程**（teacher 数据生成、质量过滤、偏好对构造、数据闭环）

### 评测与 RAG
- **评测体系与 LLM-as-Judge**（score-based、pairwise、medical safety review）
- **RAG 与 Safety-RAG 详细八股**（15 本 PDF / 11707 chunks / 五层架构 / 8 条路由规则）

### 工程落地
- vLLM 推理部署与性能优化（TTFT、TPOT、p95、continuous batching）
- RAG 工程化与低延迟优化（并行化、15 种缓存、三级动态路由）
- 并行计算与分布式训练（DDP、FSDP、DeepSpeed ZeRO）
- 分布式推理与多 GPU 部署（TP、PP、多副本、负载均衡）
- 线上监控与灰度发布（25+ 监控指标、A/B 测试、安全审计、数据闭环）

### 冲刺工具
- 152 问 Checklist（自测掌握程度）
- 262 张 Flashcards（面试前一晚速刷）
- 面试自我介绍与项目讲述模板（30s/1min/2min/3min/5min）
- 后训练算法对比总表（25 种方法一张表）
- 60+ 常见追问与避坑回答

## 🚀 快速开始

### 本地预览

```bash
# 安装依赖
pip install mkdocs-material

# 启动本地开发服务器
cd llm-interview-notes
mkdocs serve

# 浏览器打开 http://127.0.0.1:8000
```

### 一键部署到 GitHub Pages

```bash
git add .
git commit -m "Update notes"
git push origin main
# GitHub Actions 自动构建并部署到 https://tcsnyy.github.io/llm-interview-notes/
```

## 📋 复习路线建议

| 天数 | 主题 | 核心任务 |
|------|------|---------|
| Day 1 | 基础 | 大模型基础 + Transformer + Tokenizer |
| Day 2 | 核心 | SFT + LoRA / QLoRA |
| Day 3 | 核心 | DPO + Reward Model |
| Day 4 | 评测 | RAG + Safety-RAG + LLM-as-Judge |
| Day 5 | 安全 | 医疗安全对齐 + 工程落地 |
| Day 6 | 工程 | vLLM + 训练工程 + 分布式 |
| Day 7 | 冲刺 | 项目深挖 + 模拟面试 + Flashcards |

## ⚠️ 注意事项

- 阅读顺序：按左侧导航从上到下
- 每个文件末尾都有 **"背诵版总结"**——面试前一晚快速扫描
- 项目相关内容严格区分 **"已实现"** 和 **"了解原理、项目中未实现"**
- 推荐先看 `01_岗位画像与复习路线.md` 了解全局

---

> 📬 仓库地址：[github.com/tcsnyy/llm-interview-notes](https://github.com/tcsnyy/llm-interview-notes)
