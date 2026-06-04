# 03_Transformer与Attention

---

## 一、Transformer 整体结构

### Q: 画出 Transformer 的结构图，并解释每个组件的功能。star:5

Transformer 由 Encoder 和 Decoder 两部分组成（但现在 LLM 只用 Decoder 部分）。核心组件：

1. **Input Embedding**：将 token ID 映射为向量
2. **Positional Encoding**：注入位置信息（RoPE、ALiBi）
3. **Multi-Head Self-Attention**：让每个 token 关注序列中的其他 token
4. **Feed-Forward Network (FFN)**：对每个位置独立做非线性变换（SwiGLU）
5. **Residual Connection + LayerNorm/RMSNorm**：稳定训练
6. **LM Head**：将 hidden state 映射回 vocab space，输出 logits

**3 分钟版本**：在以上基础上展开每个组件的作用，特别是 causal mask 和 why decoder-only。

在项目中：
- MiniMind 从零实现了标准 Transformer（transformer.py），包括 MultiHeadAttention、PositionalEncoding、FFN 等
- MiniMind 中使用了 RMSNorm、RoPE、GQA（MiniMind2）、SwiGLU

---

## 二、Encoder-only / Decoder-only / Encoder-Decoder 区别

### Q: Bert 和 GPT 的架构区别是什么？为什么现在大模型都用 decoder-only？star:4

| 架构 | 代表模型 | Attention 类型 | 适用场景 |
|------|---------|---------------|---------|
| Encoder-only | BERT | Bidirectional self-attention | 理解任务（分类、NER） |
| Decoder-only | GPT, Llama, Qwen | Causal self-attention | 生成任务、对话 |
| Encoder-Decoder | T5, BART | Bidirectional encoder + Causal decoder | 翻译、摘要 |

Decoder-only 成为主流的原因：
1. 统一范式：所有 NLP 任务都可以用"文本 in -> 文本 out"
2. Causal attention + teacher forcing 让训练非常高效
3. Scaling law 已验证 decoder-only 在更大规模下效果持续提升

---

## 三、Self-Attention / Multi-Head Attention

### Q: 手写一下 Multi-Head Attention 的 forward 过程，包括维度变换。star:5

流程：
1. 输入 x: `[batch, seq_len, d_model]`
2. 分别通过 Q/K/V 线性投影：`[batch, seq_len, d_model] -> [batch, seq_len, d_model]`
3. Reshape 为多头：`[batch, seq_len, num_heads, head_dim]` -> transpose -> `[batch, num_heads, seq_len, head_dim]`
4. 计算 attention score：$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$
5. 乘以 V：`[batch, num_heads, seq_len, head_dim]`
6. 合并多头：transpose + reshape -> `[batch, seq_len, d_model]`
7. Output projection

**手撕代码**：见文件 16_手撕代码合集.md

---

## 四、MHA / MQA / GQA 区别

### Q: MHA、MQA、GQA 分别是什么？为什么现在都用 GQA？star:5

| 类型 | 全称 | K/V 头数 | 参数/显存 | 效果 |
|------|------|---------|----------|------|
| MHA | Multi-Head Attention | = Q 头数 | 最大 | 最好 |
| MQA | Multi-Query Attention | = 1 | 最小 | 略差 |
| GQA | Grouped-Query Attention | 1 < N < Q头数 | 折中 | 接近 MHA |

GQA 是 MHA 和 MQA 的折中方案。Q 头保持不变，K/V 头分成若干组，每组共享一对 K/V 头。这样既减少了 KV Cache（因为 K/V 头更少），又不会像 MQA 那样损失太多效果。

现在模型普遍用 GQA：Qwen3、Llama 3、Mistral 等都使用 GQA。MiniMind 项目中 `num_key_value_heads=2`（Q 头=8），使用了 GQA。

---

## 五、Q/K/V 作用

### Q: Q、K、V 分别代表什么？为什么需要三个投影？star:4

- **Q (Query)**：当前 token "我在找什么"
- **K (Key)**：每个 token "我是什么"，用来和 Q 匹配
- **V (Value)**：每个 token "我的内容是什么"，实际被聚合的信息

Q 和 K 的点积计算相关性（attention score），然后用这个 score 加权聚合 V。分离 K 和 V 的原因：匹配相关性（K）和传递内容（V）是两个不同的任务，用不同矩阵更灵活。

---

## 六、为什么 Attention Score 除以 sqrt(d)

### Q: Attention 公式里为什么除以 sqrt(d_k)？star:4

假设 Q 和 K 的每个元素独立同分布（均值 0，方差 1），则 $QK^T$ 的点积方差为 $d_k$。当 $d_k$ 很大时，点积值会很大，导致 softmax 的梯度消失（进入饱和区）。除以 $\sqrt{d_k}$ 将方差归一化到 1，保持 softmax 梯度健康。

---

## 七、Causal Mask / Padding Mask

### Q: causal mask 和 padding mask 有什么区别？为什么要区分？star:4

- **Causal Mask**：下三角矩阵（1 在下三角，0 在上三角），确保 token t 只能看到 t 及之前的 token
- **Padding Mask**：标记哪些位置是 padding（通常用 0/1 或 True/False），padding 位置不参与 attention

两者通常结合使用：causal_mask & padding_mask。

**手撕代码**：见文件 16_手撕代码合集.md

---

## 八、FFN / SwiGLU

### Q: Transformer 的 FFN 做了什么？SwiGLU 是什么？star:4

FFN 对每个位置独立做非线性变换：
$$FFN(x) = W_2 \cdot \text{Activation}(W_1 \cdot x + b_1) + b_2$$

传统用 ReLU 或 GELU。现代 LLM 普遍用 SwiGLU：
$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \odot (xW_2)$$

其中 Swish = $x \cdot \text{sigmoid}(\beta x)$。SwiGLU 效果比 ReLU/GELU 好，但参数量增加约 33%。

MiniMind 的配置中 `hidden_act='silu'`（即 Swish），是 SwiGLU 的基础。

---

## 九、Residual Connection / LayerNorm / RMSNorm / PreNorm / PostNorm

### Q: 为什么需要 Residual Connection？PreNorm 和 PostNorm 哪个好？star:4

- **Residual Connection**：$y = F(x) + x$，让梯度可以直通，解决深层网络梯度消失/爆炸
- **RMSNorm**：LayerNorm 的简化版，去除均值的归中操作，只做 scale，计算更快，效果相当。现代 LLM 普遍使用
- **PreNorm**（先 Norm 再 Attention/FFN）：训练更稳定
- **PostNorm**（先 Attention/FFN 再 Norm）：效果略好但训练不稳定，不太用了

MiniMind 使用 RMSNorm + PreNorm 结构。

---

## 十、RoPE / ALiBi

### Q: RoPE 是什么？它解决了什么问题？star:5

**RoPE (Rotary Position Embedding)** 是目前最主流的位置编码方式。它通过旋转变换将位置信息编码到 Q 和 K 中：
$$f(q, m) = q \cdot e^{im\theta}$$
$$f(k, n) = k \cdot e^{in\theta}$$

点积后：$f(q,m) \cdot f(k,n) = q \cdot k \cdot e^{i(m-n)\theta}$，只依赖于相对位置 m-n。

**优点**：
1. 天然支持相对位置
2. 通过调整 theta 可以外推到更长的上下文
3. 计算高效

Qwen3 使用 RoPE，MiniMind 配置中 `rope_theta=1,000,000`，且支持 YaRN 外推。

---

## 十一、KV Cache

### Q: KV Cache 是什么？为什么要 cache？star:5

自回归生成时，每次生成一个 token 都需要做 self-attention。如果没有 KV Cache，每次都要把之前所有 token 的 K、V 重新算一遍，导致 $O(n^2)$ 的重复计算。有 KV Cache 后，每次只需计算新 token 的 Q/K/V，之前的 K/V 直接从 cache 读取，计算量降为 $O(n)$。

**代价**：KV Cache 占用显存。对于 8B 模型，KV Cache 可能占用几 GB 到几十 GB（取决于 batch size 和序列长度）。

在我的项目中，vLLM 部署使用 PagedAttention 管理 KV Cache；压测时 TTFT 由 prefill 决定，TPOT 由 decode + KV Cache 决定。

---

## 十二、FlashAttention

### Q: FlashAttention 为什么快？star:4

FlashAttention 通过**算子融合和分块计算**来减少 HBM 读写：
1. 传统 attention：把完整的 QK^T 矩阵写入 HBM -> 读回做 softmax -> 写回 -> 读回乘以 V（大量 HBM IO）
2. FlashAttention：使用 tiling 将 attention 分块在 SRAM 中计算，online softmax 避免存储完整的 attention matrix
3. FlashAttention-2 进一步优化了并行策略，减少非矩阵乘法运算

---

## 十三、长上下文为什么难

### Q: 为什么 LLM 处理长上下文很难？star:4

1. **Attention 复杂度**：Self-attention 的复杂度是 $O(n^2)$，序列翻倍，计算量翻四倍
2. **KV Cache 显存**：随序列长度线性增长
3. **位置编码外推**：训练时的位置编码范围有限，推理时需要外推但可能不稳定
4. **Lost in the Middle**：模型对中间位置的信息利用效率低于开头和结尾

---

## 十四、MiniMind 项目中相关代码

MiniMind 项目 `model/model_minimind.py` 完整实现了 decoder-only Transformer，关键特点：
- RMSNorm（不是 LayerNorm）
- RoPE 位置编码
- GQA（num_key_value_heads=2, num_attention_heads=8）
- SwiGLU FFN
- FlashAttention（`flash_attn=True`）
- YaRN 外推支持
- 可选 MOE 架构
- 可选 MTP（Multi-Token Prediction）

MiniMind 项目 `model/transformer.py` 包含标准 Transformer 教学实现，包括 MultiHeadAttention、PositionalEncoding。

---

## 背诵版总结

1. Transformer 核心：Embedding + Attention + FFN + Residual + Norm + LM Head
2. MHA/MQA/GQA：GQA 是折中，K/V 头少于 Q 头但多于 1
3. QKV：Q 匹配相关，K 定义自身特征，V 传递内容
4. sqrt(d_k)：防止点积方差过大导致 softmax 梯度消失
5. Causal Mask 限制看到未来，Padding Mask 忽略填充
6. SwiGLU 替代 ReLU/GELU，效果更好但参数量增加
7. RMSNorm 替代 LayerNorm，去均值化
8. RoPE 用旋转变换编码相对位置，主流方案
9. KV Cache 避免重复计算，推理加速关键
10. FlashAttention 通过分块计算 + IO 优化加速 attention
11. 长上下文难在 Attention O(n^2)、KV Cache 显存、位置外推、Lost in Middle
12. MiniMind 实现了完整的 decoder-only Transformer（RMSNorm+RoPE+GQA+SwiGLU+FlashAttn）
