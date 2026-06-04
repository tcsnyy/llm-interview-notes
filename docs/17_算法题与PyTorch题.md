# 17_算法题与PyTorch题

---

## 一、后训练算法实习常见算法题类型概述

### 为什么还要考算法题？

后训练岗位的面试**不是纯工程岗**，但算法题仍然是必须环节。原因：
1. **筛人效率高**：一轮 coding 能快速判断基本编程能力
2. **考察工程思维**：DPO/SFT 代码本质上也涉及批量处理、维度变换
3. **团队协作**：后训练算法工程师需要写高质量的训练代码、数据处理 pipeline

### 考什么类型的题？

| 类型 | 占比 | 说明 |
|------|------|------|
| LeetCode 常规算法题 | 40% | 字符串、数组、哈希表、双指针为主 |
| PyTorch/张量操作题 | 30% | 维度变换、mask、gather/scatter |
| 机器学习基础手写 | 20% | cross entropy、softmax、attention |
| 场景题 | 10% | "这个 batch 怎么处理"、"这段代码有什么问题" |

### 难度定位

后训练岗位的算法题难度**低于纯 SDE 岗**，LeetCode Medium 为主，Hard 出现概率低。重点不是解题速度，而是：
- 代码风格清晰
- 能主动分析复杂度
- 对 PyTorch 操作熟练
- 能讨论边缘情况

---

## 二、LeetCode 优先级排序

按后训练面试出现的频率排序：

### 第一优先级 (几乎必考)

| 题型 | 标签 | 面试频率 | 说明 |
|------|------|---------|------|
| 哈希表 | HashMap | ⭐⭐⭐⭐⭐ | 两数之和、字母异位词分组 |
| 双指针 | Two Pointers | ⭐⭐⭐⭐⭐ | 三数之和、盛水最多容器 |
| 滑动窗口 | Sliding Window | ⭐⭐⭐⭐ | 最长无重复子串、最小覆盖子串 |
| 前缀和 | Prefix Sum | ⭐⭐⭐⭐ | 子数组和为K、区域和检索 |

### 第二优先级 (高频)

| 题型 | 标签 | 面试频率 | 说明 |
|------|------|---------|------|
| 二分查找 | Binary Search | ⭐⭐⭐⭐ | 搜索旋转排序数组、找峰值 |
| 堆/TopK | Heap | ⭐⭐⭐⭐ | TopK 高频元素、合并K个有序链表 |
| 链表 | Linked List | ⭐⭐⭐ | 反转链表、环形链表 |
| 区间合并 | Intervals | ⭐⭐⭐ | 合并区间、插入区间 |

### 第三优先级 (偶尔出现)

| 题型 | 标签 | 面试频率 | 说明 |
|------|------|---------|------|
| DFS/BFS | Graph/Tree | ⭐⭐⭐ | 岛屿数量、二叉树遍历 |
| 动态规划(基础) | DP | ⭐⭐⭐ | 爬楼梯、打家劫舍 |
| 单调栈 | Stack | ⭐⭐ | 每日温度、柱状图最大矩形 |
| 树 | Tree | ⭐⭐ | 最近公共祖先、二叉树的右视图 |

### 基本不会考 (时间有限可以跳过)

- 图论高级算法 (Dijkstra 可以了解)
- 高级 DP (背包、编辑距离可能但概率低)
- 并查集
- 线段树/树状数组
- 数论

---

## 三、必刷 10 题 (每道题给思路但不给完整代码)

### 1. 两数之和 (Two Sum)

**题目**：给定数组 nums 和目标 target，找到两数之和等于 target 的索引。

**思路**：
- 哈希表存储遍历过的 `值 → 索引`
- 遍历时检查 `target - nums[i]` 是否在哈希表中
- 时间复杂度 O(N)，空间复杂度 O(N)

**面试关键词**：哈希表、一次遍历、边查边存

**复杂度**：O(N) 时间，O(N) 空间

**追问点**：如果要求返回所有组合呢？→ 排序+双指针，但要处理重复

---

### 2. 三数之和 (3Sum)

**题目**：给定数组，找到所有 a+b+c=0 的三元组，要求不重复。

**思路**：
- 先排序 O(N log N)
- 固定第一个数，剩下两个数用双指针
- 关键：去重！固定数和移动指针时都要跳过重复值
- 时间复杂度 O(N^2)，空间复杂度 O(1)（不含输出）

**面试关键词**：排序 + 双指针 + 去重

**追问点**：四数之和怎么做？→ 再加一层循环，O(N^3)

---

### 3. 最长无重复子串 (Longest Substring Without Repeating Characters)

**题目**："abcabcbb" → 最长无重复子串长度为 3 ("abc")

**思路**：
- 滑动窗口：left 指针和 right 指针
- 用哈希表/数组记录窗口内字符最后出现的位置
- right 遇到重复字符时，left 跳到重复字符的下一个位置
- 时间复杂度 O(N)，空间 O(字符集大小)

**面试关键词**：滑动窗口、哈希表、动态调整 left

**追问点**：如果字符集是 ASCII 怎么优化？→ 用固定大小 128 的数组代替 HashMap

---

### 4. 最小覆盖子串 (Minimum Window Substring)

**题目**：s = "ADOBECODEBANC", t = "ABC" → 最小覆盖子串是 "BANC"

**思路**：
- 滑动窗口 + 两个计数器 (t 需要哪些字符 and 各需多少个，窗口里有哪些字符 and 各有多少个)
- 维护一个 `valid` 计数器，当某字符数量满足 t 的需求时 valid++
- valid == t 中不同字符数时，尝试收缩 left
- 时间复杂度 O(N)，空间 O(字符集大小)

**面试关键词**：可变长度的滑动窗口、valid 计数器、收缩左边界

**追问点**：如果要按顺序覆盖？→ 这就是另一道题了 (子序列)

---

### 5. 子数组和为 K (Subarray Sum Equals K)

**题目**：nums = [1,1,1], k=2 → 答案为 2 (前两个 1 或后两个 1)

**思路**：
- 前缀和：prefix_sum[i] = sum(nums[0:i])
- 子数组 [i:j] 的和 = prefix_sum[j] - prefix_sum[i]
- 用一个哈希表存储前缀和出现的次数
- 遍历时，查询 prefix_sum - k 是否出现过

**面试关键词**：前缀和 + 哈希表、preSum 技巧

**追问点**：如果 nums 中有负数还能用吗？→ 可以，前缀和仍然有效

---

### 6. TopK 高频元素 (Top K Frequent Elements)

**题目**：nums = [1,1,1,2,2,3], k=2 → 返回 [1, 2]

**思路**：
- 统计频率：HashMap O(N)
- 方法1：大小为 k 的最小堆，遍历频率表，堆中始终保持频率最高的 k 个 O(N log K)
- 方法2：桶排序，以频率作为下标，频率相同的放在一个桶里，从高到底取 k 个 O(N)
- 面试时先说方法1（更通用），再提方法2（是最优解）

**面试关键词**：最小堆 size=k、桶排序优化

**追问点**：如果数据流呢？→ 最小堆适用于流式场景

---

### 7. 合并区间 (Merge Intervals)

**题目**：intervals = [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]

**思路**：
- 按区间左端点排序
- 遍历：如果当前区间和 merged 的最后一个有重叠 → 扩大 merged 区间
- 否则 → 把当前区间加入 merged

**面试关键词**：排序 + 一次遍历、判断重叠 `intervals[i][0] <= merged[-1][1]`

**追问点**：插入新区间呢？→ 先加入再合并，或二分找到插入位置

---

### 8. 反转链表 (Reverse Linked List)

**题目**：1 → 2 → 3 → 4 → 5 → None → 5 → 4 → 3 → 2 → 1 → None

**思路**：
- 迭代：pre=None, cur=head
- 每步：保存 next→翻转 cur 的指向→pre 和 cur 后移
- 时间复杂度 O(N)，空间复杂度 O(1)
- 面试时迭代和递归都写一遍（面试官可能要求）

**面试关键词**：双指针迭代、pre/cur/next 三变量、递归尾递归

**追问点**：反转链表的子区间？→ 定位到区间边界，反转子区间后接回去

---

### 9. LRU Cache

**题目**：实现 LRU (最近最少使用) 缓存，get 和 put 都是 O(1)

**思路**：
- 核心数据结构：HashMap + 双向链表
- HashMap: key → 链表中节点的引用 (实现 O(1) 查找)
- 双向链表：维护访问顺序，最近使用的在头部，最久未用的在尾部
- get: 查 HashMap → 找到后移到链表头部
- put: 如果存在则更新并移到头部；如果不存在，如果满了删尾部，插入头部
- 技巧：使用虚拟头尾节点，简化和边界判断

**面试关键词**：HashMap O(1) 查找 + 双向链表 O(1) 插入/删除

**追问点**：Python 可以直接用 OrderedDict → 面试官会要求"不用 OrderedDict 实现"

---

### 10. 岛屿数量 (Number of Islands)

**题目**：grid = [["1","1","0","0","0"],["1","1","0","0","0"],...] → 岛屿数量 = 1

**思路**：
- DFS: 遍历每个格子，遇到 '1' 就岛屿数+1，然后用 DFS 把相连的 '1' 全部置为 '0'
- BFS: 同理，用队列
- 时间复杂度 O(M×N)，空间 O(M×N)（最坏递归深度）
- 面试时先写 DFS（更简洁），再提 BFS

**面试关键词**：DFS 淹没岛屿、visited 可以用 in-place 修改

**追问点**：如果 grid 很大怎么办？→ BFS 避免栈溢出；或并查集离线处理

---

## 四、PyTorch 高频题 (每题给代码+解释)

### 1. Tensor Shape/View/Reshape/Transpose/Permute/Contiguous

**问题**："view 和 reshape 有什么区别？transpose 和 permute 有什么区别？contiguous 是干什么的？"

```python
import torch

# === view vs reshape ===
x = torch.randn(2, 3, 4)           # [2, 3, 4]

# view: 要求张量在内存中是连续的 (contiguous)
y = x.view(2, 12)                  # [2, 12] — 如果连续则成功，否则报错
# y = x.view(3, 8)                 # 注意: 维度乘积必须一致

# reshape: 不要求连续，会尽量返回 view，不行则 copy
z = x.reshape(2, 12)               # [2, 12] — 总可以用

# 典型场景: transpose 后再 view 会报错
x_t = x.transpose(1, 2)            # [2, 4, 3]
# x_t.view(2, 12)                  # RuntimeError! transpose 后不再连续
x_t.reshape(2, 12)                 # OK, reshape 会内部处理

# 修复: .contiguous() 让内存重新排列为连续
x_t.contiguous().view(2, 12)       # OK, [2, 12]


# === transpose vs permute ===
# transpose: 交换两个维度
x = torch.randn(2, 3, 4)
a = x.transpose(0, 2)              # [4, 3, 2]  — 交换维0和维2
# 等价于:
a = x.permute(2, 1, 0)             # [4, 3, 2]  — 显式指定所有维度顺序

# permute: 可以重排任意多个维度
b = x.permute(2, 0, 1)             # [4, 2, 3]  — transpose 做不到


# === contiguous 的本质 ===
# contiguous 表示张量在内存中按 C 顺序（行优先）连续存储
# transpose 不移动数据，只改变"查看方式"（stride 改变）
# contiguous() 会复制一份新的、连续存储的张量

# 检查是否连续:
print(x.is_contiguous())            # True
print(x.transpose(0, 1).is_contiguous())  # False
```

**面试关键词**：
- view: 连续内存才能用，不拷贝
- reshape: 总是可用，可能拷贝
- transpose: 交换两个维度，改变 stride
- permute: 任意排列维度，比 transpose 更通用
- contiguous: 让内存变连续，会拷贝

---

### 2. Gather 和 Scatter 的用法

这两者在 DPO/RLHF 代码中极其常用，用于从 logits 中"取"概率或"放回"数值。

```python
import torch

# ============ gather: 按索引收集 ============
# 典型用途: 从 [B, S, V] 的 log_probs 中取出 label token 的 log prob

log_probs = torch.randn(2, 5, 1000)      # [B=2, S=5, V=1000]
labels = torch.randint(0, 1000, (2, 5))  # [2, 5]

# gather: 在 vocab 维度(最后一个维度)按 labels 取值
# labels 需要 unsqueeze 以匹配 gather 维度
gathered = torch.gather(
    log_probs,                           # input: [2, 5, 1000]
    dim=-1,                              # 在 vocab 维度操作
    index=labels.unsqueeze(-1)          # index: [2, 5, 1]
).squeeze(-1)                           # output: [2, 5]
# 结果: gathered[i][j] = log_probs[i][j][labels[i][j]]


# ============ scatter: 按索引写入 ============
# 典型用途: 构造 attention mask 的逆过程

# scatter_add_: 按索引累加 (DPO 中统计某些 token 的累积 log prob)
src = torch.ones(2, 5)                   # [2, 5]
index = torch.randint(0, 3, (2, 5))     # [2, 5], 索引映射到 3 个桶
output = torch.zeros(2, 3)              # [2, 3]
output.scatter_add_(dim=-1, index=index, src=src)
# output[i][k] = sum of src[i][j] for all j where index[i][j] == k

# scatter_: 按索引复制 (One-hot 编码)
labels = torch.tensor([2, 0, 1])         # [3]
one_hot = torch.zeros(3, 4)             # [3, 4]
one_hot.scatter_(dim=-1, index=labels.unsqueeze(-1), value=1)
# one_hot: [[0,0,1,0], [1,0,0,0], [0,1,0,0]]
```

**面试关键词**：
- gather: 按 index 维度在 dim 维度上"取"——阅读操作
- scatter: 按 index 维度在 dim 维度上"写"——写入操作
- gather 在 DPO 中用于"取 label token 的 log prob"

---

### 3. Mask 操作 (masked_fill, Boolean Indexing)

```python
import torch

# ============ masked_fill: 按 mask 填充 ============
# 典型用途: causal mask、忽略 padding token

logits = torch.randn(2, 3, 1000)         # [B, S, V]

# 构造 padding mask: 最后一个 token 是 padding (假设)
padding_mask = torch.tensor([
    [0, 0, 1],   # 第三个位置是 padding
    [0, 1, 1],   # 第二和第三是 padding
], dtype=torch.bool)                     # [2, 3], True=padding

# 将 padding 位置的 logits 设为 -inf
logits_masked = logits.masked_fill(
    padding_mask.unsqueeze(-1),           # [2, 3, 1] 广播到 [2, 3, 1000]
    float("-inf")
)
# 现在 padding 位置 softmax 后概率=0


# ============ Boolean Indexing: 按条件选取 ============
# 典型用途: 过滤无效标签

labels = torch.tensor([[1, 2, -100, 5], [3, -100, 7, -100]])  # [2, 4]
valid_mask = labels != -100               # [2, 4], bool
valid_labels = labels[valid_mask]         # [N_valid], 1D
# valid_labels: tensor([1, 2, 5, 3, 7])

# 注意: boolean indexing 会将结果展平为 1D
# 如果希望保持 shape: 用 masked_fill + 原来的 shape


# ============ where: 条件选择 ============
# torch.where(condition, x, y)
# condition 为 True 的位置取 x, False 取 y

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([-1.0, -2.0, -3.0])
condition = a > 1.5
result = torch.where(condition, a, b)
# result: tensor([-1., 2., 3.])
```

**面试关键词**：
- masked_fill: 按 mask 修改原位置的值
- boolean indexing: 按条件提取元素，结果展平 1D
- where: 按条件在两个张量间选择

---

### 4. Softmax dim 参数的含义

**问题**："softmax(dim=0), softmax(dim=1), softmax(dim=-1) 分别是什么意思？"

```python
import torch
import torch.nn.functional as F

logits = torch.randn(2, 3, 4)            # [B=2, S=3, V=4]

# dim=-1 (或 dim=2): 在最后一维做 softmax
# 一行内的 4 个值变成概率，每行加起来 = 1
# 用途: 从 logits 得到 token 概率分布
probs_dim2 = F.softmax(logits, dim=-1)    # [2, 3, 4]
# probs_dim2[b][s].sum() == 1.0  (每一行的概率和为 1)

# dim=0: 沿 batch 维度做 softmax
# 同一位置的 2 个 batch 元素做 softmax
# 用途: 不常见，有时用于 attention weight 的 batch-normalization
probs_dim0 = F.softmax(logits, dim=0)    # [2, 3, 4]
# probs_dim0[:, s, v].sum() == 1.0  (每个 (s,v) 位置的 batch 维概率和为 1)

# dim=1: 沿序列维度 softmax
# 同一位置不同 token 之间做 softmax
# 用途: Attention weights! QK^T [B, H, S, S] → softmax(dim=-1)
#       这里的 dim=-1 是对 key 序列维度做 softmax
attn = torch.randn(2, 8, 5, 5)           # [B, H, S, S]
attn_weights = F.softmax(attn, dim=-1)    # 每个 query 对所有 key 的概率
# attn_weights[b][h][i].sum() == 1.0  (每个 query 的关注分布和为 1)

# 记忆口诀:
# dim=-1: "每行归一化" — 最常用，token probability
# dim=1:  "每列归一化" — 序列维度
# dim=0:  "每样本归一化" — 跨 batch
```

**面试关键词**：dim 指定的是 softmax 操作的**维度**，该维度上的值变成概率分布。dim=-1 表示"每个 token 的 vocab 分布"。

---

### 5. Cross Entropy 输入要求

**问题**："F.cross_entropy 的输入 logits 需要 softmax 吗？"

```python
import torch
import torch.nn.functional as F

# ❌ 错误: logits 已经过了 softmax
logits = torch.randn(4, 1000)
probs = F.softmax(logits, dim=-1)         # 不需要! 会导致 double softmax
labels = torch.randint(0, 1000, (4,))
loss = F.cross_entropy(probs, labels)     # 虽然不会报错，但数学上错误
# cross_entropy 内部会做 log_softmax，所以输入 softmax 后会变成 log(softmax(logits)) 而不是 log_softmax(logits)
# 实际上如果输入的是概率(全>0且和为1), log_softmax 的效果等价于 log(probs) - logsumexp(log(probs))
# 这会导致错误的结果

# ✅ 正确: logits 直接输入 (未经 softmax)
loss = F.cross_entropy(logits, labels)

# ✅ 等效于:
loss_manual = -F.log_softmax(logits, dim=-1)[range(len(labels)), labels].mean()
# 或
loss_manual = F.nll_loss(F.log_softmax(logits, dim=-1), labels)

# ============ ignore_index 用法 ============
# 记住: ignore_index 跳过 label == ignore_index 的位置
labels_with_pad = torch.tensor([1, 2, -100, 3])  # 第三个位置被跳过
loss = F.cross_entropy(logits[:4], labels_with_pad, ignore_index=-100)
# loss 只在位置 0, 1, 3 上计算

# ============ weight 参数 (类别不平衡) ============
class_weights = torch.ones(1000)
class_weights[rare_class_id] = 5.0  # 给罕见类更高权重
loss = F.cross_entropy(logits, labels, weight=class_weights)
```

**面试关键词**：
- cross_entropy 内部做了 log_softmax + nll_loss
- 输入必须是**原始 logits**
- ignore_index 用于跳过无效标签（如 padding=-100）

---

### 6. DataLoader / Dataset / collate_fn 关系

```python
import torch
from torch.utils.data import Dataset, DataLoader

# ============ 1. Dataset: 定义如何获取一个样本 ============
class MySFTDataset(Dataset):
    """
    输入: data: list of dict, 每项包含 "prompt" 和 "response"
    """
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """返回一个未 padding 的样本"""
        item = self.data[idx]
        # 构造完整对话: "<|user|> prompt <|assistant|> response"
        text = f"<|user|>\n{item['prompt']}\n<|assistant|>\n{item['response']}"
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            padding=False,  # 不在这里 padding!
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)  # [S]
        return {"input_ids": input_ids}


# ============ 2. collate_fn: 定义如何把 batch 合并 ============
def collate_fn(batch):
    """
    输入: list of dict, 每个 {"input_ids": [S_i]}
    输出: dict, 每个 value 是 [B, S_max]
    """
    # 找最大长度
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len
        # padding
        input_ids = torch.cat([
            item["input_ids"],
            torch.full((pad_len,), 0, dtype=torch.long)  # pad_token_id=0
        ])
        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long)
        ])
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)

    return {
        "input_ids": torch.stack(input_ids_list),         # [B, S_max]
        "attention_mask": torch.stack(attention_mask_list), # [B, S_max]
    }


# ============ 3. DataLoader: 管理 batch 的迭代 ============
dataloader = DataLoader(
    dataset,
    batch_size=8,          # 每个 batch 8 个样本
    shuffle=True,          # 每个 epoch 随机打乱
    collate_fn=collate_fn, # 自定义合并策略
    num_workers=4,         # 多进程加载数据
    pin_memory=True,       # 将数据固定在 page-locked 内存(加快 GPU 传输)
)

# 三者的调用顺序:
# 1. DataLoader 按 batch_size 从 Dataset 取样本 (调用 __getitem__)
# 2. 将 batch_size 个样本传给 collate_fn 合并
# 3. collate_fn 返回一个 batch，传给训练循环
```

**面试关键词**：
- Dataset: 定义"一个样本是什么"（`__getitem__` 和 `__len__`）
- collate_fn: 定义"如何把 batch 个样本合并成一个 batch"
- DataLoader: 把二者组合起来，管理 shuffle、多进程、batch 迭代

---

### 7. no_grad / detach / model.train() / model.eval() 区别

```python
import torch
import torch.nn as nn

# ============ no_grad vs detach ============
model = nn.Linear(10, 5)
x = torch.randn(3, 10)

# torch.no_grad(): 上下文管理器，禁用梯度计算
with torch.no_grad():
    y = model(x)     # y.requires_grad == False, 不构建计算图
    # 好处: 节省显存和计算

# detach(): 从计算图中分离一个张量
z = model(x)          # z.requires_grad == True
z_detached = z.detach()  # z_detached 与 z 共享数据，但不需要梯度
# 注意: detach 是张量级别，no_grad 是整个 block 级别


# ============ model.train() vs model.eval() ============
# train(): 启用训练特定行为
model.train()
# - Dropout 启用 (随机将神经元置零)
# - BatchNorm 更新 running_mean 和 running_var
# - 不影响 requires_grad (通过 optimizer 管理)

# eval(): 禁用训练特定行为
model.eval()
# - Dropout 停用 (不做随机置零)
# - BatchNorm 使用训练好的 running_mean 和 running_var (不更新)
# - 推理时必须调用

# 注意: 即使 model.eval()，只要参数 requires_grad=True，梯度还是会计算
# 推理时通常需要 model.eval() + torch.no_grad()
with torch.no_grad():
    model.eval()
    output = model(test_input)  # 推理模式，不计算梯度
```

**面试关键词**：
- no_grad: 不构建计算图，省显存
- detach: 从计算图中"摘"下一个张量
- train(): Dropout 有效 / BatchNorm 更新统计
- eval(): Dropout 停用 / BatchNorm 用固定统计 / 推理时必调用

---

### 8. 自定义 Dataset 示例

```python
from torch.utils.data import Dataset
import torch

class DPODataset(Dataset):
    """
    DPO 训练数据集
    每个样本包含 prompt, chosen response, rejected response
    """
    def __init__(self, data, tokenizer, max_length=512):
        """
        data: list of dict, 每个:
            {"prompt": str, "chosen": str, "rejected": str}
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize prompt + chosen
        chosen_text = item["prompt"] + "\n" + item["chosen"]
        chosen_enc = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # 如果后续 collate 不处理，这里 pad
            return_tensors="pt",
        )

        # Tokenize prompt + rejected
        rejected_text = item["prompt"] + "\n" + item["rejected"]
        rejected_enc = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),      # [S]
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),  # [S]
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
        }
```

**面试关键词**：`__len__` 和 `__getitem__` 必须实现，每次返回 dict 或 tuple，tokenize 可以在这里做或在 collate 里做。

---

## 五、LLM 训练相关的 PyTorch 追问

### 1. 为什么 DDP 中 Gradient 要 all_reduce？

**问题**："在 DDP 中每个 GPU 各自计算梯度后，为什么要 all_reduce？怎么 reduce 的？"

**标准回答**：

DDP (DistributedDataParallel) 中，每个 GPU 持有模型的一个副本，各自处理 batch 的不同子集（mini-batch）。

1. **前向**：每个 GPU 独立计算 loss，不需要通信
2. **反向**：每个 GPU 独立计算梯度（基于自己的 mini-batch）
3. **All-Reduce**：所有 GPU 的梯度做平均（reduce: 求和；all: 结果分发回每个 GPU）
4. **更新**：每个 GPU 用平均后的梯度更新自己的参数副本

为什么需要 all_reduce？
- 如果不通信，每个 GPU 的梯度只反映自己看到的 mini-batch
- All-reduce 后，每个 GPU 的梯度反映整个 batch 的信息
- 数学上：全局梯度 = (1/N) * sum(每个 GPU 的局部梯度)

```python
# 伪代码: DDP 的 backward 钩子
def ddp_backward_hook(grad, process_group):
    # grad: 本 GPU 计算得到的参数梯度
    # all_reduce: 求和所有 GPU 的梯度, 然后除以 world_size
    dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=process_group)
    grad /= dist.get_world_size(process_group)
    # 现在 grad 是所有 GPU 上的平均梯度
    return grad
```

**追问**："all_reduce 和 reduce 的区别？"
- reduce: 结果只发送到一个指定的 GPU（root）
- all_reduce: 结果分发给所有 GPU
- DDP 用 all_reduce 因为每个 GPU 都要更新自己的参数

---

### 2. torch.cuda.amp 的使用

**问题**："混合精度训练怎么做的？为什么能加速？"

**标准回答**：

混合精度训练 (AMP: Automatic Mixed Precision) 使用 FP16 做前向和反向（速度快、省显存），但用 FP32 维护参数更新（避免精度损失）。

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # 梯度缩放器，防止 FP16 下溢

for batch in dataloader:
    optimizer.zero_grad()

    # 1. 前向: 部分运算自动用 FP16
    with autocast(device_type="cuda", dtype=torch.float16):
        loss = model(**batch).loss

    # 2. 反向: scale loss → backward → unscale grad
    # scaler.scale 将 loss 放大，避免小梯度在 FP16 下变为 0
    scaler.scale(loss).backward()

    # 3. 更新: unscale → 裁剪 → step → update scaler
    scaler.unscale_(optimizer)          # 将梯度除以 scale 恢复
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)              # optimizer.step()
    scaler.update()                     # 动态调整 scale 因子

# 为什么能加速？
# - FP16 矩阵乘法在 Tensor Core 上比 FP32 快 2-8x
# - FP16 显存占用是 FP32 的一半 → 可以用更大的 batch
# - GradScaler 防止 FP16 动态范围不足导致的小梯度消失
```

**追问**："GradScaler 为什么需要？"
- FP16 能表示的最小正数约 6e-8，很多梯度比这小
- Scale 把 loss 放大（如 2^16 倍），让小梯度落到 FP16 表示范围内
- 更新前 unscale 恢复，保证参数更新精度

---

### 3. Gradient Checkpointing 原理

**问题**："显存不够 train 一个 LLM 怎么办？Gradient checkpointing 怎么 work？"

**标准回答**：

Gradient checkpointing 是一种**用计算换显存**的技术。

**普通训练**：
- 前向时保存所有中间激活值（activations），反向时直接使用
- 显存占用：O(层数 × 激活值大小)

**Gradient checkpointing**：
- 前向时**不保存**中间激活值（除了少数 checkpoint 节点）
- 反向时**重新计算**前向过程需要的激活值
- 显存占用：O(sqrt(层数) × 激活值大小) 或 O(激活值大小)

```python
# PyTorch 中的使用
from torch.utils.checkpoint import checkpoint

class TransformerBlock(nn.Module):
    def __init__(self, ...):
        # attention, ffn, norms...
        pass

    def forward(self, x):
        # 使用 checkpoint 包装 self-attention + FFN
        # 前向时不保存中间结果，反向时重算
        x = x + checkpoint(self.self_attn, x, use_reentrant=False)
        x = x + checkpoint(self.ffn, x, use_reentrant=False)
        return x

# HuggingFace 中使用:
# model.gradient_checkpointing_enable()
# 等价于 model.config.gradient_checkpointing = True

# 效果:
# - 显存降低 ~30-50%
# - 训练速度降低 ~15-25% (因为要重算)
# - 对 batch size 的限制显著放宽
```

**追问**："checkpoint 策略有哪些？"
- 每层一个 checkpoint: 省最多显存，最慢
- 每 N 层一个: 折中
- 只对 attention 做: 常见策略，FFN 的激活值不大

**追问**："和 activation offloading 的区别？"
- checkpointing: 激活值不存，需要时重算（快但需要重算时间）
- offloading: 激活值存到 CPU（慢但不需要重算）
- 两者可结合使用

---

## 六、面试代码题答题技巧

### 写代码时的注意事项

1. **先说思路再写代码**：花 30 秒描述算法思路，面试官想看你的思维过程
2. **边写边说**：解释你在做什么，"这里用哈希表存储前缀和...""这里要处理边界条件..."
3. **主动分析复杂度**：写完主动说"时间复杂度 O(N)，空间复杂度 O(N)"
4. **写完后自测**：用一个简单例子跑一遍代码逻辑
5. **主动讨论边缘情况**："如果输入是空数组呢？""如果有重复元素呢？"

### PyTorch 题的答题策略

1. **先画张量维度**：在白板上画 `[B, S, V] → [B, S-1, V]` 这种维度变换
2. **说清楚 dim 参数**：每次用 softmax/gather/sum 都要说清楚 dim
3. **区分 in-place 和非 in-place**：`scatter_` 带下划线是 in-place
4. **注意 broadcast 规则**：哪些操作会自动广播，哪些不会

### 面试中如果遇到不会的

1. 不要说"不会"，说"这是一个很好的问题，让我想想..."
2. 从最基础的暴力解开始，逐步优化
3. 即使写不出最优解，写一个能跑的解法也比空着好
4. 主动提问澄清需求："输入范围是多少？""是否需要原地操作？"

---

## 背诵版总结

| 模块 | 必知必会 | 重要程度 |
|------|---------|---------|
| LeetCode 第一优先级 | 哈希表、双指针、滑动窗口、前缀和 | ⭐⭐⭐⭐⭐ |
| LeetCode 第二优先级 | 二分、堆、链表、区间 | ⭐⭐⭐⭐ |
| 必刷 10 题 | 两数之和/三数之和/最长无重复子串/最小覆盖子串/子数组和为K/TopK/合并区间/反转链表/LRU/岛屿数量 | ⭐⭐⭐⭐⭐ |
| PyTorch view/reshape | view 要求连续，reshape 不要求 | ⭐⭐⭐⭐ |
| PyTorch transpose/permute | transpose 交换二维, permute 重排所有维 | ⭐⭐⭐⭐ |
| PyTorch gather | dim 维度按 index 取值 | ⭐⭐⭐⭐⭐ |
| PyTorch masked_fill | 按 mask 填充 -inf, 用于 attention mask | ⭐⭐⭐⭐⭐ |
| softmax dim | dim=-1 "每行归一化", dim=1 "每列归一化" | ⭐⭐⭐⭐ |
| cross_entropy | 输入 logits, 不可过 softmax | ⭐⭐⭐⭐⭐ |
| DDP all_reduce | 所有 GPU 梯度求平均后分发 | ⭐⭐⭐ |
| Mixed Precision | GradScaler 防止 FP16 下溢 | ⭐⭐⭐ |
| Gradient Checkpointing | 用计算换显存，不存中间激活 | ⭐⭐⭐ |
| Dataset/DataLoader | __getitem__ → collate_fn → batch | ⭐⭐⭐⭐ |
