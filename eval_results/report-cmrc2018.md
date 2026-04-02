# RAG 检索评测报告（CMRC 2018 阅读理解）

> 评测日期：2026-04-03
> 数据集：CMRC 2018（台达阅读理解数据集）验证集子集
> 评测维度：检索质量（无生成评估）

---

## 1. 评测方法

### 1.1 数据集

| 属性 | 值 |
|------|-----|
| 数据集 | CMRC 2018（台达阅读理解数据集） |
| 评测记录数 | 50 |
| 有效题目（有关联 Chunk） | 45 |
| 数据格式 | Span Extraction（抽取式阅读理解） |
| 答案类型 | 文本片段（从 context 中抽取） |
| Context 平均长度 | ~600-1000 字符 |

### 1.2 评测流程

1. **索引**：将 CMRC 2018 context 按三层分块策略写入独立 Milvus collection (`eval_cmrc2018`)
2. **Ground Truth**：若 Chunk 文本与任意答案文本的 3-gram 重叠度 ≥ 30%，标记为相关
3. **检索**：使用 `question` 作为查询，在 L3 层（leaf）检索 top-5
4. **指标计算**：Precision@5, Recall@5, MRR, NDCG@5

### 1.3 消融实验

| 实验 | 说明 |
|------|------|
| Baseline | 完整流程：Hybrid Search → RRF → Cross-Encoder Rerank → Auto-Merging |
| No Rerank | 跳过 Cross-Encoder，仅靠 RRF 排序结果 |
| No Auto-merge | 跳过自动合并，rerank 后直接返回 L3 层 Chunk |

---

## 2. 评测结果

### 2.1 核心指标

| 实验 | P@5 | R@5 | MRR | NDCG@5 | 有效题目数 |
|------|------|------|------|--------|----------|
| Baseline | 0.2089 | 0.3407 | 0.5767 | 0.3259 | 45 |
| No Rerank | 0.2089 | 0.3407 | 0.5767 | 0.3259 | 45 |
| No Auto-merge | 0.2089 | 0.3407 | **0.5537** | **0.3189** | 45 |

> **关键发现**：
> - **Rerank 依然无效果**：No Rerank 与 Baseline 完全一致，Cross-Encoder 对阅读理解场景排序无额外贡献
> - **Auto-merge 作用减弱**：禁用后 MRR 下降 ~4%，NDCG 下降 ~2.2%，影响小于 CMRC 2019 填空场景

### 2.2 与 CMRC 2019（填空题）的对比

| 指标 | CMRC 2019 (填空) | CMRC 2018 (阅读理解) | 差异分析 |
|------|------|------|--------|
| MRR Baseline | 0.9020 | 0.5767 | 填空题检索更容易命中 |
| P@5 Baseline | 0.2484 | 0.2089 | 阅读理解 precision 更低 |
| Rerank 效果 | 无差异 | 无差异 | 两类任务均无效果 |
| Auto-merge MRR 下降 | -12.7% | -4.0% | 填空题受益更多 |

---

## 3. 分析与发现

### 3.1 Rerank 无效果的原因

Cross-Encoder 重排序在两种数据集上均未产生差异。可能原因：
- RRF 融合已将语义最接近的结果排在前列
- 评测 query 是 question 而非完整答案，Cross-Encoder 难以进一步区分
- 当前 rerank_top_k=5，候选数较少，重排空间有限

### 3.2 Auto-merge 在不同任务上的差异

在 CMRC 2019（填空题）中禁用 Auto-merge 导致 MRR 下降 12.7%，而 CMRC 2018（阅读理解）中仅下降 4%。这说明：
- 填空题答案短，Auto-merge 提供的上下文扩展对匹配帮助更大
- 阅读理解答案较长（完整短语/句子），L3 层 Chunk 本身已包含足够上下文

### 3.3 低 MRR 的原因（0.58 vs 0.90）

阅读理解的 question 与 context 中的答案片段语义关联较弱，不像填空题中 question 与 context 高度重合。因此语义检索更难精确召回第一相关文档。

---

## 4. 结论与建议

### 4.1 结论

1. **Rerank 组件暂非必选**：在当前配置下（top_k=5），RRF 融合已提供足够排序质量
2. **Auto-merge 建议保留**：对需要上下文扩展的场景（填空题、短答案）有正向贡献
3. **数据集特性影响显著**：填空题检索效果远好于阅读理解，需根据实际任务类型选择评测集

### 4.2 后续建议

- **扩大评测规模**：当前仅 50 条，建议用 300-500 条做完整评测以提升置信度
- **测试更大 top_k**：当前 top_k=5，可测试 top_k=10/20 观察 Recall 提升
- **引入生成评估**：接入 LLM 生成接口，评估 RAG 答案质量
- **调整 rerank 候选数**：当前 rerank 候选仅 top_k*3=15，可增大候选池观察效果

---

## 5. 输出文件

| 文件 | 说明 |
|------|------|
| `backend/eval/eval_config.py` | 评测配置（可切换 DATASET_PATH 使用不同数据集） |
| `backend/eval/eval_utils.py` | 指标计算（Precision/Recall/MRR/NDCG） |
| `backend/eval/eval_indexer.py` | CMRC 数据集索引脚本 |
| `backend/eval/eval_retrieval.py` | 批量评测脚本 + 三种消融实验 |
| `eval_results/ground_truth.json` | Ground Truth 标注 |
| `eval_results/YYYY-MM-DD_retrieval_metrics.csv` | 评测结果原始数据 |
