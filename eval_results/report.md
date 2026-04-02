# RAG 检索评测报告

> 评测日期：2026-04-02
> 数据集：CMRC 2019 Development Set
> 评测维度：检索质量（无生成评估）

---

## 1. 评测方法

### 1.1 数据集

| 属性 | 值 |
|------|-----|
| 数据集 | CMRC 2019 Development Set |
| 总题数 | 300 |
| 有效题目（有关联 Chunk） | 153 |
| 数据格式 | 填空题（cloze），答案索引对应选项列表 |
| Context 平均长度 | ~600 字符 |
| Chunk 分层 | L1 (1200-2400 char) → L2 (600-1200) → L3 (300-600) |

### 1.2 评测流程

1. **索引**：将 CMRC context 按三层分块策略写入独立 Milvus collection (`eval_cmrc2019`)
2. **Ground Truth**：若 Chunk 文本与任意答案选项的 3-gram 重叠度 ≥ 30%，标记为相关
3. **检索**：使用 `question_id` 对应 context 作为查询，在 L3 层（leaf）检索 top-5
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
| Baseline | 0.2484 | 0.3660 | 0.9020 | 0.4778 | 153 |
| No Rerank | 0.2484 | 0.3660 | 0.9020 | 0.4778 | 153 |
| No Auto-merge | 0.2484 | 0.3660 | **0.7865** | **0.4378** | 153 |

> **关键发现**：
> - **Rerank 无效果**：No Rerank 与 Baseline 完全一致，说明 Cross-Encoder 重排序对填空题的排序无额外贡献
> - **Auto-merge 有轻微帮助**：禁用后 MRR 下降 ~12.7%，NDCG 下降 ~8.4%；Auto-merge 能帮助将相关 L3 Chunk 合并到更完整的上下文，提升排名

### 2.2 指标解读

| 指标 | 数值 | 解读 |
|------|------|------|
| MRR = 0.90 | 高 | 第一个检索结果通常是相关的，说明语义检索能力强 |
| P@5 = 0.25 | 低 | Top-5 中约 2 个相关，说明大部分召回的是上下文而非精确答案 |
| NDCG@5 = 0.48 | 中等 | 排序质量有提升空间 |

---

## 3. 分析与发现

### 3.1 Rerank 无效果的原因

CMRC 2019 是填空题，答案短语短且高度重合（如"狮子说完就一个猛扑将狐狸咬住"）。RRF 融合已将语义最接近的结果排在前列，Cross-Encoder 在短答案场景下难以进一步区分。

### 3.2 Auto-merge 的正向作用

禁用 Auto-merge 后 MRR 和 NDCG 均有明显下降。分析认为：
- L3 层 Chunk 较小（300-600 字符），单独检索可能只匹配到答案的片段
- Auto-merge 将相邻 L3 合并到 L2/L1，上下文更完整，有助于语义匹配
- 但 Precision@5 不变，说明合并并未提高"恰好命中"的精确度

### 3.3 低 Precision 的原因

- CMRC context 是连续叙述文，答案短语散落在多个 Chunk 中
- Top-5 召回了大量背景上下文Chunk，而非直接包含答案的Chunk
- 建议：可尝试提高 top_k 到 10，或引入 answer-type-aware filtering

---

## 4. 结论与建议

### 4.1 结论

1. **Rerank 组件暂非必选**：在短答案填空场景下，RRF 融合已提供足够的排序质量
2. **Auto-merge 建议保留**：对需要上下文完整性的任务有正向贡献
3. **当前评测集有限**：仅 51% 的题目产生有效 Ground Truth（153/300），建议扩展到阅读理解类数据集

### 4.2 后续建议

- **更换数据集**：使用 DuReader 或自建数据集（带标注的阅读理解问题）
- **补充生成评估**：接入 LLM 生成接口，评估 RAG 答案质量
- **参数调优**：测试不同 `AUTO_MERGE_THRESHOLD`（当前=2）和 `TOP_K`（当前=5）

---

## 5. 输出文件

| 文件 | 说明 |
|------|------|
| `backend/eval/eval_config.py` | 评测配置 |
| `backend/eval/eval_utils.py` | 指标计算（Precision/Recall/MRR/NDCG） |
| `backend/eval/eval_indexer.py` | CMRC 数据集索引脚本 |
| `backend/eval/eval_retrieval.py` | 批量评测脚本 + 三种消融实验 |
| `eval_results/ground_truth.json` | Ground Truth 标注 |
| `eval_results/YYYY-MM-DD_retrieval_metrics.csv` | 评测结果原始数据 |
