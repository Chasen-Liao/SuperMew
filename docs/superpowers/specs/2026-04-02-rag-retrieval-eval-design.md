# RAG 检索评测设计

## 1. 目标

评估 SuperMew RAG 系统中各检索组件（主要是 rerank 和 auto-merge）对最终检索质量的影响，通过 Precision@K、Recall@K、MRR、NDCG@K 等指标量化贡献度。

## 2. 评测范围

**专注检索质量，不评估生成。** 所有指标基于检索结果与标注答案的匹配度计算。

## 3. 文件结构

```
backend/
├── eval_indexer.py      # 批量索引脚本：将 CMRC context 分块后写入 Milvus
├── eval_retrieval.py    # 主评测脚本：加载数据、执行检索、输出报告
├── eval_utils.py        # 评测指标计算：Precision/Recall/MRR/NDCG
├── eval_config.py        # 评测配置：数据集路径、top_k、输出目录、评测 collection 名称

eval_results/            # 评测结果输出目录
```

**评测使用独立的 Milvus collection（与生产数据隔离）：**
- `EVAL_COLLECTION = "eval_cmrc2019"`
- 评测前运行 `eval_indexer.py` 批量索引
- 评测完成后可删除 collection 或保留供下次使用

## 4. 数据集

使用 **CMRC 2019** 公开数据集（中文机器阅读理解）：
- 每条数据包含：question（问题）、context（上下文）、answers（参考答案列表）
- context 即"相关文档"，answers 中的内容应在 context 中有对应片段
- 评测时：将 context 按 chunk_size=1024 分块，question 作为查询检索，以 answers 作为相关片段的判断依据

**数据格式处理逻辑：**
- 若 answers 中任意答案的 n-gram 与某 chunk 存在重叠 → 该 chunk 标记为相关
- 重叠度超过阈值（如 30%）才认定相关，避免噪声

## 5. 评测流程

**步骤 0（索引）：评测前一次性执行**
```
eval_indexer.py:
    1. 加载 CMRC 2019 数据集
    2. 对每条数据的 context 按 chunk_size=1024, overlap=128 分块
    3. 调用 milvus_writer 将 chunk 写入 EVAL_COLLECTION
    4. 同时记录 (question_id → relevant_chunk_ids) 映射，用于 ground truth 对比
```

**步骤 1（评测）：批量执行**
```
for each (question, context, answers) in dataset:
    1. 用 question 作为查询，在 EVAL_COLLECTION 中检索
    2. 执行 baseline 检索 → top_k 结果
    3. 执行 no_rerank 检索 → top_k 结果（跳过 Cross-Encoder）
    4. 执行 no_auto_merge 检索 → top_k 结果（禁用自动合并）
    5. 计算三个实验的指标
    6. 累加到全局统计
```

## 6. 消融实验

| 实验 | 实现方式 |
|------|----------|
| Baseline | 完整检索流程：hybrid → RRF → rerank → auto_merge |
| No Rerank | 跳过 Cross-Encoder，用 RRF 分排序结果 |
| No Auto-merge | 禁用自动合并，直接输出 rerank 结果 |

## 7. 评测指标

```python
def evaluate_retrieval(pred_chunks: List[str], gt_chunk_ids: Set[str], k: int):
    """
    - Precision@K: pred 中在 gt 里的比例
    - Recall@K: gt 中被召回的比例
    - MRR: 第一个相关文档排名的倒数均值
    - NDCG@K: 标准化折扣累积增益
    """
```

K 默认取 5。

## 8. 输出

- `eval_results/YYYY-MM-DD_retrieval_metrics.csv`：各指标汇总表
- 控制台打印：各实验的指标对比摘要

## 9. 配置项（eval_config.py）

```python
DATASET_PATH = "data/cmrc2019_dev.json"  # CMRC 2019 开发集
EVAL_COLLECTION = "eval_cmrc2019"        # 评测用 Milvus collection（与生产隔离）
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 5
OVERLAP_THRESHOLD = 0.3                  # 答案与 chunk 重叠度阈值
RESULTS_DIR = "eval_results"
```

## 10. 依赖

- 复用现有 `document_loader.py`（分块）、`milvus_writer.py`（写入）、`rag_utils.py`（检索）
- 需要实例化 Milvus 和 Embedding 服务（读取现有 .env 配置）
- 新增 `eval_indexer.py`：调用 `milvus_writer` 批量写入，不修改原写入逻辑
