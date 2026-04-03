# HyDE 检索评测设计

## 目标

在现有 RAG 检索流程中引入 HyDE（Hypothetical Document Embedding），使用 CMRC 2018/2019 数据集评测 HyDE 对各指标的影响。

## 评测配置

- **HyDE 模型**：`zai-org/GLM-4.5-Air`（硬编码在评测脚本中，后期可换）
- **检索流程**：假设文档 → Hybrid Search → RRF(k=60, top-20) → Cross-Encoder rerank → Auto-merge → Top-5
- **测试集**：CMRC 2018（n=87）、CMRC 2019（n=51），与已有 Baseline 报告共用同一份测试集
- **Ground Truth**：Chunk 与答案 3-gram 重叠度 ≥ 30%

## HyDE 实现

### 假设文档生成

```python
HYDE_MODEL = "zai-org/GLM-4.5-Air"

def generate_hypothetical_document(query: str, model) -> str:
    prompt = (
        "请基于用户问题生成一段'假设性文档'，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    return (model.invoke(prompt).content or "").strip()
```

- 使用 `init_chat_model(model=HYDE_MODEL, ...)` 初始化模型
- 直接替换 `retrieve_documents` 的 query 参数，其余流程不变

### HyDE 检索函数

```python
def retrieve_with_hyde(query: str, mm: MilvusManager, hyde_model) -> tuple[list[dict], dict]:
    hyde_doc = generate_hypothetical_document(query, hyde_model)
    meta = {"hyde_generated": bool(hyde_doc), "hyde_doc": hyde_doc}
    if not hyde_doc:
        return [], meta
    result = retrieve_documents(hyde_doc, top_k=TOP_K, milvus_manager=mm, candidate_k=RRF_TOP_K)
    return result.get("docs", []), meta
```

## 评测脚本修改

### eval_retrieval_cmrc2018.py

新增：
1. `HYDE_MODEL = "zai-org/GLM-4.5-Air"` 常量
2. `retrieve_with_hyde()` 函数
3. `run_hyde_experiment()` 函数，执行完整 HyDE 评测并写入 `cmrc2018_hyde_{timestamp}_metrics.csv`

### eval_retrieval_cmrc2019.py

同上。

## 评测指标

| 指标 | 说明 |
|------|------|
| P@5, R@5, MRR, NDCG@5 | 检索标准指标 |
| HyDE 生成成功率 | 成功生成假设文档数 / 总 query 数 |

生成失败时返回空列表，该条不计入评测指标，分母按实际成功数计算。

## 评测结果文件

| 文件 | 内容 |
|------|------|
| `eval_results/cmrc2018_hyde_{timestamp}_metrics.csv` | CMRC 2018 HyDE 实验结果 |
| `eval_results/cmrc2019_hyde_{timestamp}_metrics.csv` | CMRC 2019 HyDE 实验结果 |

报告直接引用已有 Baseline 数据（report-2026-04-03.md）做横向对比。

## 预期

- 阅读理解（CMRC 2018）：HyDE 生成的假设文档可能弥合 query 与 context 的语义 gap，MRR/NDCG@5 预期提升
- 填空题（CMRC 2019）：query=context 已高度相关，HyDE 效果可能有限
