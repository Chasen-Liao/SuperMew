# HyDE 检索评测实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 CMRC 2018/2019 评测脚本中新增 HyDE 实验配置，用 `zai-org/GLM-4.5-Air` 生成假设文档进行检索，对比 HyDE vs Baseline 各指标变化。

**Architecture:** 在现有 `eval_retrieval_cmrc2018.py` / `eval_retrieval_cmrc2019.py` 中新增 `retrieve_with_hyde()` 函数，使用 `init_chat_model` 初始化 HyDE 模型，用生成的假设文档替换原始 query 走完整检索流程。评测结果写入独立 CSV，报告引用已存 Baseline 数据做对比。

**Tech Stack:** LangChain `init_chat_model`, SiliconFlow API, CMRC 2018/2019 数据集

---

## 文件映射

| 操作 | 文件 |
|------|------|
| 修改 | `backend/eval/eval_retrieval_cmrc2018.py` — 新增 HyDE 实验 |
| 修改 | `backend/eval/eval_retrieval_cmrc2019.py` — 新增 HyDE 实验 |
| 新建 | `eval_results/cmrc2018_hyde_{ts}_metrics.csv` — CMRC 2018 HyDE 结果 |
| 新建 | `eval_results/cmrc2019_hyde_{ts}_metrics.csv` — CMRC 2019 HyDE 结果 |
| 新建/更新 | `eval_results/report-YYYY-MM-DD-hyde.md` — HyDE vs Baseline 对比报告 |

---

## Task 1: 修改 eval_retrieval_cmrc2018.py — 新增 HyDE 实验

**文件:** Modify: `backend/eval/eval_retrieval_cmrc2018.py`

在文件顶部 import 区添加：

```python
from langchain.chat_models import init_chat_model
```

在硬编码配置区添加：

```python
# ============================================================
# HyDE 实验配置
# ============================================================
HYDE_MODEL = "zai-org/GLM-4.5-Air"
# ============================================================
```

在 `retrieve_no_auto_merge()` 函数之后新增两个函数：

```python
def _init_hyde_model():
    """初始化 HyDE 模型，每次调用返回新实例"""
    return init_chat_model(
        model=HYDE_MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.2,
    )


def generate_hypothetical_document(query: str, model) -> str:
    """用 HyDE 模型生成假设性文档"""
    prompt = (
        "请基于用户问题生成一段'假设性文档'，内容应像真实资料片段，"
        "用于帮助检索相关信息。文档可以包含合理推测，但需与问题语义相关。"
        "只输出文档正文，不要标题或解释。\n"
        f"用户问题：{query}"
    )
    try:
        return (model.invoke(prompt).content or "").strip()
    except Exception:
        return ""


def retrieve_with_hyde(query: str, mm: MilvusManager, hyde_model) -> tuple[list[dict], dict]:
    """
    HyDE 检索：用假设文档替代原始 query 走完整检索流程。
    返回 (docs, meta)，meta 含 hyde_generated 和 hyde_doc。
    """
    hyde_doc = generate_hypothetical_document(query, hyde_model)
    meta = {"hyde_generated": bool(hyde_doc), "hyde_doc": hyde_doc}
    if not hyde_doc:
        return [], meta
    result = retrieve_documents(hyde_doc, top_k=TOP_K, milvus_manager=mm, candidate_k=RRF_TOP_K)
    return result.get("docs", []), meta
```

在 `run()` 函数中，新增 hyde 实验分支。在 `for i, rec in enumerate(records, 1):` 循环内，对每条 record，先用 `_init_hyde_model()` 初始化模型，然后调用 `retrieve_with_hyde()`。评测结果写入 `cmrc2018_hyde_{ts}_metrics.csv`。

修改后的主循环逻辑：

```python
hyde_model = _init_hyde_model()  # 初始化一次，复用
hyde_success_count = 0
hyde_total_count = 0

for i, rec in enumerate(records, 1):
    qid = rec.get("question_id", rec.get("id", ""))
    question = rec.get("question", "")
    gt_ids = set(gt_map.get(qid, []))
    if not gt_ids:
        continue

    # Baseline/No Rerank/No Auto-merge
    for name, fn in experiments.items():
        docs = fn(question, mm)
        ids = [d.get("chunk_id", "") for d in docs]
        metrics = evaluate_single_query(ids, gt_ids, TOP_K)
        metrics["question_id"] = qid
        all_results[name].append(metrics)

    # HyDE 实验
    docs_hyde, hyde_meta = retrieve_with_hyde(question, mm, hyde_model)
    if hyde_meta["hyde_generated"]:
        hyde_total_count += 1
        ids_hyde = [d.get("chunk_id", "") for d in docs_hyde]
        if ids_hyde:  # 有召回结果才计入指标
            metrics_hyde = evaluate_single_query(ids_hyde, gt_ids, TOP_K)
            metrics_hyde["question_id"] = qid
            all_results["hyde"].append(metrics_hyde)
            hyde_success_count += 1
    else:
        hyde_total_count += 1  # 生成失败也计入总数

print(f"  HyDE 生成成功率: {hyde_success_count}/{hyde_total_count}")
```

其中 `all_results` 需要新增 `"hyde"` key，`experiments` 字典不变（只跑三种 baseline 配置），hyde 结果单独收集。

最终 CSV 输出格式不变，`experiment` 列值会包含 `hyde`。

---

## Task 2: 修改 eval_retrieval_cmrc2019.py — 新增 HyDE 实验

**文件:** Modify: `backend/eval/eval_retrieval_cmrc2019.py`

完全相同的修改逻辑，区别在于：
- CMRC 2019 使用 `context` 而非 `question` 作为查询字段
- CSV 文件名：`cmrc2019_hyde_{ts}_metrics.csv`

---

## Task 3: 运行评测脚本

**文件:** `backend/eval/eval_retrieval_cmrc2018.py` 和 `backend/eval/eval_retrieval_cmrc2019.py`

- [ ] 启动 Docker（Milvus + PostgreSQL）：`docker compose up -d`
- [ ] 运行 CMRC 2018 HyDE 评测：`uv run python backend/eval/eval_retrieval_cmrc2018.py`
- [ ] 运行 CMRC 2019 HyDE 评测：`uv run python backend/eval/eval_retrieval_cmrc2019.py`
- [ ] 确认两个 CSV 文件已生成在 `eval_results/` 目录

---

## Task 4: 撰写 HyDE 评测报告

**文件:** Create: `eval_results/report-2026-04-03-hyde.md`

报告结构：

```
# HyDE 检索评测报告

> 评测日期：2026-04-03
> HyDE 模型：zai-org/GLM-4.5-Air
> 测试集：CMRC 2018（n=87）、CMRC 2019（n=51）

## 1. HyDE vs Baseline 对比

### CMRC 2018 阅读理解（n=87）

| 实验 | P@5 | R@5 | MRR | NDCG@5 | HyDE生成成功率 |
|------|------|------|------|--------|---------------|
| Baseline | 0.2023 | 0.3343 | 0.5565 | 0.3143 | - |
| No Rerank | 0.2023 | 0.3343 | 0.5571 | 0.3145 | - |
| No Auto-merge | 0.2023 | 0.3343 | 0.5707 | 0.3189 | - |
| **HyDE** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | XX/X |

### CMRC 2019 填空题（n=51）

| 实验 | P@5 | R@5 | MRR | NDCG@5 | HyDE生成成功率 |
|------|------|------|------|--------|---------------|
| Baseline | 0.2510 | 0.3693 | 0.9314 | 0.4899 | - |
| No Rerank | 0.2510 | 0.3693 | 0.9314 | 0.4899 | - |
| No Auto-merge | 0.2510 | 0.3693 | 0.8039 | 0.4458 | - |
| **HyDE** | X.XXXX | X.XXXX | X.XXXX | X.XXXX | XX/X |

## 2. 分析

- HyDE 对阅读理解（CMRC 2018）的影响：[分析]
- HyDE 对填空题（CMRC 2019）的影响：[分析]
- 生成成功率分析：[分析]

## 3. 结论
```

Baseline 数据从 `eval_results/report-2026-04-03.md` 中复制，HyDE 数据从刚生成的 CSV 中读取并填入。

---

## 自查清单

- [ ] `HYDE_MODEL` 硬编码在脚本顶部，模型可替换
- [ ] `retrieve_with_hyde` 返回 `(docs, meta)`，meta 包含 `hyde_generated`
- [ ] 生成失败时该条不计入指标，但计入 HyDE 生成成功率的分子分母
- [ ] CSV 文件名带 `hyde` 标识，与已有 Baseline CSV 区分
- [ ] 报告引用已有 Baseline 数据，不重复跑实验
