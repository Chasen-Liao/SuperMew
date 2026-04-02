# RAG 检索评测实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 RAG 检索质量评测框架，支持 Baseline / No Rerank / No Auto-merge 三种消融实验，输出 Precision@K、Recall@K、MRR、NDCG@K 指标。

**Architecture:** 评测流程分两步：1) eval_indexer.py 批量索引 CMRC context 到独立 Milvus collection；2) eval_retrieval.py 执行批量检索并计算指标。评测脚本内部实例化独立的服务，不依赖全局单例。

**Tech Stack:** 复用现有 backend 模块（embedding.py、milvus_client.py、milvus_writer.py、rag_utils.py）；评测指标手工实现，不引入新依赖。

---

## 文件清单

| 文件 | 职责 |
|------|------|
| `backend/eval_config.py` | 评测配置：数据集路径、collection 名称、top_k、重叠阈值 |
| `backend/eval_indexer.py` | 批量索引脚本：从 CMRC 数据生成 chunks，写入 eval collection，输出 ground_truth.json |
| `backend/eval_utils.py` | 评测指标计算：compute_precision / compute_recall / compute_mrr / compute_ndcg |
| `backend/eval_retrieval.py` | 主评测脚本：加载 ground_truth，执行三种检索，计算指标，输出 CSV |

辅助：
- `data/cmrc2019_dev.json` — 需要手动下载放置
- `eval_results/` — 评测输出目录（自动创建）
- `data/eval_parent_chunks.json` — eval 专用的 parent chunk store（隔离生产数据）

---

## Task 1: 创建 eval_config.py

**Files:**
- Create: `backend/eval_config.py`

- [ ] **Step 1: 创建 eval_config.py**

```python
"""评测配置"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "data" / "cmrc2019_dev.json"
EVAL_COLLECTION = "eval_cmrc2019"
PARENT_CHUNK_STORE_PATH = BASE_DIR / "data" / "eval_parent_chunks.json"

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
TOP_K = 5
OVERLAP_THRESHOLD = 0.3
RESULTS_DIR = BASE_DIR / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True)
```

- [ ] **Step 2: Commit**

```bash
git add backend/eval_config.py && git commit -m "feat(eval): add eval configuration"
```

---

## Task 2: 创建 eval_utils.py

**Files:**
- Create: `backend/eval_utils.py`

- [ ] **Step 1: 创建 eval_utils.py（指标计算）**

```python
"""检索评测指标计算"""
from typing import List, Set
import numpy as np


def compute_precision(pred_chunk_ids: List[str], gt_chunk_ids: Set[str], k: int) -> float:
    """Precision@K: 预测结果中相关文档的比例"""
    top_k = pred_chunk_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for cid in top_k if cid in gt_chunk_ids) / k


def compute_recall(pred_chunk_ids: List[str], gt_chunk_ids: Set[str], k: int) -> float:
    """Recall@K: 召回的相关文档占全部相关文档的比例"""
    top_k = pred_chunk_ids[:k]
    if not gt_chunk_ids:
        return 0.0
    return sum(1 for cid in top_k if cid in gt_chunk_ids) / len(gt_chunk_ids)


def compute_mrr(pred_chunk_ids: List[str], gt_chunk_ids: Set[str], k: int) -> float:
    """MRR: 第一个相关文档排名的倒数均值"""
    for i, cid in enumerate(pred_chunk_ids[:k], 1):
        if cid in gt_chunk_ids:
            return 1.0 / i
    return 0.0


def _dcg_at_k(relevances: List[int], k: int) -> float:
    """DCG@K: 折扣累积增益"""
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], 1):
        dcg += rel / np.log2(i + 1)
    return dcg


def compute_ndcg(pred_chunk_ids: List[str], gt_chunk_ids: Set[str], k: int) -> float:
    """NDCG@K: 标准化折扣累积增益"""
    # 所有相关文档 relevance=1，不相关=0
    relevances = [1 if cid in gt_chunk_ids else 0 for cid in pred_chunk_ids[:k]]
    actual_dcg = _dcg_at_k(relevances, k)

    # 理想情况：所有相关文档排在前面
    ideal_relevances = [1] * min(len(gt_chunk_ids), k)
    ideal_dcg = _dcg_at_k(ideal_relevances, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def evaluate_single_query(
    pred_chunk_ids: List[str],
    gt_chunk_ids: Set[str],
    k: int = 5
) -> dict:
    """对单条查询计算全部指标"""
    return {
        "precision": compute_precision(pred_chunk_ids, gt_chunk_ids, k),
        "recall": compute_recall(pred_chunk_ids, gt_chunk_ids, k),
        "mrr": compute_mrr(pred_chunk_ids, gt_chunk_ids, k),
        "ndcg": compute_ndcg(pred_chunk_ids, gt_chunk_ids, k),
    }


def aggregate_metrics(results: List[dict]) -> dict:
    """对多条查询结果计算均值"""
    if not results:
        return {}
    keys = ["precision", "recall", "mrr", "ndcg"]
    return {k: np.mean([r[k] for r in results]) for k in keys}
```

- [ ] **Step 2: Commit**

```bash
git add backend/eval_utils.py && git commit -m "feat(eval): add retrieval metrics computation"
```

---

## Task 3: 创建 eval_indexer.py

**Files:**
- Create: `backend/eval_indexer.py`
- Reference: `backend/document_loader.py`, `backend/milvus_writer.py`, `backend/parent_chunk_store.py`, `backend/embedding.py`, `backend/milvus_client.py`

**注意：** CMRC 数据是 (question_id, context, answers) 结构，context 是一个长字符串。我将其当作只有一个 "page" 的文档来处理。

- [ ] **Step 1: 创建 eval_indexer.py**

```python
"""批量索引脚本：将 CMRC context 分块后写入独立的 eval Milvus collection"""
import json
import sys
from pathlib import Path

# 确保 backend 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_config import (
    DATASET_PATH, EVAL_COLLECTION, PARENT_CHUNK_STORE_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, RESULTS_DIR
)
from embedding import EmbeddingService
from milvus_client import MilvusManager
from milvus_writer import MilvusWriter
from parent_chunk_store import ParentChunkStore


def load_cmrc_data(path: Path) -> list[dict]:
    """加载 CMRC 2019 JSON 数据"""
    if not path.exists():
        raise FileNotFoundError(f"CMRC 数据未找到: {path}\n请从 https://github.com/ymcui/cmrc2019 下载 dev.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError(f"未知的 CMRC 数据格式: {path}")


def build_chunks_from_context(
    context: str,
    question_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """
    将 CMRC context 字符串拆分成三层 chunks（L1/L2/L3）。
    模拟 document_loader.py 的三层分块逻辑，但不依赖文件路径。
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    level_1_size = max(1200, chunk_size * 2)
    level_1_overlap = max(240, chunk_overlap * 2)
    level_2_size = max(600, chunk_size)
    level_2_overlap = max(120, chunk_overlap)
    level_3_size = max(300, chunk_size // 2)
    level_3_overlap = max(60, chunk_overlap // 2)

    splitter_l1 = RecursiveCharacterTextSplitter(
        chunk_size=level_1_size, chunk_overlap=level_1_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
    )
    splitter_l2 = RecursiveCharacterTextSplitter(
        chunk_size=level_2_size, chunk_overlap=level_2_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
    )
    splitter_l3 = RecursiveCharacterTextSplitter(
        chunk_size=level_3_size, chunk_overlap=level_3_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
    )

    def build_chunk_id(qid: str, level: int, idx: int) -> str:
        return f"{qid}::l{level}::{idx}"

    chunks = []
    global_idx = 0

    l1_docs = splitter_l1.create_documents([context])
    for l1_idx, l1_doc in enumerate(l1_docs):
        l1_text = (l1_doc.page_content or "").strip()
        if not l1_text:
            continue
        l1_id = build_chunk_id(question_id, 1, l1_idx)

        chunks.append({
            "text": l1_text,
            "filename": question_id,
            "file_type": "CMRC",
            "file_path": "",
            "page_number": 0,
            "chunk_id": l1_id,
            "parent_chunk_id": "",
            "root_chunk_id": l1_id,
            "chunk_level": 1,
            "chunk_idx": global_idx,
        })
        global_idx += 1

        l2_docs = splitter_l2.create_documents([l1_text])
        for l2_idx, l2_doc in enumerate(l2_docs):
            l2_text = (l2_doc.page_content or "").strip()
            if not l2_text:
                continue
            l2_id = build_chunk_id(question_id, 2, l2_idx)

            chunks.append({
                "text": l2_text,
                "filename": question_id,
                "file_type": "CMRC",
                "file_path": "",
                "page_number": 0,
                "chunk_id": l2_id,
                "parent_chunk_id": l1_id,
                "root_chunk_id": l1_id,
                "chunk_level": 2,
                "chunk_idx": global_idx,
            })
            global_idx += 1

            l3_docs = splitter_l3.create_documents([l2_text])
            for l3_idx, l3_doc in enumerate(l3_docs):
                l3_text = (l3_doc.page_content or "").strip()
                if not l3_text:
                    continue
                l3_id = build_chunk_id(question_id, 3, l3_idx)

                chunks.append({
                    "text": l3_text,
                    "filename": question_id,
                    "file_type": "CMRC",
                    "file_path": "",
                    "page_number": 0,
                    "chunk_id": l3_id,
                    "parent_chunk_id": l2_id,
                    "root_chunk_id": l1_id,
                    "chunk_level": 3,
                    "chunk_idx": global_idx,
                })
                global_idx += 1

    return chunks


def compute_answer_overlap(chunk_text: str, answer: str, threshold: float = 0.3) -> bool:
    """
    判断 chunk 与 answer 的 n-gram 重叠度是否超过阈值。
    """
    def get_ngrams(text: str, n: int = 3) -> set:
        text = text.lower()
        if len(text) < n:
            return {text}
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    chunk_ngrams = get_ngrams(chunk_text)
    answer_ngrams = get_ngrams(answer)
    if not answer_ngrams:
        return False
    overlap = len(chunk_ngrams & answer_ngrams)
    return overlap / len(answer_ngrams) >= threshold


def build_ground_truth(records: list[dict], threshold: float = 0.3) -> dict:
    """
    为每条 record 构建 question_id → relevant_chunk_ids 映射。
    chunk 为 relevant 当且仅当其 text 与任意 answer 的重叠度 >= threshold。
    """
    from collections import defaultdict

    # question_id → list of (chunk_id, chunk_text)
    all_chunks = {}

    # 第一步：分块
    for record in records:
        qid = record.get("question_id", record.get("id", ""))
        context = record.get("context", "")
        answers = record.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        chunks = build_chunks_from_context(
            context, qid, CHUNK_SIZE, CHUNK_OVERLAP
        )
        for chunk in chunks:
            if chunk["chunk_id"] not in all_chunks:
                all_chunks[chunk["chunk_id"]] = chunk["text"]

    # 第二步：判断相关
    gt_map = defaultdict(set)
    for record in records:
        qid = record.get("question_id", record.get("id", ""))
        answers = record.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        for chunk_id, chunk_text in all_chunks.items():
            # 判断该 chunk 是否属于这个 question
            if not chunk_id.startswith(f"{qid}::"):
                continue
            for answer in answers:
                if compute_answer_overlap(chunk_text, answer, threshold):
                    gt_map[qid].add(chunk_id)
                    break

    return gt_map


def run_indexer():
    print(f"加载数据集: {DATASET_PATH}")
    records = load_cmrc_data(DATASET_PATH)
    print(f"加载 {len(records)} 条数据")

    # 建立 ground truth
    print("构建 ground truth 标注...")
    gt_map = build_ground_truth(records, threshold=OVERLAP_THRESHOLD)

    gt_output = RESULTS_DIR / "ground_truth.json"
    gt_output.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_output, "w", encoding="utf-8") as f:
        json.dump(gt_map, f, ensure_ascii=False)
    print(f"Ground truth 已保存: {gt_output}")

    # 收集所有 chunks
    all_chunks = []
    for record in records:
        qid = record.get("question_id", record.get("id", ""))
        context = record.get("context", "")
        chunks = build_chunks_from_context(context, qid, CHUNK_SIZE, CHUNK_OVERLAP)
        all_chunks.extend(chunks)

    print(f"总 chunks 数量: {len(all_chunks)}")

    # 初始化 eval 专用的 Milvus 和 parent chunk store
    eval_mm = MilvusManager(collection_name=EVAL_COLLECTION)
    eval_pc_store = ParentChunkStore(store_path=PARENT_CHUNK_STORE_PATH)

    # 重建 collection
    print(f"重建 eval collection: {EVAL_COLLECTION}")
    eval_mm.drop_collection()
    eval_mm.init_collection()

    # 写入 parent chunks（L1 和 L2 需要用于 auto-merge）
    parent_chunks = [c for c in all_chunks if c["chunk_level"] in (1, 2)]
    eval_pc_store.upsert_documents(parent_chunks)
    print(f"Parent chunks 已写入: {len(parent_chunks)}")

    # 写入 Milvus
    print(f"写入 Milvus: {EVAL_COLLECTION}")
    writer = MilvusWriter(
        embedding_service=EmbeddingService(),
        milvus_manager=eval_mm,
    )
    writer.write_documents(all_chunks, batch_size=50)
    print(f"索引完成: {len(all_chunks)} chunks")


if __name__ == "__main__":
    run_indexer()
```

- [ ] **Step 2: 测试索引脚本（验证数据格式）**

```bash
cd /mnt/e/MyProjects/Agent_Projects/SuperMew
# 检查数据是否存在
ls -la data/cmrc2019_dev.json 2>/dev/null || echo "CMRC 数据不存在，需要下载"
```

如果数据不存在，跳过运行，但代码逻辑正确。

- [ ] **Step 3: Commit**

```bash
git add backend/eval_indexer.py && git commit -m "feat(eval): add CMRC indexer for eval collection"
```

---

## Task 4: 创建 eval_retrieval.py

**Files:**
- Create: `backend/eval_retrieval.py`
- Modify: `backend/rag_utils.py:237-293` — 给 `retrieve_documents` 添加 `collection` 参数
- Reference: `backend/rag_utils.py` 的 `_rerank_documents` 和 `_auto_merge_documents` 函数

**消融实验实现方式：**
- `Baseline` → 调用 `retrieve_documents(query, top_k, collection=EVAL_COLLECTION)`
- `No Rerank` → 调用 retrieval 后直接跳过 rerank，用 RRF 排序结果
- `No Auto-merge` → 修改 rag_utils 内部逻辑或新建简化版 retrieval

**注意：** `rag_utils.py` 中的 `retrieve_documents` 使用全局 `_milvus_manager`，需要改造为支持传入自定义 collection。最简方案：在 `retrieve_documents` 函数中加一个 `milvus_manager` 可选参数。

- [ ] **Step 1: 修改 rag_utils.py — 给 retrieve_documents 添加 collection 支持**

在 `backend/rag_utils.py` 找到 `retrieve_documents` 函数（约 line 237），在函数签名添加 `milvus_manager=None` 参数，内部调用时如果传入了就用传入的，否则用全局的。

```python
def retrieve_documents(
    query: str,
    top_k: int = 5,
    milvus_manager=None,  # 新增：允许外部注入 MilvusManager（用于评测）
) -> Dict[str, Any]:
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    mm = milvus_manager or _milvus_manager  # 使用注入的或全局的
    try:
        dense_embeddings = _embedding_service.get_embeddings([query])
        dense_embedding = dense_embeddings[0]
        sparse_embedding = _embedding_service.get_sparse_embedding(query)

        retrieved = mm.hybrid_retrieve(  # <-- 改用 mm
            dense_embedding=dense_embedding,
            sparse_embedding=sparse_embedding,
            top_k=candidate_k,
            filter_expr=filter_expr,
        )
        # ... 其余不变
```

- [ ] **Step 2: 创建 eval_retrieval.py**

```python
"""RAG 检索评测主脚本"""
import json
import sys
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_config import (
    DATASET_PATH, EVAL_COLLECTION, PARENT_CHUNK_STORE_PATH,
    TOP_K, OVERLAP_THRESHOLD, RESULTS_DIR
)
from eval_utils import evaluate_single_query, aggregate_metrics
from rag_utils import retrieve_documents, _rerank_documents, _auto_merge_documents
from milvus_client import MilvusManager
from parent_chunk_store import ParentChunkStore
from config import LEAF_RETRIEVE_LEVEL


def load_cmrc_data(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError(f"未知数据格式: {path}")


def retrieve_baseline(query: str, top_k: int, mm: MilvusManager) -> list[dict]:
    """完整流程：hybrid → rerank → auto_merge"""
    result = retrieve_documents(query, top_k=top_k, milvus_manager=mm)
    return result.get("docs", [])


def retrieve_no_rerank(query: str, top_k: int, mm: MilvusManager) -> list[dict]:
    """跳过 rerank：hybrid → RRF排序 → auto_merge"""
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    from embedding import EmbeddingService
    es = EmbeddingService()
    dense_embeddings = es.get_embeddings([query])
    dense_embedding = dense_embeddings[0]
    sparse_embedding = es.get_sparse_embedding(query)

    retrieved = mm.hybrid_retrieve(
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        top_k=candidate_k,
        filter_expr=filter_expr,
    )
    # 跳过 rerank，直接用 RRF 排序结果（已按 score 排序）
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(retrieved, 1)]
    rerank_meta = {
        "rerank_enabled": False,
        "rerank_applied": False,
        "rerank_model": None,
        "rerank_endpoint": None,
        "rerank_error": None,
        "candidate_count": len(docs_with_rank),
    }
    merged_docs, merge_meta = _auto_merge_documents(docs=docs_with_rank, top_k=top_k)
    rerank_meta.update(merge_meta)
    return merged_docs


def retrieve_no_auto_merge(query: str, top_k: int, mm: MilvusManager) -> list[dict]:
    """跳过 auto_merge：hybrid → rerank → 直接返回"""
    candidate_k = max(top_k * 3, top_k)
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    from embedding import EmbeddingService
    es = EmbeddingService()
    dense_embeddings = es.get_embeddings([query])
    dense_embedding = dense_embeddings[0]
    sparse_embedding = es.get_sparse_embedding(query)

    retrieved = mm.hybrid_retrieve(
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        top_k=candidate_k,
        filter_expr=filter_expr,
    )
    reranked, rerank_meta = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
    # 不调用 auto_merge，直接返回 top_k
    return reranked[:top_k]


def run_evaluation():
    print(f"加载数据集: {DATASET_PATH}")
    records = load_cmrc_data(DATASET_PATH)

    gt_path = RESULTS_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(
            f"Ground truth 未找到: {gt_path}\n请先运行 eval_indexer.py"
        )
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map = json.load(f)

    # 初始化 eval 专用的 MilvusManager
    eval_mm = MilvusManager(collection_name=EVAL_COLLECTION)
    if not eval_mm.has_collection():
        raise RuntimeError(f"Eval collection 不存在: {EVAL_COLLECTION}\n请先运行 eval_indexer.py")

    experiments = {
        "baseline": retrieve_baseline,
        "no_rerank": retrieve_no_rerank,
        "no_auto_merge": retrieve_no_auto_merge,
    }

    all_results = {exp_name: [] for exp_name in experiments}

    print(f"\n开始评测，共 {len(records)} 条数据")
    for i, record in enumerate(records, 1):
        qid = record.get("question_id", record.get("id", ""))
        question = record.get("question", "")
        gt_chunk_ids = set(gt_map.get(qid, []))

        if not gt_chunk_ids:
            # 没有相关 chunk，跳过
            continue

        for exp_name, retrieve_fn in experiments.items():
            pred_docs = retrieve_fn(question, TOP_K, eval_mm)
            pred_ids = [doc.get("chunk_id", "") for doc in pred_docs]

            metrics = evaluate_single_query(pred_ids, gt_chunk_ids, TOP_K)
            metrics["question_id"] = qid
            metrics["retrieved_count"] = len(pred_ids)
            all_results[exp_name].append(metrics)

        if i % 100 == 0:
            print(f"  已处理 {i}/{len(records)} 条")

    # 汇总
    print("\n" + "=" * 60)
    print("评测结果汇总")
    print("=" * 60)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = RESULTS_DIR / f"{timestamp}_retrieval_metrics.csv"

    fieldnames = ["experiment", "precision", "recall", "mrr", "ndcg", "query_count"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp_name, results in all_results.items():
            if not results:
                continue
            agg = aggregate_metrics(results)
            row = {
                "experiment": exp_name,
                "precision": round(agg["precision"], 4),
                "recall": round(agg["recall"], 4),
                "mrr": round(agg["mrr"], 4),
                "ndcg": round(agg["ndcg"], 4),
                "query_count": len(results),
            }
            writer.writerow(row)
            print(f"  {exp_name}: P@5={row['precision']:.4f} R@5={row['recall']:.4f} "
                  f"MRR={row['mrr']:.4f} NDCG@5={row['ndcg']:.4f} (n={row['query_count']})")

    print(f"\nCSV 已保存: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
```

- [ ] **Step 3: 测试 retrieval 脚本（需要 Milvus 运行）**

```bash
cd backend
# 验证语法
python -m py_compile eval_retrieval.py && echo "语法正确"
```

- [ ] **Step 4: Commit**

```bash
git add backend/eval_retrieval.py backend/rag_utils.py
git commit -m "feat(eval): add retrieval evaluation script with ablation experiments"
```

---

## Task 5: 验证完整流程

**Files:**
- Test: 完整端到端运行

- [ ] **Step 1: 下载 CMRC 2019 dev 数据集**

```bash
mkdir -p /mnt/e/MyProjects/Agent_Projects/SuperMew/data
# 使用 huggingface 或 github 下载 cmrc2019 dev.json
# 验证
python -c "import json; json.load(open('data/cmrc2019_dev.json'))" && echo "CMRC 数据有效"
```

- [ ] **Step 2: 运行索引**

```bash
cd backend
uv run python eval_indexer.py
```

预期：输出 chunks 数量，建立 eval collection，写入 parent chunks

- [ ] **Step 3: 运行评测**

```bash
uv run python eval_retrieval.py
```

预期：输出三种实验的 P@5/R@5/MRR/NDCG@5，写入 CSV

- [ ] **Step 4: Commit 最终状态**

---

## 依赖清单

无新增外部依赖。所有模块均复用现有代码：
- `langchain_text_splitters` — 已在 document_loader 中使用
- `pymilvus` — 已在 milvus_client 中使用
- `numpy` — 已在项目依赖中

## 验证要点

1. `eval_indexer.py` 能成功将 CMRC context 索引到独立 collection
2. `eval_retrieval.py` 的三种消融实验都能正确执行
3. ground_truth.json 中的相关 chunk ID 与检索返回的 chunk_id 能对应上
4. CSV 输出包含所有四个指标，且数值合理（0~1 之间）
