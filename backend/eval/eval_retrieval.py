"""RAG 检索评测主脚本"""
import json
import sys
import csv
from pathlib import Path
from datetime import datetime

# backend/ 是 eval/ 的父目录，需要将其加入 path 以便导入 embedding 等模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_config import (
    DATASET_PATH, EVAL_COLLECTION, PARENT_CHUNK_STORE_PATH,
    TOP_K, RRF_TOP_K, OVERLAP_THRESHOLD, RESULTS_DIR
)
from eval.eval_utils import evaluate_single_query, aggregate_metrics
from rag_utils import retrieve_documents, _rerank_documents, _auto_merge_documents
from milvus_client import MilvusManager
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
    """完整流程：RRF(20) → rerank → auto_merge"""
    result = retrieve_documents(query, top_k=top_k, milvus_manager=mm, candidate_k=RRF_TOP_K)
    return result.get("docs", [])


def retrieve_no_rerank(query: str, top_k: int, mm: MilvusManager) -> list[dict]:
    """跳过 rerank：RRF(20) → RRF排序 → auto_merge"""
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    from embedding import EmbeddingService
    es = EmbeddingService()
    dense_embeddings = es.get_embeddings([query])
    dense_embedding = dense_embeddings[0]
    sparse_embedding = es.get_sparse_embedding(query)

    retrieved = mm.hybrid_retrieve(
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        top_k=RRF_TOP_K,
        filter_expr=filter_expr,
    )
    docs_with_rank = [{**doc, "rrf_rank": i} for i, doc in enumerate(retrieved, 1)]
    rerank_meta = {
        "rerank_enabled": False,
        "rerank_applied": False,
    }
    merged_docs, merge_meta = _auto_merge_documents(docs=docs_with_rank, top_k=top_k)
    rerank_meta.update(merge_meta)
    return merged_docs


def retrieve_no_auto_merge(query: str, top_k: int, mm: MilvusManager) -> list[dict]:
    """跳过 auto_merge：RRF(20) → rerank → 直接返回"""
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"

    from embedding import EmbeddingService
    es = EmbeddingService()
    dense_embeddings = es.get_embeddings([query])
    dense_embedding = dense_embeddings[0]
    sparse_embedding = es.get_sparse_embedding(query)

    retrieved = mm.hybrid_retrieve(
        dense_embedding=dense_embedding,
        sparse_embedding=sparse_embedding,
        top_k=RRF_TOP_K,
        filter_expr=filter_expr,
    )
    reranked, _ = _rerank_documents(query=query, docs=retrieved, top_k=top_k)
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
        qid = record.get("question_id", record.get("id", record.get("context_id", "")))
        # CMRC 2019 用 context 作为查询（填空题），其他数据集用 question
        question = record.get("question", record.get("context", ""))
        gt_chunk_ids = set(gt_map.get(qid, []))

        if not gt_chunk_ids:
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
