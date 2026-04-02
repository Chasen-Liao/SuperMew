"""CMRC 2019 填空题 - 检索评测脚本"""
import json
import sys
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import evaluate_single_query, aggregate_metrics
from milvus_client import MilvusManager
from config import LEAF_RETRIEVE_LEVEL
from embedding import EmbeddingService
from rag_utils import retrieve_documents, _rerank_documents, _auto_merge_documents

# ============================================================
# 硬编码配置 - CMRC 2019 填空题
# ============================================================
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cmrc2019_dev_100.json"
EVAL_COLLECTION = "eval_cmrc2019"
TOP_K = 5
RRF_TOP_K = 20      # RRF 融合阶段取 20 条候选，rerank 后输出 top-5
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
# ============================================================


def load_data(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError(f"未知数据格式: {path}")


def retrieve_baseline(query: str, mm: MilvusManager) -> list[dict]:
    """完整流程：RRF(20) → Cross-Encoder rerank → auto_merge → 输出 top-5"""
    result = retrieve_documents(query, top_k=TOP_K, milvus_manager=mm, candidate_k=RRF_TOP_K)
    return result.get("docs", [])


def retrieve_no_rerank(query: str, mm: MilvusManager) -> list[dict]:
    """跳过 rerank：RRF(20) → RRF 排序 → auto_merge → top-5"""
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    es = EmbeddingService()
    dense = es.get_embeddings([query])[0]
    sparse = es.get_sparse_embedding(query)
    retrieved = mm.hybrid_retrieve(dense_embedding=dense, sparse_embedding=sparse,
                                   top_k=RRF_TOP_K, filter_expr=filter_expr)
    docs = [{**d, "rrf_rank": i} for i, d in enumerate(retrieved, 1)]
    merged, _ = _auto_merge_documents(docs=docs, top_k=TOP_K)
    return merged


def retrieve_no_auto_merge(query: str, mm: MilvusManager) -> list[dict]:
    """跳过 auto_merge：RRF(20) → rerank → 直接输出 top-5"""
    filter_expr = f"chunk_level == {LEAF_RETRIEVE_LEVEL}"
    es = EmbeddingService()
    dense = es.get_embeddings([query])[0]
    sparse = es.get_sparse_embedding(query)
    retrieved = mm.hybrid_retrieve(dense_embedding=dense, sparse_embedding=sparse,
                                   top_k=RRF_TOP_K, filter_expr=filter_expr)
    reranked, _ = _rerank_documents(query=query, docs=retrieved, top_k=TOP_K)
    return reranked[:TOP_K]


def run():
    print(f"[CMRC 2019] 加载数据集: {DATASET_PATH}")
    records = load_data(DATASET_PATH)

    gt_path = RESULTS_DIR / "ground_truth_cmrc2019.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground truth 未找到，请先运行 eval_indexer_cmrc2019.py: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map: dict = json.load(f)

    mm = MilvusManager(collection_name=EVAL_COLLECTION)
    if not mm.has_collection():
        raise RuntimeError(f"collection 不存在: {EVAL_COLLECTION}，请先运行 eval_indexer_cmrc2019.py")

    experiments = {
        "baseline": retrieve_baseline,
        "no_rerank": retrieve_no_rerank,
        "no_auto_merge": retrieve_no_auto_merge,
    }
    all_results = {k: [] for k in experiments}

    print(f"\n[CMRC 2019] 开始评测，共 {len(records)} 条数据")
    for i, rec in enumerate(records, 1):
        # CMRC 2019 用 context 作为查询（填空题）
        qid = rec.get("context_id", rec.get("id", ""))
        question = rec.get("context", "")
        gt_ids = set(gt_map.get(qid, []))
        if not gt_ids:
            continue
        for name, fn in experiments.items():
            docs = fn(question, mm)
            ids = [d.get("chunk_id", "") for d in docs]
            metrics = evaluate_single_query(ids, gt_ids, TOP_K)
            metrics["question_id"] = qid
            all_results[name].append(metrics)
        if i % 100 == 0:
            print(f"  已处理 {i}/{len(records)} 条")

    # 输出
    print("\n" + "=" * 60)
    print("CMRC 2019 评测结果汇总")
    print("=" * 60)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = RESULTS_DIR / f"cmrc2019_{ts}_metrics.csv"
    fields = ["experiment", "precision", "recall", "mrr", "ndcg", "query_count"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, results in all_results.items():
            if not results:
                continue
            agg = aggregate_metrics(results)
            row = {**{k: round(agg[k], 4) for k in ["precision", "recall", "mrr", "ndcg"]},
                   "experiment": name, "query_count": len(results)}
            writer.writerow(row)
            print(f"  {name}: P@5={row['precision']} R@5={row['recall']} "
                  f"MRR={row['mrr']} NDCG@5={row['ndcg']} (n={row['query_count']})")

    print(f"\nCSV 已保存: {csv_path}")


if __name__ == "__main__":
    run()
