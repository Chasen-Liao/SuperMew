"""CMRC 2018 阅读理解 - 检索评测脚本"""
import json
import logging
import sys
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_utils import evaluate_single_query, aggregate_metrics
from milvus_client import MilvusManager
from config import API_KEY, BASE_URL, LEAF_RETRIEVE_LEVEL
from embedding import EmbeddingService
from rag_utils import retrieve_documents, _rerank_documents, _auto_merge_documents
from langchain.chat_models import init_chat_model

# ============================================================
# 硬编码配置 - CMRC 2018 阅读理解
# ============================================================
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cmrc2018_dev_100.json"
EVAL_COLLECTION = "eval_cmrc2018"
TOP_K = 5
RRF_TOP_K = 20      # RRF 融合阶段取 20 条候选，rerank 后输出 top-5
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# HyDE 实验配置
# ============================================================
HYDE_MODEL = "zai-org/GLM-4.5-Air"
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


def _init_hyde_model():
    """初始化 HyDE 模型，每次调用返回新实例"""
    return init_chat_model(
        model=HYDE_MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.2,
        stream_usage=True,
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
    except Exception as e:
        logging.warning(f"HyDE document generation failed for query '{query[:50]}...': {e}")
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


def run_hyde_experiment(mm: MilvusManager, records: list[dict], gt_map: dict) -> tuple[list[dict], int, int]:
    """
    运行 HyDE 评测实验，返回 (hyde_results, hyde_success_count, hyde_total_count)。
    hyde_success_count：成功生成假设文档的 query 数
    hyde_total_count：参与评测的 query 总数（含生成失败）
    """
    hyde_model = _init_hyde_model()
    hyde_results = []
    hyde_success_count = 0
    hyde_total_count = 0

    for i, rec in enumerate(records, 1):
        qid = rec.get("question_id", rec.get("id", ""))
        question = rec.get("question", "")
        gt_ids = set(gt_map.get(qid, []))
        if not gt_ids:
            continue

        docs_hyde, hyde_meta = retrieve_with_hyde(question, mm, hyde_model)
        hyde_total_count += 1  # 每条 query 都计入分母

        if hyde_meta["hyde_generated"]:
            hyde_success_count += 1
            ids_hyde = [d.get("chunk_id", "") for d in docs_hyde]
            if ids_hyde:  # 有召回结果才计入指标
                metrics = evaluate_single_query(ids_hyde, gt_ids, TOP_K)
                metrics["question_id"] = qid
                hyde_results.append(metrics)

        if i % 50 == 0:
            print(f"  HyDE 已处理 {i}/{len(records)} 条")

    print(f"  HyDE 生成成功率: {hyde_success_count}/{hyde_total_count}")
    return hyde_results, hyde_success_count, hyde_total_count


def run():
    print(f"[CMRC 2018] 加载数据集: {DATASET_PATH}")
    records = load_data(DATASET_PATH)

    gt_path = RESULTS_DIR / "ground_truth_cmrc2018.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"ground truth 未找到，请先运行 eval_indexer_cmrc2018.py: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_map: dict = json.load(f)

    mm = MilvusManager(collection_name=EVAL_COLLECTION)
    if not mm.has_collection():
        raise RuntimeError(f"collection 不存在: {EVAL_COLLECTION}，请先运行 eval_indexer_cmrc2018.py")

    experiments = {
        "baseline": retrieve_baseline,
        "no_rerank": retrieve_no_rerank,
        "no_auto_merge": retrieve_no_auto_merge,
    }
    all_results = {k: [] for k in experiments}

    print(f"\n[CMRC 2018] 开始评测，共 {len(records)} 条数据")
    for i, rec in enumerate(records, 1):
        qid = rec.get("question_id", rec.get("id", ""))
        question = rec.get("question", "")
        gt_ids = set(gt_map.get(qid, []))
        if not gt_ids:
            continue
        for name, fn in experiments.items():
            docs = fn(question, mm)
            ids = [d.get("chunk_id", "") for d in docs]
            metrics = evaluate_single_query(ids, gt_ids, TOP_K)
            metrics["question_id"] = qid
            all_results[name].append(metrics)
        if i % 50 == 0:
            print(f"  已处理 {i}/{len(records)} 条")

    # 输出
    print("\n" + "=" * 60)
    print("CMRC 2018 评测结果汇总")
    print("=" * 60)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = RESULTS_DIR / f"cmrc2018_{ts}_metrics.csv"
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

    # 单独运行 HyDE 实验
    print("\n[CMRC 2018] 开始 HyDE 实验...")
    hyde_results, hyde_success, hyde_total = run_hyde_experiment(mm, records, gt_map)

    # 写入 HyDE 独立 CSV（复用同一个 ts）
    hyde_csv_path = RESULTS_DIR / f"cmrc2018_hyde_{ts}_metrics.csv"
    fields = ["experiment", "precision", "recall", "mrr", "ndcg", "query_count", "hyde_success_rate"]
    with open(hyde_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        if hyde_results:
            agg = aggregate_metrics(hyde_results)
            row = {**{k: round(agg[k], 4) for k in ["precision", "recall", "mrr", "ndcg"]},
                   "experiment": "hyde", "query_count": len(hyde_results),
                   "hyde_success_rate": f"{hyde_success}/{hyde_total}"}
            writer.writerow(row)
    print(f"  HyDE CSV 已保存: {hyde_csv_path}")


if __name__ == "__main__":
    run()
