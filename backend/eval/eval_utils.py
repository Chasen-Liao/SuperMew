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
    relevances = [1 if cid in gt_chunk_ids else 0 for cid in pred_chunk_ids[:k]]
    actual_dcg = _dcg_at_k(relevances, k)
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
