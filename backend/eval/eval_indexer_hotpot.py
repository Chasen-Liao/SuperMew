"""HotpotQA 多跳问答 - 检索评测 Indexer"""
import json
import sys
import hashlib
from pathlib import Path
from pymilvus import MilvusClient, DataType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding import EmbeddingService
from config import MILVUS_HOST, MILVUS_PORT

# ============================================================
# 硬编码配置 - HotpotQA 多跳问答
# ============================================================
EVAL_COLLECTION = "eval_hotpotqa"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
# ============================================================


def _download_dataset():
    """下载 HotpotQA dev distractor 数据集（从 HuggingFace）"""
    from datasets import load_dataset
    if DATASET_PATH.exists():
        print(f"[HotpotQA] 数据集已存在: {DATASET_PATH}")
        return
    print(f"[HotpotQA] 从 HuggingFace 加载数据集...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    # 保存前 100 条为 JSON（与原格式兼容）
    records = []
    for rec in ds.select(range(100)):
        records.append({
            "id": rec["id"],
            "question": rec["question"],
            "answer": rec["answer"],
            "supporting_facts": rec["supporting_facts"],
            "context": rec["context"],
        })
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"[HotpotQA] 下载完成: {DATASET_PATH}")


DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "hotpot_dev_100.json"


def _make_chunk_id(qid: str, title: str, sent_idx: int) -> str:
    """qid_title_sentIdx 的 MD5 前16位"""
    raw = f"{qid}|{title}|{sent_idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def run():
    _download_dataset()

    print(f"[HotpotQA] 加载数据集: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data[:100]  # 取前 100 条
    print(f"[HotpotQA] 使用前 {len(records)} 条数据")

    docs_to_write = []
    gt_map: dict[str, list[str]] = {}

    for rec in records:
        qid = rec.get("id", "")
        ctx = rec.get("context", {})  # HuggingFace: dict with title/sentences lists
        question = rec.get("question", "")
        answer = rec.get("answer", "")
        supporting_facts = rec.get("supporting_facts", {})

        # 构建 supporting_facts 集合：(title, sent_idx)
        # HuggingFace 格式: {"title": [...], "sent_id": [...]}
        sf_set = set()
        sf_titles = supporting_facts.get("title", [])
        sf_sent_ids = supporting_facts.get("sent_id", [])
        for title, sent_id in zip(sf_titles, sf_sent_ids):
            sf_set.add((title, sent_id))

        # 每个 sentence 作为一个 L3 chunk
        # HuggingFace 格式: context["title"][i] -> title string, context["sentences"][i] -> list of sentences
        titles = ctx.get("title", [])
        sentences_list = ctx.get("sentences", [])

        for idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
            parent_id = f"{qid}:{title}"

            for sent_idx, sentence in enumerate(sentences):
                chunk_text = sentence.strip()
                if not chunk_text:
                    continue
                chunk_id = _make_chunk_id(qid, title, sent_idx)

                docs_to_write.append({
                    "text": chunk_text,
                    "filename": qid,
                    "file_type": "HotpotQA",
                    "file_path": "",
                    "page_number": 0,
                    "chunk_idx": sent_idx,
                    "chunk_id": chunk_id,
                    "parent_chunk_id": parent_id,
                    "root_chunk_id": qid,
                    "chunk_level": 3,  # L3 叶子块
                    "title": title,
                    "question": question,
                    "answer": answer,
                })

        # Ground truth: supporting_facts 中的 sentence 对应的 chunk_id
        gt_chunk_ids = []
        for idx, (title, sentences) in enumerate(zip(titles, sentences_list)):
            for sent_idx in range(len(sentences)):
                if (title, sent_idx) in sf_set:
                    chunk_id = _make_chunk_id(qid, title, sent_idx)
                    gt_chunk_ids.append(chunk_id)
        if gt_chunk_ids:
            gt_map[qid] = gt_chunk_ids

    print(f"[HotpotQA] 总 chunks: {len(docs_to_write)}")
    print(f"[HotpotQA] Ground truth: {len(gt_map)} 条")

    # Milvus collection
    print(f"[HotpotQA] 重建 collection: {EVAL_COLLECTION}")
    client = MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
    if client.has_collection(EVAL_COLLECTION):
        client.drop_collection(EVAL_COLLECTION)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=2560)
    schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("text", DataType.VARCHAR, max_length=4000)
    schema.add_field("filename", DataType.VARCHAR, max_length=255)
    schema.add_field("file_type", DataType.VARCHAR, max_length=50)
    schema.add_field("file_path", DataType.VARCHAR, max_length=1024)
    schema.add_field("page_number", DataType.INT64)
    schema.add_field("chunk_idx", DataType.INT64)
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=512)
    schema.add_field("parent_chunk_id", DataType.VARCHAR, max_length=512)
    schema.add_field("root_chunk_id", DataType.VARCHAR, max_length=512)
    schema.add_field("chunk_level", DataType.INT64)

    idx_params = client.prepare_index_params()
    idx_params.add_index("dense_embedding", index_type="HNSW", metric_type="IP", params={"M": 16, "efConstruction": 256})
    idx_params.add_index("sparse_embedding", index_type="SPARSE_INVERTED_INDEX", metric_type="IP", params={"drop_ratio_build": 0.2})
    client.create_collection(EVAL_COLLECTION, schema=schema, index_params=idx_params)

    # 写入
    es = EmbeddingService()
    es.fit_corpus([c["text"] for c in docs_to_write])
    for i in range(0, len(docs_to_write), 50):
        batch = docs_to_write[i:i + 50]
        texts = [c["text"][:3999] for c in batch]
        dense_embs, sparse_embs = es.get_all_embeddings(texts)
        rows = []
        for c, d, s in zip(batch, dense_embs, sparse_embs):
            rows.append({
                "dense_embedding": d, "sparse_embedding": s,
                "text": c["text"][:3999], "filename": c["filename"],
                "file_type": c["file_type"], "file_path": c.get("file_path", ""),
                "page_number": c.get("page_number", 0), "chunk_idx": c.get("chunk_idx", 0),
                "chunk_id": c["chunk_id"], "parent_chunk_id": c.get("parent_chunk_id", ""),
                "root_chunk_id": c.get("root_chunk_id", ""), "chunk_level": c.get("chunk_level", 0),
                "title": c.get("title", ""), "question": c.get("question", ""),
                "answer": c.get("answer", ""),
            })
        client.insert(EVAL_COLLECTION, rows)
        print(f"  批次 {i//50 + 1}/{(len(docs_to_write) + 49)//50} 完成")

    print(f"[HotpotQA] 索引完成: {len(docs_to_write)} chunks")

    # 保存 ground truth
    gt_path = RESULTS_DIR / "ground_truth_hotpotqa.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_map, f, ensure_ascii=False)
    print(f"[HotpotQA] Ground truth 已保存: {gt_path}")


if __name__ == "__main__":
    run()
