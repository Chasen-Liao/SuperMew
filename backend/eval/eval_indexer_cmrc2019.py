"""CMRC 2019 填空题 - 批量索引脚本"""
import json
import sys
from pathlib import Path
from pymilvus import MilvusClient, DataType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from embedding import EmbeddingService
from parent_chunk_store import ParentChunkStore
from config import MILVUS_HOST, MILVUS_PORT

# ============================================================
# 硬编码配置 - CMRC 2019 填空题
# ============================================================
DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "cmrc2019_dev_100.json"
EVAL_COLLECTION = "eval_cmrc2019"
PARENT_CHUNK_STORE_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval_parent_chunks_2019.json"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 128
OVERLAP_THRESHOLD = 0.3
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
# ============================================================


def load_data(path: Path) -> list[dict]:
    """加载 CMRC 2019 JSON（填空题格式：context + answers 是整数索引）"""
    if not path.exists():
        raise FileNotFoundError(f"数据未找到: {path}\n请从 https://github.com/ymcui/cmrc2019 下载")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    raise ValueError(f"未知数据格式: {path}")


def build_chunks_from_context(context: str, question_id: str) -> list[dict]:
    """三层分块：L1(2400) → L2(1024) → L3(512)"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter_l1 = RecursiveCharacterTextSplitter(
        chunk_size=max(1200, CHUNK_SIZE * 2),
        chunk_overlap=max(240, CHUNK_OVERLAP * 2),
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
    )
    splitter_l2 = RecursiveCharacterTextSplitter(
        chunk_size=max(600, CHUNK_SIZE),
        chunk_overlap=max(120, CHUNK_OVERLAP),
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
    )
    splitter_l3 = RecursiveCharacterTextSplitter(
        chunk_size=max(300, CHUNK_SIZE // 2),
        chunk_overlap=max(60, CHUNK_OVERLAP // 2),
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
    )

    def make_id(qid: str, level: int, idx: int) -> str:
        return f"{qid}::l{level}::{idx}"

    chunks = []
    global_idx = 0

    for l1_idx, l1_doc in enumerate(splitter_l1.create_documents([context])):
        l1_text = (l1_doc.page_content or "").strip()
        if not l1_text:
            continue
        l1_id = make_id(question_id, 1, l1_idx)
        chunks.append({
            "text": l1_text, "filename": question_id, "file_type": "CMRC2019",
            "file_path": "", "page_number": 0, "chunk_id": l1_id,
            "parent_chunk_id": "", "root_chunk_id": l1_id,
            "chunk_level": 1, "chunk_idx": global_idx,
        })
        global_idx += 1

        for l2_idx, l2_doc in enumerate(splitter_l2.create_documents([l1_text])):
            l2_text = (l2_doc.page_content or "").strip()
            if not l2_text:
                continue
            l2_id = make_id(question_id, 2, l2_idx)
            chunks.append({
                "text": l2_text, "filename": question_id, "file_type": "CMRC2019",
                "file_path": "", "page_number": 0, "chunk_id": l2_id,
                "parent_chunk_id": l1_id, "root_chunk_id": l1_id,
                "chunk_level": 2, "chunk_idx": global_idx,
            })
            global_idx += 1

            for l3_idx, l3_doc in enumerate(splitter_l3.create_documents([l2_text])):
                l3_text = (l3_doc.page_content or "").strip()
                if not l3_text:
                    continue
                chunks.append({
                    "text": l3_text, "filename": question_id, "file_type": "CMRC2019",
                    "file_path": "", "page_number": 0, "chunk_id": make_id(question_id, 3, l3_idx),
                    "parent_chunk_id": l2_id, "root_chunk_id": l1_id,
                    "chunk_level": 3, "chunk_idx": global_idx,
                })
                global_idx += 1

    return chunks


def compute_overlap(chunk_text: str, answer: str) -> bool:
    """3-gram 重叠度判断"""
    def ngrams(text: str, n: int = 3) -> set:
        text = text.lower()
        return {text[i:i+n] for i in range(max(0, len(text) - n + 1))}
    if not answer:
        return False
    ng = ngrams(chunk_text)
    ag = ngrams(answer)
    return len(ng & ag) / len(ag) >= OVERLAP_THRESHOLD if ag else False


def build_ground_truth(records: list[dict]) -> dict:
    """
    question_id → [relevant_chunk_ids]
    CMRC 2019 格式：answers 是整数列表（choices 的索引）
    """
    from collections import defaultdict
    all_chunks = {}
    for rec in records:
        qid = rec.get("context_id", rec.get("id", ""))
        ctx = rec.get("context", "")
        for c in build_chunks_from_context(ctx, qid):
            if c["chunk_id"] not in all_chunks:
                all_chunks[c["chunk_id"]] = c["text"]

    gt = defaultdict(list)
    for rec in records:
        qid = rec.get("context_id", rec.get("id", ""))
        answers = rec.get("answers", [])
        choices = rec.get("choices", [])
        # 整数索引 → 取 choices 对应文本
        answer_texts = []
        for ans in answers:
            if isinstance(ans, int) and choices and 0 <= ans < len(choices):
                answer_texts.append(choices[ans])
            elif isinstance(ans, str):
                answer_texts.append(ans)
        for cid, ctext in all_chunks.items():
            if not cid.startswith(f"{qid}::"):
                continue
            for ans in answer_texts:
                if compute_overlap(ctext, ans):
                    gt[qid].append(cid)
                    break
    return gt


def run():
    print(f"[CMRC 2019] 加载数据集: {DATASET_PATH}")
    records = load_data(DATASET_PATH)
    print(f"[CMRC 2019] {len(records)} 条记录")

    print("[CMRC 2019] 构建 ground truth...")
    gt_map = build_ground_truth(records)
    gt_path = RESULTS_DIR / "ground_truth_cmrc2019.json"
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_map, f, ensure_ascii=False)
    print(f"[CMRC 2019] ground truth 已保存: {gt_path}")

    print("[CMRC 2019] 分块...")
    all_chunks = []
    for rec in records:
        qid = rec.get("context_id", rec.get("id", ""))
        ctx = rec.get("context", "")
        all_chunks.extend(build_chunks_from_context(ctx, qid))
    print(f"[CMRC 2019] 总 chunks: {len(all_chunks)}")

    # Milvus
    print(f"[CMRC 2019] 重建 collection: {EVAL_COLLECTION}")
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

    # parent chunks
    ParentChunkStore(store_path=PARENT_CHUNK_STORE_PATH).upsert_documents(
        [c for c in all_chunks if c["chunk_level"] in (1, 2)]
    )

    # 写入
    es = EmbeddingService()
    es.fit_corpus([c["text"] for c in all_chunks])
    for i in range(0, len(all_chunks), 50):
        batch = all_chunks[i:i + 50]
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
            })
        client.insert(EVAL_COLLECTION, rows)
        print(f"  批次 {i//50 + 1}/{(len(all_chunks) + 49)//50} 完成")
    print(f"[CMRC 2019] 索引完成: {len(all_chunks)} chunks")


if __name__ == "__main__":
    run()
