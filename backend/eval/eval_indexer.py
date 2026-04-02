"""批量索引脚本：将 CMRC context 分块后写入独立的 eval Milvus collection"""
import json
import sys
from pathlib import Path

# backend/ 是 eval/ 的父目录，需要将其加入 path 以便导入 embedding 等模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.eval_config import (
    DATASET_PATH, EVAL_COLLECTION, PARENT_CHUNK_STORE_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, OVERLAP_THRESHOLD, RESULTS_DIR
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

    all_chunks = {}

    # 第一步：分块
    for record in records:
        qid = record.get("question_id", record.get("id", ""))
        context = record.get("context", "")

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
