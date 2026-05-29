"""Temporary Milvus vector store for Tavily web search chunks."""
from __future__ import annotations

from pymilvus import AnnSearchRequest, DataType, MilvusClient, RRFRanker

from config import MILVUS_HOST, MILVUS_PORT, WEB_SEARCH_COLLECTION
from embedding import EmbeddingService


class WebSearchVectorStore:
    """Store one web search's chunks in Milvus and retrieve within that search_id."""

    def __init__(
        self,
        collection_name: str = None,
        client=None,
        embedding_service: EmbeddingService = None,
    ):
        self.collection_name = collection_name or WEB_SEARCH_COLLECTION
        self.client = client or MilvusClient(uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}")
        self.embedding_service = embedding_service or EmbeddingService()

    def init_collection(self, dense_dim: int = 2560):
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("dense_embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("sparse_embedding", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field("search_id", DataType.VARCHAR, max_length=64)
        schema.add_field("title", DataType.VARCHAR, max_length=512)
        schema.add_field("url", DataType.VARCHAR, max_length=2048)
        schema.add_field("snippet", DataType.VARCHAR, max_length=2000)
        schema.add_field("text", DataType.VARCHAR, max_length=4000)
        schema.add_field("source_rank", DataType.INT64)
        schema.add_field("chunk_idx", DataType.INT64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_embedding",
            index_type="HNSW",
            metric_type="IP",
            params={"M": 16, "efConstruction": 256},
        )
        index_params.add_index(
            field_name="sparse_embedding",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"drop_ratio_build": 0.2},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def write_chunks(self, chunks: list[dict]):
        if not chunks:
            return {"insert_count": 0}

        self.init_collection()
        texts = [chunk["text"] for chunk in chunks]
        self.embedding_service.fit_corpus(texts)
        dense_embeddings, sparse_embeddings = self.embedding_service.get_all_embeddings(texts)

        insert_data = []
        for chunk, dense_embedding, sparse_embedding in zip(chunks, dense_embeddings, sparse_embeddings):
            insert_data.append(
                {
                    "dense_embedding": dense_embedding,
                    "sparse_embedding": sparse_embedding,
                    "search_id": chunk["search_id"],
                    "title": chunk.get("title", "")[:511],
                    "url": chunk.get("url", "")[:2047],
                    "snippet": chunk.get("snippet", "")[:1999],
                    "text": chunk.get("text", "")[:3999],
                    "source_rank": int(chunk.get("source_rank") or 0),
                    "chunk_idx": int(chunk.get("chunk_idx") or 0),
                    "source_score": float(chunk.get("source_score") or 0.0),
                }
            )
        return self.client.insert(self.collection_name, insert_data)

    def hybrid_retrieve(self, query: str, search_id: str, top_k: int) -> list[dict]:
        dense_embedding = self.embedding_service.get_embeddings([query])[0]
        sparse_embedding = self.embedding_service.get_sparse_embedding(query)
        filter_expr = f'search_id == "{search_id}"'
        output_fields = [
            "search_id",
            "title",
            "url",
            "snippet",
            "text",
            "source_rank",
            "chunk_idx",
            "source_score",
        ]

        dense_search = AnnSearchRequest(
            data=[dense_embedding],
            anns_field="dense_embedding",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k * 2,
            expr=filter_expr,
        )
        sparse_search = AnnSearchRequest(
            data=[sparse_embedding],
            anns_field="sparse_embedding",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=top_k * 2,
            expr=filter_expr,
        )

        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_search, sparse_search],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=output_fields,
        )

        formatted = []
        for hits in results:
            for rank, hit in enumerate(hits, 1):
                formatted.append(
                    {
                        "title": hit.get("title", ""),
                        "url": hit.get("url", ""),
                        "snippet": hit.get("snippet", ""),
                        "text": hit.get("text", ""),
                        "source_rank": hit.get("source_rank", 0),
                        "chunk_idx": hit.get("chunk_idx", 0),
                        "source_score": hit.get("source_score", 0.0),
                        "score": hit.get("distance", 0.0),
                        "rrf_rank": rank,
                    }
                )
        return formatted

    def delete_search(self, search_id: str):
        if not search_id:
            return {"delete_count": 0}
        return self.client.delete(
            collection_name=self.collection_name,
            filter=f'search_id == "{search_id}"',
        )
