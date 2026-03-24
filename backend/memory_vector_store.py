"""记忆向量存储 — 管理 user_memory collection 的创建、检索和重建"""
from pymilvus import MilvusClient, DataType
from config import MILVUS_HOST, MILVUS_PORT, MEMORY_COLLECTION_NAME, MEMORY_TOP_K


class MemoryVectorStore:
    """记忆向量存储"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        embedding_service=None,
    ):
        self.host = host or MILVUS_HOST
        self.port = port or MILVUS_PORT
        self.collection_name = collection_name or MEMORY_COLLECTION_NAME
        self.embedding_service = embedding_service
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")

    def init_collection(self, dense_dim: int = 2560):
        """初始化 user_memory collection（幂等）"""
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("memory_type", DataType.VARCHAR, max_length=32)
        schema.add_field("source_key", DataType.VARCHAR, max_length=512)
        schema.add_field("text", DataType.VARCHAR, max_length=2000)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("created_at", DataType.VARCHAR, max_length=64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert(self, data: list[dict]):
        """插入记忆向量"""
        return self.client.insert(self.collection_name, data)

    def search(
        self, query_vector: list[float], top_k: int = MEMORY_TOP_K
    ) -> list[dict]:
        """语义检索记忆"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["memory_type", "source_key", "text", "created_at"],
        )
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.get("id"),
                    "memory_type": hit.get("memory_type", ""),
                    "source_key": hit.get("source_key", ""),
                    "text": hit.get("text", ""),
                    "created_at": hit.get("created_at", ""),
                    "score": hit.get("distance", 0.0),
                })
        return formatted

    def delete_all(self):
        """清空所有记忆（用于重建前）"""
        if self.client.has_collection(self.collection_name):
            self.client.delete(self.collection_name, filter="id >= 0")

    def collection_exists(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def drop_collection(self):
        """删除 collection"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
