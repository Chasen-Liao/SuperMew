from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from web_search import WebSearchService, normalize_tavily_response, build_web_chunks
from web_search_vector_store import WebSearchVectorStore


def test_normalize_tavily_response_keeps_core_fields():
    response = {
        "query": "LangChain Tavily",
        "results": [
            {
                "title": "LangChain - Tavily Docs",
                "url": "https://docs.tavily.com/documentation/integrations/langchain",
                "content": "Tavily provides a LangChain integration.",
                "raw_content": "Full page text about Tavily and LangChain.",
                "score": 0.91,
            }
        ],
    }

    docs = normalize_tavily_response(response)

    assert len(docs) == 1
    assert docs[0]["title"] == "LangChain - Tavily Docs"
    assert docs[0]["url"] == "https://docs.tavily.com/documentation/integrations/langchain"
    assert docs[0]["content"] == "Tavily provides a LangChain integration."
    assert docs[0]["raw_content"] == "Full page text about Tavily and LangChain."
    assert docs[0]["score"] == 0.91
    assert docs[0]["source_rank"] == 1


def test_normalize_tavily_response_accepts_result_list():
    response = [
        {
            "title": "Example",
            "url": "https://example.com",
            "content": "short snippet",
            "score": 0.5,
        }
    ]

    docs = normalize_tavily_response(response)

    assert len(docs) == 1
    assert docs[0]["title"] == "Example"
    assert docs[0]["raw_content"] == ""
    assert docs[0]["source_rank"] == 1


def test_build_web_chunks_prefers_raw_content_and_keeps_source_metadata():
    docs = [
        {
            "title": "Source A",
            "url": "https://example.com/a",
            "content": "snippet text",
            "raw_content": "alpha " * 500,
            "score": 0.8,
            "source_rank": 1,
        }
    ]

    chunks = build_web_chunks(
        docs,
        chunk_size=200,
        chunk_overlap=20,
        max_chunks_per_result=2,
    )

    assert len(chunks) == 2
    assert chunks[0]["title"] == "Source A"
    assert chunks[0]["url"] == "https://example.com/a"
    assert chunks[0]["source_rank"] == 1
    assert chunks[0]["chunk_idx"] == 0
    assert chunks[0]["text"].startswith("alpha")
    assert chunks[1]["chunk_idx"] == 1


class FakeMilvusClient:
    def __init__(self):
        self.collections = set()
        self.inserted = []
        self.deleted_filters = []

    def has_collection(self, collection_name):
        return collection_name in self.collections

    def create_schema(self, auto_id=True, enable_dynamic_field=True):
        return FakeSchema()

    def prepare_index_params(self):
        return FakeIndexParams()

    def create_collection(self, collection_name, schema, index_params):
        self.collections.add(collection_name)

    def insert(self, collection_name, data):
        self.inserted.extend(data)
        return {"insert_count": len(data)}

    def delete(self, collection_name, filter):
        self.deleted_filters.append(filter)
        return {"delete_count": 1}


class FakeSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, *args, **kwargs):
        self.fields.append((args, kwargs))


class FakeIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, **kwargs):
        self.indexes.append(kwargs)


class FakeEmbeddingService:
    def fit_corpus(self, texts):
        self.fitted_texts = texts

    def get_all_embeddings(self, texts):
        dense = [[0.1, 0.2, 0.3] for _ in texts]
        sparse = [{1: 0.4} for _ in texts]
        return dense, sparse


def test_web_search_vector_store_inserts_embeddings_and_metadata():
    client = FakeMilvusClient()
    embedder = FakeEmbeddingService()
    store = WebSearchVectorStore(
        collection_name="test_web_search",
        client=client,
        embedding_service=embedder,
    )
    chunks = [
        {
            "search_id": "search-1",
            "title": "Title",
            "url": "https://example.com",
            "snippet": "snippet",
            "text": "chunk text",
            "source_score": 0.9,
            "source_rank": 1,
            "chunk_idx": 0,
        }
    ]

    result = store.write_chunks(chunks)

    assert result == {"insert_count": 1}
    assert embedder.fitted_texts == ["chunk text"]
    assert client.inserted[0]["dense_embedding"] == [0.1, 0.2, 0.3]
    assert client.inserted[0]["sparse_embedding"] == {1: 0.4}
    assert client.inserted[0]["url"] == "https://example.com"
    assert client.inserted[0]["search_id"] == "search-1"


class FakeTavilySearch:
    def invoke(self, payload):
        assert payload == {"query": "latest LangChain Tavily integration"}
        return {
            "results": [
                {
                    "title": "Tavily LangChain",
                    "url": "https://docs.tavily.com/documentation/integrations/langchain",
                    "content": "LangChain integration snippet",
                    "raw_content": "LangChain Tavily integration raw content " * 80,
                    "score": 0.88,
                }
            ]
        }


class FakeWebSearchStore:
    def __init__(self):
        self.written_chunks = []
        self.deleted_search_ids = []

    def write_chunks(self, chunks):
        self.written_chunks = chunks
        return {"insert_count": len(chunks)}

    def hybrid_retrieve(self, query, search_id, top_k):
        assert query == "latest LangChain Tavily integration"
        assert search_id == self.written_chunks[0]["search_id"]
        assert top_k == 2
        return [
            {
                "title": "Tavily LangChain",
                "url": "https://docs.tavily.com/documentation/integrations/langchain",
                "snippet": "LangChain integration snippet",
                "text": "LangChain Tavily integration raw content",
                "source_rank": 1,
                "chunk_idx": 0,
                "score": 0.77,
                "rrf_rank": 1,
            }
        ]

    def delete_search(self, search_id):
        self.deleted_search_ids.append(search_id)
        return {"delete_count": 1}


def test_web_search_service_searches_indexes_retrieves_and_cleans_up():
    store = FakeWebSearchStore()
    service = WebSearchService(
        tavily_tool=FakeTavilySearch(),
        vector_store=store,
        max_results=3,
        top_k=2,
        chunk_size=300,
        chunk_overlap=20,
        max_chunks_per_result=2,
    )

    result = service.search_and_retrieve("latest LangChain Tavily integration")

    assert result["query"] == "latest LangChain Tavily integration"
    assert result["source_count"] == 1
    assert result["chunk_count"] >= 1
    assert result["retrieved_chunks"][0]["url"] == "https://docs.tavily.com/documentation/integrations/langchain"
    assert store.deleted_search_ids == [store.written_chunks[0]["search_id"]]
