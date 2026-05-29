from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from web_search import normalize_tavily_response, build_web_chunks
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
