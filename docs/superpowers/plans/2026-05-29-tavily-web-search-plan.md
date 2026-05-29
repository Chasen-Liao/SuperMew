# Tavily Web Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tavily-powered web search tool that lets the LangChain Agent answer current internet questions through Tavily search plus Milvus vector retrieval with visible source URLs.

**Architecture:** Add a focused web-search pipeline separate from the existing document RAG pipeline. Tavily fetches search results and raw page content, the pipeline chunks and embeds those results into a temporary Milvus collection keyed by `search_id`, then retrieves the most relevant chunks for the Agent. The existing Agent tool registration, RAG step queue, `rag_trace`, and frontend source display are reused instead of creating a new chat flow.

**Tech Stack:** LangChain `create_agent`, `langchain-tavily` `TavilySearch`, Milvus `pymilvus`, existing `EmbeddingService`, FastAPI SSE trace path, Vue 3 CDN frontend.

---

## Scope

This plan implements the smallest working version of 【选题4】:

- LangChain Agent can call a `search_web` tool.
- `search_web` uses Tavily as the web search provider.
- Web results are converted into chunks and embedded into a Milvus collection.
- The final answer can cite web source URLs.
- The frontend can show web-search trace and clickable source URLs.

This plan does not implement a long-term crawler, scheduled web refresh, login-required page extraction, multi-user web cache retention, or advanced source trust scoring.

Official Tavily references checked on 2026-05-29:

- `langchain-tavily` is Tavily's current LangChain integration package.
- `TavilySearch` supports `max_results`, `topic`, `search_depth`, `include_raw_content`, and direct `.invoke({"query": ...})`.
- The old `langchain_community.tools.tavily_search` path is deprecated by Tavily's docs, so this plan uses `langchain_tavily`.

Reference URLs:

- https://docs.tavily.com/documentation/integrations/langchain
- https://docs.tavily.com/documentation/api-reference/endpoint/search

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `pyproject.toml` | Modify | Add `langchain-tavily` dependency. |
| `.env.example` | Modify | Document `TAVILY_API_KEY` and web-search tuning env vars. |
| `backend/config.py` | Modify | Load Tavily and web-search configuration. |
| `backend/web_search.py` | Create | Tavily invocation, result normalization, chunk creation, service orchestration. |
| `backend/web_search_vector_store.py` | Create | Temporary Milvus collection for web-search chunks and hybrid retrieval. |
| `backend/tools.py` | Modify | Add `search_web` LangChain tool and per-turn guard. |
| `backend/agent.py` | Modify | Register `search_web` in the Agent. |
| `backend/schemas.py` | Modify | Let `rag_trace.retrieved_chunks` include web source title, URL, and source rank. |
| `backend/soul/soul.md` | Modify | Tell the Agent when to use web search and how to cite sources. |
| `frontend/index.html` | Modify | Render web source title and URL in RAG trace. |
| `frontend/style.css` | Modify | Add compact URL/source styling. |
| `tests/backend/test_web_search.py` | Create | Unit tests for normalization, chunking, service orchestration, and tool guard behavior. |

---

## Task 1: Add Tavily Configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `backend/config.py`

- [ ] **Step 1: Add dependency**

Modify the dependencies list in `pyproject.toml` by adding this entry near the LangChain dependencies:

```toml
    "langchain-tavily>=0.2.0",
```

Expected dependency block excerpt:

```toml
dependencies = [
    "rich>=14.2.0",
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "python-dotenv>=1.0.1",
    "requests>=2.32.0",
    "pymilvus>=2.5.0",
    "python-multipart>=0.0.9",
    "pydantic>=2.8.0",
    "langchain>=0.3",
    "langchain-core>=0.2.37",
    "langchain-community>=0.2.12",
    "langchain-text-splitters>=0.2.2",
    "langchain-openai>=0.1.22",
    "langchain-tavily>=0.2.0",
    "langgraph>=0.2.31",
    "langgraph-checkpoint-postgres>=0.1.0",
    "pypdf>=4.3.1",
    "docx2txt>=0.8",
    "langchain-postgres>=0.0.17",
    "langsmith>=0.7.20",
]
```

- [ ] **Step 2: Document env vars**

Add this block under `.env.example` section `# ===== Tools (可选) =====`:

```env
# Tavily Web Search
TAVILY_API_KEY=your_tavily_api_key
WEB_SEARCH_COLLECTION=web_search_cache
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TOP_K=5
WEB_SEARCH_CHUNK_SIZE=1200
WEB_SEARCH_CHUNK_OVERLAP=150
WEB_SEARCH_MAX_CHUNKS_PER_RESULT=4
```

- [ ] **Step 3: Load config values**

In `backend/config.py`, replace the current tools block:

```python
# ===== Tools =====
AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")
```

with:

```python
# ===== Tools =====
AMAP_WEATHER_API = os.getenv("AMAP_WEATHER_API")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# ===== Tavily Web Search =====
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
WEB_SEARCH_COLLECTION = os.getenv("WEB_SEARCH_COLLECTION", "web_search_cache")
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5"))
WEB_SEARCH_TOP_K = int(os.getenv("WEB_SEARCH_TOP_K", "5"))
WEB_SEARCH_CHUNK_SIZE = int(os.getenv("WEB_SEARCH_CHUNK_SIZE", "1200"))
WEB_SEARCH_CHUNK_OVERLAP = int(os.getenv("WEB_SEARCH_CHUNK_OVERLAP", "150"))
WEB_SEARCH_MAX_CHUNKS_PER_RESULT = int(os.getenv("WEB_SEARCH_MAX_CHUNKS_PER_RESULT", "4"))
```

- [ ] **Step 4: Sync dependencies**

Run:

```bash
uv sync
```

Expected: command completes and `uv.lock` updates with `langchain-tavily`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .env.example backend/config.py
git commit -m "chore: add Tavily web search configuration"
```

---

## Task 2: Create Web Search Normalization and Chunking

**Files:**
- Create: `backend/web_search.py`
- Create: `tests/backend/test_web_search.py`

- [ ] **Step 1: Write failing tests for result normalization and chunking**

Create `tests/backend/test_web_search.py`:

```python
from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from web_search import normalize_tavily_response, build_web_chunks


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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/backend/test_web_search.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'web_search'`.

- [ ] **Step 3: Implement normalization and chunking**

Create `backend/web_search.py`:

```python
"""Tavily web search pipeline helpers."""
from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_web_text(text: str) -> str:
    """Collapse whitespace and remove empty text from web content."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_tavily_response(response: Any) -> list[dict]:
    """Normalize TavilySearch output into a predictable list of source documents."""
    if isinstance(response, dict):
        results = response.get("results", [])
    elif isinstance(response, list):
        results = response
    else:
        results = []

    normalized = []
    for rank, item in enumerate(results, 1):
        if not isinstance(item, dict):
            continue
        url = (item.get("url") or "").strip()
        title = clean_web_text(item.get("title") or url or "Untitled")
        content = clean_web_text(item.get("content") or "")
        raw_content = clean_web_text(item.get("raw_content") or "")
        if not url or not (content or raw_content):
            continue
        normalized.append(
            {
                "title": title,
                "url": url,
                "content": content,
                "raw_content": raw_content,
                "score": float(item.get("score") or 0.0),
                "source_rank": rank,
            }
        )
    return normalized


def build_web_chunks(
    docs: list[dict],
    chunk_size: int,
    chunk_overlap: int,
    max_chunks_per_result: int,
) -> list[dict]:
    """Split normalized web documents into chunks that can be embedded."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    search_id = str(uuid4())
    chunks = []
    for doc in docs:
        source_text = doc.get("raw_content") or doc.get("content") or ""
        text_parts = splitter.split_text(source_text)
        for chunk_idx, text in enumerate(text_parts[:max_chunks_per_result]):
            cleaned = clean_web_text(text)
            if not cleaned:
                continue
            chunks.append(
                {
                    "search_id": search_id,
                    "title": doc.get("title", ""),
                    "url": doc.get("url", ""),
                    "snippet": doc.get("content", ""),
                    "text": cleaned,
                    "source_score": float(doc.get("score") or 0.0),
                    "source_rank": int(doc.get("source_rank") or 0),
                    "chunk_idx": chunk_idx,
                }
            )
    return chunks
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/backend/test_web_search.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/web_search.py tests/backend/test_web_search.py
git commit -m "feat: normalize Tavily web search results"
```

---

## Task 3: Add Temporary Web Search Vector Store

**Files:**
- Create: `backend/web_search_vector_store.py`
- Modify: `tests/backend/test_web_search.py`

- [ ] **Step 1: Add store payload test**

Append this test to `tests/backend/test_web_search.py`:

```python
from web_search_vector_store import WebSearchVectorStore


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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/backend/test_web_search.py::test_web_search_vector_store_inserts_embeddings_and_metadata -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'web_search_vector_store'`.

- [ ] **Step 3: Implement WebSearchVectorStore**

Create `backend/web_search_vector_store.py`:

```python
"""Temporary Milvus vector store for Tavily web search chunks."""
from __future__ import annotations

from pymilvus import MilvusClient, DataType, AnnSearchRequest, RRFRanker

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
        output_fields = ["search_id", "title", "url", "snippet", "text", "source_rank", "chunk_idx", "source_score"]

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
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/backend/test_web_search.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/web_search_vector_store.py tests/backend/test_web_search.py
git commit -m "feat: add temporary web search vector store"
```

---

## Task 4: Add Tavily Search Service Orchestration

**Files:**
- Modify: `backend/web_search.py`
- Modify: `tests/backend/test_web_search.py`

- [ ] **Step 1: Add service orchestration test**

Append this test to `tests/backend/test_web_search.py`:

```python
from web_search import WebSearchService


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
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
uv run pytest tests/backend/test_web_search.py::test_web_search_service_searches_indexes_retrieves_and_cleans_up -q
```

Expected: FAIL with `ImportError: cannot import name 'WebSearchService'`.

- [ ] **Step 3: Implement WebSearchService**

Append these imports to the top of `backend/web_search.py`:

```python
from langchain_tavily import TavilySearch

from config import (
    TAVILY_API_KEY,
    WEB_SEARCH_CHUNK_OVERLAP,
    WEB_SEARCH_CHUNK_SIZE,
    WEB_SEARCH_MAX_CHUNKS_PER_RESULT,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_TOP_K,
)
from web_search_vector_store import WebSearchVectorStore
```

Append this class to `backend/web_search.py`:

```python
class WebSearchService:
    """Run Tavily search, temporary Milvus indexing, and semantic retrieval."""

    def __init__(
        self,
        tavily_tool=None,
        vector_store: WebSearchVectorStore = None,
        max_results: int = None,
        top_k: int = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_chunks_per_result: int = None,
    ):
        self.max_results = max_results or WEB_SEARCH_MAX_RESULTS
        self.top_k = top_k or WEB_SEARCH_TOP_K
        self.chunk_size = chunk_size or WEB_SEARCH_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else WEB_SEARCH_CHUNK_OVERLAP
        self.max_chunks_per_result = max_chunks_per_result or WEB_SEARCH_MAX_CHUNKS_PER_RESULT
        self.tavily_tool = tavily_tool or TavilySearch(
            max_results=self.max_results,
            topic="general",
            search_depth="basic",
            include_raw_content=True,
            include_answer=False,
        )
        self.vector_store = vector_store or WebSearchVectorStore()

    def search_and_retrieve(self, query: str) -> dict:
        if not TAVILY_API_KEY and self.tavily_tool.__class__.__name__ == "TavilySearch":
            return {
                "query": query,
                "source_count": 0,
                "chunk_count": 0,
                "retrieved_chunks": [],
                "error": "Tavily 未配置：缺少 TAVILY_API_KEY",
            }

        response = self.tavily_tool.invoke({"query": query})
        docs = normalize_tavily_response(response)
        chunks = build_web_chunks(
            docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            max_chunks_per_result=self.max_chunks_per_result,
        )
        if not chunks:
            return {
                "query": query,
                "source_count": len(docs),
                "chunk_count": 0,
                "retrieved_chunks": [],
                "error": "Tavily 搜索未返回可索引正文",
            }

        search_id = chunks[0]["search_id"]
        try:
            self.vector_store.write_chunks(chunks)
            retrieved = self.vector_store.hybrid_retrieve(query=query, search_id=search_id, top_k=self.top_k)
        finally:
            self.vector_store.delete_search(search_id)

        return {
            "query": query,
            "source_count": len(docs),
            "chunk_count": len(chunks),
            "retrieved_chunks": retrieved,
            "error": None,
        }
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```bash
uv run pytest tests/backend/test_web_search.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/web_search.py tests/backend/test_web_search.py
git commit -m "feat: add Tavily web search retrieval service"
```

---

## Task 5: Add `search_web` LangChain Tool

**Files:**
- Modify: `backend/tools.py`
- Modify: `tests/backend/test_web_search.py`

- [ ] **Step 1: Add tool behavior tests**

Append these tests to `tests/backend/test_web_search.py`:

```python
import tools


class FakeWebSearchServiceForTool:
    def search_and_retrieve(self, query):
        return {
            "query": query,
            "source_count": 1,
            "chunk_count": 1,
            "retrieved_chunks": [
                {
                    "title": "Tavily LangChain",
                    "url": "https://docs.tavily.com/documentation/integrations/langchain",
                    "text": "Tavily provides a current LangChain integration.",
                    "score": 0.91,
                    "rrf_rank": 1,
                    "source_rank": 1,
                }
            ],
            "error": None,
        }


def test_search_web_tool_formats_sources_and_sets_trace(monkeypatch):
    tools.reset_tool_call_guards()
    tools.get_last_rag_context(clear=True)
    monkeypatch.setattr(tools, "WebSearchService", lambda: FakeWebSearchServiceForTool())

    result = tools.search_web.invoke({"query": "LangChain Tavily integration"})
    context = tools.get_last_rag_context(clear=True)

    assert "【联网搜索结果】" in result
    assert "Tavily LangChain" in result
    assert "https://docs.tavily.com/documentation/integrations/langchain" in result
    assert context["rag_trace"]["tool_name"] == "search_web"
    assert context["rag_trace"]["retrieved_chunks"][0]["url"] == "https://docs.tavily.com/documentation/integrations/langchain"


def test_search_web_tool_guard_allows_one_call_per_turn(monkeypatch):
    tools.reset_tool_call_guards()
    monkeypatch.setattr(tools, "WebSearchService", lambda: FakeWebSearchServiceForTool())

    first = tools.search_web.invoke({"query": "first query"})
    second = tools.search_web.invoke({"query": "second query"})

    assert "【联网搜索结果】" in first
    assert "TOOL_CALL_LIMIT_REACHED" in second
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
uv run pytest tests/backend/test_web_search.py::test_search_web_tool_formats_sources_and_sets_trace tests/backend/test_web_search.py::test_search_web_tool_guard_allows_one_call_per_turn -q
```

Expected: FAIL with `AttributeError: module 'tools' has no attribute 'search_web'`.

- [ ] **Step 3: Modify imports and guard state**

In `backend/tools.py`, replace:

```python
from config import AMAP_WEATHER_API, AMAP_API_KEY
```

with:

```python
from config import AMAP_WEATHER_API, AMAP_API_KEY
from web_search import WebSearchService
```

Replace:

```python
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
```

with:

```python
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_WEB_SEARCH_TOOL_CALLS_THIS_TURN = 0
```

Replace `reset_tool_call_guards` with:

```python
def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN, _WEB_SEARCH_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
    _WEB_SEARCH_TOOL_CALLS_THIS_TURN = 0
```

- [ ] **Step 4: Add search_web tool**

Add this function before `@tool("search_knowledge_base")` in `backend/tools.py`:

```python
@tool("search_web")
def search_web(query: str) -> str:
    """Search the public web using Tavily, index results into Milvus, and return relevant source chunks.

    Use this tool for current events, latest information, public internet facts,
    or questions that explicitly require web search. Cite source URLs in the final answer.
    """
    global _WEB_SEARCH_TOOL_CALLS_THIS_TURN
    if _WEB_SEARCH_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_web has already been called once in this turn. "
            "Use the existing web search result and provide the final answer directly."
        )
    _WEB_SEARCH_TOOL_CALLS_THIS_TURN += 1

    if not query or not query.strip():
        return "query 参数不能为空"

    try:
        emit_rag_step("🌐", "正在联网搜索...", f"查询: {query[:50]}")
        service = WebSearchService()
        result = service.search_and_retrieve(query.strip())
        chunks = result.get("retrieved_chunks", [])
        error = result.get("error")

        rag_trace = {
            "tool_used": True,
            "tool_name": "search_web",
            "query": result.get("query", query),
            "retrieval_stage": "web_search",
            "retrieval_mode": "tavily_milvus_hybrid",
            "candidate_k": result.get("chunk_count", 0),
            "retrieved_chunks": chunks,
            "initial_retrieved_chunks": chunks,
            "web_source_count": result.get("source_count", 0),
            "web_chunk_count": result.get("chunk_count", 0),
            "web_error": error,
        }
        _set_last_rag_context({"rag_trace": rag_trace})

        if error:
            emit_rag_step("⚠️", "联网搜索失败", error)
            return f"联网搜索失败：{error}"

        emit_rag_step(
            "✅",
            f"联网搜索完成，找到 {len(chunks)} 个相关片段",
            f"来源: {result.get('source_count', 0)} 个网页",
        )

        if not chunks:
            return "No relevant web results found."

        formatted = []
        for i, item in enumerate(chunks, 1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            text = item.get("text", "")
            score = item.get("score", 0)
            formatted.append(f"[{i}] {title}\nURL: {url}\n相关度: {score:.3f}\n{text}")
        return "【联网搜索结果】\n" + "\n\n---\n\n".join(formatted)
    except Exception as e:
        emit_rag_step("⚠️", "联网搜索异常", str(e))
        return f"联网搜索失败: {e}"
```

- [ ] **Step 5: Run tests and verify pass**

Run:

```bash
uv run pytest tests/backend/test_web_search.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/tools.py tests/backend/test_web_search.py
git commit -m "feat: add Tavily search_web tool"
```

---

## Task 6: Register Tool in Agent and Prompt

**Files:**
- Modify: `backend/agent.py`
- Modify: `backend/soul/soul.md`

- [ ] **Step 1: Register tool import**

In `backend/agent.py`, replace the current tools import:

```python
from tools import get_current_weather, search_knowledge_base, search_memory, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
```

with:

```python
from tools import get_current_weather, search_knowledge_base, search_memory, search_web, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
```

- [ ] **Step 2: Register tool in create_agent**

In `backend/agent.py`, replace:

```python
        tools=[get_current_weather, search_knowledge_base, search_memory],
```

with:

```python
        tools=[get_current_weather, search_knowledge_base, search_memory, search_web],
```

- [ ] **Step 3: Update system prompt**

Open `backend/soul/soul.md` and add these rules after the existing knowledge-base rules:

```markdown

当用户询问最新、实时、联网、新闻、公开网页、互联网资料，或知识库明显无法覆盖的问题时，使用 search_web 工具。
收到 search_web 结果后，必须基于搜索结果生成最终答案，并在答案中保留关键来源 URL。
search_web 的结果只用于当前问题，不代表长期知识库已经更新。
如果 search_web 返回失败信息，应说明联网搜索不可用，并基于已有知识给出有限回答。
```

- [ ] **Step 4: Syntax check import path**

Run:

```bash
uv run python -c "import sys; sys.path.insert(0, 'backend'); from agent import agent; print('agent loaded')"
```

Expected: prints `agent loaded`.

If PostgreSQL or Milvus is not running, this command can fail during existing startup imports. In that case run the narrower import check:

```bash
uv run python -c "import sys; sys.path.insert(0, 'backend'); from tools import search_web; print(search_web.name)"
```

Expected: prints `search_web`.

- [ ] **Step 5: Commit**

```bash
git add backend/agent.py backend/soul/soul.md
git commit -m "feat: register Tavily web search tool"
```

---

## Task 7: Extend Trace Schema and Frontend Source Display

**Files:**
- Modify: `backend/schemas.py`
- Modify: `frontend/index.html`
- Modify: `frontend/style.css`

- [ ] **Step 1: Add web fields to schemas**

In `backend/schemas.py`, replace `RetrievedChunk` with:

```python
class RetrievedChunk(BaseModel):
    """检索到的文档块"""
    filename: Optional[str] = Field(default=None, description="来源文件名")
    title: Optional[str] = Field(default=None, description="网页标题")
    url: Optional[str] = Field(default=None, description="网页URL")
    page_number: Optional[str | int] = Field(default=None, description="页码")
    text: Optional[str] = Field(default=None, description="文档块文本内容")
    score: Optional[float] = Field(default=None, description="相似度分数")
    rrf_rank: Optional[int] = Field(default=None, description="RRF排名")
    rerank_score: Optional[float] = Field(default=None, description="重排序分数")
    source_rank: Optional[int] = Field(default=None, description="搜索结果原始排名")
```

In `RagTrace`, add these fields after `candidate_k`:

```python
    web_source_count: Optional[int] = Field(default=None, description="联网搜索来源网页数")
    web_chunk_count: Optional[int] = Field(default=None, description="联网搜索索引分块数")
    web_error: Optional[str] = Field(default=None, description="联网搜索错误")
```

- [ ] **Step 2: Render web trace counts**

In `frontend/index.html`, find the RAG trace panel lines that render `candidate_k`. Add this block after it:

```html
                                <div v-if="msg.ragTrace.web_source_count !== null && msg.ragTrace.web_source_count !== undefined" class="trace-line">
                                    <strong>网页来源：</strong>{{ msg.ragTrace.web_source_count }}
                                </div>
                                <div v-if="msg.ragTrace.web_chunk_count !== null && msg.ragTrace.web_chunk_count !== undefined" class="trace-line">
                                    <strong>网页分块：</strong>{{ msg.ragTrace.web_chunk_count }}
                                </div>
                                <div v-if="msg.ragTrace.web_error" class="trace-line">
                                    <strong>联网错误：</strong>{{ msg.ragTrace.web_error }}
                                </div>
```

- [ ] **Step 3: Render URL source fields**

In each `source-item` block that currently renders:

```html
                                            <strong>{{ chunk.filename }}</strong>
                                            <span v-if="chunk.page_number"> - Page {{ chunk.page_number }}</span>
```

replace it with:

```html
                                            <strong>{{ chunk.title || chunk.filename || 'Unknown Source' }}</strong>
                                            <span v-if="chunk.page_number"> - Page {{ chunk.page_number }}</span>
                                            <a v-if="chunk.url" class="source-url" :href="chunk.url" target="_blank" rel="noopener noreferrer">
                                                {{ chunk.url }}
                                            </a>
```

There are three source lists in the current template: initial retrieved chunks, expanded retrieved chunks, and fallback retrieved chunks. Apply the same replacement in all three places.

- [ ] **Step 4: Add compact source URL styling**

Append to `frontend/style.css` near the existing source styles:

```css
.source-url {
    display: block;
    margin-top: 4px;
    color: #2563eb;
    font-size: 12px;
    line-height: 1.4;
    overflow-wrap: anywhere;
    text-decoration: none;
}

.source-url:hover {
    text-decoration: underline;
}
```

- [ ] **Step 5: Run syntax checks**

Run:

```bash
uv run python -m py_compile backend/schemas.py
```

Expected: no output and exit code 0.

- [ ] **Step 6: Commit**

```bash
git add backend/schemas.py frontend/index.html frontend/style.css
git commit -m "feat: show web search sources in trace"
```

---

## Task 8: End-to-End Manual Verification

**Files:**
- No code changes unless a previous task failed verification.

- [ ] **Step 1: Ensure `.env` contains Tavily key**

In local `.env`, add:

```env
TAVILY_API_KEY=tvly-your-real-key
WEB_SEARCH_COLLECTION=web_search_cache
WEB_SEARCH_MAX_RESULTS=5
WEB_SEARCH_TOP_K=5
WEB_SEARCH_CHUNK_SIZE=1200
WEB_SEARCH_CHUNK_OVERLAP=150
WEB_SEARCH_MAX_CHUNKS_PER_RESULT=4
```

- [ ] **Step 2: Start infrastructure**

Run:

```bash
docker compose up -d
docker compose ps
```

Expected: PostgreSQL and Milvus services are healthy or running.

- [ ] **Step 3: Start backend**

Run:

```bash
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Expected: Uvicorn starts and serves `http://127.0.0.1:8000/`.

- [ ] **Step 4: Verify tool directly**

In a second terminal, run:

```bash
uv run python -c "import sys; sys.path.insert(0, 'backend'); from tools import search_web; print(search_web.invoke({'query': 'Tavily LangChain integration current docs'})[:500])"
```

Expected output contains:

```text
【联网搜索结果】
URL:
```

- [ ] **Step 5: Verify chat API**

Run:

```bash
curl -N -X POST "http://127.0.0.1:8000/chat/stream" ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"请联网搜索 Tavily 的 LangChain 集成现在怎么用，并给出来源链接\",\"user_id\":\"demo\",\"session_id\":\"tavily-web-search\"}"
```

Expected SSE events include:

```text
data: {"type": "rag_step"
data: {"type": "trace"
data: [DONE]
```

Expected answer behavior:

- The assistant uses `search_web`.
- The final answer includes at least one URL.
- The trace contains `tool_name` equal to `search_web`.
- The frontend source panel shows clickable URLs.

- [ ] **Step 6: Run unit tests**

Run:

```bash
uv run pytest tests/backend/test_web_search.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit verification-only doc changes if any**

If no code changed during verification, do not create a commit. If verification required a small fix, commit that fix with:

```bash
git add <changed-files>
git commit -m "fix: stabilize Tavily web search verification"
```

---

## Acceptance Criteria

1. `search_web` is registered in the LangChain Agent tool list.
2. When the user asks for latest/current/web information, the Agent can call Tavily.
3. Tavily result content is indexed into Milvus before being used for answer context.
4. Retrieval is scoped by `search_id`, so one search does not retrieve chunks from another search.
5. Temporary web chunks are deleted after each search attempt.
6. The final answer includes source URLs when web search succeeds.
7. The frontend RAG trace shows `search_web`, source counts, chunk counts, and clickable URLs.
8. Unit tests for normalization, chunking, service orchestration, vector-store payloads, and tool guard behavior pass.

---

## Risk Notes

- Tavily can return short `content` without `raw_content` for some pages; `build_web_chunks` falls back to snippet content so the tool still works.
- Some model providers may not reliably choose `search_web`; the prompt update makes the routing rule explicit.
- The current `EmbeddingService` BM25 vocabulary is in-memory and fitted per search, which is acceptable for temporary per-search retrieval.
- The direct `agent` import check can fail when PostgreSQL or Milvus is offline because the current project initializes those clients at import time; use the narrower `tools.search_web` check in that case.

---

## Self-Review

- Spec coverage: Tavily search, LangChain tool registration, Milvus vector indexing, trace display, and source URLs are covered by Tasks 1-8.
- Placeholder scan: The plan contains concrete paths, code blocks, commands, and expected outputs.
- Type consistency: `retrieved_chunks` use `title`, `url`, `text`, `score`, `rrf_rank`, and `source_rank` consistently across service, tool, schema, and frontend.
