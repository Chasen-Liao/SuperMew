"""Tavily web search pipeline helpers."""
from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
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


def clean_web_text(text: str) -> str:
    """Collapse whitespace in web content."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_tavily_response(response: Any) -> list[dict]:
    """Normalize TavilySearch output into source documents."""
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
    """Split normalized web documents into embeddable chunks."""
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
            retrieved = self.vector_store.hybrid_retrieve(
                query=query,
                search_id=search_id,
                top_k=self.top_k,
            )
        finally:
            self.vector_store.delete_search(search_id)

        return {
            "query": query,
            "source_count": len(docs),
            "chunk_count": len(chunks),
            "retrieved_chunks": retrieved,
            "error": None,
        }
