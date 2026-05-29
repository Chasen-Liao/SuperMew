"""Tavily web search pipeline helpers."""
from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter


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
