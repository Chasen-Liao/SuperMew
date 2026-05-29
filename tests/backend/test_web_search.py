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
