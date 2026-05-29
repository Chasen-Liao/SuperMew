from typing import Optional
import requests
try:
    from langchain_core.tools import tool
except ImportError:
    from langchain_core.tools import tool

from config import AMAP_WEATHER_API, AMAP_API_KEY
from web_search import WebSearchService

_LAST_RAG_CONTEXT = None
_KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
_WEB_SEARCH_TOOL_CALLS_THIS_TURN = 0
_RAG_STEP_QUEUE = None  # asyncio.Queue, set by agent before streaming
_RAG_STEP_LOOP = None   # asyncio loop, captured when setting queue


def _set_last_rag_context(context: dict):
    global _LAST_RAG_CONTEXT
    _LAST_RAG_CONTEXT = context


def get_last_rag_context(clear: bool = True) -> Optional[dict]:
    """获取最近一次 RAG 检索上下文，默认读取后清空。"""
    global _LAST_RAG_CONTEXT
    context = _LAST_RAG_CONTEXT
    if clear:
        _LAST_RAG_CONTEXT = None
    return context


def reset_tool_call_guards():
    """每轮对话开始时重置工具调用计数。"""
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN, _WEB_SEARCH_TOOL_CALLS_THIS_TURN
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN = 0
    _WEB_SEARCH_TOOL_CALLS_THIS_TURN = 0


def set_rag_step_queue(queue):
    """设置 RAG 步骤队列，并捕获当前事件循环以便跨线程调度。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    _RAG_STEP_QUEUE = queue
    if queue:
        import asyncio
        try:
            _RAG_STEP_LOOP = asyncio.get_running_loop()
        except RuntimeError:
            _RAG_STEP_LOOP = asyncio.get_event_loop()
    else:
        _RAG_STEP_LOOP = None


def emit_rag_step(icon: str, label: str, detail: str = ""):
    """向队列发送一个 RAG 检索步骤。支持跨线程安全调用。"""
    global _RAG_STEP_QUEUE, _RAG_STEP_LOOP
    if _RAG_STEP_QUEUE is not None and _RAG_STEP_LOOP is not None:
        step = {"icon": icon, "label": label, "detail": detail}
        try:
            if not _RAG_STEP_LOOP.is_closed():
                _RAG_STEP_LOOP.call_soon_threadsafe(_RAG_STEP_QUEUE.put_nowait, step)
        except Exception:
            pass


def get_current_weather(location: str, extensions: Optional[str] = "base") -> str:
    """获取天气信息"""
    if not location:
        return "location参数不能为空"
    if extensions not in ("base", "all"):
        return "extensions参数错误，请输入base或all"

    if not AMAP_WEATHER_API or not AMAP_API_KEY:
        return "天气服务未配置（缺少 AMAP_WEATHER_API 或 AMAP_API_KEY）"

    params = {
        "key": AMAP_API_KEY,
        "city": location,
        "extensions": extensions,
        "output": "json",
    }

    try:
        resp = requests.get(AMAP_WEATHER_API, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "1":
            return f"查询失败：{data.get('info', '未知错误')}"

        if extensions == "base":
            lives = data.get("lives", [])
            if not lives:
                return f"未查询到 {location} 的天气数据"
            w = lives[0]
            return (
                f"【{w.get('city', location)} 实时天气】\n"
                f"天气状况：{w.get('weather', '未知')}\n"
                f"温度：{w.get('temperature', '未知')}℃\n"
                f"湿度：{w.get('humidity', '未知')}%\n"
                f"风向：{w.get('winddirection', '未知')}\n"
                f"风力：{w.get('windpower', '未知')}级\n"
                f"更新时间：{w.get('reporttime', '未知')}"
            )

        forecasts = data.get("forecasts", [])
        if not forecasts:
            return f"未查询到 {location} 的天气预报数据"
        f0 = forecasts[0]
        out = [f"【{f0.get('city', location)} 天气预报】", f"更新时间：{f0.get('reporttime', '未知')}", ""]
        today = (f0.get("casts") or [])[0] if f0.get("casts") else {}
        out += [
            "今日天气：",
            f"  白天：{today.get('dayweather','未知')}",
            f"  夜间：{today.get('nightweather','未知')}",
            f"  气温：{today.get('nighttemp','未知')}~{today.get('daytemp','未知')}℃",
        ]
        return "\n".join(out)

    except requests.exceptions.Timeout:
        return "错误：请求天气服务超时"
    except requests.exceptions.RequestException as e:
        return f"错误：天气服务请求失败 - {e}"
    except Exception as e:
        return f"错误：解析天气数据失败 - {e}"


@tool("search_memory")
def search_memory(query: str) -> str:
    """Search the user's past conversation memories using dense+sparse hybrid retrieval (RRF fusion).

    Use this tool when the user asks about something they discussed before,
    wants to recall past conversations, or refers to "what I told you earlier", etc.

    NOTE: The retrieval results are automatically used by the assistant.
    Do NOT reveal the raw retrieval content to the user in your response.
    """
    from memory_vector_store import MemoryVectorStore
    from embedding import EmbeddingService
    from config import MEMORY_TOP_K

    try:
        emit_rag_step("🔍", "正在检索记忆...", f"查询: {query[:50]}")

        store = MemoryVectorStore()
        store.init_collection()  # 确保 collection 存在
        embedder = EmbeddingService()

        dense_vec = embedder.get_embeddings([query])[0]
        sparse_vec = embedder.get_sparse_embedding(query)
        results = store.hybrid_search(dense_vec, sparse_vec, top_k=MEMORY_TOP_K)

        emit_rag_step("✅", f"记忆检索完成，找到 {len(results)} 条相关记忆")

        if not results:
            return "No relevant memories found."

        formatted = []
        for i, result in enumerate(results, 1):
            mem_type = result.get("memory_type", "")
            text = result.get("text", "")
            score = result.get("score", 0)
            formatted.append(f"[{i}] [{mem_type}] {text}（相关度: {score:.3f}）")

        return "【相关记忆】\n" + "\n\n".join(formatted)

    except Exception as e:
        return f"记忆检索失败: {e}"


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


@tool("search_knowledge_base")
def search_knowledge_base(query: str) -> str:
    """Search for information in the knowledge base using hybrid retrieval (dense + sparse vectors)."""
    # ... guards omitted ...
    global _KNOWLEDGE_TOOL_CALLS_THIS_TURN
    if _KNOWLEDGE_TOOL_CALLS_THIS_TURN >= 1:
        return (
            "TOOL_CALL_LIMIT_REACHED: search_knowledge_base has already been called once in this turn. "
            "Use the existing retrieval result and provide the final answer directly."
        )
    _KNOWLEDGE_TOOL_CALLS_THIS_TURN += 1

    from rag_pipeline import run_rag_graph

    # 在同步工具中获取当前的 Loop 可能不可靠，但我们之前是通过 call_soon_threadsafe 调度的。
    # 这里 _RAG_STEP_QUEUE 是在主线程/Loop 设置的全局变量。
    # 如果工具运行在线程池中，它是可以访问到全局变量 _RAG_STEP_QUEUE 的。
    # emit_rag_step 内部做了 try-except 和 get_event_loop()。

    # 问题可能出在 asyncio.get_event_loop() 在子线程中调用会报错或者拿不到主线程的loop。
    # 我们应该在 set_rag_step_queue 时也保存 loop 引用，或者在 emit_rag_step 中更健壮地获取 loop。

    rag_result = run_rag_graph(query)

    docs = rag_result.get("docs", []) if isinstance(rag_result, dict) else []
    rag_trace = rag_result.get("rag_trace", {}) if isinstance(rag_result, dict) else {}
    if rag_trace:
        _set_last_rag_context({"rag_trace": rag_trace})

    if not docs:
        return "No relevant documents found in the knowledge base."

    formatted = []
    for i, result in enumerate(docs, 1):
        source = result.get("filename", "Unknown")
        page = result.get("page_number", "N/A")
        text = result.get("text", "")
        formatted.append(f"[{i}] {source} (Page {page}):\n{text}")

    return "Retrieved Chunks:\n" + "\n\n---\n\n".join(formatted)
