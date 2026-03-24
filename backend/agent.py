import os
import sys
import json
import asyncio
import queue
import threading
from pathlib import Path

# 确保 backend 目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent))

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from middleware import memory_summary_middleware, extract_and_save_user_memory_async, load_user_memory_for_prompt, system_prompt_middleware, Context
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from tools import get_current_weather, search_knowledge_base, get_last_rag_context, reset_tool_call_guards, set_rag_step_queue
from datetime import datetime
from config import API_KEY, MODEL, BASE_URL, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB


def create_checkpointer():
    """创建 PostgresSaver checkpointer"""
    import psycopg
    conn = psycopg.connect(
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        autocommit=True,
        prepare_threshold=0,
        row_factory=psycopg.rows.dict_row
    )
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer


checkpointer = create_checkpointer()

# 创建 Store（长期记忆，持久化到 PostgreSQL）
_store = PostgresStore.from_conn_string(
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
store = _store.__enter__()
store.setup()

class ConversationStorage:
    """会话列表存储（仅管理会话元数据，不存储消息）

    消息由 checkpointer 自动管理，此处只记录会话的存在和更新时间，
    用于 API 的会话列表查询和删除功能。
    """

    def __init__(self, storage_file: str = None):
        if storage_file:
            storage_path = os.path.abspath(storage_file)
        else:
            package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            data_dir = os.path.join(package_root, "data")
            os.makedirs(data_dir, exist_ok=True)
            storage_path = os.path.join(data_dir, "customer_service_history.json")

        self.storage_file = storage_path

    def touch_session(self, user_id: str, session_id: str):
        """更新会话的最后访问时间（checkpointer 会在首次对话时自动创建会话）"""
        data = self._load()

        if user_id not in data:
            data[user_id] = {}

        if session_id not in data[user_id]:
            data[user_id][session_id] = {"updated_at": datetime.now().isoformat()}
        else:
            data[user_id][session_id]["updated_at"] = datetime.now().isoformat()

        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def list_sessions(self, user_id: str) -> list:
        """列出用户的所有会话"""
        data = self._load()
        if user_id not in data:
            return []
        return list(data[user_id].keys())

    def delete_session(self, user_id: str, session_id: str) -> bool:
        """删除指定用户的会话，返回是否删除成功"""
        data = self._load()
        if user_id not in data or session_id not in data[user_id]:
            return False

        del data[user_id][session_id]
        if not data[user_id]:
            del data[user_id]

        with open(self.storage_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True

    def _load(self) -> dict:
        """加载数据"""
        if not os.path.exists(self.storage_file):
            return {}
        try:
            with open(self.storage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}



def create_agent_instance():
    model = init_chat_model(
        model=MODEL,
        model_provider="openai",  # 兼容 SiliconFlow 等兼容 OpenAI API 的服务商
        api_key=API_KEY,
        base_url=BASE_URL,
        temperature=0.3,
        stream_usage=True,
    )

    # 摘要模型 - 使用与主模型相同的配置
    summary_model = init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        checkpointer=checkpointer,
        store=store,
        context_schema=Context,
        middleware=[
            SummarizationMiddleware(
                model=summary_model,
                trigger=("tokens", 8000),
                keep=("messages", 12),
            ),
            memory_summary_middleware,
            system_prompt_middleware,
        ],
    )
    return agent, model


agent, model = create_agent_instance()#

storage = ConversationStorage()


def chat_with_agent(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并返回响应"""
    # 使用 thread_id 标识会话，checkpointer 自动管理会话历史
    config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}, "recursion_limit": 8}

    # 更新会话列表（用于 API 的会话查询）
    storage.touch_session(user_id, session_id)

    # 清理可能残留的 RAG 上下文，避免跨请求污染
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 提取用户信息并保存到 Store（后台异步执行）
    extract_and_save_user_memory_async(user_text)

    # 系统提示词由 dynamic_prompt 中间件自动生成，无需手动拼接
    from langchain_core.messages import HumanMessage
    messages = [HumanMessage(content=user_text)]

    result = agent.invoke(
        {"messages": messages},
        config=config,
        context={"user_id": user_id},
    )

    response_content = ""
    if isinstance(result, dict):
        if "output" in result:
            response_content = result["output"]
        elif "messages" in result and result["messages"]:
            msg = result["messages"][-1]
            response_content = getattr(msg, "content", str(msg))
        else:
            response_content = str(result)
    elif hasattr(result, "content"):
        response_content = result.content
    else:
        response_content = str(result)

    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    return {
        "response": response_content,
        "rag_trace": rag_trace,
    }


def get_session_messages(user_id: str, session_id: str) -> list:
    """从 checkpointer 获取会话消息"""
    try:
        config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
        checkpoint = checkpointer.get(config)
        if checkpoint is None:
            return []
        channel_values = checkpoint.get("channel_values", {})
        messages = channel_values.get("messages", [])
        return messages
    except Exception as e:
        print(f"[get_session_messages] 错误: {e}")
        return []


def delete_session_from_checkpointer(user_id: str, session_id: str):
    """从 checkpointer 删除会话"""
    config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
    checkpointer.delete(config)


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。

    架构：使用线程安全的 queue.Queue 传递 RAG 步骤和 LLM 内容。
    """
    # 使用 thread_id 标识会话，checkpointer 自动管理会话历史
    config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}, "recursion_limit": 8}

    # 更新会话列表（用于 API 的会话查询）
    storage.touch_session(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 提取用户信息并保存到 Store（后台异步执行，不阻塞）
    extract_and_save_user_memory_async(user_text)

    # 统一输出队列：使用线程安全的 queue.Queue
    output_queue = queue.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    full_response = ""

    def _agent_sync_stream():
        """同步运行 agent.stream() 并将 chunk 推入队列。"""
        nonlocal full_response
        try:
            from langchain_core.messages import HumanMessage

            # 系统提示词由 dynamic_prompt 中间件自动生成，无需手动拼接
            messages = [HumanMessage(content=user_text)]

            for mode, data in agent.stream(
                {"messages": messages},
                config=config,
                stream_mode=["messages", "updates"],
                context={"user_id": user_id},
            ):
                if mode == "messages":
                    msg, metadata = data
                    if not isinstance(msg, AIMessageChunk):
                        continue
                    if getattr(msg, "tool_call_chunks", None):
                        continue

                    content = ""
                    if isinstance(msg.content, str):
                        content = msg.content
                    elif isinstance(msg.content, list):
                        for block in msg.content:
                            if isinstance(block, str):
                                content += block
                            elif isinstance(block, dict) and block.get("type") == "text":
                                content += block.get("text", "")

                    if content:
                        full_response += content
                        output_queue.put({"type": "content", "content": content})
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            output_queue.put({"type": "error", "content": error_msg})
        finally:
            output_queue.put(None)

    # 启动后台线程运行 agent
    agent_thread = threading.Thread(target=_agent_sync_stream, daemon=True)
    agent_thread.start()

    try:
        # 主循环：从队列读取事件并 yield SSE
        while True:
            try:
                event = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: output_queue.get(timeout=1.0)
                )
            except Exception:
                # queue.get 超时会抛出 queue.Empty，表示 1 秒内没有事件
                # 继续循环，等待下一个事件
                continue

            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        set_rag_step_queue(None)
        agent_thread.join(timeout=2.0)

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"
