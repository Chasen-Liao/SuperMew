import os
import json
import asyncio
from pathlib import Path
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessageChunk
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
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

# 创建 Store（长期记忆）
store = InMemoryStore()

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

    # 读取 soul.md 作为系统提示词
    # 这样可以方便用户修改，和后续记忆添加和改进
    soul_prompt_path = Path(__file__).parent / "soul" / "soul.md"
    system_prompt = soul_prompt_path.read_text(encoding="utf-8")

    agent = create_agent(
        model=model,
        tools=[get_current_weather, search_knowledge_base],
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        store=store,
        middleware=[
            SummarizationMiddleware(
                model=summary_model,
                trigger=("tokens", 80000),
                keep=("messages", 12),  # 6 轮交互 = 12 条消息
            ),
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

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_text}]},
        config=config,
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
    config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
    checkpoint = checkpointer.get(config)
    if checkpoint is None:
        return []
    # checkpointer 返回格式: {"channel_values": {"messages": [...]}}
    channel_values = checkpoint.get("channel_values", {})
    messages = channel_values.get("messages", [])
    return messages


def delete_session_from_checkpointer(user_id: str, session_id: str):
    """从 checkpointer 删除会话"""
    config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}}
    checkpointer.delete(config)


async def chat_with_agent_stream(user_text: str, user_id: str = "default_user", session_id: str = "default_session"):
    """使用 Agent 处理用户消息并流式返回响应。

    架构：使用统一输出队列 + 后台任务，确保 RAG 检索步骤在工具执行期间实时推送，
    而非等待工具完成后才显示。
    """
    # 使用 thread_id 标识会话，checkpointer 自动管理会话历史
    config = {"configurable": {"thread_id": f"{user_id}_{session_id}"}, "recursion_limit": 8}

    # 更新会话列表（用于 API 的会话查询）
    storage.touch_session(user_id, session_id)

    # 清理可能残留的 RAG 上下文
    get_last_rag_context(clear=True)
    reset_tool_call_guards()

    # 统一输出队列：所有事件（content / rag_step）都汇入这里
    output_queue = asyncio.Queue()

    class _RagStepProxy:
        """代理对象：将 emit_rag_step 的原始 step dict 包装后放入统一输出队列。"""
        def put_nowait(self, step):
            output_queue.put_nowait({"type": "rag_step", "step": step})

    set_rag_step_queue(_RagStepProxy())

    full_response = ""

    async def _agent_worker():
        """后台任务：运行 agent 并将内容 chunk 推入输出队列。"""
        nonlocal full_response
        try:
            # 使用同步 stream 方法配合 asyncio.to_thread
            for chunk in agent.stream(
                {"messages": [{"role": "user", "content": user_text}]},
                config=config,
            ):
                # chunk 格式: {'model': {'messages': [AIMessage, ...]}}
                if isinstance(chunk, dict) and "model" in chunk:
                    model_data = chunk["model"]
                    if isinstance(model_data, dict) and "messages" in model_data:
                        messages = model_data["messages"]
                        if messages and hasattr(messages[-1], "content"):
                            content = messages[-1].content
                            if content:
                                full_response += content
                                await output_queue.put({"type": "content", "content": content})
        except Exception as e:
            await output_queue.put({"type": "error", "content": str(e)})
        finally:
            # 哨兵：通知主循环 agent 已完成
            await output_queue.put(None)

    # 启动后台任务
    agent_task = asyncio.create_task(_agent_worker())

    try:
        # 主循环：持续从统一队列取事件并 yield SSE
        # RAG 步骤在工具执行期间通过 call_soon_threadsafe 实时入队，不需要等 agent 产出 chunk
        while True:
            event = await output_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except GeneratorExit:
        # 客户端断开连接（AbortController）时，FastAPI 会向此生成器抛出 GeneratorExit
        # 我们必须在此处取消后台任务
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass  # 任务已成功取消
        raise  # 重新抛出 GeneratorExit 以便 FastAPI 正确处理关闭
    finally:
        # 正常结束或异常退出时清理
        set_rag_step_queue(None)
        if not agent_task.done():
             agent_task.cancel()

    # 获取 RAG trace
    rag_context = get_last_rag_context(clear=True)
    rag_trace = rag_context.get("rag_trace") if rag_context else None

    # 发送 trace 信息
    if rag_trace:
        yield f"data: {json.dumps({'type': 'trace', 'rag_trace': rag_trace})}\n\n"

    # 发送结束信号
    yield "data: [DONE]\n\n"
