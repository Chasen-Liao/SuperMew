from langchain.agents.middleware import after_model, AgentState, dynamic_prompt, ModelRequest
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from typing import Any, Optional
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel, Field
from config import MODEL, BASE_URL, API_KEY, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
from dataclasses import dataclass
from pathlib import Path
import os
import sys

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from memory_vector_store import MemoryVectorStore
from embedding import EmbeddingService

TRIGGER_TURNS = 2  # 每 25 轮触发一次摘要

@dataclass
class Context:
    """Agent 运行时上下文 schema"""
    user_id: str
    thread_id: str


# LLM 提取用的 Pydantic 模型
class UserInfo(BaseModel):
    """用户信息结构化提取"""
    name: Optional[str] = Field(default=None, description="用户的名字或昵称")
    school: Optional[str] = Field(default=None, description="用户就读或工作的学校/公司")
    major: Optional[str] = Field(default=None, description="用户的专业或从事的领域")
    grade: Optional[str] = Field(default=None, description="用户的年级，如：大一、大三、研一、博二")
    identity: Optional[str] = Field(default=None, description="用户的身份，如：学生、工程师、老师")
    relationship: Optional[str] = Field(default=None, description="用户的重要关系，如：女朋友、男朋友、朋友、家人")
    interest: Optional[str] = Field(default=None, description="用户的兴趣爱好")
    location: Optional[str] = Field(default=None, description="用户所在的城市或地区")
    other: Optional[str] = Field(default=None, description="其他重要信息")


class UserMemoryManager:
    """用户记忆管理器：使用 LLM 提取并存储到 PostgresStore"""

    """
    Store 存储的是结构化的 UserInfo 模型
    当Human发送的消息有包含用户信息时，调用 LLM 提取并存储到 PostgresStore
    """

    _instance = None
    _conn_string = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if UserMemoryManager._conn_string is None:
            UserMemoryManager._conn_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

    def _get_store(self):
        """获取 Store 实例"""
        return PostgresStore.from_conn_string(UserMemoryManager._conn_string)

    def _get_llm(self):
        """获取用于提取的 LLM"""
        from langchain.chat_models import init_chat_model
        return init_chat_model(
            model=MODEL,
            model_provider="openai",
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
        )

    def extract_from_text(self, text: str) -> Optional[UserInfo]:
        """调用 LLM 从用户消息中提取结构化信息"""
        prompt = f"""你是一个用户信息提取助手。请从用户的发言中提取结构化的个人信息。

提取规则：
- 只提取明确提到的信息，不要推测
- 如果某项信息没有提到，留空
- 如果提到了关系人的名字，一定要包含在提取结果中
  例如："女朋友叫做xxx" → relationship: "女朋友"
       "我男朋友是xxx" → relationship: "男朋友"
- name: 名字/昵称
- school: 学校或公司
- major: 专业或工作领域
- grade: 年级（大一/大二/大三/大四/研一/研二/研三/博一/博二/博三）
- identity: 身份（学生/工程师/老师等）
- relationship: 重要关系及对方名字（格式：关系（名字），如：女朋友）
- interest: 兴趣爱好
- location: 所在地
- other: 其他重要信息

用户发言：
{text}

请以 JSON 格式输出，直接返回 JSON，不要有其他文字："""

        try:
            llm = self._get_llm()
            response = llm.with_structured_output(UserInfo).invoke(prompt)
            return response
        except Exception as e:
            print(f"[UserMemory] LLM 提取失败: {e}")
            return None

    def save_user_info(self, user_info: UserInfo) -> bool:
        """保存用户信息到 Store（合并模式，保留已有值）"""
        if not user_info:
            return False

        # 将 Pydantic model 转为 dict，排除 None 值
        info_dict = user_info.model_dump(exclude_none=True)
        if not info_dict:
            return False

        try:
            namespace = ("user_memory", "global")
            with self._get_store() as store:
                # 先加载已有信息
                existing = {}
                try:
                    existing_result = store.get(namespace, "profile")
                    if existing_result:
                        existing = dict(existing_result.value)
                except Exception:
                    pass  # 如果没有已有信息，忽略

                # 合并：新值非空时才覆盖已有值
                merged = {**existing}
                for key, value in info_dict.items():
                    if value and str(value).strip():  # 新值非空才覆盖
                        merged[key] = value

                merged["updated_at"] = datetime.now().isoformat()

                store.put(namespace, "profile", merged)
                print(f"[UserMemory] 已保存用户信息到 Store: {merged}")

            return True
        except Exception as e:
            print(f"[UserMemory] 保存失败: {e}")
            return False

    def load_user_info(self) -> dict:
        """从 Store 加载用户信息"""
        try:
            namespace = ("user_memory", "global")
            with self._get_store() as store:
                results = store.get(namespace, "profile")
                if results:
                    return dict(results.value)
        except Exception as e:
            print(f"[UserMemory] 加载失败: {e}")
        return {}


# 全局单例
user_memory_manager = UserMemoryManager()


def extract_and_save_user_memory(user_text: str) -> bool:
    """从用户文本提取信息并保存到 Store"""
    user_info = user_memory_manager.extract_from_text(user_text)
    if user_info:
        return user_memory_manager.save_user_info(user_info)
    return False


def should_extract_memory(text: str) -> bool:
    """快速检查文本是否可能包含需要提取的个人信息"""
    keywords = [
        # 名字相关
        '我叫', '名字叫', '我是', '叫', '昵称', '称呼',
        # 学校相关
        '学校', '大学', '学院', '上学', '读书',
        # 专业相关
        '专业', '学',
        # 身份相关
        '学生', '工程师', '老师', '设计师', '医生', '研究生', '硕士', '博士',
        # 关系相关
        '女朋友', '男朋友', '朋友', '家人', '老公', '老婆', '男友', '女友',
        # 位置相关
        '住在', '位于', '在', # 这些太泛化了，注释掉
        # 兴趣相关
        '喜欢', '爱好', '兴趣',
    ]
    text_lower = text.lower()
    # 检查是否包含至少2个关键词（减少误触发）
    matches = sum(1 for kw in keywords if kw in text_lower)
    return matches >= 2


def extract_and_save_user_memory_async(user_text: str):
    """后台线程执行：检查并提取用户信息"""
    import threading
    def _run():
        if should_extract_memory(user_text):
            extract_and_save_user_memory(user_text)
    threading.Thread(target=_run, daemon=True).start()


def _format_conversation_for_summary(messages, last_n_turns: int = 25) -> str:
    """将消息格式化为可读文本，只保留最近 N 轮用户对话"""
    lines = []
    user_msg_count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            if user_msg_count >= last_n_turns:
                break
            lines.append(f"用户: {msg.content}")
            user_msg_count += 1
        elif isinstance(msg, AIMessage):
            lines.append(f"助手: {msg.content}")
        elif isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
    return "\n".join(reversed(lines))


def _get_summary_model():
    """获取用于总结的 LLM"""
    from langchain.chat_models import init_chat_model
    return init_chat_model(
        model=MODEL,
        model_provider="openai",
        api_key=API_KEY,
        base_url=BASE_URL,
    )


def _summarize_conversation(conversation_text: str) -> str:
    """调用 LLM 总结对话"""
    prompt = f"""请总结以下对话的要点，包括：
1. 用户讨论的主题
2. 用户的需求或问题
3. 提供的帮助或解决方案
4. 任何重要的上下文信息

对话内容：
{conversation_text}

请用简洁的语言总结（不超过500字）："""

    try:
        llm = _get_summary_model()
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"[memory_summary_hook] 总结失败: {e}")
        return ""


def _save_summary_to_milvus(summary_text: str, thread_id: str, turn_count: int):
    """将摘要写入 Milvus"""
    if not summary_text:
        return

    try:
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        store = _get_memory_vector_store()
        embedder = _get_embedding_service()

        formatted_text = f"[对话摘要] {summary_text}（{turn_count}轮对话，{timestamp}）"
        dense_vec = embedder.get_embeddings([formatted_text])[0]
        sparse_vec = embedder.get_sparse_embedding(formatted_text)

        store.insert([{
            "memory_type": "summary",
            "source_key": f"memory/{thread_id}/{turn_count}",
            "text": formatted_text,
            "created_at": timestamp,
            "embedding": dense_vec,
            "sparse_embedding": sparse_vec,
        }])
        print(f"[memory_summary_hook] 已写入 Milvus: {formatted_text[:50]}...")
    except Exception as e:
        print(f"[memory_summary_hook] 写入 Milvus 失败: {e}")


@after_model
def memory_summary_hook(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    """每 N 轮总结对话并写入 Milvus"""
    messages = state.get("messages", [])
    if not messages:
        return None

    user_turns = sum(1 for msg in messages if isinstance(msg, HumanMessage))

    if user_turns == 0 or user_turns % TRIGGER_TURNS != 0:
        return None

    thread_id = runtime.context.thread_id
    print(f"[memory_summary_hook] 触发摘要ID: {thread_id}")

    conversation_text = _format_conversation_for_summary(messages, TRIGGER_TURNS)
    summary_text = _summarize_conversation(conversation_text)
    _save_summary_to_milvus(summary_text, thread_id, user_turns)

    return None


def _load_soul_prompt() -> str:
    """加载 soul.md 系统提示词"""
    soul_path = Path(__file__).parent / "soul" / "soul.md"
    return soul_path.read_text(encoding="utf-8")


_memory_vector_store = None
_embedding_service_for_memory = None


def _get_memory_vector_store():
    global _memory_vector_store
    if _memory_vector_store is None:
        _memory_vector_store = MemoryVectorStore()
    # 确保 collection 存在（删除后可重建）
    _memory_vector_store.init_collection()
    return _memory_vector_store


def _get_embedding_service():
    global _embedding_service_for_memory
    if _embedding_service_for_memory is None:
        _embedding_service_for_memory = EmbeddingService()
    return _embedding_service_for_memory


def _load_profile_for_prompt() -> str:
    """从 Store 加载用户画像，拼接到系统提示词"""
    info = user_memory_manager.load_user_info()
    if not info:
        return ""

    lines = ["【用户档案】"]
    if info.get("name"):
        lines.append(f"- 名字：{info['name']}")
    if info.get("identity"):
        lines.append(f"- 身份：{info['identity']}")
    if info.get("school"):
        lines.append(f"- 学校/公司：{info['school']}")
    if info.get("major"):
        lines.append(f"- 专业/领域：{info['major']}")
    if info.get("grade"):
        lines.append(f"- 年级：{info['grade']}")
    if info.get("relationship"):
        lines.append(f"- 重要关系：{info['relationship']}")
    if info.get("interest"):
        lines.append(f"- 兴趣爱好：{info['interest']}")
    if info.get("location"):
        lines.append(f"- 所在地：{info['location']}")
    # if info.get("other"):
    # 这里去检索
    #     lines.append(f"- 其他信息：{info['other']}")

    return "\n".join(lines)


@dynamic_prompt
def system_prompt_middleware(request: ModelRequest) -> str:
    """动态生成系统提示词：soul.md + 用户档案（来自 PostgresStore）"""
    soul_prompt = _load_soul_prompt()
    profile_text = _load_profile_for_prompt()

    if not profile_text:
        return soul_prompt

    return f"{soul_prompt}\n\n{profile_text}"
