from langchain.agents.middleware import before_model, AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
from typing import Any, Optional
from langgraph.types import Command
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel, Field
from config import MODEL, BASE_URL, API_KEY, POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
import os


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
  例如："女朋友叫做洪海艳" → relationship: "女朋友（洪海艳）"
       "我男朋友是张三" → relationship: "男朋友（张三）"
- name: 名字/昵称
- school: 学校或公司
- major: 专业或工作领域
- grade: 年级（大一/大二/大三/大四/研一/研二/研三/博一/博二/博三）
- identity: 身份（学生/工程师/老师等）
- relationship: 重要关系及对方名字（格式：关系（名字），如：女朋友（洪海艳））
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

    def format_for_system_prompt(self) -> str:
        """格式化用户信息为系统提示词字符串"""
        info = self.load_user_info()
        if not info:
            return ""

        lines = ["【用户记忆】"]
        if "name" in info and info["name"]:
            lines.append(f"- 名字：{info['name']}")
        if "identity" in info and info["identity"]:
            lines.append(f"- 身份：{info['identity']}")
        if "school" in info and info["school"]:
            lines.append(f"- 学校/公司：{info['school']}")
        if "major" in info and info["major"]:
            lines.append(f"- 专业/领域：{info['major']}")
        if "grade" in info and info["grade"]:
            lines.append(f"- 年级：{info['grade']}")
        if "relationship" in info and info["relationship"]:
            lines.append(f"- 重要关系：{info['relationship']}")
        if "interest" in info and info["interest"]:
            lines.append(f"- 兴趣爱好：{info['interest']}")
        if "location" in info and info["location"]:
            lines.append(f"- 所在地：{info['location']}")
        if "other" in info and info["other"]:
            lines.append(f"- 其他：{info['other']}")

        return "\n".join(lines)


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


def load_user_memory_for_prompt() -> str:
    """加载用户记忆用于拼接到系统提示词"""
    return user_memory_manager.format_for_system_prompt()


class MemorySummaryMiddleware(AgentMiddleware):
    """记忆总结中间件

    每隔指定轮次（默认25轮），对对话历史进行总结，
    并将总结存入 Store（长期记忆）。
    """

    def __init__(self, trigger_turns: int = 25):
        self._summary_model = None
        self._initialized = False
        self._turn_counts = {}  # thread_id -> turn_count
        self._trigger_turns = trigger_turns

    @property
    def summary_model(self):
        if not self._initialized:
            from langchain.chat_models import init_chat_model
            self._summary_model = init_chat_model(
                model=MODEL,
                model_provider="openai",
                api_key=API_KEY,
                base_url=BASE_URL,
            )
            self._initialized = True
        return self._summary_model

    @before_model
    def __call__(self, state: AgentState, config: dict) -> Optional[Command]:
        """在模型调用前执行，检查是否需要总结记忆"""
        messages = state.get("messages", [])
        if not messages:
            return None

        thread_id = config.get("configurable", {}).get("thread_id", "default")

        user_turns = sum(
            1 for msg in messages
            if isinstance(msg, (HumanMessage, dict)) and
            (isinstance(msg, dict) and msg.get("role") == "user" or
             isinstance(msg, HumanMessage) and msg.type == "human")
        )

        current_count = self._turn_counts.get(thread_id, 0)

        if user_turns > current_count:
            self._turn_counts[thread_id] = user_turns

        if user_turns > 0 and user_turns % self._trigger_turns == 0:
            return self._summarize_and_store(messages, thread_id, config)

        return None

    def _summarize_and_store(self, messages, thread_id: str, config: dict) -> Optional[Command]:
        """总结对话并存储到 Store"""
        from langgraph.store.postgres import PostgresStore

        # 开始总结
        print(f"[MemorySummary] 开始总结线程 {thread_id} 的记忆")

        conversation_text = self._format_conversation(messages)

        summary_prompt = f"""请总结以下对话的要点，包括：
1. 用户讨论的主题
2. 用户的需求或问题
3. 提供的帮助或解决方案
4. 任何重要的上下文信息

对话内容：
{conversation_text}

请用简洁的语言总结（不超过200字）："""

        try:
            summary_response = self.summary_model.invoke(summary_prompt)
            summary_text = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        except Exception as e:
            print(f"[MemorySummary] 总结失败: {e}")
            return None

        conn_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

        try:
            with PostgresStore.from_conn_string(conn_string) as store:
                store.setup()

                # 命名空间 为 (memory, thread_id) 区分不同的线程记忆
                namespace = ("memory", thread_id)
                summary_key = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                store.put(namespace, summary_key, {
                    "summary": summary_text,
                    "timestamp": datetime.now().isoformat(),
                    "turn_count": self._turn_counts.get(thread_id, 0),
                    "conversation_text": conversation_text[:1000] if len(conversation_text) > 1000 else conversation_text,
                })

                print(f"[MemorySummary] 已保存总结到 Store: {namespace}/{summary_key}")
        except Exception as e:
            print(f"[MemorySummary] 存储失败: {e}")

        return None

    def _format_conversation(self, messages) -> str:
        """将消息格式化为可读文本"""
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"助手: {msg.content}")
            elif isinstance(msg, dict):
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
        return "\n".join(lines[-50:])

# 测试一下记忆总结中间件，每2轮对话总结一次，后面改为每25轮总结一次
memory_summary_middleware = MemorySummaryMiddleware(trigger_turns=25)
