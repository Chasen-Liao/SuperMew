"""记忆向量索引定时重建任务"""
import threading
from datetime import datetime, timedelta
from config import MEMORY_REBUILD_HOUR
from memory_vector_store import MemoryVectorStore
from embedding import EmbeddingService
from middleware import user_memory_manager


class MemoryRebuildTask:
    """记忆向量索引重建任务（in-place 清空重建）"""

    def __init__(self):
        self.store = MemoryVectorStore()
        self.embedder = EmbeddingService()
        self._rebuilding = False
        self._last_rebuilt_at: str | None = None

    def format_profile_text(self, profile: dict) -> str:
        """将用户画像字典格式化为可向量化文本"""
        parts = []
        if profile.get("name"):
            parts.append(f"用户名叫{profile['name']}")
        if profile.get("identity"):
            parts.append(f"身份是{profile['identity']}")
        if profile.get("school"):
            parts.append(f"就读于{profile['school']}")
        if profile.get("major"):
            parts.append(f"专业是{profile['major']}")
        if profile.get("grade"):
            parts.append(f"年级是{profile['grade']}")
        if profile.get("relationship"):
            parts.append(f"{profile['relationship']}")
        if profile.get("interest"):
            parts.append(f"兴趣爱好是{profile['interest']}")
        if profile.get("location"):
            parts.append(f"所在地是{profile['location']}")
        if profile.get("other"):
            parts.append(profile["other"])
        return "，".join(parts) if parts else ""

    def format_summary_text(self, summary_data: dict) -> str:
        """将摘要数据格式化为可向量化文本"""
        summary = summary_data.get("summary", "")
        timestamp = summary_data.get("timestamp", "")
        turn_count = summary_data.get("turn_count", 0)
        return f"[对话摘要] {summary}（{turn_count}轮对话，{timestamp}）"

    def _iterate_summaries(self):
        """从 PostgresStore 遍历所有对话摘要

        ⚠️ LangGraph PostgresStore API 待验证。
        预期返回格式：[{"key": "summary_20260323", "value": {...}}, ...]
        """
        from langgraph.store.postgres import PostgresStore
        from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

        conn_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        results = []
        with PostgresStore.from_conn_string(conn_string) as pg_store:
            pg_store.setup()
            # LangGraph PostgresStore namespace 是 tuple，如 ("memory", "thread_id")
            # 遍历所有 thread_id 下的 summary_* key
            # 实现时需要：
            # 1. 列出所有 namespace 或直接 SQL 查询
            # 2. 按 key 前缀搜索
            # 伪代码（待验证）：
            # for ns in pg_store._list_namespaces():
            #     if ns[0] == "memory":
            #         for item in pg_store.search(ns, query="summary_", limit=100):
            #             results.append({"key": item.key, "value": item.value})
            return results

    def rebuild_index(self) -> dict:
        """执行全量重建索引，返回统计信息"""
        if self._rebuilding:
            return {"status": "skipped", "reason": "rebuild already in progress"}

        self._rebuilding = True
        try:
            store = self.store
            store.init_collection()
            store.delete_all()

            all_vectors = []
            all_metadatas = []

            # 1. 读取并向量化用户画像
            profile = user_memory_manager.load_user_info()
            if profile:
                text = self.format_profile_text(profile)
                if text.strip():
                    vec = self.embedder.get_embeddings([text])[0]
                    all_vectors.append(vec)
                    all_metadatas.append({
                        "memory_type": "profile",
                        "source_key": "user_memory/profile",
                        "text": text,
                        "created_at": profile.get("updated_at", datetime.now().isoformat()),
                    })

            # 2. 读取并向量化对话摘要（PostgresStore API 待验证）
            try:
                for item in self._iterate_summaries():
                    if item.get("key", "").startswith("summary_"):
                        text = self.format_summary_text(item["value"])
                        if text.strip():
                            vec = self.embedder.get_embeddings([text])[0]
                            all_vectors.append(vec)
                            all_metadatas.append({
                                "memory_type": "summary",
                                "source_key": f"memory/{item['key']}",
                                "text": text,
                                "created_at": item["value"].get("timestamp", ""),
                            })
            except Exception as e:
                print(f"[MemoryRebuild] 遍历摘要失败: {e}")

            # 3. 批量插入
            rebuilt_count = 0
            if all_vectors:
                data = [
                    {**meta, "embedding": vec}
                    for meta, vec in zip(all_metadatas, all_vectors)
                ]
                store.insert(data)
                rebuilt_count = len(data)

            self._last_rebuilt_at = datetime.now().isoformat()
            print(f"[MemoryRebuild] 重建完成，共 {rebuilt_count} 条记忆")
            return {"status": "ok", "rebuilt_count": rebuilt_count, "last_rebuilt_at": self._last_rebuilt_at}
        finally:
            self._rebuilding = False

    def is_rebuilding(self) -> bool:
        return self._rebuilding


_memory_rebuild_task: MemoryRebuildTask | None = None


def get_memory_rebuild_task() -> MemoryRebuildTask:
    global _memory_rebuild_task
    if _memory_rebuild_task is None:
        _memory_rebuild_task = MemoryRebuildTask()
    return _memory_rebuild_task


def run_rebuild_in_background():
    """在后台线程执行重建（不阻塞主线程）"""
    def _bg():
        task = get_memory_rebuild_task()
        task.rebuild_index()
    threading.Thread(target=_bg, daemon=True).start()


def _schedule_daily_rebuild():
    """每天凌晨定时重建（使用 threading.Timer 实现）"""
    task = get_memory_rebuild_task()
    task.rebuild_index()

    # 计算距离明天凌晨3点还有多少秒
    now = datetime.now()
    target_hour = MEMORY_REBUILD_HOUR
    next_run = now.replace(hour=target_hour, minute=0, second=0, microsecond=0)
    if now.hour >= target_hour:
        next_run += timedelta(days=1)
    delay_seconds = (next_run - now).total_seconds()

    t = threading.Timer(delay_seconds, _schedule_daily_rebuild)
    t.daemon = True
    t.start()
    print(f"[MemoryRebuild] 已调度下次重建: {next_run.isoformat()}")
