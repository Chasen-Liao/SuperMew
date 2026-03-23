# 记忆检索能力增强 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为记忆系统增加语义检索能力，每次对话时从 Milvus 向量库动态检索 Top-3 相关记忆拼入提示词，而非一股脑拼接全部记忆。

**Architecture:** PostgresStore 作为 source of truth，定期全量重建 Milvus 向量索引；读取时仅查 Milvus 语义检索。用户画像和对话摘要分别格式化文本后向量化存入同一个 collection。

**Tech Stack:** Milvus (pymilvus)、Jina Embedding API、APScheduler（定时任务）、PostgresStore（已有）

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `backend/memory_vector_store.py` | 新增：Milvus 向量存储操作（collection 管理、检索、重建） |
| `backend/memory_tasks.py` | 新增：定时重建任务（每天凌晨 3 点） |
| `backend/agent.py:113-118` | 修改：`build_system_message()` 增加向量检索 + 降级回退 |
| `backend/config.py:51` | 修改：新增 5 个配置项 |
| `backend/api.py:271` | 修改：新增 `/api/memory/rebuild` 接口 |

> **注意：** `backend/tasks.py` 改为 `backend/memory_tasks.py`，避免与 `backend/` 下其他 task 相关文件混淆。

---

## Task 1: 配置项

**Files:**
- Modify: `backend/config.py:51`（文件末尾添加）

- [ ] **Step 1: 添加配置项**

在 `config.py` 末尾添加：

```python
# ===== 记忆向量库 =====
MEMORY_MILVUS_COLLECTION = os.getenv("MEMORY_MILVUS_COLLECTION", "user_memory")
MEMORY_TOP_K = 5           # 向量库检索候选数
MEMORY_RECALL_LIMIT = 3    # 最终召回使用数
MEMORY_TOKEN_LIMIT = 500   # 记忆文本 token 上限

# 重建策略
MEMORY_REBUILD_HOUR = 3    # 每天凌晨 3 点重建
```

- [ ] **Step 2: 提交**

```bash
git add backend/config.py
git commit -m "feat: 添加记忆向量库配置项"
```

---

## Task 2: memory_vector_store.py — Milvus 向量存储模块

**Files:**
- Create: `backend/memory_vector_store.py`
- Test: `tests/backend/test_memory_vector_store.py`

### 2.1 模块接口设计

```python
class MemoryVectorStore:
    """记忆向量存储 — 管理 user_memory collection"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        embedding_service: EmbeddingService = None,
    ): ...

    def init_collection(self): ...         # 初始化 collection（幂等）
    def insert(self, data: list[dict]): ...  # 插入记忆向量
    def search(self, query_vector: list[float], top_k: int) -> list[dict]: ...  # 语义检索
    def delete_all(self): ...              # 清空所有记忆
    def collection_exists(self) -> bool: ...  # 检查 collection 是否存在
    def drop_collection(self): ...          # 删除 collection
```

### 2.2 Collection Schema

| Field | Type | Dim/Config |
|-------|------|------------|
| `id` | int64 | auto_id, primary_key |
| `memory_type` | varchar | max_length=32（`profile` 或 `summary`） |
| `source_key` | varchar | max_length=512 |
| `text` | varchar | max_length=2000 |
| `embedding` | FLOAT_VECTOR | dim=1024（Jina v3） |
| `created_at` | datetime | - |

### 2.3 实现步骤

- [ ] **Step 1: 编写测试**

```python
# tests/backend/test_memory_vector_store.py
import pytest
from backend.memory_vector_store import MemoryVectorStore
from backend.embedding import EmbeddingService

def test_memory_vector_store_init_and_search(milvus_client):
    """测试初始化 collection 和语义检索"""
    store = MemoryVectorStore()
    store.init_collection()

    # 插入测试数据
    test_data = [
        {
            "memory_type": "profile",
            "source_key": "user_memory/profile",
            "text": "用户名叫张三，在清华大学读计算机系大三",
            "embedding": [0.1] * 1024,
            "created_at": "2026-03-23T00:00:00",
        },
        {
            "memory_type": "summary",
            "source_key": "memory/session1/summary_20260323",
            "text": "用户讨论了 RAG 系统的架构设计",
            "embedding": [0.2] * 1024,
            "created_at": "2026-03-23T01:00:00",
        },
    ]
    store.insert(test_data)

    # 检索
    query_vec = [0.1] * 1024
    results = store.search(query_vec, top_k=2)
    assert len(results) <= 2
    assert all("text" in r for r in results)
    assert all("memory_type" in r for r in results)
```

- [ ] **Step 2: 运行测试验证失败（需要先实现）**

Run: `pytest tests/backend/test_memory_vector_store.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: 实现 memory_vector_store.py**

```python
# backend/memory_vector_store.py
"""记忆向量存储 — 管理 user_memory collection 的创建、检索和重建"""
from datetime import datetime
from pymilvus import MilvusClient, DataType
from config import MILVUS_HOST, MILVUS_PORT, MEMORY_MILVUS_COLLECTION, MEMORY_TOP_K


class MemoryVectorStore:
    """记忆向量存储"""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        collection_name: str = None,
        embedding_service=None,
    ):
        self.host = host or MILVUS_HOST
        self.port = port or MILVUS_PORT
        self.collection_name = collection_name or MEMORY_MILVUS_COLLECTION
        self.embedding_service = embedding_service
        self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")

    def init_collection(self, dense_dim: int = 1024):
        """初始化 user_memory collection（幂等）"""
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("memory_type", DataType.VARCHAR, max_length=32)
        schema.add_field("source_key", DataType.VARCHAR, max_length=512)
        schema.add_field("text", DataType.VARCHAR, max_length=2000)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dense_dim)
        schema.add_field("created_at", DataType.VARCHAR, max_length=64)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def insert(self, data: list[dict]):
        """插入记忆向量"""
        return self.client.insert(self.collection_name, data)

    def search(
        self, query_vector: list[float], top_k: int = MEMORY_TOP_K
    ) -> list[dict]:
        """语义检索记忆"""
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="embedding",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=["memory_type", "source_key", "text", "created_at"],
        )
        formatted = []
        for hits in results:
            for hit in hits:
                formatted.append({
                    "id": hit.get("id"),
                    "memory_type": hit.get("memory_type", ""),
                    "source_key": hit.get("source_key", ""),
                    "text": hit.get("text", ""),
                    "created_at": hit.get("created_at", ""),
                    "score": hit.get("distance", 0.0),
                })
        return formatted

    def delete_all(self):
        """清空所有记忆（用于重建前）"""
        if self.client.has_collection(self.collection_name):
            self.client.delete(self.collection_name, filter="id >= 0")

    def collection_exists(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def drop_collection(self):
        """删除 collection"""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/backend/test_memory_vector_store.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/memory_vector_store.py tests/backend/test_memory_vector_store.py
git commit -m "feat: 添加记忆向量存储模块 memory_vector_store"
```

---

## Task 3: 修改 agent.py — 集成记忆检索

**Files:**
- Modify: `backend/agent.py:113-118`（`build_system_message` 函数）

### 3.1 修改 `build_system_message()`

**原逻辑（直接拼接）：**
```python
def build_system_message() -> str:
    memory = load_user_memory_for_prompt()  # 从 PostgresStore 读取
    if memory:
        return f"{_SOUL_PROMPT}\n\n{memory}"
    return _SOUL_PROMPT
```

**新逻辑（语义检索）：**
1. 从 PostgresStore 读取原始记忆
2. 如果有记忆，用 Jina 向量化后去 Milvus 检索 Top-5
3. 取 Top-3，拼入提示词
4. Milvus 不可用时降级回退到原始拼接

### 3.2 实现步骤

- [ ] **Step 1: 添加导入和全局实例**

在 `backend/agent.py` 文件顶部（其他导入附近）添加：

```python
from memory_vector_store import MemoryVectorStore
from embedding import EmbeddingService
```

在 `_SOUL_PROMPT` 定义下方添加：

```python
_memory_vector_store = None
_embedding_service_for_memory = None

def _get_memory_vector_store():
    global _memory_vector_store
    if _memory_vector_store is None:
        _memory_vector_store = MemoryVectorStore()
        _memory_vector_store.init_collection()
    return _memory_vector_store

def _get_embedding_service():
    global _embedding_service_for_memory
    if _embedding_service_for_memory is None:
        _embedding_service_for_memory = EmbeddingService()
    return _embedding_service_for_memory
```

- [ ] **Step 2: 修改 build_system_message()**

将原函数替换为：

```python
def build_system_message(user_text: str = "") -> str:
    """构建完整的系统提示词（soul.md + 检索到的记忆）"""
    from middleware import load_user_memory_for_prompt

    memory_text = load_user_memory_for_prompt()

    # 如果没有记忆，直接使用 soul.md
    if not memory_text:
        return _SOUL_PROMPT

    # 如果有记忆，尝试向量检索
    try:
        store = _get_memory_vector_store()
        embedder = _get_embedding_service()

        # 生成查询向量
        query_vec = embedder.get_embeddings([user_text])[0]

        # Milvus 检索 Top-5，取 Top-3
        candidates = store.search(query_vec, top_k=5)

        if candidates:
            # 按相关性得分排序，取 Top-3
            top_memories = candidates[:3]
            # 格式化记忆文本
            memory_lines = []
            for m in top_memories:
                mem_type = m.get("memory_type", "")
                text = m.get("text", "")
                if text:
                    memory_lines.append(f"[{mem_type}] {text}")
            if memory_lines:
                # 控制 token 上限
                combined = "\n".join(memory_lines)
                if len(combined) > MEMORY_TOKEN_LIMIT:
                    combined = combined[:MEMORY_TOKEN_LIMIT]
                return f"{_SOUL_PROMPT}\n\n【相关记忆】\n{combined}"

        # Milvus 无数据，降级回退到原始拼接
        return f"{_SOUL_PROMPT}\n\n{memory_text}"

    except Exception as e:
        # 任何异常都降级回退
        print(f"[build_system_message] 记忆检索失败，降级回退: {e}")
        return f"{_SOUL_PROMPT}\n\n{memory_text}"
```

- [ ] **Step 3: 在 chat_with_agent 和 chat_with_agent_stream 中传入 user_text**

找到 `chat_with_agent` 函数中的调用：

```python
# 原
combined_message = f"{system_content}\n\n用户问题：{user_text}"

# 改
combined_message = f"{system_content}\n\n用户问题：{user_text}"
```

`build_system_message()` 现在需要接收 `user_text` 参数。修改两处调用：

在 `chat_with_agent`（约 line 179）：
```python
system_content = build_system_message(user_text=user_text)
```

在 `chat_with_agent_stream` 里的 `_agent_sync_stream`（约 line 269）：
```python
system_content = build_system_message(user_text=user_text)
```

- [ ] **Step 4: 提交**

```bash
git add backend/agent.py
git commit -m "feat: 集成记忆向量检索到 build_system_message"
```

---

## Task 4: memory_tasks.py — 定时重建任务

**Files:**
- Create: `backend/memory_tasks.py`
- Modify: `backend/app.py`（注册启动事件）

### 4.1 模块设计

```python
class MemoryRebuildTask:
    """记忆向量索引重建任务"""

    def __init__(self, store: MemoryVectorStore, embedding_service, user_memory_manager): ...

    def rebuild_index(self): ...      # 核心重建逻辑
    def rebuild_if_needed(self): ...  # 检查并执行重建
    def format_profile_text(self, profile: dict) -> str: ...   # 格式化用户画像
    def format_summary_text(self, summary: dict) -> str: ...   # 格式化摘要
```

### 4.2 实现步骤

- [ ] **Step 1: 实现 memory_tasks.py**

```python
# backend/memory_tasks.py
"""记忆向量索引定时重建任务"""
import threading
from datetime import datetime
from config import MEMORY_REBUILD_HOUR, MEMORY_MILVUS_COLLECTION
from memory_vector_store import MemoryVectorStore
from embedding import EmbeddingService
from middleware import user_memory_manager


class MemoryRebuildTask:
    """记忆向量索引重建任务（双 collection 交替策略）"""

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

    def rebuild_index(self) -> dict:
        """执行全量重建索引，返回统计信息"""
        if self._rebuilding:
            return {"status": "skipped", "reason": "rebuild already in progress"}

        self._rebuilding = True
        try:
            store = self.store
            store.init_collection()
            store.delete_all()  # 清空旧数据

            all_texts = []
            all_vectors = []
            all_metadatas = []

            # 1. 读取并向量化用户画像
            profile = user_memory_manager.load_user_info()
            if profile:
                text = self.format_profile_text(profile)
                if text.strip():
                    vec = self.embedder.get_embeddings([text])[0]
                    all_texts.append(text)
                    all_vectors.append(vec)
                    all_metadatas.append({
                        "memory_type": "profile",
                        "source_key": "user_memory/profile",
                        "text": text,
                        "created_at": profile.get("updated_at", datetime.now().isoformat()),
                    })

            # 2. 读取并向量化对话摘要（从 PostgresStore 遍历所有 namespace）
            from langgraph.store.postgres import PostgresStore
            from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB

            conn_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
            try:
                with PostgresStore.from_conn_string(conn_string) as pg_store:
                    pg_store.setup()
                    # 遍历 memory namespace 下的所有摘要
                    try:
                        all_items = pg_store.search(["memory", ""], query="", limit=1000)
                        for item in all_items:
                            if item.key.startswith("summary_"):
                                text = self.format_summary_text(item.value)
                                if text.strip():
                                    vec = self.embedder.get_embeddings([text])[0]
                                    all_texts.append(text)
                                    all_vectors.append(vec)
                                    all_metadatas.append({
                                        "memory_type": "summary",
                                        "source_key": f"memory/{item.key}",
                                        "text": text,
                                        "created_at": item.value.get("timestamp", ""),
                                    })
                    except Exception:
                        pass  # 没有摘要时忽略
            except Exception as e:
                print(f"[MemoryRebuild] PostgresStore 连接失败: {e}")

            # 3. 批量插入 Milvus
            rebuilt_count = 0
            if all_vectors:
                data = [
                    {
                        **meta,
                        "embedding": vec,
                    }
                    for meta, vec in zip(all_metadatas, all_vectors)
                ]
                store.insert(data)
                rebuilt_count = len(data)

            self._last_rebuilt_at = datetime.now().isoformat()
            print(f"[MemoryRebuild] 重建完成，共 {rebuilt_count} 条记忆")

            return {
                "status": "ok",
                "rebuilt_count": rebuilt_count,
                "last_rebuilt_at": self._last_rebuilt_at,
            }
        finally:
            self._rebuilding = False

    def is_rebuilding(self) -> bool:
        return self._rebuilding


# 全局单例
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
    t = threading.Thread(target=_bg, daemon=True)
    t.start()
```

> **注意：** `PostgresStore.search` API 需要确认。上面的遍历方式可能需要根据实际 API 调整。实际实现时需要检查 LangGraph PostgresStore 的 search 语法。

- [ ] **Step 2: 注册定时任务**

在 `backend/app.py` 中，在 `create_app()` 末尾添加启动事件：

```python
# 启动后执行一次记忆索引重建（后台）
@app.on_event("startup")
async def startup_memory_rebuild():
    from memory_tasks import run_rebuild_in_background
    run_rebuild_in_background()
```

同时在 `backend/app.py` 顶部添加导入：

```python
import threading
from memory_tasks import get_memory_rebuild_task, MemoryRebuildTask
from memory_vector_store import MemoryVectorStore
```

并添加定时调度（在 startup 事件中设置）：

```python
import schedulers  # 需要确认项目中是否有 scheduler 库

# 在 startup_memory_rebuild 函数中添加：
import time
scheduler = schedulers.ThreadScheduler()
def _daily_rebuild():
    task = get_memory_rebuild_task()
    task.rebuild_index()

# 每天凌晨3点执行
scheduler.repeat(_daily_rebuild, hour=MEMORY_REBUILD_HOUR, minute=0)
scheduler.start()
```

> **备选方案（如果项目没有 schedulers 库）：** 使用 APScheduler，在 `requirements.txt` 中添加 `apscheduler`，或在 `app.py` 中用 `threading.Timer` 实现简单定时。

- [ ] **Step 3: 提交**

```bash
git add backend/memory_tasks.py backend/app.py
git commit -m "feat: 添加记忆索引定时重建任务"
```

---

## Task 5: API 接口 — POST /api/memory/rebuild

**Files:**
- Modify: `backend/api.py`（在文件末尾添加路由）

### 5.1 实现步骤

- [ ] **Step 1: 添加 API 路由**

在 `backend/api.py` 末尾添加：

```python
@router.post("/memory/rebuild")
async def rebuild_memory_index():
    """手动触发记忆向量索引重建"""
    try:
        from memory_tasks import get_memory_rebuild_task
        task = get_memory_rebuild_task()

        if task.is_rebuilding():
            raise HTTPException(status_code=400, detail="rebuild already in progress")

        result = task.rebuild_index()
        return {
            "status": result.get("status"),
            "rebuilt_count": result.get("rebuilt_count", 0),
            "last_rebuilt_at": result.get("last_rebuilt_at"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 2: 提交**

```bash
git add backend/api.py
git commit -m "feat: 添加记忆索引重建 API 接口 POST /api/memory/rebuild"
```

---

## Task 6: 端到端测试

- [ ] **Step 1: 验证 collection 创建和数据插入**

启动服务后，调用重建 API：
```bash
curl -X POST http://127.0.0.1:8000/api/memory/rebuild
```

预期：`{"status": "ok", "rebuilt_count": N, "last_rebuilt_at": "..."}`

- [ ] **Step 2: 验证对话时记忆检索**

发送对话请求：
```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我之前说的那个项目叫什么来着？", "user_id": "test", "session_id": "test"}'
```

检查返回的 response 中是否只包含检索到的相关记忆而非全部记忆。

- [ ] **Step 3: 验证降级回退**

停止 Milvus 服务，发送对话请求，验证是否降级回退到直接拼接 PostgresStore 记忆（服务不应报错）。

---

## 实施顺序

1. **Task 1** — 配置项（无依赖）
2. **Task 2** — `memory_vector_store.py`（无依赖）
3. **Task 3** — 修改 `agent.py`（依赖 Task 2）
4. **Task 4** — `memory_tasks.py`（依赖 Task 2）
5. **Task 5** — API（依赖 Task 4）
6. **Task 6** — 端到端测试（依赖 Task 1-5）

---

## 附录：关键依赖确认

| 依赖 | 确认方式 |
|------|---------|
| `apscheduler` | 检查 `requirements.txt` 是否已有，没有则需添加 |
| `pymilvus` | 已有（在 `milvus_client.py` 使用） |
| LangGraph `PostgresStore.search` API | 需确认 search 的 namespace 和返回格式 |
| Jina Embedding dim | 当前 `embedding.py` 用的是 Qwen embedding，dim 需确认为 1024 还是其他值 |
