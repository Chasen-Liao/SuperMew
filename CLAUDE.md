# CLAUDE.md

## 常用命令

```bash
# 安装依赖
uv sync

# 启动服务（PostgreSQL + Milvus）
docker compose up -d

# 启动后端
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

## 项目架构

```
SuperMew/
├── backend/
│   ├── app.py              # FastAPI 入口、定时任务调度
│   ├── api.py             # 聊天/会话/文档/记忆重建 API
│   ├── agent.py           # Agent + PostgresSaver checkpointer
│   ├── middleware.py      # UserMemoryManager + MemorySummaryMiddleware + system_prompt_middleware
│   ├── memory_vector_store.py  # Milvus user_memory collection 管理
│   ├── memory_tasks.py    # 记忆索引定时重建任务
│   ├── config.py          # 环境变量
│   ├── schemas.py         # Pydantic 模型
│   ├── tools.py           # 工具函数
│   ├── rag_pipeline.py    # LangGraph RAG 工作流
│   ├── rag_utils.py       # 检索、查询重写、HyDE
│   ├── embedding.py       # 稠密 + BM25 稀疏向量
│   ├── milvus_client.py   # Milvus 混合检索
│   ├── milvus_writer.py   # 向量写入
│   ├── document_loader.py  # PDF/Word 分块
│   ├── parent_chunk_store.py  # 父级分块 DocStore
│   ├── migrate_to_checkpointer.py  # 旧数据迁移
│   └── soul/soul.md       # Agent 系统提示词
├── frontend/
├── data/
└── docker-compose.yml     # PostgreSQL + Milvus + Attu
```

## 核心模块

- **Agent 记忆** (`agent.py`)：PostgresSaver checkpointer 持久化会话到 PostgreSQL
- **用户画像记忆** (`middleware.py` UserMemoryManager)：LLM 提取 + PostgresStore 持久化
- **记忆向量检索** (`middleware.py` system_prompt_middleware)：Milvus user_memory 向量召回，拼接到系统提示词
- **对话摘要记忆** (`middleware.py` MemorySummaryMiddleware)：每25轮自动总结，存入 PostgresStore
- **记忆索引重建** (`memory_tasks.py`)：每日凌晨3点自动 + API 手动触发 in-place 清空重建
- **RAG 管道** (`rag_pipeline.py`)：LangGraph 工作流：retrieve → grade → rewrite → retrieve_expanded
- **检索** (`rag_utils.py`)：Hybrid Search + Jina Rerank + 三级分块 + Auto-merging

## 调试记录

### RAG 实时步骤显示
- `emit_rag_step()` 直接用 `queue.Queue.put_nowait()`，不用 `call_soon_threadsafe`
- `agent.stream()` 在工具执行期间阻塞，RAG 步骤必须在后台线程同步写入队列

### create_agent system prompt 冲突
- 不要手动添加 `SystemMessage`，会导致 `"System message must be at the beginning"` 错误
- 改用：将系统提示词作为文本前缀拼接到用户消息中

### PostgresStore 用法
- `PostgresStore.from_conn_string()` 是 context manager，用 `with` 语句

### 用户记忆 LLM 提取
- `save_user_info()` 必须用合并模式，否则空字段会覆盖已有值

### 启动命令
```bash
uv sync                      # 安装依赖
docker compose up -d        # 启动 PostgreSQL + Milvus
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
