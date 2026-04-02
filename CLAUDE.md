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
- **记忆向量检索** (`middleware.py` system\_prompt\_middleware)：Milvus user\_memory 向量召回，拼接到系统提示词
- **对话摘要记忆** (`middleware.py` MemorySummaryMiddleware)：每25轮自动总结，存入 PostgresStore
- **记忆索引重建** (`memory_tasks.py`)：每日凌晨3点自动 + API 手动触发 in-place 清空重建
- **RAG 管道** (`rag_pipeline.py`)：LangGraph 工作流：retrieve → grade → rewrite → retrieve\_expanded
- **检索** (`rag_utils.py`)：Hybrid Search + Jina Rerank + 三级分块 + Auto-merging

