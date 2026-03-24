# 记忆摘要中间件实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用函数式 `@after_model` 钩子实现每 25 轮对话自动总结并存入 Milvus

**Architecture:** 在 `middleware.py` 添加 `memory_summary_hook` 函数，注册到 `agent.py` 的 middleware 列表，替换失效的 `MemorySummaryMiddleware`

**Tech Stack:** LangChain agents, `@after_model` decorator, Milvus, EmbeddingService

---

## 变更文件

- `backend/middleware.py` - 添加 `memory_summary_hook` 函数
- `backend/agent.py` - 注册新中间件，移除旧的 `memory_summary_middleware`

---

## 任务列表

### Task 1: 添加 memory_summary_hook 到 middleware.py

**Files:**
- Modify: `backend/middleware.py`

- [ ] **Step 1: 添加 import**

在 `backend/middleware.py` 顶部添加：
```python
from langgraph.runtime import Runtime
```

- [ ] **Step 2: 添加 TRIGGER_TURNS 常量**

```python
TRIGGER_TURNS = 25  # 每 25 轮触发一次摘要
```

- [ ] **Step 3: 添加辅助函数 `_format_conversation_for_summary`**

在 `MemorySummaryMiddleware` 类之前添加：
```python
def _format_conversation_for_summary(messages, last_n_turns: int = 25) -> str:
    """将消息格式化为可读文本，只保留最近 N 轮用户对话"""
    lines = []
    user_msg_count = 0
    # 从后向前收集，直到收集够 last_n_turns 条用户消息
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
```

- [ ] **Step 4: 添加 LLM 总结函数**

在 `_format_conversation_for_summary` 之后添加：
```python
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
```

- [ ] **Step 5: 添加写入 Milvus 的函数**

```python
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
```

- [ ] **Step 6: 添加 memory_summary_hook 函数**

```python
@after_model
def memory_summary_hook(state: AgentState, runtime: Runtime) -> dict | None:
    """每 N 轮总结对话并写入 Milvus"""
    messages = state.get("messages", [])
    if not messages:
        return None

    # 获取 thread_id
    thread_id = "default"
    if runtime.run_config and runtime.run_config.get("configurable"):
        thread_id = runtime.run_config["configurable"].get("thread_id", "default")

    # 统计用户轮数
    user_turns = sum(1 for msg in messages if isinstance(msg, HumanMessage))

    # 检查是否达到触发轮数
    if user_turns == 0 or user_turns % TRIGGER_TURNS != 0:
        return None

    print(f"[memory_summary_hook] 触发摘要 thread={thread_id} user_turns={user_turns}")

    # 提取最近 25 轮对话
    conversation_text = _format_conversation_for_summary(messages, TRIGGER_TURNS)

    # LLM 总结
    summary_text = _summarize_conversation(conversation_text)

    # 写入 Milvus
    _save_summary_to_milvus(summary_text, thread_id, user_turns)

    return None
```

- [ ] **Step 7: Commit**

```bash
git add backend/middleware.py
git commit -m "feat: 添加 memory_summary_hook 函数式中间件"
```

---

### Task 2: 修改 agent.py 注册新中间件

**Files:**
- Modify: `backend/agent.py`

- [ ] **Step 1: 更新 import**

将：
```python
from middleware import memory_summary_middleware, extract_and_save_user_memory_async, system_prompt_middleware, Context
```

改为：
```python
from middleware import memory_summary_hook, extract_and_save_user_memory_async, system_prompt_middleware, Context
```

- [ ] **Step 2: 更新 middleware 列表**

将：
```python
middleware=[
    SummarizationMiddleware(...),
    memory_summary_middleware,
    system_prompt_middleware,
],
```

改为：
```python
middleware=[
    SummarizationMiddleware(...),
    memory_summary_hook,
    system_prompt_middleware,
],
```

- [ ] **Step 3: Commit**

```bash
git add backend/agent.py
git commit -m "feat: 注册 memory_summary_hook 替换失效的 memory_summary_middleware"
```

---

### Task 3: 验证

**Files:**
- Test: `docker compose up -d` + `uv run uvicorn backend.app:app --reload`

- [ ] **Step 1: 启动服务**

```bash
docker compose up -d
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

- [ ] **Step 2: 测试对话（至少 25 轮）**

与 Agent 对话 25 轮，观察日志是否输出 `[memory_summary_hook] 触发摘要`

- [ ] **Step 3: 检查 Milvus**

通过前端或 API 验证摘要是否写入 Milvus

---

## 验收标准

1. 对话 25 轮后，控制台输出 `触发摘要` 相关日志
2. Milvus `user_memory` collection 中有条 `memory_type="summary"` 的记录
3. 对话 50 轮时，再次触发（验证周期性触发）
