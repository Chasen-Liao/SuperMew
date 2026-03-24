# 记忆摘要中间件设计方案

## 目标

修复 `MemorySummaryMiddleware` 不生效的问题，改用函数式 `@after_model` 钩子实现每 N 轮对话自动总结并存入 Milvus。

## 现状问题

现有 `MemorySummaryMiddleware` 使用类 + `@after_model` 装饰器，但方法签名错误（`runtime` 写成 `config`），导致中间件静默失效。

## 设计方案

### 实现位置

`backend/middleware.py`

### 核心代码

```python
from langchain.agents.middleware import after_model, AgentState
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

TRIGGER_TURNS = 25  # 可配置

@after_model
def memory_summary_hook(state: AgentState, runtime: Runtime) -> dict | None:
    """每 N 轮总结对话并写入 Milvus"""
    messages = state.get("messages", [])
    if not messages:
        return None

    # 获取 thread_id
    thread_id = runtime.run_config.get("configurable", {}).get("thread_id", "default") if runtime.run_config else "default"

    # 统计用户轮数
    user_turns = sum(1 for msg in messages if isinstance(msg, HumanMessage))

    # 检查是否达到触发轮数
    if user_turns == 0 or user_turns % TRIGGER_TURNS != 0:
        return None

    # 提取最近 25 轮对话
    conversation_text = _format_conversation_for_summary(messages, last_n_turns=TRIGGER_TURNS)

    # LLM 总结
    summary_text = _summarize_conversation(conversation_text)

    # 写入 Milvus
    _save_summary_to_milvus(summary_text, thread_id, user_turns)

    return None
```

### 注册方式

`backend/agent.py` 中的 `create_agent_instance()`：

```python
from middleware import memory_summary_hook

agent = create_agent(
    ...
    middleware=[
        SummarizationMiddleware(...),
        memory_summary_hook,  # 替换原来的 memory_summary_middleware
        system_prompt_middleware,
    ],
)
```

### 依赖模块

- `MemoryVectorStore.insert()` - 写入 Milvus
- `EmbeddingService` - 生成稠密/稀疏向量
- LLM 总结模型（复用 `summary_model` 配置）

## 数据流

1. 用户每轮对话 → agent 处理
2. `after_model` 钩子触发 → 统计用户轮数
3. 达到 25 轮 → 提取最近 25 轮对话文本
4. 调用 LLM 总结
5. 生成稠密 + 稀疏向量
6. 写入 Milvus `user_memory` collection（memory_type="summary"）

## 兼容性

- 保留 `UserMemoryManager` 和 `extract_and_save_user_memory_async`（用户画像提取，走 PostgresStore）
- 新中间件专注对话摘要，走 Milvus
