# search_memory 修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 search_memory：添加 RAG 步骤显示，修改 tool description 防止原文暴露给用户

**Architecture:** 修改 `backend/tools.py` 中的 `search_memory` 函数，添加 `emit_rag_step` 调用，并更新 tool description 告知 LLM 不要暴露检索结果

**Tech Stack:** LangChain tools, emit_rag_step, SSE

---

## 变更文件

- `backend/tools.py` - 修改 `search_memory` 函数

---

## 任务列表

### Task 1: 修改 search_memory 函数

**Files:**
- Modify: `backend/tools.py:123-156`

- [ ] **Step 1: 修改 tool description**

将 `@tool("search_memory")` 的 description 改为：
```python
@tool("search_memory")
def search_memory(query: str) -> str:
    """Search the user's past conversation memories using dense+sparse hybrid retrieval (RRF fusion).

    Use this tool when the user asks about something they discussed before,
    wants to recall past conversations, or refers to "what I told you earlier", etc.

    NOTE: The retrieval results are automatically used by the assistant.
    Do NOT reveal the raw retrieval content to the user in your response.
    """
```

- [ ] **Step 2: 添加 emit_rag_step 调用**

在 `search_memory` 函数体内，检索前添加：
```python
    emit_rag_step("🔍", "正在检索记忆...", f"查询: {query[:50]}")
```

在检索完成后添加：
```python
    emit_rag_step("✅", f"记忆检索完成，找到 {len(results)} 条相关记忆")
```

- [ ] **Step 3: Commit**

```bash
git add backend/tools.py
git commit -m "fix: search_memory 添加 RAG 步骤显示并禁止暴露检索结果"
```

---

## 验收标准

1. 触发记忆检索时，前端显示 RAG 步骤气泡（🔍 正在检索记忆...）
2. AI 回复中不包含 "【相关记忆】" 等原始检索内容
