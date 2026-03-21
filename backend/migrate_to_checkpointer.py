"""
迁移脚本：将 customer_service_history.json 中的历史数据迁移到 checkpointer

使用方法：
    python migrate_to_checkpointer.py [--dry-run]

注意：InMemorySaver 是内存存储，重启后会丢失。
如需持久化，请改用 SqliteSaver 或其他持久化 checkpointer。
"""

import json
import os
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 将 backend 目录添加到 path
sys.path.insert(0, str(Path(__file__).parent))

from langgraph.checkpoint.memory import InMemorySaver

# 创建 checkpointer 实例（与 agent.py 中相同）
checkpointer = InMemorySaver()


def load_history_json():
    """加载历史数据"""
    package_root = Path(__file__).parent
    data_dir = package_root / "data"
    storage_path = data_dir / "customer_service_history.json"

    if not os.path.exists(storage_path):
        print(f"历史文件不存在: {storage_path}")
        return {}

    with open(storage_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def deserialize_message(msg_data: dict):
    """反序列化消息"""
    msg_type = msg_data.get("type", "")
    content = msg_data.get("content", "")

    if msg_type == "human":
        return HumanMessage(content=content)
    elif msg_type == "ai":
        return AIMessage(content=content)
    elif msg_type == "system":
        return SystemMessage(content=content)
    else:
        # 默认当作 AI 消息
        return AIMessage(content=content)


def migrate_history(dry_run: bool = True):
    """迁移历史数据到 checkpointer"""
    data = load_history_json()

    if not data:
        print("没有历史数据需要迁移")
        return

    total_users = len(data)
    total_sessions = sum(len(sessions) for sessions in data.values())
    total_messages = sum(
        sum(len(s.get("messages", [])) for s in sessions.values())
        for sessions in data.values()
    )

    print(f"发现历史数据:")
    print(f"  - 用户数: {total_users}")
    print(f"  - 会话数: {total_sessions}")
    print(f"  - 消息数: {total_messages}")
    print()

    if dry_run:
        print("[DRY RUN] 以下数据将被迁移：")
    else:
        print("[MIGRATING] 开始迁移...")

    migrated_sessions = 0
    migrated_messages = 0

    for user_id, sessions in data.items():
        for session_id, session_data in sessions.items():
            messages = session_data.get("messages", [])
            if not messages:
                continue

            thread_id = f"{user_id}_{session_id}"
            config = {"configurable": {"thread_id": thread_id}}

            # 反序列化消息
            deserialized_messages = [deserialize_message(m) for m in messages]

            if dry_run:
                print(f"  - {thread_id}: {len(deserialized_messages)} 条消息")
            else:
                # 构建 checkpoint 格式并存储
                checkpoint = {
                    "values": {"messages": deserialized_messages},
                    "next_tasks": [],
                    "parent_config": None,
                }
                checkpointer.put(config, checkpoint)
                print(f"  迁移: {thread_id} ({len(deserialized_messages)} 条消息)")

            migrated_sessions += 1
            migrated_messages += len(deserialized_messages)

    print()
    print(f"迁移完成:")
    print(f"  - 会话数: {migrated_sessions}")
    print(f"  - 消息数: {migrated_messages}")

    if dry_run:
        print()
        print("提示：使用 --dry-run 参数查看迁移计划")
        print("      移除 --dry-run 参数执行实际迁移")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate_history(dry_run=dry_run)
