"""
迁移脚本：将 customer_service_history.json 的会话元数据迁移到 PostgreSQL

用法：uv run python backend/migrate_sessions_to_pg.py
"""

import json
import os
from pathlib import Path
import psycopg
from datetime import datetime
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB


def migrate():
    # 连接 PG
    conn = psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
        autocommit=True,
        prepare_threshold=0,
        row_factory=psycopg.rows.dict_row
    )

    # 读取 JSON 文件
    json_path = Path(__file__).parent.parent / "data" / "customer_service_history.json"
    if not json_path.exists():
        print(f"JSON 文件不存在: {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data:
        print("JSON 文件为空，无需迁移")
        return

    # 迁移数据
    count = 0
    with conn.cursor() as cur:
        for user_id, sessions in data.items():
            for session_id, info in sessions.items():
                updated_at = info.get("updated_at", datetime.now().isoformat())
                # 解析 ISO 格式时间
                try:
                    dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                except Exception:
                    dt = datetime.now()

                cur.execute("""
                    INSERT INTO conversations (user_id, session_id, updated_at, created_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id, session_id) DO NOTHING
                """, (user_id, session_id, dt, dt))
                count += 1

    print(f"迁移完成，共处理 {count} 条会话记录")

    # 备份旧文件
    backup_path = json_path.with_suffix('.json.bak')
    os.rename(json_path, backup_path)
    print(f"已将原文件备份为: {backup_path}")

    conn.close()


if __name__ == "__main__":
    migrate()
