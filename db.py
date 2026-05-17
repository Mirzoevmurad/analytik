"""SQLite persistence: история чатов и настройки пользователей."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    INTEGER PRIMARY KEY,
                username   TEXT,
                first_name TEXT,
                last_name  TEXT,
                system_prompt TEXT DEFAULT NULL,
                voice_responses INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    INTEGER NOT NULL,
                role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content    TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );
            CREATE INDEX IF NOT EXISTS idx_messages_user
                ON messages(user_id, id DESC);
        """)
        self._conn.commit()

    def upsert_user(
        self,
        user_id: int,
        username: str | None,
        first_name: str | None,
        last_name: str | None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO users (user_id, username, first_name, last_name)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                   username=excluded.username,
                   first_name=excluded.first_name,
                   last_name=excluded.last_name""",
            (user_id, username, first_name, last_name),
        )
        self._conn.commit()

    def add_message(self, user_id: int, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
            (user_id, role, content),
        )
        self._conn.commit()

    def get_context(self, user_id: int, limit: int = 20) -> list[dict[str, str]]:
        """Возвращает последние N сообщений как [{role, content}, ...]."""
        rows = self._conn.execute(
            """SELECT role, content FROM (
                   SELECT role, content, id FROM messages
                   WHERE user_id = ?
                   ORDER BY id DESC
                   LIMIT ?
               ) sub ORDER BY id ASC""",
            (user_id, limit),
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in rows]

    def clear_context(self, user_id: int) -> int:
        """Удаляет всю историю сообщений пользователя. Возвращает кол-во удалённых."""
        cur = self._conn.execute(
            "DELETE FROM messages WHERE user_id = ?", (user_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def get_system_prompt(self, user_id: int) -> str | None:
        row = self._conn.execute(
            "SELECT system_prompt FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else None

    def set_system_prompt(self, user_id: int, prompt: str | None) -> None:
        self._conn.execute(
            "UPDATE users SET system_prompt = ? WHERE user_id = ?",
            (prompt, user_id),
        )
        self._conn.commit()

    def get_voice_responses(self, user_id: int) -> bool:
        row = self._conn.execute(
            "SELECT voice_responses FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        return bool(row[0]) if row else False

    def set_voice_responses(self, user_id: int, enabled: bool) -> None:
        self._conn.execute(
            "UPDATE users SET voice_responses = ? WHERE user_id = ?",
            (1 if enabled else 0, user_id),
        )
        self._conn.commit()

    def export_history(self, user_id: int) -> list[dict[str, str]]:
        """Возвращает всю историю сообщений для экспорта."""
        rows = self._conn.execute(
            "SELECT role, content, created_at FROM messages WHERE user_id = ? ORDER BY id ASC",
            (user_id,),
        ).fetchall()
        return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in rows]

    def message_count(self, user_id: int) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE user_id = ?", (user_id,)
        ).fetchone()
        return row[0] if row else 0
