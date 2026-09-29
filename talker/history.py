from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatHistory:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations(
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages(
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS attachments(
                    id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    mime_type TEXT,
                    parsed_text TEXT NOT NULL DEFAULT '',
                    images_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(message_id) REFERENCES messages(id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_attachments_message
                    ON attachments(message_id);
                """
            )

    def new_conversation(self, title: str = "New chat") -> dict:
        conversation_id = uuid.uuid4().hex
        now = _now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (conversation_id, title, now, now),
            )
        return {
            "id": conversation_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def list_conversations(self) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE conversations SET title=?,updated_at=? WHERE id=?",
                (title[:120], _now(), conversation_id),
            )

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> str:
        message_id = uuid.uuid4().hex
        now = _now()
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO messages(id,conversation_id,role,content,created_at) "
                "VALUES(?,?,?,?,?)",
                (message_id, conversation_id, role, content, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?",
                (now, conversation_id),
            )
        return message_id

    def add_attachment(
        self,
        message_id: str,
        *,
        filename: str,
        stored_path: str,
        mime_type: str | None,
        parsed_text: str,
        images: list[str],
    ) -> str:
        attachment_id = uuid.uuid4().hex
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO attachments(
                    id,message_id,filename,stored_path,mime_type,
                    parsed_text,images_json,created_at
                ) VALUES(?,?,?,?,?,?,?,?)
                """,
                (
                    attachment_id,
                    message_id,
                    filename,
                    stored_path,
                    mime_type,
                    parsed_text,
                    json.dumps(images, ensure_ascii=False),
                    _now(),
                ),
            )
        return attachment_id

    def get_messages(self, conversation_id: str) -> list[dict]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at,id",
                (conversation_id,),
            ).fetchall()
            result = []
            for row in rows:
                message = dict(row)
                attachments = conn.execute(
                    "SELECT * FROM attachments WHERE message_id=? ORDER BY created_at,id",
                    (row["id"],),
                ).fetchall()
                message["attachments"] = []
                for attachment in attachments:
                    item = dict(attachment)
                    item["images"] = json.loads(item.pop("images_json") or "[]")
                    message["attachments"].append(item)
                result.append(message)
        return result

    def ensure_conversation(self, conversation_id: str | None) -> str:
        if conversation_id:
            with self.connect() as conn:
                row = conn.execute(
                    "SELECT id FROM conversations WHERE id=?",
                    (conversation_id,),
                ).fetchone()
            if row is not None:
                return conversation_id
        return self.new_conversation()["id"]
