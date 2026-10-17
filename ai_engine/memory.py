"""
Persistent memory manager for Sunnatbek's Multimodal Telegram AI Executive Assistant.
Combines local SQLite persistent storage (works 100% offline and across restarts)
with optional Supabase PostgreSQL + pgvector RAG semantic search.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from supabase import Client, create_client

from config import BASE_DIR, config

logger = logging.getLogger("Userbot.Memory")


class MemoryManager:
    """Manages persistent SQLite memory with optional Supabase pgvector RAG."""

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.openai_client = None
        if config.openai_api_key:
            self.openai_client = openai_client or AsyncOpenAI(
                api_key=config.openai_api_key
            )
        self.supabase: Optional[Client] = None
        self._in_memory_fallback: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        # Initialize persistent SQLite storage
        self.db_dir = BASE_DIR / "data"
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.db_dir / "chat_history.db"
        self._init_sqlite()

        # Initialize Supabase if configured
        self._init_supabase()

    def _init_sqlite(self) -> None:
        """Initialize local persistent SQLite database for conversation history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        chat_id INTEGER NOT NULL,
                        telegram_message_id INTEGER,
                        sender_id INTEGER NOT NULL,
                        sender_name TEXT,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        message_type TEXT DEFAULT 'text',
                        is_owner INTEGER DEFAULT 0,
                        created_at TEXT,
                        metadata TEXT
                    )
                """)
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_chat_msg_unique
                    ON chat_messages(chat_id, telegram_message_id)
                    WHERE telegram_message_id IS NOT NULL
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chat_id_order
                    ON chat_messages(chat_id, id DESC)
                """)
                conn.commit()
            logger.info(f"SQLite persistent chat history database ready at: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}", exc_info=True)

    def _init_supabase(self) -> None:
        """Initialize the Supabase client if credentials are configured."""
        if not config.is_supabase_configured():
            logger.info(
                "Supabase not configured. Using local SQLite persistent database for all chat history."
            )
            return

        try:
            self.supabase = create_client(config.supabase_url, config.supabase_key)
            logger.info("Successfully connected to Supabase pgvector database.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}. Falling back to SQLite.")
            self.supabase = None

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding vector using OpenAI's embedding model."""
        if not self.openai_client or not text or not text.strip():
            return None

        safe_text = text.strip()[:8000]
        try:
            response = await self.openai_client.embeddings.create(
                model=config.openai_embedding_model,
                input=safe_text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return None

    async def store_message(
        self,
        chat_id: int,
        sender_id: int,
        sender_name: str,
        role: str,
        content: str,
        message_type: str = "text",
        is_owner: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        telegram_message_id: Optional[int] = None,
        created_at: Optional[str] = None,
    ) -> bool:
        """
        Store a message permanently into local SQLite database and Supabase (if active).
        Prevents duplicate entries if telegram_message_id already exists for this chat.
        """
        if not content or not content.strip():
            return False

        metadata = metadata or {}
        now_iso = created_at or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. Store into SQLite asynchronously
        def _sqlite_insert():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    if telegram_message_id is not None:
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO chat_messages 
                            (chat_id, telegram_message_id, sender_id, sender_name, role, content, message_type, is_owner, created_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chat_id,
                                telegram_message_id,
                                sender_id,
                                sender_name,
                                role,
                                content.strip(),
                                message_type,
                                1 if is_owner else 0,
                                now_iso,
                                json.dumps(metadata, ensure_ascii=False),
                            ),
                        )
                    else:
                        cursor.execute(
                            """
                            INSERT INTO chat_messages 
                            (chat_id, sender_id, sender_name, role, content, message_type, is_owner, created_at, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                chat_id,
                                sender_id,
                                sender_name,
                                role,
                                content.strip(),
                                message_type,
                                1 if is_owner else 0,
                                now_iso,
                                json.dumps(metadata, ensure_ascii=False),
                            ),
                        )
                    conn.commit()
                return True
            except Exception as sql_err:
                logger.error(f"SQLite store error: {sql_err}")
                return False

        await asyncio.to_thread(_sqlite_insert)

        # 2. In-memory cache update
        record = {
            "chat_id": chat_id,
            "telegram_message_id": telegram_message_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "is_owner": is_owner,
            "role": role,
            "message_type": message_type,
            "content": content.strip(),
            "created_at": now_iso,
            "metadata": metadata,
        }
        self._in_memory_fallback[chat_id].append(record)
        if len(self._in_memory_fallback[chat_id]) > 50:
            self._in_memory_fallback[chat_id].pop(0)

        # 3. Supabase sync if configured
        if self.supabase is not None:
            try:
                embedding = await self.get_embedding(content)
                data_to_insert = {
                    "chat_id": chat_id,
                    "sender_id": sender_id,
                    "sender_name": sender_name,
                    "is_owner": is_owner,
                    "role": role,
                    "message_type": message_type,
                    "content": content.strip(),
                    "metadata": metadata,
                }
                if embedding is not None:
                    data_to_insert["embedding"] = embedding

                def _insert():
                    return (
                        self.supabase.table(config.supabase_table_name)
                        .insert(data_to_insert)
                        .execute()
                    )

                await asyncio.to_thread(_insert)
            except Exception as e:
                logger.debug(f"Supabase store error: {e}")

        return True

    async def update_last_assistant_telegram_id(
        self, chat_id: int, telegram_message_id: int
    ) -> None:
        """Update the telegram_message_id of the most recent assistant message in SQLite."""
        def _update():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        UPDATE chat_messages 
                        SET telegram_message_id = ? 
                        WHERE id = (
                            SELECT id FROM chat_messages 
                            WHERE chat_id = ? AND role = 'assistant' 
                            ORDER BY id DESC LIMIT 1
                        )
                        """,
                        (telegram_message_id, chat_id),
                    )
                    conn.commit()
            except Exception as e:
                logger.debug(f"Update last assistant id error: {e}")

        await asyncio.to_thread(_update)

    async def get_recent_messages(
        self,
        chat_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages in chronological order (oldest to newest)
        from persistent SQLite database (with fallback to Supabase or in-memory).
        """
        limit = limit or config.max_recent_messages

        def _sqlite_fetch():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT id, chat_id, telegram_message_id, sender_id, sender_name, 
                               role, message_type, content, is_owner, created_at, metadata
                        FROM chat_messages
                        WHERE chat_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                        """,
                        (chat_id, limit),
                    )
                    rows = cursor.fetchall()
                    result = []
                    for row in rows:
                        result.append({
                            "id": row["id"],
                            "chat_id": row["chat_id"],
                            "telegram_message_id": row["telegram_message_id"],
                            "sender_id": row["sender_id"],
                            "sender_name": row["sender_name"],
                            "role": row["role"],
                            "message_type": row["message_type"],
                            "content": row["content"],
                            "is_owner": bool(row["is_owner"]),
                            "created_at": row["created_at"],
                        })
                    return list(reversed(result))
            except Exception as sql_err:
                logger.error(f"SQLite fetch error: {sql_err}")
                return []

        sqlite_records = await asyncio.to_thread(_sqlite_fetch)
        if sqlite_records:
            return sqlite_records

        # Fallback to Supabase if SQLite returned empty
        if self.supabase is not None:
            try:
                def _query():
                    return (
                        self.supabase.table(config.supabase_table_name)
                        .select(
                            "id, chat_id, sender_id, sender_name, role, message_type, content, is_owner, created_at"
                        )
                        .eq("chat_id", chat_id)
                        .order("created_at", desc=True)
                        .limit(limit)
                        .execute()
                    )

                response = await asyncio.to_thread(_query)
                records = response.data or []
                return list(reversed(records))
            except Exception as e:
                logger.error(f"Error fetching recent messages from Supabase: {e}")

        # Fallback to in-memory store
        return self._in_memory_fallback[chat_id][-limit:]

    async def retrieve_semantic_context(
        self,
        chat_id: int,
        query_text: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform RAG search: retrieve semantic similar memories using vector cosine similarity.
        """
        if self.supabase is None:
            return []

        top_k = top_k or config.memory_top_k
        threshold = threshold or config.similarity_threshold

        embedding = await self.get_embedding(query_text)
        if not embedding:
            return []

        try:
            def _rpc_call():
                params = {
                    "query_embedding": embedding,
                    "match_threshold": threshold,
                    "match_count": top_k,
                    "filter_chat_id": chat_id,
                }
                return self.supabase.rpc("match_chat_memories", params).execute()

            response = await asyncio.to_thread(_rpc_call)
            results = response.data or []
            logger.debug(f"Retrieved {len(results)} semantic memories for chat {chat_id}.")
            return results
        except Exception as e:
            logger.warning(
                f"RPC 'match_chat_memories' failed or not set up ({e}). "
                "Ensure supabase_schema.sql was executed in Supabase SQL editor."
            )
            return []

    async def clear_chat_memory(self, chat_id: int) -> bool:
        """Clear memories for a given chat ID from SQLite, Supabase, and local cache."""
        self._in_memory_fallback.pop(chat_id, None)

        def _sqlite_delete():
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM chat_messages WHERE chat_id = ?", (chat_id,))
                    conn.commit()
            except Exception as e:
                logger.error(f"SQLite delete error: {e}")

        await asyncio.to_thread(_sqlite_delete)

        if self.supabase is not None:
            try:
                def _delete():
                    return (
                        self.supabase.table(config.supabase_table_name)
                        .delete()
                        .eq("chat_id", chat_id)
                        .execute()
                    )

                await asyncio.to_thread(_delete)
                logger.info(f"Cleared chat memory for chat_id={chat_id} from Supabase.")
            except Exception as e:
                logger.error(f"Error deleting chat memory from Supabase: {e}")
                return False

        return True
