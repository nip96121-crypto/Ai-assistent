"""
Supabase PostgreSQL + pgvector integration for long-term memory and RAG.
Stores chat history with vector embeddings and retrieves semantic context.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI
from supabase import Client, create_client

from config import config

logger = logging.getLogger("Userbot.Memory")


class MemoryManager:
    """Manages long-term vector memory (Supabase + pgvector) with in-memory fallback."""

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.openai_client = None
        if config.openai_api_key:
            self.openai_client = openai_client or AsyncOpenAI(
                api_key=config.openai_api_key
            )
        self.supabase: Optional[Client] = None
        self._in_memory_fallback: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._init_supabase()

    def _init_supabase(self) -> None:
        """Initialize the Supabase client if credentials are configured."""
        if not config.is_supabase_configured():
            logger.warning(
                "Supabase URL or Key not set or using placeholders. "
                "Running in fallback in-memory mode. Set SUPABASE_URL and SUPABASE_KEY in .env for persistent RAG."
            )
            return

        try:
            self.supabase = create_client(config.supabase_url, config.supabase_key)
            logger.info("Successfully connected to Supabase pgvector database.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}. Falling back to in-memory store.")
            self.supabase = None

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate text embedding vector using OpenAI's embedding model."""
        if not self.openai_client or not text or not text.strip():
            return None

        # Truncate text if excessively long to avoid token limits
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
    ) -> bool:
        """
        Store a message into Supabase with its vector embedding.
        If Supabase is unavailable, stores in local in-memory fallback.
        """
        if not content or not content.strip():
            return False

        metadata = metadata or {}
        embedding: Optional[List[float]] = None

        # Only generate embeddings if Supabase is active
        if self.supabase is not None:
            embedding = await self.get_embedding(content)

        record = {
            "chat_id": chat_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "is_owner": is_owner,
            "role": role,
            "message_type": message_type,
            "content": content,
            "metadata": metadata,
        }

        # Local in-memory backup always updated for instant session recall
        self._in_memory_fallback[chat_id].append(
            {**record, "created_at": None}
        )
        if len(self._in_memory_fallback[chat_id]) > 50:
            self._in_memory_fallback[chat_id].pop(0)

        if self.supabase is None:
            return True

        # Store in Supabase asynchronously
        try:
            data_to_insert = {**record}
            if embedding is not None:
                data_to_insert["embedding"] = embedding

            def _insert():
                return (
                    self.supabase.table(config.supabase_table_name)
                    .insert(data_to_insert)
                    .execute()
                )

            await asyncio.to_thread(_insert)
            logger.debug(f"Stored message [{role}] from {sender_name} ({chat_id}) into Supabase.")
            return True
        except Exception as e:
            logger.error(f"Error inserting message to Supabase: {e}")
            return False

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

    async def get_recent_messages(
        self,
        chat_id: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent messages in chronological order to supply direct context window.
        """
        limit = limit or config.max_recent_messages

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
                # Reverse to maintain chronological order (oldest -> newest)
                return list(reversed(records))
            except Exception as e:
                logger.error(f"Error fetching recent messages from Supabase: {e}")

        # Fallback to in-memory store
        return self._in_memory_fallback[chat_id][-limit:]

    async def clear_chat_memory(self, chat_id: int) -> bool:
        """Clear memories for a given chat ID from Supabase and local cache."""
        self._in_memory_fallback.pop(chat_id, None)

        if self.supabase is None:
            return True

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
            return True
        except Exception as e:
            logger.error(f"Error deleting chat memory from Supabase: {e}")
            return False
