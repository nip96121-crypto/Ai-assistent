"""
Multimodal LLM handler supporting Google Gemini (gemini-2.5-flash / gemini-2.0-flash)
and OpenAI GPT-4o with contextual long-term memory (RAG).
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import logging
from typing import Any, Dict, List, Optional
import zoneinfo

from openai import AsyncOpenAI

from config import config
from .memory import MemoryManager

logger = logging.getLogger("Userbot.LLM")


class LLMHandler:
    """Handles multimodal completions via Google Gemini or OpenAI GPT-4o with contextual memory."""

    def __init__(
        self,
        memory_manager: MemoryManager,
        openai_client: Optional[AsyncOpenAI] = None,
    ):
        self.memory = memory_manager

        # Initialize OpenAI client if configured
        self.openai_client = None
        if config.openai_api_key:
            self.openai_client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)

        # Initialize Google Gemini client if configured
        self.gemini_client = None
        if config.gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=config.gemini_api_key)
                logger.info(f"Initialized Google Gemini Client ({config.gemini_model})")
            except Exception as e:
                logger.warning(f"Failed to initialize Google GenAI client: {e}")

    def _get_current_time_str(self) -> str:
        """Get formatted current time in the configured timezone."""
        try:
            tz = zoneinfo.ZoneInfo(config.timezone)
            now = datetime.datetime.now(tz)
        except Exception:
            now = datetime.datetime.now()
        return now.strftime("%A, %B %d, %Y at %I:%M %p (%Z)")

    def _format_rag_context(self, retrieved_memories: List[Dict[str, Any]]) -> str:
        """Format semantic memories into a prompt block."""
        if not retrieved_memories:
            return ""

        lines = ["--- RELEVANT PAST CONTEXT RETRIEVED FROM LONG-TERM MEMORY ---"]
        for idx, mem in enumerate(retrieved_memories, start=1):
            role = mem.get("role", "user")
            sender = mem.get("sender_name", role)
            content = mem.get("content", "")
            created_at = mem.get("created_at", "")
            time_tag = f" [{created_at}]" if created_at else ""
            lines.append(f"{idx}. {sender}{time_tag}: {content}")
        lines.append("-------------------------------------------------------------")
        return "\n".join(lines)

    async def generate_response(
        self,
        chat_id: int,
        sender_id: int,
        sender_name: str,
        text_content: Optional[str] = None,
        audio_transcript: Optional[str] = None,
        video_frames_b64: Optional[List[str]] = None,
        is_owner: bool = False,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Main pipeline to generate autonomous response using Gemini or GPT-4o.
        Coordinates RAG retrieval, history reconstruction, and multimodal message formatting.
        """
        metadata = metadata or {}

        # Synthesize primary query text for memory retrieval
        query_text = ""
        if text_content:
            query_text += text_content
        if audio_transcript:
            query_text += f" [Transcript]: {audio_transcript}"

        if not query_text and not video_frames_b64:
            return "I received your message, but could not detect any readable content or audio."

        # 1. Fetch relevant long-term RAG memories and recent chat history in parallel
        rag_task = self.memory.retrieve_semantic_context(
            chat_id=chat_id,
            query_text=query_text or "video note message",
            top_k=config.memory_top_k,
            threshold=config.similarity_threshold,
        )
        recent_task = self.memory.get_recent_messages(
            chat_id=chat_id,
            limit=config.max_recent_messages,
        )

        retrieved_memories, recent_messages = await asyncio.gather(rag_task, recent_task)

        # 2. Build system prompt
        base_system_prompt = config.get_base_system_prompt(is_owner_chat=is_owner)
        current_time = self._get_current_time_str()
        rag_context_str = self._format_rag_context(retrieved_memories)

        system_instruction = (
            f"{base_system_prompt}\n\n"
            f"CURRENT TIME: {current_time}\n"
            f"ACTIVE CONTACT: {sender_name} (User ID: {sender_id})\n"
        )
        if rag_context_str:
            system_instruction += f"\n{rag_context_str}\n"

        # 3. Build User modality prompt text
        prompt_text_parts = []
        if message_type == "voice":
            prompt_text_parts.append(
                f"[Voice Message received from {sender_name}]\n"
                f"Audio Transcription:\n\"{audio_transcript}\""
            )
            if text_content:
                prompt_text_parts.append(f"Accompanying Text/Caption: {text_content}")
        elif message_type == "video":
            prompt_text_parts.append(
                f"[Video / Video Note received from {sender_name}]\n"
                f"{len(video_frames_b64 or [])} sampled keyframes are attached below in chronological order."
            )
            if audio_transcript:
                prompt_text_parts.append(f"Audio Transcription from Video: \"{audio_transcript}\"")
            else:
                prompt_text_parts.append("(No spoken audio detected in the video)")
            if text_content:
                prompt_text_parts.append(f"Accompanying Text/Caption: {text_content}")
            prompt_text_parts.append(
                "Please analyze both what is visible in the video frames and any spoken transcript "
                "to provide an intelligent, context-aware reply."
            )
        else:
            prompt_text_parts.append(text_content or "")

        user_prompt_text = "\n\n".join(prompt_text_parts)

        # 4. Dispatch to either Gemini or OpenAI
        use_gemini = (config.ai_provider == "gemini" and self.gemini_client is not None) or (
            self.gemini_client is not None and self.openai_client is None
        )

        clean_reply = ""

        try:
            if use_gemini:
                clean_reply = await self._call_gemini(
                    system_instruction=system_instruction,
                    recent_messages=recent_messages,
                    user_prompt_text=user_prompt_text,
                    video_frames_b64=video_frames_b64,
                )
            else:
                clean_reply = await self._call_openai(
                    system_instruction=system_instruction,
                    recent_messages=recent_messages,
                    user_prompt_text=user_prompt_text,
                    video_frames_b64=video_frames_b64,
                )

            # Apply configurable prefix if not empty and not already present
            final_reply = clean_reply
            if config.reply_prefix and not is_owner:
                if not clean_reply.startswith(config.reply_prefix):
                    final_reply = f"{config.reply_prefix}{clean_reply}"

            # 5. Store interaction into memory asynchronously
            user_stored_content = (
                f"[Voice Transcript]: {audio_transcript}"
                if message_type == "voice"
                else (
                    f"[Video Note Transcript]: {audio_transcript or '(No audio)'} | Text: {text_content or ''}"
                    if message_type == "video"
                    else text_content or ""
                )
            )

            await self.memory.store_message(
                chat_id=chat_id,
                sender_id=sender_id,
                sender_name=sender_name,
                role="user",
                content=user_stored_content,
                message_type=message_type,
                is_owner=is_owner,
                metadata=metadata,
            )

            await self.memory.store_message(
                chat_id=chat_id,
                sender_id=0,
                sender_name=config.assistant_name if is_owner else config.owner_name,
                role="assistant",
                content=clean_reply,
                message_type="text",
                is_owner=False,
                metadata={"reply_to_type": message_type},
            )

            return final_reply

        except Exception as e:
            logger.error(f"AI response generation error: {e}", exc_info=True)
            return (
                f"Kechirasiz, xabaringizni qayta ishlashda vaqtinchalik xatolik yuz berdi. "
                f"{config.owner_name} bu haqda xabardor qilindi."
            )

    async def _call_gemini(
        self,
        system_instruction: str,
        recent_messages: List[Dict[str, Any]],
        user_prompt_text: str,
        video_frames_b64: Optional[List[str]] = None,
    ) -> str:
        """Execute multimodal completion using Google Gemini SDK with automatic model fallback."""
        from google.genai import types

        contents: List[types.Content] = []

        # Reconstruct chat turns
        for msg in recent_messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            sender = msg.get("sender_name", role)
            text = msg.get("content", "")
            prefix = f"[{sender}]: " if role == "user" else ""
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=f"{prefix}{text}")],
                )
            )

        # Current user turn
        current_parts = [types.Part.from_text(text=user_prompt_text)]
        if video_frames_b64:
            for b64 in video_frames_b64:
                try:
                    img_bytes = base64.b64decode(b64)
                    current_parts.append(
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")
                    )
                except Exception as decode_err:
                    logger.debug(f"Failed to decode image frame for Gemini: {decode_err}")

        contents.append(types.Content(role="user", parts=current_parts))

        # Build list of models to try in order of preference
        models_to_try = [config.gemini_model]
        for fallback_m in ["gemini-3.5-flash-lite", "gemini-3.8-flash", "gemini-flash-latest"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        last_error = None
        for model_name in models_to_try:
            try:
                logger.info(f"Calling Google Gemini ({model_name})...")
                response = await self.gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                        max_output_tokens=1000,
                    ),
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini model '{model_name}' failed: {e}. Trying fallback...")
                last_error = e

        if last_error:
            raise last_error
        return ""

    async def _call_openai(
        self,
        system_instruction: str,
        recent_messages: List[Dict[str, Any]],
        user_prompt_text: str,
        video_frames_b64: Optional[List[str]] = None,
    ) -> str:
        """Execute multimodal completion using OpenAI GPT-4o."""
        if self.openai_client is None:
            raise ValueError("OpenAI client is not configured.")

        messages_payload: List[Dict[str, Any]] = [
            {"role": "system", "content": system_instruction}
        ]

        for msg in recent_messages:
            msg_role = "assistant" if msg.get("role") == "assistant" else "user"
            msg_sender = msg.get("sender_name", msg_role)
            msg_content = msg.get("content", "")
            prefix = f"[{msg_sender}]: " if msg_role == "user" else ""
            messages_payload.append({
                "role": msg_role,
                "content": f"{prefix}{msg_content}",
            })

        content_items: List[Dict[str, Any]] = [
            {"type": "text", "text": user_prompt_text}
        ]

        if video_frames_b64:
            for b64_frame in video_frames_b64:
                content_items.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_frame}",
                        "detail": "low",
                    },
                })

        messages_payload.append({
            "role": "user",
            "content": content_items,
        })

        logger.info(f"Calling OpenAI ({config.openai_model})...")
        completion = await self.openai_client.chat.completions.create(
            model=config.openai_model,
            messages=messages_payload,
            temperature=0.7,
            max_tokens=1000,
        )
        raw_reply = completion.choices[0].message.content or ""
        return raw_reply.strip()
