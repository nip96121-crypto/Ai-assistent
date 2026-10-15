"""
Telegram MTProto helper utilities:
Simulates live chat actions ('Typing...', 'Recording voice...') and safe chunked message dispatch.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, Union

from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.custom import Message

logger = logging.getLogger("Userbot.TelegramHelpers")


class TelegramChatAction:
    """
    Asynchronous context manager that maintains an active Telegram chat action status
    (e.g., 'typing', 'record-audio', 'record-round', 'upload-document') by refreshing it periodically.
    """

    def __init__(
        self,
        client: TelegramClient,
        chat_id: Union[int, str],
        action: str = "typing",
        interval: float = 4.0,
    ):
        self.client = client
        self.chat_id = chat_id
        self.action = action
        self.interval = interval
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def _action_loop(self) -> None:
        """Continuously refresh the chat action until stopped."""
        try:
            while self._running:
                try:
                    await self.client.action(self.chat_id, self.action)
                except FloodWaitError as e:
                    logger.warning(f"FloodWait while sending chat action: {e.seconds}s")
                    await asyncio.sleep(e.seconds)
                except Exception as e:
                    logger.debug(f"Could not refresh chat action '{self.action}': {e}")
                await asyncio.sleep(self.interval)
        except asyncio.CancelledError:
            pass

    async def __aenter__(self):
        self._running = True
        self._task = asyncio.create_task(self._action_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def safe_send_message(
    client: TelegramClient,
    chat_id: Union[int, str],
    text: str,
    reply_to: Optional[Union[int, Message]] = None,
) -> Optional[Message]:
    """
    Sends message safely, splitting chunks if text exceeds Telegram's 4096 character limit,
    handling flood waits, and falling back to plain text if Markdown parsing fails.
    """
    if not text or not text.strip():
        return None

    MAX_CHUNK_SIZE = 4000  # Conservative safety margin below 4096

    # Split text into chunks if too long
    chunks = []
    if len(text) <= MAX_CHUNK_SIZE:
        chunks.append(text)
    else:
        current_chunk = []
        current_len = 0
        for line in text.splitlines(keepends=True):
            if current_len + len(line) > MAX_CHUNK_SIZE:
                chunks.append("".join(current_chunk))
                current_chunk = [line]
                current_len = len(line)
            else:
                current_chunk.append(line)
                current_len += len(line)
        if current_chunk:
            chunks.append("".join(current_chunk))

    last_sent_msg = None
    target_reply = reply_to

    for chunk in chunks:
        sent = False
        while not sent:
            try:
                last_sent_msg = await client.send_message(
                    entity=chat_id,
                    message=chunk,
                    reply_to=target_reply,
                    parse_mode="md",
                )
                sent = True
                # Subsequent chunks don't need reply_to parameter
                target_reply = None
            except FloodWaitError as e:
                logger.warning(f"Telegram FloodWait encountered: sleeping {e.seconds}s")
                await asyncio.sleep(e.seconds + 1)
            except Exception as e:
                logger.warning(f"Failed to send with Markdown ({e}). Falling back to plain text.")
                try:
                    last_sent_msg = await client.send_message(
                        entity=chat_id,
                        message=chunk,
                        reply_to=target_reply,
                        parse_mode=None,
                    )
                    sent = True
                    target_reply = None
                except Exception as final_err:
                    logger.error(f"Failed to dispatch message to {chat_id}: {final_err}")
                    break

    return last_sent_msg
