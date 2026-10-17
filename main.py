"""
Main entry point for Sunnatbek's Multimodal Telegram AI Executive Assistant.
Handles MTProto client initialization, event routing, media detection, and AI dispatch.
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure logging early so INFO level is respected across all modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("Userbot.Main")

from typing import Optional

from openai import AsyncOpenAI
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
import base64
import ipaddress
import re
import struct
from telethon.crypto import AuthKey
from telethon.sessions.string import StringSession, _STRUCT_PREFORMAT, CURRENT_VERSION


class SafeStringSession(StringSession):
    """Robust StringSession parser that handles trailing spaces, quotes, and prevents struct unpack errors."""

    def __init__(self, string: str = None):
        super(StringSession, self).__init__()
        if string:
            cleaned = str(string).strip().strip('"').strip("'").strip()
            m = re.search(r"1[A-Za-z0-9_\-]{351}=", cleaned)
            if m:
                cleaned = m.group(0)

            if not cleaned or cleaned[0] != CURRENT_VERSION:
                raise ValueError("Not a valid string")

            payload = cleaned[1:]
            decoded = StringSession.decode(payload)
            ip_len = 4 if len(decoded) == 263 else 16
            self._dc_id, ip, self._port, key = struct.unpack(
                _STRUCT_PREFORMAT.format(ip_len), decoded
            )
            self._server_address = ipaddress.ip_address(ip).compressed
            if any(key):
                self._auth_key = AuthKey(key)

from telethon.tl.custom import Message

from config import config
from ai_engine.memory import MemoryManager
from ai_engine.llm_handler import LLMHandler
from media_processor.audio_handler import AudioHandler
from media_processor.video_handler import VideoHandler
from utils.filters import FilterManager
from utils.telegram_helpers import TelegramChatAction, safe_send_message


class PendingBatch:
    """Holds unread messages from a contact while waiting for the owner to see or reply."""

    def __init__(self):
        self.messages: list[Message] = []
        self.task: Optional[asyncio.Task] = None


class TelegramAIUserbot:
    """Master controller orchestrating Telethon MTProto and Multimodal AI services."""

    def __init__(self):
        # Validate critical configuration
        validation_errors = config.validate_core()
        if validation_errors:
            logger.error("Configuration validation failed:")
            for err in validation_errors:
                logger.error(f"  - {err}")
            logger.error("Please configure the missing keys in .env and restart.")
            sys.exit(1)

        # Initialize MTProto Telethon client (supports StringSession for cloud/Render or file session)
        session_str = config.telegram_string_session
        print(f"📡 SESSION CONFIG: TELEGRAM_STRING_SESSION present={bool(session_str)}, length={len(session_str)}")
        if session_str:
            logger.info("Using TELEGRAM_STRING_SESSION (SafeStringSession) for authentication.")
            session_target = SafeStringSession(session_str)
        else:
            logger.warning(f"No TELEGRAM_STRING_SESSION found! Using local file session: {config.telegram_session_name}")
            session_target = config.telegram_session_name

        self.client = TelegramClient(
            session=session_target,
            api_id=config.telegram_api_id,
            api_hash=config.telegram_api_hash,
            auto_reconnect=True,
            connection_retries=None,
            retry_delay=3,
        )

        # Initialize OpenAI async client if key is provided
        self.openai_client = None
        if config.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)

        # Tracking state for real-time read, manual outgoing events, and smart message debouncing
        self.last_read_inbox: dict[int, int] = {}
        self.last_manual_outgoing_time: dict[int, float] = {}
        self.last_global_manual_outgoing_time: float = 0.0
        self.bot_sent_msg_ids: set[int] = set()
        self.pending_batches: dict[int, PendingBatch] = {}
        self._cached_owner_online: bool = False

        # Initialize Subsystems
        self.memory_manager = MemoryManager(self.openai_client)
        self.llm_handler = LLMHandler(self.memory_manager, self.openai_client)
        self.audio_handler = AudioHandler(self.openai_client)
        self.video_handler = VideoHandler(self.audio_handler)
        self.filter_manager = FilterManager(self.client)

        self._register_handlers()

    def _register_handlers(self) -> None:
        """Register Telethon event listeners."""

        @self.client.on(events.NewMessage(incoming=True, outgoing=True))
        async def message_router(event: events.NewMessage.Event):
            await self._handle_message(event.message)

        @self.client.on(events.UserUpdate)
        async def on_user_status_update(event: events.UserUpdate.Event):
            # Listen for owner online/offline status changes in real-time
            if self.filter_manager.owner_id and event.user_id == self.filter_manager.owner_id:
                if event.online is not None:
                    self._cached_owner_online = event.online
                    state_str = "ONLAYN (Telegramda)" if event.online else "OFLAYN (Chiqib ketgan)"
                    logger.info(f"👤 Egasi holati yangilandi: {state_str}")

        @self.client.on(events.MessageRead(inbox=True))
        async def on_message_read(event: events.MessageRead.Event):
            current_max = self.last_read_inbox.get(event.chat_id, 0)
            self.last_read_inbox[event.chat_id] = max(current_max, event.max_id)
            logger.debug(f"Chat {event.chat_id} read up to message ID {event.max_id}")

            # If the owner personally opened and read the pending messages, cancel AI response immediately!
            if event.chat_id in self.pending_batches:
                batch = self.pending_batches.pop(event.chat_id, None)
                if batch and batch.task and not batch.task.done():
                    batch.task.cancel()
                logger.info(f"Chat {event.chat_id} was READ by owner. Cancelled pending AI auto-reply.")

        @self.client.on(events.NewMessage(outgoing=True))
        async def on_outgoing_message(event: events.NewMessage.Event):
            if not event.message:
                return
            # If this message was dispatched by our bot, do not treat it as manual owner typing
            if event.message.id in self.bot_sent_msg_ids:
                return

            if event.message.date:
                import time
                self.last_global_manual_outgoing_time = time.time()
                self.last_manual_outgoing_time[event.chat_id] = event.message.date.timestamp()
                # If owner typed personally in this chat, cancel pending AI auto-reply immediately!
                if event.chat_id in self.pending_batches:
                    batch = self.pending_batches.pop(event.chat_id, None)
                    if batch and batch.task and not batch.task.done():
                        batch.task.cancel()
                    logger.info(f"Owner personally replied in chat {event.chat_id}. Cancelled pending AI batch.")

            # Record owner's manual message into persistent memory so AI remembers what the owner said!
            try:
                out_content = event.message.text or ""
                out_type = "text"
                if not out_content:
                    if event.message.voice:
                        out_content = "[Ovozli xabar]"
                        out_type = "voice"
                    elif event.message.video or event.message.video_note:
                        out_content = "[Video xabar]"
                        out_type = "video"
                    elif event.message.photo:
                        out_content = "[Rasm]"
                        out_type = "photo"

                if out_content:
                    created_str = (
                        event.message.date.strftime("%Y-%m-%d %H:%M:%S")
                        if event.message.date
                        else None
                    )
                    await self.memory_manager.store_message(
                        chat_id=event.chat_id,
                        sender_id=getattr(self.me, "id", 0) if hasattr(self, "me") and self.me else 0,
                        sender_name=config.owner_name,
                        role="assistant",
                        content=out_content,
                        message_type=out_type,
                        is_owner=True,
                        telegram_message_id=event.message.id,
                        created_at=created_str,
                    )
            except Exception as store_err:
                logger.debug(f"Could not store manual outgoing message: {store_err}")

    async def is_owner_online(self) -> bool:
        """Check whether the owner (Ayubxon) is currently active/online on Telegram."""
        import time
        # 1. If owner sent a manual message in Telegram anywhere within the last 60 seconds, owner is definitely active
        if time.time() - getattr(self, "last_global_manual_outgoing_time", 0.0) < 60:
            return True

        # 2. Query Telegram server directly for latest UserStatus of the owner
        try:
            from telethon.tl.types import InputUserSelf, UserStatusOnline
            from telethon.tl.functions.users import GetUsersRequest
            import datetime

            users = await self.client(GetUsersRequest(id=[InputUserSelf()]))
            if users:
                status = getattr(users[0], "status", None)
                if isinstance(status, UserStatusOnline):
                    now = datetime.datetime.now(datetime.timezone.utc)
                    if status.expires and status.expires > now:
                        self._cached_owner_online = True
                        return True
                    self._cached_owner_online = False
                    return False
                else:
                    self._cached_owner_online = False
                    return False
        except Exception as e:
            logger.debug(f"Error checking owner online status: {e}")

        return getattr(self, "_cached_owner_online", False)

    async def is_message_read_on_server(self, chat_id: int, message_id: int) -> bool:
        """Check Telegram server dialog read state."""
        try:
            from telethon.tl.functions.messages import GetPeerDialogsRequest
            from telethon.tl.types import InputDialogPeer

            peer = await self.client.get_input_entity(chat_id)
            res = await self.client(GetPeerDialogsRequest(peers=[InputDialogPeer(peer=peer)]))
            if res and res.dialogs:
                dialog = res.dialogs[0]
                read_max = getattr(dialog, "read_inbox_max_id", 0)
                if read_max >= message_id:
                    return True
        except Exception as e:
            logger.debug(f"Error checking dialog read state on server: {e}")
        return False

    async def _handle_message(self, message: Message) -> None:
        """Central message inspection, online check, filter check, and AI reply dispatch."""
        chat_id = message.chat_id

        # 1. Handle Owner Management Commands (.ai ...)
        is_cmd, cmd_output = await self.filter_manager.check_command(message)
        if is_cmd:
            if cmd_output == "__CLEAR_MEMORY_SIGNAL__":
                await self.memory_manager.clear_chat_memory(chat_id)
                await message.reply("🧹 **Conversation memory cleared for this chat.**")
            elif cmd_output:
                await message.reply(cmd_output)
            return

        # 2. Check routing policies and filters
        if not await self.filter_manager.should_process_message(message):
            return

        is_owner = self.filter_manager.is_saved_messages(message)

        # 3. If in Saved Messages (owner copilot mode), process immediately
        if is_owner:
            await self._process_messages_and_reply(chat_id, [message], is_owner=True)
            return

        # 4. If in group chat (if groups enabled in config)
        if not message.is_private:
            await self._process_messages_and_reply(chat_id, [message], is_owner=False)
            return

        # 5. External Private Chat:
        # AGAR EGASI TELEGRAMGA KIRIB ONLAYN BO'LSA - AI JAVOB YOZMAYDI (Avtomatik to'xtaydi)!
        if config.only_reply_when_offline and await self.is_owner_online():
            sender = await message.get_sender()
            sender_name = (
                getattr(sender, "first_name", "") or getattr(sender, "title", "User")
            ).strip() or "User"
            logger.info(
                f"👤 Egasi ({config.owner_name}) hozir Telegramda ONLAYN! "
                f"AI {sender_name} ga javob bermaydi (egasi o'zi yozishi mumkin)."
            )
            # Yozishmalar tarixi to'liq saqlanib borishi uchun xabarni xotiraga yozib qo'yamiz
            try:
                inc_text = message.text or ("[Ovozli xabar]" if message.voice else "[Media]")
                created_str = message.date.strftime("%Y-%m-%d %H:%M:%S") if message.date else None
                await self.memory_manager.store_message(
                    chat_id=chat_id,
                    sender_id=message.sender_id or 0,
                    sender_name=sender_name,
                    role="user",
                    content=inc_text,
                    message_type="text",
                    is_owner=False,
                    telegram_message_id=message.id,
                    created_at=created_str,
                )
            except Exception as e:
                logger.debug(f"Could not record online incoming message: {e}")
            return

        # Agar bu chatda egasining o'zi oxirgi 60 soniya ichida yozgan bo'lsa - AI aralashmaydi
        import time
        if time.time() - self.last_manual_outgoing_time.get(chat_id, 0.0) < 60:
            logger.info(f"Egasi bu chatda ({chat_id}) yaqinda o'zi yozgan. AI aralashmaydi.")
            try:
                sender = await message.get_sender()
                sender_name = (
                    getattr(sender, "first_name", "") or getattr(sender, "title", "User")
                ).strip() or "User"
                inc_text = message.text or ("[Ovozli xabar]" if message.voice else "[Media]")
                created_str = message.date.strftime("%Y-%m-%d %H:%M:%S") if message.date else None
                await self.memory_manager.store_message(
                    chat_id=chat_id,
                    sender_id=message.sender_id or 0,
                    sender_name=sender_name,
                    role="user",
                    content=inc_text,
                    message_type="text",
                    is_owner=False,
                    telegram_message_id=message.id,
                    created_at=created_str,
                )
            except Exception as e:
                logger.debug(f"Could not record recent manual outgoing incoming message: {e}")
            return

        # 6. Agar egasi oflayn bo'lsa (telegramdan chiqqan):
        # A. Agar kutish vaqti 0 bo'lsa - SRAZI (darhol) javob qaytaradi!
        if config.reply_delay_seconds <= 0:
            logger.info(f"Egasi oflayn. {chat_id} dan kelgan xabarga AI darhol (srazi) javob tayyorlamoqda...")
            await self._process_messages_and_reply(chat_id, [message], is_owner=False)
            return

        # B. Agar kechiktirish belgilangan bo'lsa (reply_delay_seconds > 0):
        batch = self.pending_batches.setdefault(chat_id, PendingBatch())
        batch.messages.append(message)

        # Reset timer task on each new message from the contact so we wait after the LAST message
        if batch.task and not batch.task.done():
            batch.task.cancel()

        batch.task = asyncio.create_task(self._wait_and_dispatch_batch(chat_id, batch))

    async def _wait_and_dispatch_batch(self, chat_id: int, batch: PendingBatch) -> None:
        """Wait for grace period while monitoring if owner reads or replies."""
        try:
            wait_seconds = config.reply_delay_seconds
            logger.info(
                f"Waiting {wait_seconds}s for chat {chat_id} (collected {len(batch.messages)} message(s)) "
                f"to check if owner sees or replies..."
            )
            loop = asyncio.get_event_loop()
            start_time = loop.time()

            while loop.time() - start_time < wait_seconds:
                await asyncio.sleep(1.5)

                # Check 1: Did owner come online?
                if config.only_reply_when_offline and await self.is_owner_online():
                    logger.info(f"Owner {config.owner_name} came ONLINE during wait. Cancelling AI reply.")
                    self.pending_batches.pop(chat_id, None)
                    return

                # Check 2: Did owner read this chat up to the latest message?
                latest_id = max((m.id for m in batch.messages), default=0)
                if self.last_read_inbox.get(chat_id, 0) >= latest_id:
                    logger.info(f"Chat {chat_id} was READ by owner. Cancelling AI reply.")
                    self.pending_batches.pop(chat_id, None)
                    return

                # Check 3: Did owner manually reply in this chat?
                earliest_msg_time = min(
                    (m.date.timestamp() for m in batch.messages if m.date),
                    default=0.0,
                )
                if self.last_manual_outgoing_time.get(chat_id, 0.0) >= earliest_msg_time:
                    logger.info(f"Owner manually replied in chat {chat_id}. Cancelling AI reply.")
                    self.pending_batches.pop(chat_id, None)
                    return

            # Double check with server dialog read status before generating response
            latest_id = max((m.id for m in batch.messages), default=0)
            if await self.is_message_read_on_server(chat_id, latest_id):
                logger.info(f"Chat {chat_id} was confirmed READ on Telegram server. Cancelling AI reply.")
                self.pending_batches.pop(chat_id, None)
                return

            messages_to_process = list(batch.messages)
            self.pending_batches.pop(chat_id, None)
            if not messages_to_process:
                return

            await self._process_messages_and_reply(chat_id, messages_to_process, is_owner=False)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in _wait_and_dispatch_batch for chat {chat_id}: {e}", exc_info=True)
            self.pending_batches.pop(chat_id, None)

    async def sync_telegram_chat_history(
        self, chat_id: int, limit: int = 15, max_id: Optional[int] = None
    ) -> None:
        """
        Synchronizes past chat history from Telegram servers into local persistent SQLite database.
        Ensures AI has full awareness of previous conversations, even if they took place
        while the bot was offline or directly from the owner's Telegram mobile app.
        """
        try:
            kwargs: dict[str, Any] = {"limit": limit}
            if max_id:
                kwargs["max_id"] = max_id

            tg_messages = await self.client.get_messages(chat_id, **kwargs)
            if not tg_messages:
                return

            for msg in reversed(tg_messages):
                if not msg:
                    continue

                content = msg.text or ""
                msg_type = "text"
                if not content:
                    if msg.voice or (msg.audio and getattr(msg.audio, "voice", False)):
                        content = "[Ovozli xabar]"
                        msg_type = "voice"
                    elif msg.video_note or msg.video:
                        content = "[Video xabar]"
                        msg_type = "video"
                    elif msg.photo:
                        content = "[Rasm yuborildi]"
                        msg_type = "photo"
                    elif msg.document:
                        content = "[Fayl / Hujjat]"
                        msg_type = "document"

                if not content.strip():
                    continue

                is_owner = bool(msg.out)
                role = "assistant" if is_owner else "user"

                sender_id = msg.sender_id or 0
                if is_owner:
                    sender_name = config.owner_name
                else:
                    try:
                        sender = await msg.get_sender()
                        sender_name = (
                            getattr(sender, "first_name", "")
                            or getattr(sender, "title", "User")
                        ).strip() or "User"
                    except Exception:
                        sender_name = "User"

                created_at_str = (
                    msg.date.strftime("%Y-%m-%d %H:%M:%S") if msg.date else None
                )

                await self.memory_manager.store_message(
                    chat_id=chat_id,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    role=role,
                    content=content,
                    message_type=msg_type,
                    is_owner=is_owner,
                    telegram_message_id=msg.id,
                    created_at=created_at_str,
                )
        except Exception as e:
            logger.debug(f"History sync note for chat {chat_id}: {e}")

    async def _process_messages_and_reply(
        self,
        chat_id: int,
        messages: list[Message],
        is_owner: bool = False,
    ) -> None:
        """Processes one or more incoming messages, generates AI response, and sends reply."""
        if not messages:
            return

        latest_message = messages[-1]
        sender = await latest_message.get_sender()
        sender_id = latest_message.sender_id or 0
        sender_name = (
            getattr(sender, "first_name", "") or getattr(sender, "title", "User")
        ).strip()

        # Synchronize earlier conversation history from Telegram before processing
        try:
            earliest_id = min((m.id for m in messages), default=None)
            await self.sync_telegram_chat_history(
                chat_id=chat_id,
                limit=config.max_recent_messages,
                max_id=earliest_id,
            )
        except Exception as sync_err:
            logger.debug(f"History sync note for chat {chat_id}: {sync_err}")

        # Extract contents across messages
        text_parts: list[str] = []
        audio_transcript: Optional[str] = None
        video_frames_b64: Optional[list[str]] = None
        message_type = "text"
        simulated_action = "typing"

        for msg in messages:
            # Voice message
            if msg.voice or (msg.audio and getattr(msg.audio, "voice", False)):
                message_type = "voice"
                simulated_action = "record-audio"
                logger.info(f"Processing voice message from {sender_name} ({sender_id})...")
                async with TelegramChatAction(self.client, chat_id, action="record-audio"):
                    v_trans, _ = await self.audio_handler.process_voice_message(msg)
                    if v_trans:
                        audio_transcript = v_trans
                        text_parts.append(f"[Ovozli xabar]: \"{v_trans}\"")
                    if msg.text:
                        text_parts.append(msg.text)
            # Video note or regular video
            elif msg.video_note or msg.video:
                message_type = "video"
                simulated_action = "typing"
                logger.info(f"Processing video message from {sender_name} ({sender_id})...")
                async with TelegramChatAction(self.client, chat_id, action="typing"):
                    v_res = await self.video_handler.process_video_message(msg)
                    v_trans = v_res.get("transcription")
                    video_frames_b64 = v_res.get("frames_b64")
                    if v_trans:
                        audio_transcript = v_trans
                        text_parts.append(f"[Video xabar ovozi]: \"{v_trans}\"")
                    if msg.text:
                        text_parts.append(msg.text)
            else:
                if msg.text:
                    text_parts.append(msg.text)

        combined_text = "\n".join(text_parts).strip()
        if not combined_text and not video_frames_b64:
            logger.debug(f"No textual or visual content detected for chat {chat_id}.")
            return

        # Generate autonomous response (as Ayubxon himself if external contact, or copilot if is_owner)
        async with TelegramChatAction(self.client, chat_id, action=simulated_action):
            reply_text = await self.llm_handler.generate_response(
                chat_id=chat_id,
                sender_id=sender_id,
                sender_name=sender_name,
                text_content=combined_text,
                audio_transcript=audio_transcript,
                video_frames_b64=video_frames_b64,
                is_owner=is_owner,
                message_type=message_type,
                metadata={"message_id": latest_message.id},
            )

        if reply_text and reply_text.strip():
            sent_msg = await safe_send_message(
                client=self.client,
                chat_id=chat_id,
                text=reply_text,
                reply_to=latest_message,
            )
            if sent_msg:
                self.bot_sent_msg_ids.add(sent_msg.id)
                try:
                    await self.memory_manager.update_last_assistant_telegram_id(
                        chat_id=chat_id, telegram_message_id=sent_msg.id
                    )
                except Exception:
                    pass

            # Mark message as read AFTER the AI userbot has replied
            if config.auto_read_messages and not latest_message.out:
                try:
                    await latest_message.mark_read()
                except Exception as e:
                    logger.debug(f"Could not mark message as read: {e}")

            # Send Saved Messages notification for Ayubxon so he knows what was answered on his behalf!
            if not is_owner and latest_message.is_private:
                try:
                    summary_text = (
                        f"📩 **Sizning nomingizdan javob berildi:**\n\n"
                        f"👤 **Kimdan:** {sender_name} (ID: `{sender_id}`)\n"
                        f"💬 **Kelgan xabar(lar):**\n{combined_text}\n\n"
                        f"💬 **Sizdek berilgan javob:**\n{reply_text}"
                    )
                    await self.client.send_message("me", summary_text)
                except Exception as notif_err:
                    logger.debug(f"Could not send Saved Messages notification: {notif_err}")

    async def start(self) -> None:
        """Start the MTProto userbot."""
        print("=" * 70)
        print("🚀 STARTING AYUBXON'S MULTIMODAL TELEGRAM AI EXECUTIVE USERBOT")
        print("=" * 70)

        # Connect and authenticate
        await self.client.connect()
        is_auth = await self.client.is_user_authorized()
        print(f"📡 AUTH STATUS: is_user_authorized = {is_auth}")
        if not is_auth:
            from telethon import functions
            try:
                await self.client(functions.updates.GetStateRequest())
            except Exception as auth_err:
                print(f"📡 AUTH ERROR: {type(auth_err).__name__}: {auth_err}")

            if not sys.stdin or not sys.stdin.isatty():
                logger.critical(
                    "❌ Telegram akkaunt avtorizatsiyadan o'tmagan!\n"
                    "Render serverida TELEGRAM_STRING_SESSION topilmadi yoki noto'g'ri kiritilgan.\n"
                    "Iltimos, Render boshqaruv panelidagi 'Environment' bo'limiga TELEGRAM_STRING_SESSION kalitini to'g'ri qo'shing."
                )
                sys.exit(1)

            phone = config.telegram_phone_number or input("Telefon raqamingizni kiriting (+998...): ").strip()
            print(f"\n📡 Telegram'dan login kodi so'ralmoqda ({phone})...")
            sent_code = await self.client.send_code_request(phone)

            code_type_name = type(sent_code.type).__name__
            if "Email" in code_type_name:
                pattern = getattr(sent_code.type, "email_pattern", "Gmail pochtangiz")
                print(f"\n📧 DIQQAT: Kod ELEKTRON POCHTANGIZGA ({pattern}) yuborildi!")
                print("   Iltimos, pochtangizni (Inbox yoki Spam) ochib yangi kelgan kodni tekshiring.")
            elif "Sms" in code_type_name:
                print("\n📩 DIQQAT: Kod telefoningizga oddiy SMS xabar orqali yuborildi!")
                print("   Iltimos, telefoningizdagi Xabarlar (SMS) ilovasini tekshiring.")
            elif "App" in code_type_name:
                print("\n📲 DIQQAT: Kod Telegram ilovasidagi rasmiy 'Telegram' (Service Notifications) chatiga yuborildi!")
            else:
                print(f"\nℹ️ Telegram kod yubordi (Kanal turi: {code_type_name}).")

            code = input("\n👉 Telegram yuborgan kodni kiriting: ").strip()
            try:
                await self.client.sign_in(phone=phone, code=code)
            except SessionPasswordNeededError:
                pwd = getpass.getpass("🔒 Akkauntingizning ikki bosqichli paroli (2FA)ni kiriting: ")
                await self.client.sign_in(password=pwd)

        # Cache owner identity
        await self.filter_manager.initialize()

        me = await self.client.get_me()
        self.me = me
        print("\n" + "=" * 70)
        print(f"✅ Logged in successfully as: {me.first_name} (@{me.username}) [ID: {me.id}]")
        engine_name = (
            f"Google Gemini ({config.gemini_model})"
            if (config.ai_provider == "gemini" and config.gemini_api_key) or not config.openai_api_key
            else f"OpenAI ({config.openai_model})"
        )
        print(f"🤖 Assistant Persona: {config.assistant_name}")
        print(f"🧠 AI Engine: {engine_name}")
        print(
            f"🗄️ Database Memory: "
            f"{'Supabase pgvector (RAG) + SQLite' if config.is_supabase_configured() else 'Persistent SQLite (data/chat_history.db)'}"
        )
        print("=" * 70)
        # ----------------------------------------------------------------------
        # Keep-Alive 24/7: Lightweight HTTP Healthcheck Server & Heartbeat Pinger
        # ----------------------------------------------------------------------
        import os
        import time
        import datetime
        from aiohttp import web

        bot_start_time = time.time()
        port = int(os.getenv("PORT", "8080"))

        async def handle_root(request):
            uptime_min = int((time.time() - bot_start_time) / 60)
            return web.Response(
                text=f"🟢 Sunnatbek AI Userbot 24/7 faol ishlamoqda!\nIsh vaqti (Uptime): {uptime_min} daqiqa\nHolat: OK\n",
                content_type="text/plain; charset=utf-8",
            )

        async def handle_health(request):
            uptime_sec = int(time.time() - bot_start_time)
            return web.json_response(
                {
                    "status": "healthy",
                    "service": "Telegram AI Userbot",
                    "uptime_seconds": uptime_sec,
                    "uptime_minutes": uptime_sec // 60,
                    "server_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        async def handle_ping(request):
            return web.Response(text="pong", content_type="text/plain; charset=utf-8")

        try:
            http_app = web.Application()
            http_app.router.add_get("/", handle_root)
            http_app.router.add_get("/health", handle_health)
            http_app.router.add_get("/ping", handle_ping)
            runner = web.AppRunner(http_app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", port)
            await site.start()
            logger.info(f"🚀 24/7 Keep-alive HTTP server faol ishga tushdi (Port: {port})")
        except Exception as e:
            logger.warning(f"⚠️ HTTP serverni ishga tushirishda ogohlantirish: {e}")

        # Keep-alive background task to prevent server free tier from sleeping
        async def keep_alive_pinger():
            """
            Keep bot alive 24/7:
            1. Sends HTTP ping to Render/Cloud Web Service to reset inactivity sleep timer.
            2. Sends heartbeat status message to Telegram Saved Messages ('me') every 13 minutes.
            """
            import aiohttp

            # Heartbeat interval (default 13 minutes = 780 seconds)
            interval_minutes = int(os.getenv("HEARTBEAT_INTERVAL_MINUTES", "13"))
            interval_seconds = max(60, interval_minutes * 60)

            url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("PING_URL")
            target_url = None
            if url:
                if not url.startswith("http"):
                    url = f"https://{url}"
                target_url = f"{url.rstrip('/')}/health"
                logger.info(f"🚀 Keep-alive tashqi ping manzili: {target_url}")

            send_tg_msg = os.getenv("SEND_TELEGRAM_HEARTBEAT", "True").lower() in ("true", "1", "yes")

            # Dastlabki 30 soniya kutib, birinchi salomlashish/ishga tushish xabarini yuboramiz
            await asyncio.sleep(30)
            if send_tg_msg:
                try:
                    start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    welcome_msg = (
                        f"🚀 **Bot serverda 24/7 muvaffaqiyatli ishga tushdi!**\n\n"
                        f"⏰ Boshlangan vaqt: `{start_str}`\n"
                        f"🔄 Har **{interval_minutes} daqiqada** o'ziga o'zi holat xabari tashlab turadi.\n"
                        f"✅ Server uxlab qolmasligi uchun tizim to'liq sozlangan."
                    )
                    await self.client.send_message("me", welcome_msg)
                    logger.info("📩 Dastlabki ishga tushish xabari Saved Messages ga yuborildi.")
                except Exception as first_msg_err:
                    logger.debug(f"Birinchi heartbeat xabarini yuborishda xatolik: {first_msg_err}")

            # Asosiy 13 daqiqalik davriy sikl
            while True:
                try:
                    await asyncio.sleep(interval_seconds)

                    # 1. HTTP ping (tashqi server yoki local healthcheck)
                    ping_target = target_url or f"http://127.0.0.1:{port}/health"
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(ping_target, timeout=25) as resp:
                                logger.info(f"⏰ Keep-alive HTTP ping yuborildi: {ping_target} (HTTP {resp.status})")
                    except Exception as ping_err:
                        logger.debug(f"Keep-alive HTTP ping xatosi: {ping_err}")

                    # 2. Telegram 'Saved Messages' ('me') ga har 13 daqiqada xabar tashlash
                    if send_tg_msg:
                        try:
                            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            uptime_total_min = int((time.time() - bot_start_time) / 60)
                            uptime_hours = uptime_total_min // 60
                            uptime_mins = uptime_total_min % 60
                            uptime_display = f"{uptime_hours} soat {uptime_mins} daqiqa" if uptime_hours > 0 else f"{uptime_mins} daqiqa"

                            heartbeat_msg = (
                                f"💓 **Bot faol (24/7 holat tekshiruvi)**\n"
                                f"━━━━━━━━━━━━━━━━━━━━\n"
                                f"✅ Server to'xtovsiz ishlamoqda.\n"
                                f"⏰ Hozirgi vaqt: `{now_str}`\n"
                                f"⏳ Ish vaqti (Uptime): `{uptime_display}`\n"
                                f"🔄 Keyingi xabar: {interval_minutes} daqiqadan so'ng"
                            )
                            await self.client.send_message("me", heartbeat_msg)
                            logger.info("📩 13 daqiqalik holat xabari Saqlangan xabarlarga (me) yuborildi.")
                        except Exception as tg_err:
                            logger.warning(f"Telegram heartbeat xabarini yuborishda xatolik: {tg_err}")

                except asyncio.CancelledError:
                    break
                except Exception as loop_err:
                    logger.error(f"Keep-alive pinger siklida kutilmagan xatolik: {loop_err}")

        asyncio.create_task(keep_alive_pinger())

        await self.client.run_until_disconnected()


def main():
    """Application launcher."""
    userbot = TelegramAIUserbot()
    try:
        userbot.client.loop.run_until_complete(userbot.start())
    except KeyboardInterrupt:
        print("\n👋 Userbot gracefully stopped.")
    except Exception as e:
        logger.critical(f"Fatal error running userbot: {e}", exc_info=True)


if __name__ == "__main__":
    main()
