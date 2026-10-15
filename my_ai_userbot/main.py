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

from typing import Optional

from openai import AsyncOpenAI
from telethon import TelegramClient, events
from telethon.errors import SessionPasswordNeededError
from telethon.tl.custom import Message

from config import config
from ai_engine.memory import MemoryManager
from ai_engine.llm_handler import LLMHandler
from media_processor.audio_handler import AudioHandler
from media_processor.video_handler import VideoHandler
from utils.filters import FilterManager
from utils.telegram_helpers import TelegramChatAction, safe_send_message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Userbot.Main")


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

        # Initialize MTProto Telethon client
        self.client = TelegramClient(
            session=config.telegram_session_name,
            api_id=config.telegram_api_id,
            api_hash=config.telegram_api_hash,
        )

        # Initialize OpenAI async client if key is provided
        self.openai_client = None
        if config.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)

        # Tracking dictionaries for real-time read and outgoing events
        self.last_read_inbox: dict[int, int] = {}
        self.last_outgoing_time: dict[int, float] = {}

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

        @self.client.on(events.MessageRead(inbox=True))
        async def on_message_read(event: events.MessageRead.Event):
            current_max = self.last_read_inbox.get(event.chat_id, 0)
            self.last_read_inbox[event.chat_id] = max(current_max, event.max_id)
            logger.debug(f"Chat {event.chat_id} read up to message ID {event.max_id}")

        @self.client.on(events.NewMessage(outgoing=True))
        async def on_outgoing_message(event: events.NewMessage.Event):
            if event.message and event.message.date:
                self.last_outgoing_time[event.chat_id] = event.message.date.timestamp()

    async def is_owner_online(self) -> bool:
        """Check whether the owner (Ayubxon) is currently active/online on Telegram."""
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
                        return True
        except Exception as e:
            logger.debug(f"Error checking owner online status: {e}")
        return False

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
        """Central message inspection, filter check, media processing, and AI reply dispatch."""
        chat_id = message.chat_id
        sender = await message.get_sender()
        sender_id = message.sender_id or 0
        sender_name = (
            getattr(sender, "first_name", "") or getattr(sender, "title", "User")
        ).strip()

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

        # 3. Check Owner Status & Grace Period for External Private Chats
        if not is_owner and message.is_private:
            # A. If owner is currently online on Telegram, do NOT reply (owner can answer personally)
            if config.only_reply_when_offline and await self.is_owner_online():
                logger.info(
                    f"Owner {config.owner_name} is currently ONLINE on Telegram. "
                    f"Skipping AI reply to {sender_name} ({sender_id}) so owner can reply personally."
                )
                return

            # B. If delay is configured, wait and monitor if owner sees or replies to the message
            if config.reply_delay_seconds > 0:
                logger.info(
                    f"Message from {sender_name} received while offline. "
                    f"Waiting {config.reply_delay_seconds}s to check if owner sees or replies..."
                )
                loop = asyncio.get_event_loop()
                start_time = loop.time()
                while loop.time() - start_time < config.reply_delay_seconds:
                    await asyncio.sleep(2)

                    # Check 1: Did owner come online?
                    if config.only_reply_when_offline and await self.is_owner_online():
                        logger.info(
                            f"Owner {config.owner_name} came ONLINE during wait. "
                            f"Cancelling AI reply for {sender_name}."
                        )
                        return

                    # Check 2: Did owner read this message?
                    if self.last_read_inbox.get(chat_id, 0) >= message.id:
                        logger.info(
                            f"Message {message.id} was READ by owner. "
                            f"Cancelling AI reply for {sender_name}."
                        )
                        return

                    # Check 3: Did owner personally reply in this chat?
                    msg_time = message.date.timestamp() if message.date else 0.0
                    if self.last_outgoing_time.get(chat_id, 0.0) >= msg_time:
                        logger.info(
                            f"Owner personally replied in chat {chat_id}. "
                            f"Cancelling AI reply for {sender_name}."
                        )
                        return

                # Double check with server dialog read status before generating response
                if await self.is_message_read_on_server(chat_id, message.id):
                    logger.info(
                        f"Message {message.id} was confirmed READ on Telegram server. "
                        f"Cancelling AI reply for {sender_name}."
                    )
                    return

        # 4. Modality Detection & Pre-processing
        text_content: Optional[str] = message.text or None
        audio_transcript: Optional[str] = None
        video_frames_b64 = None
        message_type = "text"
        simulated_action = "typing"

        # Check for Voice Note / Audio
        if message.voice or (message.audio and getattr(message.audio, "voice", False)):
            message_type = "voice"
            simulated_action = "record-audio"
            logger.info(f"Processing voice message from {sender_name} ({sender_id})...")
            async with TelegramChatAction(self.client, chat_id, action="record-audio"):
                audio_transcript, _ = await self.audio_handler.process_voice_message(message)

        # Check for Video Note (Round video) or regular Video
        elif message.video_note or message.video:
            message_type = "video"
            simulated_action = "typing"
            logger.info(f"Processing video message from {sender_name} ({sender_id})...")
            async with TelegramChatAction(self.client, chat_id, action="typing"):
                video_res = await self.video_handler.process_video_message(message)
                audio_transcript = video_res.get("transcription")
                video_frames_b64 = video_res.get("frames_b64")

        # 5. Generate Autonomous Executive AI Response
        async with TelegramChatAction(self.client, chat_id, action=simulated_action):
            reply_text = await self.llm_handler.generate_response(
                chat_id=chat_id,
                sender_id=sender_id,
                sender_name=sender_name,
                text_content=text_content,
                audio_transcript=audio_transcript,
                video_frames_b64=video_frames_b64,
                is_owner=is_owner,
                message_type=message_type,
                metadata={"message_id": message.id},
            )

        # 6. Dispatch reply safely to Telegram
        if reply_text and reply_text.strip():
            await safe_send_message(
                client=self.client,
                chat_id=chat_id,
                text=reply_text,
                reply_to=message,
            )

            # Mark message as read AFTER the AI assistant has replied
            if config.auto_read_messages and not message.out:
                try:
                    await message.mark_read()
                except Exception as e:
                    logger.debug(f"Could not mark message as read: {e}")

            # If an external contact sent a message in private chat,
            # send a private report to Ayubxon's Saved Messages ("me") so ONLY Ayubxon sees it!
            if not is_owner and message.is_private:
                try:
                    summary_text = (
                        f"📩 **Yangi xabar keldi (Faqat siz uchun hisobot):**\n\n"
                        f"👤 **Kimdan:** {sender_name} (ID: `{sender_id}`)\n"
                        f"💬 **Xabar:** {text_content or audio_transcript or '(Media)'}\n\n"
                        f"🤖 **Bot bergan javob:**\n{reply_text}"
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
        if not await self.client.is_user_authorized():
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
            f"{'Supabase pgvector (RAG)' if config.is_supabase_configured() else 'In-Memory Cache'}"
        )
        print("=" * 70)
        print("\nListening for incoming messages... Press Ctrl+C to terminate.")

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
