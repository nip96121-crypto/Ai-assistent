"""
Filtering and access control logic for contacts, groups, channels, and commands.
Enforces whitelist, blacklist, and executive bot controls.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from telethon import TelegramClient
from telethon.tl.custom import Message
from telethon.tl.types import User

from config import config

logger = logging.getLogger("Userbot.Filters")


class FilterManager:
    """Evaluates incoming Telegram messages against whitelist, blacklist, and routing policies."""

    def __init__(self, client: TelegramClient):
        self.client = client
        self.owner_id: Optional[int] = None
        self.owner_username: Optional[str] = None

    async def initialize(self) -> None:
        """Fetch and cache owner's Telegram profile identity."""
        try:
            me: User = await self.client.get_me()
            self.owner_id = me.id
            self.owner_username = me.username
            logger.info(f"Initialized owner identity: ID={self.owner_id}, Username=@{self.owner_username}")
        except Exception as e:
            logger.error(f"Failed to fetch userbot owner profile: {e}")

    def is_owner(self, user_id: int) -> bool:
        """Check if a given Telegram user ID corresponds to Sunnatbek (the owner)."""
        return self.owner_id is not None and user_id == self.owner_id

    async def check_command(self, message: Message) -> Tuple[bool, Optional[str]]:
        """
        Check if message contains a userbot control command (prefixed with '.ai').
        Returns (is_command, command_reply_text).
        """
        text = (message.raw_text or "").strip()
        if not text.startswith(".ai"):
            return False, None

        # Only allow commands issued by the owner (or in Saved Messages)
        sender_id = message.sender_id
        if sender_id and not self.is_owner(sender_id) and not message.out:
            return False, None

        parts = text.split()
        subcmd = parts[1].lower() if len(parts) > 1 else "help"

        if subcmd == "on":
            config.is_active = True
            return True, "✅ **AI Executive Assistant activated.** Auto-replies are now ON."

        elif subcmd == "off":
            config.is_active = False
            return True, "⏸️ **AI Executive Assistant paused.** Auto-replies are now OFF."

        elif subcmd == "status":
            state = "🟢 Active" if config.is_active else "🔴 Paused"
            supabase_state = "Connected" if config.is_supabase_configured() else "In-Memory Fallback"
            model_in_use = config.gemini_model if config.ai_provider == "gemini" else config.openai_model
            status_text = (
                f"📊 **Executive Assistant Status:**\n\n"
                f"• **State:** {state}\n"
                f"• **Owner:** {config.owner_name} (`{self.owner_id}`)\n"
                f"• **Model:** `{model_in_use}`\n"
                f"• **Memory (RAG):** `{supabase_state}`\n"
                f"• **Whitelist Filter:** {len(config.whitelist_user_ids)} users\n"
                f"• **Blacklist Filter:** {len(config.blacklist_user_ids)} users\n"
                f"• **Group Replies:** {'Enabled' if config.respond_to_groups else 'Disabled'}\n"
            )
            return True, status_text

        elif subcmd == "whitelist":
            if len(parts) >= 4 and parts[2].lower() == "add":
                try:
                    uid = int(parts[3])
                    config.whitelist_user_ids.add(uid)
                    return True, f"✅ Added `{uid}` to whitelist."
                except ValueError:
                    return True, "❌ Invalid user ID format."
            elif len(parts) >= 4 and parts[2].lower() == "remove":
                try:
                    uid = int(parts[3])
                    config.whitelist_user_ids.discard(uid)
                    return True, f"✅ Removed `{uid}` from whitelist."
                except ValueError:
                    return True, "❌ Invalid user ID format."
            else:
                return True, f"📋 **Whitelisted IDs:** {list(config.whitelist_user_ids) or 'None (All contacts allowed)'}"

        elif subcmd == "blacklist":
            if len(parts) >= 4 and parts[2].lower() == "add":
                try:
                    uid = int(parts[3])
                    config.blacklist_user_ids.add(uid)
                    return True, f"🚫 Added `{uid}` to blacklist."
                except ValueError:
                    return True, "❌ Invalid user ID format."
            elif len(parts) >= 4 and parts[2].lower() == "remove":
                try:
                    uid = int(parts[3])
                    config.blacklist_user_ids.discard(uid)
                    return True, f"✅ Removed `{uid}` from blacklist."
                except ValueError:
                    return True, "❌ Invalid user ID format."
            else:
                return True, f"📋 **Blacklisted IDs:** {list(config.blacklist_user_ids) or 'None'}"

        elif subcmd == "clear":
            return True, "__CLEAR_MEMORY_SIGNAL__"

        else:
            help_text = (
                "🤖 **AI Executive Assistant Commands:**\n\n"
                "• `.ai on` - Enable assistant auto-replies\n"
                "• `.ai off` - Pause assistant auto-replies\n"
                "• `.ai status` - View operational status & memory\n"
                "• `.ai clear` - Clear conversation memory for this chat\n"
                "• `.ai whitelist add/remove <id>` - Manage allowed users\n"
                "• `.ai blacklist add/remove <id>` - Manage ignored users\n"
                "• `.ai help` - Show this command manual\n"
            )
            return True, help_text

    def is_saved_messages(self, message: Message) -> bool:
        """
        Check if message is located in Saved Messages (chat with oneself).
        Only true when private chat and chat_id strictly matches the owner's ID.
        """
        return (
            message.is_private
            and self.owner_id is not None
            and message.chat_id == self.owner_id
        )

    async def should_process_message(self, message: Message) -> bool:
        """
        Verify whether the incoming message qualifies for AI assistant processing.
        """
        # If userbot is paused globally
        if not config.is_active:
            return False

        # Drop empty or action/service messages (e.g. pinned, user joined)
        if message.action:
            return False

        chat_id = message.chat_id
        sender_id = message.sender_id

        # Determine if chat is Saved Messages (chat with oneself)
        if self.is_saved_messages(message):
            return config.respond_to_saved_messages

        # Anti-looping: Do not process outgoing messages in chats with other people
        # (Only Ayubxon's inbound communications from others get autonomous replies)
        if message.out:
            return False

        # Ignore messages without valid sender
        if not sender_id:
            return False

        # Blacklist check
        if sender_id in config.blacklist_user_ids or chat_id in config.blacklist_user_ids:
            logger.debug(f"Ignoring message from blacklisted ID: {sender_id}")
            return False

        # Whitelist check (if whitelist is configured, only whitelisted IDs get replies)
        if config.whitelist_user_ids:
            if sender_id not in config.whitelist_user_ids and chat_id not in config.whitelist_user_ids:
                logger.debug(f"Sender {sender_id} not in whitelist; skipping.")
                return False

        # Channel check
        if message.is_channel and not message.is_group:
            return config.respond_to_channels

        # Group check
        if message.is_group:
            if not config.respond_to_groups:
                return False
            if config.allowed_group_ids and chat_id not in config.allowed_group_ids:
                return False
            # In groups, only respond if Sunnatbek is mentioned or replied to
            is_reply_to_me = False
            if message.is_reply:
                reply_target = await message.get_reply_message()
                if reply_target and reply_target.sender_id == self.owner_id:
                    is_reply_to_me = True

            has_mention = False
            if self.owner_username and message.text:
                has_mention = f"@{self.owner_username}".lower() in message.text.lower()

            return is_reply_to_me or has_mention

        # Private chat with an external contact
        return message.is_private
