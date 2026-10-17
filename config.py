"""
Configuration module for Sunnatbek's Multimodal Telegram AI Executive Assistant.
Handles environment variables, validation, and system prompt definitions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

# Locate and load .env file
BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()  # Fallback to standard environment lookup


def _parse_int_set(raw_value: str | None) -> Set[int]:
    """Helper to parse comma-separated integers into a set."""
    if not raw_value:
        return set()
    result = set()
    for item in raw_value.split(","):
        cleaned = item.strip()
        if cleaned:
            try:
                result.add(int(cleaned))
            except ValueError:
                pass
    return result


def _parse_int(raw_value: str | None, default: int = 0) -> int:
    """Helper to safely parse integers from env vars even if empty string."""
    if not raw_value or not raw_value.strip():
        return default
    try:
        return int(raw_value.strip())
    except ValueError:
        return default


def _parse_float(raw_value: str | None, default: float = 0.0) -> float:
    """Helper to safely parse floats from env vars even if empty string."""
    if not raw_value or not raw_value.strip():
        return default
    try:
        return float(raw_value.strip())
    except ValueError:
        return default


def _parse_bool(raw_value: str | None, default: bool = False) -> bool:
    """Helper to parse boolean environment variables."""
    if raw_value is None or not raw_value.strip():
        return default
    return raw_value.strip().lower() in ("true", "1", "yes", "y", "t", "on")


@dataclass
class AppConfig:
    """Global configuration container."""

    # --------------------------------------------------------------------------
    # Telegram MTProto Credentials
    # --------------------------------------------------------------------------
    telegram_api_id: int = field(
        default_factory=lambda: _parse_int(os.getenv("TELEGRAM_API_ID"), default=0)
    )
    telegram_api_hash: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_API_HASH", "").strip()
    )
    telegram_session_name: str = field(
        default_factory=lambda: os.getenv(
            "TELEGRAM_SESSION_NAME", "ayubxon_userbot"
        ).strip()
    )
    telegram_phone_number: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_PHONE_NUMBER", "").strip()
    )
    telegram_string_session: str = field(
        default_factory=lambda: os.getenv("TELEGRAM_STRING_SESSION", "")
        .strip()
        .strip('"')
        .strip("'")
        .strip()
    )

    # --------------------------------------------------------------------------
    # AI Engine Settings (Google Gemini or OpenAI)
    # --------------------------------------------------------------------------
    ai_provider: str = field(
        default_factory=lambda: os.getenv(
            "AI_PROVIDER", "gemini" if os.getenv("GEMINI_API_KEY") else "openai"
        ).strip().lower()
    )
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip()
    )
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "").strip()
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o").strip()
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ).strip()
    )
    openai_whisper_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_WHISPER_MODEL", "whisper-1"
        ).strip()
    )

    # --------------------------------------------------------------------------
    # Supabase (PostgreSQL + pgvector RAG)
    # --------------------------------------------------------------------------
    supabase_url: str = field(
        default_factory=lambda: os.getenv("SUPABASE_URL", "").strip()
    )
    supabase_key: str = field(
        default_factory=lambda: os.getenv("SUPABASE_KEY", "").strip()
    )
    supabase_table_name: str = field(
        default_factory=lambda: os.getenv(
            "SUPABASE_TABLE_NAME", "chat_memory"
        ).strip()
    )
    similarity_threshold: float = field(
        default_factory=lambda: _parse_float(
            os.getenv("SIMILARITY_THRESHOLD"), default=0.65
        )
    )
    memory_top_k: int = field(
        default_factory=lambda: _parse_int(
            os.getenv("MEMORY_TOP_K"), default=5
        )
    )
    max_recent_messages: int = field(
        default_factory=lambda: _parse_int(
            os.getenv("MAX_RECENT_MESSAGES"), default=8
        )
    )

    # --------------------------------------------------------------------------
    # Persona & Executive Settings
    # --------------------------------------------------------------------------
    owner_name: str = field(
        default_factory=lambda: os.getenv("OWNER_NAME", "Ayubxon").strip()
    )
    assistant_name: str = field(
        default_factory=lambda: os.getenv(
            "ASSISTANT_NAME", "Ayubxon's Executive Assistant"
        ).strip()
    )
    timezone: str = field(
        default_factory=lambda: os.getenv("TIMEZONE", "Asia/Tashkent").strip()
    )
    reply_prefix: str = field(
        default_factory=lambda: os.getenv("REPLY_PREFIX", "")
    )
    auto_read_messages: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("AUTO_READ_MESSAGES"), default=True
        )
    )
    only_reply_when_offline: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("ONLY_REPLY_WHEN_OFFLINE"), default=True
        )
    )
    reply_delay_seconds: int = field(
        default_factory=lambda: _parse_int(
            os.getenv("REPLY_DELAY_SECONDS"), default=0
        )
    )
    owner_custom_instructions: str = field(
        default_factory=lambda: os.getenv("OWNER_CUSTOM_INSTRUCTIONS", "").strip()
    )

    # --------------------------------------------------------------------------
    # Filtering & Access Control
    # --------------------------------------------------------------------------
    whitelist_user_ids: Set[int] = field(
        default_factory=lambda: _parse_int_set(os.getenv("WHITELIST_USER_IDS"))
    )
    blacklist_user_ids: Set[int] = field(
        default_factory=lambda: _parse_int_set(os.getenv("BLACKLIST_USER_IDS"))
    )
    respond_to_groups: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("RESPOND_TO_GROUPS"), default=False
        )
    )
    respond_to_channels: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("RESPOND_TO_CHANNELS"), default=False
        )
    )
    respond_to_saved_messages: bool = field(
        default_factory=lambda: _parse_bool(
            os.getenv("RESPOND_TO_SAVED_MESSAGES"), default=True
        )
    )
    allowed_group_ids: Set[int] = field(
        default_factory=lambda: _parse_int_set(os.getenv("ALLOWED_GROUP_IDS"))
    )

    # --------------------------------------------------------------------------
    # Media Processing Settings
    # --------------------------------------------------------------------------
    temp_dir: Path = field(
        default_factory=lambda: BASE_DIR
        / os.getenv("TEMP_DIR", "temp_media").strip()
    )
    max_video_frames: int = field(
        default_factory=lambda: _parse_int(
            os.getenv("MAX_VIDEO_FRAMES"), default=8
        )
    )
    frame_sample_interval_sec: float = field(
        default_factory=lambda: _parse_float(
            os.getenv("FRAME_SAMPLE_INTERVAL_SEC"), default=1.5
        )
    )
    max_audio_size_mb: int = field(
        default_factory=lambda: _parse_int(
            os.getenv("MAX_AUDIO_SIZE_MB"), default=25
        )
    )
    max_video_size_mb: int = field(
        default_factory=lambda: _parse_int(
            os.getenv("MAX_VIDEO_SIZE_MB"), default=50
        )
    )

    # Runtime toggle
    is_active: bool = True

    def validate_core(self) -> list[str]:
        """Validate required configuration and return missing parameter errors."""
        missing = []
        if not self.telegram_api_id:
            missing.append("TELEGRAM_API_ID is required (from https://my.telegram.org)")
        if not self.telegram_api_hash:
            missing.append("TELEGRAM_API_HASH is required (from https://my.telegram.org)")
        if not self.openai_api_key and not self.gemini_api_key:
            missing.append(
                "AI Key is required: Please set either GEMINI_API_KEY (Free from https://aistudio.google.com/app/apikey) "
                "or OPENAI_API_KEY (https://platform.openai.com) in .env"
            )
        return missing

    def is_supabase_configured(self) -> bool:
        """Check whether Supabase credentials are provided."""
        return bool(
            self.supabase_url
            and self.supabase_key
            and not self.supabase_url.startswith("https://your-project-id")
        )

    def get_base_system_prompt(self, is_owner_chat: bool = False) -> str:
        """
        Generates persona-specific system prompts depending on whether the caller
        is Ayubxon himself (personal copilot mode) or another contact (executive assistant mode).
        """
        if is_owner_chat:
            return (
                f"Siz {self.owner_name}ning shaxsiy AI yordamchisisiz (Chief of Staff).\n"
                f"Sizning vazifangiz {self.owner_name}ga ishlarni boshqarish, xabarlarni tahlil qilish, "
                f"rejalashtirish va vazifalarni tezkor va aniq bajarishda yordam berishdir.\n\n"
                f"QOIDALAR:\n"
                f"- {self.owner_name} bilan doimo hurmat bilan va aniq, lo'nda muloqot qiling.\n"
                f"- Asosan o'zbek tilida gaplashing (yoki {self.owner_name} qaysi tilda yozsa, shu tilda).\n"
                f"- Ovozli yoki video xabarlarni tahlil qilib, eng asosiy mazmunini ajratib bering.\n"
                f"- Xotiradagi o'tmishdagi suhbatlar va ma'lumotlarga tayaning.\n"
            )

        custom_extra = (
            f"\nQO'SHIMCHA SHAXSIY MA'LUMOTLAR VA KO'RSATMALAR:\n{self.owner_custom_instructions}\n"
            if self.owner_custom_instructions
            else ""
        )

        return (
            f"Siz — {self.owner_name}siz! Telegramda o'z shaxsiy profilingizdan shaxsan o'zingiz nomingizdan yozmoqdasiz.\n"
            f"Ayni vaqtda biroz muhim ishlar bilan band bo'lganingiz yoki xabarni darhol ko'rmaganingiz sababli, "
            f"sizga yozgan odamga (do'stingizga, tanishingizga yoki hamkasbingizga) o'zingiz kabi samimiy, tabiiy va do'stona javob qaytarmoqdasiz.\n\n"
            f"ASOSIY QOIDALAR VA USLUB:\n"
            f"1. HECH QACHON O'ZINGIZNI BOT, SUN'IY INTELLEKT YOKI YORDAMCHI DEMANG! Siz — haqiqiy {self.owner_name}siz. "
            f"Agar kimdir 'Sen botsanmi?' deb so'rasa ham, 'Yo'g'e, o'zimman, sal band edim nima gap?' kabi tabiiy insondek javob bering.\n"
            f"2. TABIIY VA JONLI TIL: Do'stona, samimiy va jonli kundalik o'zbek tilida muloqot qiling. "
            f"Agar suhbatdosh boshqa tilda (rus, ingliz) yozsa, uning tilida javob bering.\n"
            f"3. QISQA VA LO'NDA: Telegramdagi oddiy jonli chat kabi 1-2 jumlada qisqa va lo'nda javob bering. "
            f"Hech qachon cho'zib, rasmiy yoki kitobiy ma'ruza qilmang.\n"
            f"4. DO'STLAR VA TANISHLAR BILAN: Samimiy, erkin va do'stona muloqot qiling. "
            f"Suhbatdosh nima deb yozgan bo'lsa yoki qanday savol bergan bo'lsa, uning xabariga mos ravishda to'g'ri, mantiqiy va o'zingizdek javob bering hamda u bilan tabiiy suhbatlashing (dialog olib boring).\n"
            f"Masalan: agar biror narsa so'rasa, javobini bering yoki 'Hozir sal ishim bor edi, kechroq aniq aytaman' deb ayting. "
            f"Oddiy holatda 'Salom, nima gap?', 'Tinchlikmi, o'zingda nima yangiliklar?', 'Hozir sal band edim, aytaver' kabi tabiiy muloqot qiling.\n"
            f"5. BEGONALAR BILAN: Xushmuomala, madaniyatli va bosiq javob bering.\n"
            f"6. OVOZLI VA VIDEO XABARLAR: Suhbatdosh ovozli yoki video xabar yuborgan bo'lsa, "
            f"uning aytganlarini to'liq tushunib, xuddi tinglagandek mavzu yuzasidan tabiiy javob qaytaring ('Eshitdim, hozir sal bandroq edim...').\n"
            f"7. XAVFSIZLIK: Hech qachon bank karta ma'lumotlari, SMS kodlar, parollar yoki maxfiy narsalarni bermang. "
            f"Moliyaviy yoki jiddiy va'dalar bermang, 'Kechroq o'zim ko'rishganda gaplashamiz' deb ayting.\n"
            f"{custom_extra}"
        )


# Global singleton instance
config = AppConfig()

# Ensure temporary media directory exists
config.temp_dir.mkdir(parents=True, exist_ok=True)
