"""
Audio processing module for Telegram voice notes and audio clips.
Transcribes speech using OpenAI Whisper or Google Gemini multimodal audio.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Optional, Tuple

from openai import AsyncOpenAI
from telethon.tl.custom import Message

from config import config

logger = logging.getLogger("Userbot.Audio")


class AudioHandler:
    """Handles downloading voice messages and transcribing them via Whisper or Gemini."""

    def __init__(self, openai_client: Optional[AsyncOpenAI] = None):
        self.openai_client = None
        if config.openai_api_key:
            self.openai_client = openai_client or AsyncOpenAI(api_key=config.openai_api_key)

        self.gemini_client = None
        if config.gemini_api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=config.gemini_api_key)
            except Exception as e:
                logger.warning(f"Could not load Gemini client in AudioHandler: {e}")

    async def process_voice_message(self, message: Message) -> Tuple[Optional[str], Optional[float]]:
        """
        Download a Telegram voice or audio message and transcribe it.

        Returns:
            Tuple[Optional[str], Optional[float]]: (transcription_text, duration_seconds)
        """
        if not message.media:
            return None, None

        # Determine audio duration if available in Telegram metadata
        duration = None
        if hasattr(message, "voice") and message.voice:
            duration = getattr(message.voice.attributes[0], "duration", None)
        elif hasattr(message, "audio") and message.audio:
            for attr in message.audio.attributes:
                if hasattr(attr, "duration"):
                    duration = attr.duration
                    break

        temp_filename = f"voice_{uuid.uuid4().hex[:8]}.ogg"
        temp_path = config.temp_dir / temp_filename

        try:
            logger.info(f"Downloading voice message to {temp_path}...")
            downloaded_file = await message.download_media(file=str(temp_path))

            if not downloaded_file or not os.path.exists(downloaded_file):
                logger.error("Failed to download voice file from Telegram.")
                return None, duration

            file_size_mb = os.path.getsize(downloaded_file) / (1024 * 1024)
            if file_size_mb > config.max_audio_size_mb:
                logger.warning(
                    f"Voice message size ({file_size_mb:.2f} MB) exceeds maximum limit "
                    f"of {config.max_audio_size_mb} MB."
                )
                return "[Voice message exceeds size limit]", duration

            transcript = await self.transcribe_audio_file(Path(downloaded_file))
            return transcript, duration

        except Exception as e:
            logger.error(f"Error processing voice message: {e}", exc_info=True)
            return None, duration

        finally:
            if temp_path.exists():
                try:
                    os.remove(temp_path)
                    logger.debug(f"Cleaned up temporary audio file: {temp_path}")
                except Exception as cleanup_err:
                    logger.warning(f"Failed to delete temp file {temp_path}: {cleanup_err}")

    async def transcribe_audio_file(self, file_path: Path) -> Optional[str]:
        """
        Transcribe audio using OpenAI Whisper or Google Gemini.
        """
        if not file_path.exists() or os.path.getsize(file_path) == 0:
            logger.warning(f"Audio file {file_path} is missing or empty.")
            return None

        # 1. If OpenAI is available and configured
        if self.openai_client is not None:
            try:
                logger.info(f"Transcribing audio file {file_path.name} via OpenAI Whisper...")
                with open(file_path, "rb") as audio_file:
                    transcription = await self.openai_client.audio.transcriptions.create(
                        model=config.openai_whisper_model,
                        file=audio_file,
                    )
                result_text = transcription.text.strip() if transcription.text else ""
                logger.info(f"Whisper transcription: \"{result_text[:60]}...\"")
                return result_text
            except Exception as e:
                logger.warning(f"OpenAI Whisper transcription failed: {e}. Trying Gemini if available...")

        # 2. If Gemini is available
        if self.gemini_client is not None:
            try:
                logger.info(f"Transcribing audio file {file_path.name} via Google Gemini...")
                from google.genai import types

                audio_bytes = file_path.read_bytes()
                mime = "audio/ogg" if file_path.suffix.lower() in (".ogg", ".oga") else "audio/mp3"

                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime)
                prompt = (
                    "Transcribe the spoken words in this audio recording accurately word for word. "
                    "Output only the transcribed text, nothing else."
                )

                response = await self.gemini_client.aio.models.generate_content(
                    model=config.gemini_model,
                    contents=[audio_part, prompt],
                )
                result_text = response.text.strip() if response.text else ""
                logger.info(f"Gemini audio transcription: \"{result_text[:60]}...\"")
                return result_text
            except Exception as e:
                logger.error(f"Gemini audio transcription failed: {e}", exc_info=True)

        return None
