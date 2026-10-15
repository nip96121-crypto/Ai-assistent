"""
Video processing module for Telegram video notes and video clips.
Extracts audio for Whisper transcription and samples keyframes with OpenCV for GPT-4o Vision.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
from telethon.tl.custom import Message

from config import config
from .audio_handler import AudioHandler

logger = logging.getLogger("Userbot.Video")


class VideoHandler:
    """Processes video notes and video files into Whisper transcripts and OpenCV frames."""

    def __init__(self, audio_handler: AudioHandler):
        self.audio_handler = audio_handler
        self._ffmpeg_path = shutil.which("ffmpeg")

    async def process_video_message(self, message: Message) -> Dict[str, Any]:
        """
        Download video or video note, extract audio transcript, and extract visual keyframes.

        Returns:
            Dict[str, Any] with keys:
                - transcription: Optional[str]
                - frames_b64: List[str]
                - duration: float
                - frame_count: int
        """
        temp_video_name = f"video_{uuid.uuid4().hex[:8]}.mp4"
        temp_video_path = config.temp_dir / temp_video_name
        temp_audio_path = config.temp_dir / f"audio_extracted_{uuid.uuid4().hex[:8]}.mp3"

        result: Dict[str, Any] = {
            "transcription": None,
            "frames_b64": [],
            "duration": 0.0,
            "frame_count": 0,
        }

        try:
            logger.info(f"Downloading video message to {temp_video_path}...")
            downloaded_file = await message.download_media(file=str(temp_video_path))

            if not downloaded_file or not os.path.exists(downloaded_file):
                logger.error("Failed to download video file from Telegram.")
                return result

            video_path = Path(downloaded_file)
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            if file_size_mb > config.max_video_size_mb:
                logger.warning(
                    f"Video size ({file_size_mb:.2f} MB) exceeds maximum allowed "
                    f"limit of {config.max_video_size_mb} MB."
                )
                result["transcription"] = "[Video size exceeded processing limit]"
                return result

            # Run audio extraction and visual frame extraction concurrently
            audio_task = self._extract_and_transcribe_audio(video_path, temp_audio_path, file_size_mb)
            frames_task = asyncio.to_thread(self._extract_frames_opencv, video_path)

            transcript, (frames_b64, duration, frame_count) = await asyncio.gather(
                audio_task, frames_task
            )

            result["transcription"] = transcript
            result["frames_b64"] = frames_b64
            result["duration"] = duration
            result["frame_count"] = frame_count

            logger.info(
                f"Video processed successfully: {len(frames_b64)} frames, "
                f"duration={duration:.1f}s, transcript_len={len(transcript or '')}"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing video message: {e}", exc_info=True)
            return result

        finally:
            # Clean up temporary video and extracted audio files
            for p in (temp_video_path, temp_audio_path):
                if p.exists():
                    try:
                        os.remove(p)
                    except Exception as err:
                        logger.warning(f"Failed to delete temp file {p}: {err}")

    async def _extract_and_transcribe_audio(
        self, video_path: Path, temp_audio_path: Path, file_size_mb: float
    ) -> Optional[str]:
        """Extract audio track via FFmpeg and transcribe with Whisper (with fallback)."""
        has_ffmpeg = self._ffmpeg_path is not None

        if has_ffmpeg:
            try:
                logger.debug(f"Extracting audio using FFmpeg from {video_path.name}...")
                cmd = [
                    self._ffmpeg_path,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-acodec",
                    "libmp3lame",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(temp_audio_path),
                ]

                def _run_ffmpeg():
                    return subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=30,
                    )

                proc = await asyncio.to_thread(_run_ffmpeg)
                if proc.returncode == 0 and temp_audio_path.exists() and os.path.getsize(temp_audio_path) > 0:
                    return await self.audio_handler.transcribe_audio_file(temp_audio_path)
                else:
                    logger.warning("FFmpeg did not produce audio (video may be silent or audio track missing).")
            except Exception as e:
                logger.warning(f"FFmpeg extraction failed ({e}). Attempting direct upload fallback...")

        # Fallback: Whisper directly supports mp4 video files up to 25MB
        if file_size_mb <= 25.0:
            logger.info("Transcribing video file directly with Whisper...")
            return await self.audio_handler.transcribe_audio_file(video_path)
        else:
            logger.warning(
                f"Video is {file_size_mb:.1f} MB and FFmpeg is not available to extract audio. "
                "Install FFmpeg to support large video audio extraction."
            )
            return None

    def _extract_frames_opencv(self, video_path: Path) -> Tuple[List[str], float, int]:
        """
        Extract sampled keyframes uniformly across the video using OpenCV.
        Downscales frames and converts to base64 JPEG strings for GPT-4o.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"OpenCV could not open video: {video_path}")
            return [], 0.0, 0

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = total_frames / fps if fps > 0 else 0.0

            if total_frames <= 0 or duration <= 0:
                logger.warning(f"Empty or unreadable video frames in {video_path.name}")
                return [], duration, total_frames

            # Determine timestamps to sample
            max_frames = config.max_video_frames
            sample_interval = config.frame_sample_interval_sec

            # Calculate target timestamps
            timestamps: List[float] = []
            curr_t = 0.5  # Start 0.5s in to skip initial dark/empty frames
            while curr_t < duration and len(timestamps) < max_frames:
                timestamps.append(curr_t)
                curr_t += sample_interval

            # If video is very short, sample at least the middle frame
            if not timestamps:
                timestamps.append(duration / 2.0)

            frames_b64: List[str] = []

            for t_sec in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
                success, frame = cap.read()
                if not success or frame is None:
                    continue

                # Resize to max 768px on the longest dimension
                h, w = frame.shape[:2]
                max_dim = 768
                if max(h, w) > max_dim:
                    scale = max_dim / float(max(h, w))
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Encode to JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                success_enc, buffer = cv2.imencode(".jpg", frame, encode_param)
                if success_enc:
                    b64_str = base64.b64encode(buffer).decode("utf-8")
                    frames_b64.append(b64_str)

            return frames_b64, duration, total_frames

        finally:
            cap.release()
