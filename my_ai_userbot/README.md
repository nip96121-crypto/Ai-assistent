# 🧠 Sunnatbek's Multimodal Telegram AI Executive Assistant

A production-grade, multimodal Telegram AI Userbot powered by the **MTProto protocol** (Telethon), **OpenAI GPT-4o & Whisper**, and **Supabase pgvector (RAG)**.

This userbot functions as a personal executive assistant on **Sunnatbek's** personal Telegram account. It autonomously processes incoming text, voice notes, and video notes, reasons about them with contextual long-term memory, and dispatches articulate replies.

---

## 🌟 Key Features

- **MTProto Protocol (Telethon)**: Direct integration with Telegram user accounts without standard bot API constraints.
- **Multimodal AI Brain (OpenAI GPT-4o)**:
  - **Text Understanding**: Context-aware executive replies matching the sender's tone and language.
  - **Voice Notes (Whisper STT)**: Automatically downloads and transcribes incoming voice messages.
  - **Video Notes & Clips (OpenCV + GPT-4o Vision)**: Extracts keyframes and transcribes audio track for combined visual + auditory reasoning.
- **Persistent Long-Term Memory (RAG via Supabase pgvector)**:
  - Semantic vector search using `text-embedding-3-small`.
  - Automatically recalls past discussions, commitments, and user details across chats.
  - Graceful in-memory fallback if Supabase is offline.
- **Executive Persona & Dual Mode**:
  - **Copilot Mode**: When chatting in *Saved Messages* or with oneself, acts as a direct Chief of Staff for Sunnatbek.
  - **Executive Assistant Mode**: When answering external contacts, acts as Sunnatbek's polite and articulate executive assistant.
- **Realistic Human Simulation**:
  - Displays `"Typing..."`, `"Recording voice..."`, or `"Recording video..."` status while processing media and generating responses.
  - Automatically marks incoming messages as read.
- **Filtering & Security Controls**:
  - Whitelist & Blacklist support.
  - Group chat mention/reply filters to prevent accidental spam.
  - Anti-loop safeguard (never replies to its own outgoing messages).
  - Runtime management commands (`.ai on`, `.ai off`, `.ai status`, `.ai clear`).

---

## 🏗️ Architecture & Modular Structure

```text
my_ai_userbot/
├── .env                    # Environment variables (API keys, credentials)
├── .env.example            # Configuration template with full documentation
├── main.py                 # Telethon client initialization and event routing
├── config.py               # Strongly-typed environment configuration & persona
├── requirements.txt        # Python package dependencies
├── supabase_schema.sql     # PostgreSQL + pgvector database migration & RPC functions
├── ai_engine/
│   ├── __init__.py
│   ├── llm_handler.py      # OpenAI GPT-4o integration, prompt engineering & RAG
│   └── memory.py           # Supabase vector DB integration (semantic search & storage)
├── media_processor/
│   ├── __init__.py
│   ├── audio_handler.py    # Voice note downloader & OpenAI Whisper STT
│   └── video_handler.py    # FFmpeg audio extraction & OpenCV frame sampling (GPT-4V)
└── utils/
    ├── __init__.py
    ├── filters.py          # Whitelist/Blacklist logic, anti-loop rules & .ai commands
    └── telegram_helpers.py # Simulating chat actions ('typing...') & safe chunked sender
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.11+**
- **FFmpeg**: (Recommended for extracting audio from large video files)
  - **Windows**: `winget install Gyan.FFmpeg` or `choco install ffmpeg`
  - **macOS**: `brew install ffmpeg`
  - **Linux (Ubuntu/Debian)**: `sudo apt update && sudo apt install -y ffmpeg`

### 2. Installation

Clone or open the project folder in your terminal:

```bash
cd my_ai_userbot
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

---

### 3. Setup Credentials in `.env`

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

#### A. Telegram MTProto Credentials
1. Visit [my.telegram.org](https://my.telegram.org) and log in with your phone number.
2. Go to **API development tools**.
3. Create a new app to obtain your `api_id` and `api_hash`.
4. Enter them in `.env`:
   ```env
   TELEGRAM_API_ID=12345678
   TELEGRAM_API_HASH=your_api_hash_here
   TELEGRAM_PHONE_NUMBER=+998901234567
   ```

#### B. OpenAI API Key
1. Go to [platform.openai.com](https://platform.openai.com/api-keys) and generate an API key.
2. Enter in `.env`:
   ```env
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxx
   OPENAI_MODEL=gpt-4o
   ```

#### C. Supabase Database & pgvector (RAG Memory)
1. Create a free project at [supabase.com](https://supabase.com).
2. Go to the **SQL Editor** in your Supabase dashboard.
3. Open [`supabase_schema.sql`](supabase_schema.sql), paste the entire script into the SQL Editor, and click **Run**.
4. In Supabase Dashboard -> **Project Settings** -> **API**, copy:
   - **Project URL** (`SUPABASE_URL`)
   - **Service Role Key** or **anon key** (`SUPABASE_KEY`)
5. Enter them in `.env`:
   ```env
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your-supabase-service-key
   ```

---

### 4. Running the Userbot

Launch the userbot:

```bash
python main.py
```

On first launch, Telethon will ask for your login code sent to your Telegram app (and your 2FA password if enabled).
Once authenticated, a session file (`sunnatbek_userbot.session`) is created so you won't need to log in again.

---

## 🕹️ In-App Control Commands

Sunnatbek can send these commands in any Telegram chat (or Saved Messages) to manage the assistant on the fly:

| Command | Action |
|---|---|
| `.ai status` | Displays userbot health, active model, and memory state |
| `.ai on` | Enables autonomous AI replies |
| `.ai off` | Pauses autonomous AI replies |
| `.ai clear` | Clears conversation memory for the current chat |
| `.ai whitelist add <ID>` | Whitelists a user ID to receive AI replies |
| `.ai whitelist remove <ID>` | Removes a user ID from the whitelist |
| `.ai blacklist add <ID>` | Blocks a user ID from triggering AI replies |
| `.ai help` | Displays the command manual |

---

## ⚙️ Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `OWNER_NAME` | `Sunnatbek` | Name of the Telegram account owner |
| `ASSISTANT_NAME` | `Sunnatbek's Executive Assistant` | Persona name for the AI |
| `TIMEZONE` | `Asia/Tashkent` | Timezone used for timestamps and reminders |
| `REPLY_PREFIX` | `🤖 ` | Prefix added to replies (leave empty for invisible stealth replies) |
| `RESPOND_TO_GROUPS` | `False` | Whether to reply in groups (only triggers on mention/reply) |
| `RESPOND_TO_SAVED_MESSAGES` | `True` | Enables personal copilot mode in Saved Messages |
| `MAX_VIDEO_FRAMES` | `8` | Max keyframes extracted per video note for GPT-4o Vision |
| `FRAME_SAMPLE_INTERVAL_SEC` | `1.5` | Interval in seconds between sampled video frames |
