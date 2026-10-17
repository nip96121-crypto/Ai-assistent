"""
Telegram String Session Generator for Render / Cloud deployment.
Yangi xavfsiz TELEGRAM_STRING_SESSION yaratish skripti.
"""
from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from telethon import TelegramClient
from telethon.sessions import StringSession

from config import config


async def main():
    print("=" * 70)
    print("🔑 TELEGRAM YANGI STRING SESSION YARATISH")
    print("=" * 70)
    
    api_id = config.telegram_api_id
    api_hash = config.telegram_api_hash
    phone = config.telegram_phone_number or input("Telefon raqamingiz (+998...): ").strip()

    print(f"📱 Telefon: {phone}")
    print(f"📡 Telegram'ga ulanmoqda...")

    # StringSession() bo'sh holda berilsa, yangi toza sessiya kaliti yaratadi
    client = TelegramClient(StringSession(), api_id, api_hash)
    await client.start(phone=phone)

    session_str = client.session.save()
    me = await client.get_me()

    print("\n" + "=" * 70)
    print(f"✅ Muvaffaqiyatli login qilindi: {me.first_name} (@{me.username}) [ID: {me.id}]")
    print("=" * 70)
    print("\n👇 Yangi TELEGRAM_STRING_SESSION kodingiz:\n")
    print(session_str)
    print("\n" + "=" * 70)

    # .env faylini avtomatik yangilash
    try:
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            content = env_path.read_text(encoding="utf-8")
            if "TELEGRAM_STRING_SESSION=" in content:
                new_content = re.sub(
                    r"TELEGRAM_STRING_SESSION=.*",
                    f"TELEGRAM_STRING_SESSION={session_str}",
                    content,
                )
            else:
                new_content = content + f"\nTELEGRAM_STRING_SESSION={session_str}\n"
            env_path.write_text(new_content, encoding="utf-8")
            print("✅ Lokal .env faylidagi TELEGRAM_STRING_SESSION avtomatik yangilandi!")
    except Exception as e:
        print(f"⚠️ .env faylini yangilashda ogohlantirish: {e}")

    print("\n🚀 RENDER'DA ISHLATISH UCHUN QADAMLAR:")
    print("1. Yuqoridagi uzun session kodni to'liq nusxalab oling (Copy).")
    print("2. Render.com saytidagi botingiz sahifasiga kiring.")
    print("3. Chap menyudan 'Environment' bo'limini oching.")
    print("4. 'TELEGRAM_STRING_SESSION' qatoriga shu yangi kodni qo'ying (Paste).")
    print("5. 'Save changes' tugmasini bosing.")
    print("6. Render yangi kalit bilan botni avtomatik ishga tushiradi!")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
