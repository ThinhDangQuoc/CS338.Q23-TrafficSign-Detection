"""Text-to-speech utilities for Streamlit audio playback."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path

from traffic_sign_app.config import AUDIO_DIR


_LAST_SPOKEN_AT: dict[int, float] = {}


def should_speak(class_id, cooldown_seconds: int = 8, track_id: int | None = None) -> bool:
    """Return True when a class (or track_id) can be spoken again after cooldown."""
    if track_id is not None:
        key = f"track_{track_id}"
    else:
        try:
            key = int(class_id)
        except (TypeError, ValueError):
            key = -1

    now = time.time()
    previous = _LAST_SPOKEN_AT.get(key, 0)
    if now - previous >= cooldown_seconds:
        _LAST_SPOKEN_AT[key] = now
        return True
    return False


async def _edge_tts_save(text: str, output_path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice="vi-VN-HoaiMyNeural")
    await communicate.save(str(output_path))


def text_to_speech(text, output_dir: str | Path = AUDIO_DIR) -> str | None:
    """Convert text to mp3, returning the path or None if TTS fails."""
    if not text:
        return None

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()[:16]
    output_path = out_dir / f"warning_{digest}.mp3"
    if output_path.exists():
        return str(output_path)

    try:
        asyncio.run(_edge_tts_save(str(text), output_path))
        return str(output_path)
    except Exception:
        pass

    try:
        from gtts import gTTS

        gTTS(str(text), lang="vi").save(str(output_path))
        return str(output_path)
    except Exception:
        return None

