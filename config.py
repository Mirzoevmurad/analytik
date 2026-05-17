"""Конфигурация analytik-bot из переменных окружения."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Config:
    bot_token: str
    groq_api_key: str
    allowed_user_ids: frozenset[int]
    db_path: Path
    stt_model: str
    llm_model: str
    max_audio_mb: int
    default_lang: str
    max_context_messages: int
    voice_responses: bool

    @classmethod
    def from_env(cls) -> "Config":
        token = os.getenv("ANALYTIK_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        if not token:
            raise RuntimeError("ANALYTIK_BOT_TOKEN не задан")
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY не задан")

        raw_ids = os.getenv("ALLOWED_USER_IDS", "").strip()
        ids: set[int] = set()
        for chunk in raw_ids.split(","):
            chunk = chunk.strip()
            if chunk:
                try:
                    ids.add(int(chunk))
                except ValueError as exc:
                    raise RuntimeError(f"ALLOWED_USER_IDS: невалидный id {chunk!r}") from exc
        if not ids:
            raise RuntimeError("ALLOWED_USER_IDS должен содержать хотя бы один id")

        return cls(
            bot_token=token,
            groq_api_key=groq_key,
            allowed_user_ids=frozenset(ids),
            db_path=Path(os.getenv("DB_PATH", "data/analytik.sqlite")),
            stt_model=os.getenv("STT_MODEL", "whisper-large-v3-turbo"),
            llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            max_audio_mb=int(os.getenv("MAX_AUDIO_MB", "25")),
            default_lang=os.getenv("DEFAULT_LANG", "auto").lower(),
            max_context_messages=int(os.getenv("MAX_CONTEXT_MESSAGES", "20")),
            voice_responses=_bool(os.getenv("VOICE_RESPONSES", "false")),
        )


def _bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}
