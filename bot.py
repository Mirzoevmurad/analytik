"""Analytik-bot: Telegram чат-бот аналитик на Groq LLM."""
from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from telegram import BotCommand, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from config import Config
from db import Database
from llm import DEFAULT_SYSTEM_PROMPT, GroqLLM, LLMError
from stt import GroqSTT, STTError


try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001
    pass


logger = logging.getLogger("analytik")


# ---- rate limiter ------------------------------------------------------

class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self._requests: dict[int, list[float]] = defaultdict(list)
        self._max = max_requests
        self._window = window_seconds

    def is_allowed(self, user_id: int) -> bool:
        now = time.time()
        self._requests[user_id] = [
            t for t in self._requests[user_id] if now - t < self._window
        ]
        if len(self._requests[user_id]) >= self._max:
            return False
        self._requests[user_id].append(now)
        return True


# ---- helpers -----------------------------------------------------------

def _is_allowed(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
    cfg: Config = context.bot_data["cfg"]
    return user_id in cfg.allowed_user_ids


async def _deny(update: Update) -> None:
    await update.effective_message.reply_text(
        "Этот бот приватный. Обратитесь к владельцу."
    )


def _split_for_telegram(text: str, limit: int = 4000) -> list[str]:
    """Делит длинное сообщение на куски <= limit символов."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        # ищем последний перенос строки в пределах лимита
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = text.rfind(" ", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


async def _convert_to_wav(src: Path, dst: Path) -> None:
    """OGG/Opus -> 16kHz mono WAV."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    returncode = await proc.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {returncode}")


async def _tts_reply(msg, text: str) -> None:
    """Отправляет голосовой ответ через gTTS."""
    try:
        from gtts import gTTS
    except ImportError:
        logger.warning("gTTS не установлен, голосовой ответ пропущен")
        return

    # определяем язык: если есть кириллица — ru, иначе en
    lang = "ru"
    for ch in text[:200]:
        if "a" <= ch.lower() <= "z":
            lang = "en"
            break

    # ограничиваем длину текста для TTS (gTTS имеет лимиты)
    tts_text = text[:3000] if len(text) > 3000 else text

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tts_path = Path(f.name)

    try:
        tts = gTTS(text=tts_text, lang=lang)
        tts.save(str(tts_path))
        with tts_path.open("rb") as audio:
            await msg.reply_voice(voice=audio)
    except Exception as e:  # noqa: BLE001
        logger.warning("TTS failed: %s", e)
    finally:
        tts_path.unlink(missing_ok=True)


# ---- command handlers --------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return
    db: Database = context.bot_data["db"]
    db.upsert_user(user.id, user.username, user.first_name, user.last_name)
    await update.effective_message.reply_text(
        "Привет! Я Analytik — твой ИИ-аналитик данных.\n\n"
        "Просто напиши или отправь голосовое сообщение — "
        "я отвечу как аналитик данных.\n\n"
        "Команды:\n"
        "/clear — очистить контекст беседы\n"
        "/system — показать / сменить системный промпт\n"
        "/export — экспорт истории чата\n"
        "/voice — вкл/выкл голосовые ответы\n"
        "/help — справка",
        parse_mode=None,
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(context, update.effective_user.id):
        await _deny(update)
        return
    await update.effective_message.reply_text(
        "Analytik Bot — чат с ИИ-аналитиком\n\n"
        "Как пользоваться:\n"
        "— Просто напишите текст или отправьте голосовое\n"
        "— Бот помнит контекст беседы (последние сообщения)\n"
        "— Можно спрашивать про SQL, Python, данные, статистику\n\n"
        "Команды:\n"
        "/start — приветствие\n"
        "/clear — очистить контекст (начать заново)\n"
        "/system — показать текущий системный промпт\n"
        "/system <текст> — установить новый системный промпт\n"
        "/system reset — вернуть промпт по умолчанию\n"
        "/export — экспорт всей истории чата в файл\n"
        "/voice — вкл/выкл голосовые ответы\n"
        "/help — эта справка",
        parse_mode=None,
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return
    db: Database = context.bot_data["db"]
    count = db.clear_context(user.id)
    await update.effective_message.reply_text(
        f"Контекст очищен ({count} сообщений удалено). Начинаем с чистого листа.",
        parse_mode=None,
    )


async def cmd_system(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return
    db: Database = context.bot_data["db"]

    args_text = " ".join(context.args or []).strip() if context.args else ""

    if not args_text:
        # показать текущий промпт
        current = db.get_system_prompt(user.id) or DEFAULT_SYSTEM_PROMPT
        label = "(пользовательский)" if db.get_system_prompt(user.id) else "(по умолчанию)"
        await update.effective_message.reply_text(
            f"Текущий системный промпт {label}:\n\n{current}",
            parse_mode=None,
        )
        return

    if args_text.lower() == "reset":
        db.set_system_prompt(user.id, None)
        await update.effective_message.reply_text(
            "Системный промпт сброшен на стандартный (аналитик данных).",
            parse_mode=None,
        )
        return

    db.set_system_prompt(user.id, args_text)
    await update.effective_message.reply_text(
        f"Системный промпт обновлён:\n\n{args_text}",
        parse_mode=None,
    )


async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return
    db: Database = context.bot_data["db"]
    history = db.export_history(user.id)

    if not history:
        await update.effective_message.reply_text(
            "История чата пуста.", parse_mode=None,
        )
        return

    lines: list[str] = []
    for msg_data in history:
        role = "Вы" if msg_data["role"] == "user" else "Analytik"
        ts = msg_data["created_at"] or ""
        lines.append(f"[{ts}] {role}:\n{msg_data['content']}\n")

    export_text = "\n".join(lines)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="analytik_export_", delete=False, encoding="utf-8",
    ) as f:
        f.write(export_text)
        export_path = Path(f.name)

    try:
        with export_path.open("rb") as f:
            await update.effective_message.reply_document(
                document=f,
                filename=f"analytik_chat_{user.id}.txt",
                caption=f"История чата ({len(history)} сообщений)",
            )
    finally:
        export_path.unlink(missing_ok=True)


async def cmd_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return
    db: Database = context.bot_data["db"]
    current = db.get_voice_responses(user.id)
    new_state = not current
    db.set_voice_responses(user.id, new_state)
    status = "включены" if new_state else "выключены"
    await update.effective_message.reply_text(
        f"Голосовые ответы {status}.", parse_mode=None,
    )


# ---- message handlers --------------------------------------------------

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return

    rate_limiter: RateLimiter = context.bot_data["rate_limiter"]
    if not rate_limiter.is_allowed(user.id):
        await update.effective_message.reply_text(
            "Слишком много запросов. Подождите минуту.", parse_mode=None,
        )
        return

    question = (update.effective_message.text or "").strip()
    if not question:
        return

    await _process_chat(update, context, question=question, source="text")


async def on_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return

    rate_limiter: RateLimiter = context.bot_data["rate_limiter"]
    if not rate_limiter.is_allowed(user.id):
        await update.effective_message.reply_text(
            "Слишком много запросов. Подождите минуту.", parse_mode=None,
        )
        return

    cfg: Config = context.bot_data["cfg"]
    stt: GroqSTT = context.bot_data["stt"]
    msg = update.effective_message
    voice = msg.voice or msg.audio
    if voice is None:
        return

    # проверка размера
    if voice.file_size and voice.file_size > cfg.max_audio_mb * 1024 * 1024:
        await msg.reply_text(
            f"Аудио слишком большое (лимит {cfg.max_audio_mb} МБ).", parse_mode=None,
        )
        return

    placeholder = await msg.reply_text("Распознаю голос...", parse_mode=None)
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    try:
        tg_file = await voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as f:
            ogg_path = Path(f.name)
        await tg_file.download_to_drive(ogg_path)

        wav_path = ogg_path.with_suffix(".wav")
        await _convert_to_wav(ogg_path, wav_path)

        result = await stt.transcribe(wav_path, lang=cfg.default_lang)
        transcript = result["text"]
    except STTError as e:
        await placeholder.edit_text(f"Не удалось распознать речь: {e}")
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("STT failed")
        await placeholder.edit_text(f"Ошибка распознавания ({type(e).__name__}).")
        return
    finally:
        for p in (ogg_path, wav_path):
            try:
                p.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    await placeholder.edit_text(f"Распознано: {transcript[:200]}...")
    await _process_chat(
        update, context, question=transcript, source="voice", placeholder=placeholder,
    )


async def _process_chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    question: str,
    source: str,
    placeholder=None,
) -> None:
    """Основной пайплайн: получает вопрос, добавляет контекст, отправляет LLM, сохраняет ответ."""
    cfg: Config = context.bot_data["cfg"]
    db: Database = context.bot_data["db"]
    llm: GroqLLM = context.bot_data["llm"]
    user = update.effective_user
    msg = update.effective_message

    # обновляем пользователя
    db.upsert_user(user.id, user.username, user.first_name, user.last_name)

    # получаем контекст
    chat_context = db.get_context(user.id, limit=cfg.max_context_messages)

    # получаем системный промпт
    system_prompt = db.get_system_prompt(user.id) or DEFAULT_SYSTEM_PROMPT

    # показываем "печатает..."
    if placeholder is None:
        placeholder = await msg.reply_text("Думаю...", parse_mode=None)
    else:
        await placeholder.edit_text("Думаю...")
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    try:
        answer = await llm.chat(question, system_prompt, context=chat_context)
    except LLMError as e:
        await placeholder.edit_text(f"Не смог ответить: {e}")
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("LLM chat failed")
        await placeholder.edit_text(f"Ошибка ИИ ({type(e).__name__}).")
        return

    # сохраняем в историю
    db.add_message(user.id, "user", question)
    db.add_message(user.id, "assistant", answer)

    # отправляем ответ
    chunks = _split_for_telegram(answer)
    for i, chunk in enumerate(chunks):
        if i == 0:
            try:
                await placeholder.edit_text(chunk, parse_mode=None)
            except Exception:  # noqa: BLE001
                await msg.reply_text(chunk, parse_mode=None)
        else:
            await msg.reply_text(chunk, parse_mode=None)

    # голосовой ответ
    voice_enabled = db.get_voice_responses(user.id)
    if voice_enabled:
        await _tts_reply(msg, answer)

    logger.info(
        "chat user=%d source=%s q_len=%d a_len=%d ctx=%d",
        user.id, source, len(question), len(answer), len(chat_context),
    )


# ---- bootstrap ---------------------------------------------------------

BOT_COMMANDS: list[tuple[str, str]] = [
    ("start", "Приветствие"),
    ("help", "Справка"),
    ("clear", "Очистить контекст беседы"),
    ("system", "Показать / сменить системный промпт"),
    ("export", "Экспорт истории чата"),
    ("voice", "Вкл/выкл голосовые ответы"),
]


async def _on_post_init(app: Application) -> None:
    try:
        await app.bot.set_my_commands(
            [BotCommand(name, desc) for name, desc in BOT_COMMANDS]
        )
        logger.info("set %d bot commands", len(BOT_COMMANDS))
    except Exception:  # noqa: BLE001
        logger.exception("failed to set bot commands")


def build_app() -> Application:
    cfg = Config.from_env()
    db = Database(cfg.db_path)
    stt = GroqSTT(cfg.groq_api_key, cfg.stt_model)
    llm = GroqLLM(cfg.groq_api_key, cfg.llm_model)

    app = (
        ApplicationBuilder()
        .token(cfg.bot_token)
        .post_init(_on_post_init)
        .build()
    )
    app.bot_data["cfg"] = cfg
    app.bot_data["db"] = db
    app.bot_data["stt"] = stt
    app.bot_data["llm"] = llm
    app.bot_data["rate_limiter"] = RateLimiter(max_requests=10, window_seconds=60)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("system", cmd_system))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(CommandHandler("voice", cmd_voice))

    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, on_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    app.add_error_handler(_error)
    return app


async def _error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled update error", exc_info=context.error)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    app = build_app()
    cfg: Config = app.bot_data["cfg"]
    logger.info(
        "Analytik-bot запущен. STT=%s, LLM=%s, пользователей: %d",
        cfg.stt_model, cfg.llm_model, len(cfg.allowed_user_ids),
    )
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
