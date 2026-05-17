"""Analytik-bot: Telegram чат-бот аналитик на Groq LLM."""
from __future__ import annotations

import asyncio
import logging
import signal
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from gtts import gTTS
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
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

MAX_SYSTEM_PROMPT_LEN = 2000


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


# ---- inline keyboards --------------------------------------------------

def _main_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🗑 Очистить контекст", callback_data="menu_clear"),
            InlineKeyboardButton("🎙 Голос вкл/выкл", callback_data="menu_voice"),
        ],
        [
            InlineKeyboardButton("⚙️ Системный промпт", callback_data="menu_system"),
            InlineKeyboardButton("📤 Экспорт чата", callback_data="menu_export"),
        ],
        [
            InlineKeyboardButton("❓ Справка", callback_data="menu_help"),
        ],
    ])


def _confirm_clear_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("✅ Да, очистить", callback_data="clear_yes"),
            InlineKeyboardButton("❌ Отмена", callback_data="clear_no"),
        ],
    ])


def _retry_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Повторить", callback_data="retry")],
    ])


# ---- helpers -----------------------------------------------------------

def _is_allowed(context: ContextTypes.DEFAULT_TYPE, user_id: int) -> bool:
    cfg: Config = context.bot_data["cfg"]
    return user_id in cfg.allowed_user_ids


async def _deny(update: Update) -> None:
    target = update.effective_message or (update.callback_query and update.callback_query.message)
    if target:
        await target.reply_text("Этот бот приватный. Обратитесь к владельцу.")


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


def _escape_md(text: str) -> str:
    """Мягкое экранирование для MarkdownV2 — только вне блоков кода."""
    special = r"_*[]()~`>#+-=|{}.!"
    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue
        if in_code_block:
            result.append(line)
            continue
        escaped = ""
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "`":
                end = line.find("`", i + 1)
                if end != -1:
                    escaped += line[i:end + 1]
                    i = end + 1
                    continue
            if ch in special:
                escaped += "\\" + ch
            else:
                escaped += ch
            i += 1
        result.append(escaped)

    return "\n".join(result)


async def _safe_reply(msg, text: str, reply_markup=None) -> None:
    """Отправляет ответ, пробуя MarkdownV2, затем plain text."""
    try:
        escaped = _escape_md(text)
        await msg.edit_text(
            escaped,
            parse_mode=ParseMode.MARKDOWN_V2,
            reply_markup=reply_markup,
        )
    except Exception:  # noqa: BLE001
        try:
            await msg.edit_text(text, parse_mode=None, reply_markup=reply_markup)
        except Exception:  # noqa: BLE001
            chunks = _split_for_telegram(text)
            for chunk in chunks:
                await msg.reply_text(chunk, parse_mode=None)


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
    lang = "en"
    for ch in text[:200]:
        if "\u0400" <= ch <= "\u04FF":
            lang = "ru"
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

    msg_count = db.message_count(user.id)
    ctx_info = f"\n\n💬 В контексте: {msg_count} сообщений" if msg_count > 0 else ""

    await update.effective_message.reply_text(
        f"Привет, {user.first_name}! Я Analytik — твой ИИ-аналитик данных.\n\n"
        "Просто напиши или отправь голосовое сообщение — "
        "я отвечу как аналитик данных."
        + ctx_info,
        parse_mode=None,
        reply_markup=_main_keyboard(),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(context, update.effective_user.id):
        await _deny(update)
        return
    await update.effective_message.reply_text(
        "📖 Analytik Bot — справка\n\n"
        "Как пользоваться:\n"
        "— Просто напишите текст или отправьте голосовое\n"
        "— Бот помнит контекст беседы\n"
        "— Спрашивайте про SQL, Python, данные, статистику\n\n"
        "Команды:\n"
        "/start — приветствие и меню\n"
        "/clear — очистить контекст\n"
        "/system — показать / сменить системный промпт\n"
        "/system <текст> — установить новый промпт\n"
        "/system reset — вернуть промпт по умолчанию\n"
        "/export — экспорт истории чата\n"
        "/voice — вкл/выкл голосовые ответы\n"
        "/help — эта справка",
        parse_mode=None,
        reply_markup=_main_keyboard(),
    )


async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return
    db: Database = context.bot_data["db"]
    msg_count = db.message_count(user.id)
    if msg_count == 0:
        await update.effective_message.reply_text(
            "Контекст уже пуст.", parse_mode=None,
        )
        return
    await update.effective_message.reply_text(
        f"Удалить {msg_count} сообщений из контекста?",
        parse_mode=None,
        reply_markup=_confirm_clear_keyboard(),
    )


async def cmd_system(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return
    db: Database = context.bot_data["db"]

    args_text = " ".join(context.args or []).strip() if context.args else ""

    if not args_text:
        current = db.get_system_prompt(user.id) or DEFAULT_SYSTEM_PROMPT
        label = "(пользовательский)" if db.get_system_prompt(user.id) else "(по умолчанию)"
        display = current[:500] + "..." if len(current) > 500 else current
        await update.effective_message.reply_text(
            f"⚙️ Системный промпт {label}:\n\n{display}",
            parse_mode=None,
            reply_markup=_main_keyboard(),
        )
        return

    if args_text.lower() == "reset":
        db.set_system_prompt(user.id, None)
        await update.effective_message.reply_text(
            "✅ Системный промпт сброшен на стандартный (аналитик данных).",
            parse_mode=None,
            reply_markup=_main_keyboard(),
        )
        return

    if len(args_text) > MAX_SYSTEM_PROMPT_LEN:
        await update.effective_message.reply_text(
            f"❌ Промпт слишком длинный ({len(args_text)} символов). "
            f"Максимум: {MAX_SYSTEM_PROMPT_LEN}.",
            parse_mode=None,
        )
        return

    db.set_system_prompt(user.id, args_text)
    await update.effective_message.reply_text(
        f"✅ Системный промпт обновлён:\n\n{args_text}",
        parse_mode=None,
        reply_markup=_main_keyboard(),
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
                caption=f"📤 История чата ({len(history)} сообщений)",
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
    icon = "🔊" if new_state else "🔇"
    status = "включены" if new_state else "выключены"
    await update.effective_message.reply_text(
        f"{icon} Голосовые ответы {status}.",
        parse_mode=None,
        reply_markup=_main_keyboard(),
    )


# ---- callback query handler -------------------------------------------

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    user = update.effective_user

    if not _is_allowed(context, user.id):
        await _deny(update)
        return

    data = query.data or ""
    db: Database = context.bot_data["db"]

    if data == "menu_clear":
        msg_count = db.message_count(user.id)
        if msg_count == 0:
            await query.message.reply_text("Контекст уже пуст.", parse_mode=None)
            return
        await query.message.reply_text(
            f"Удалить {msg_count} сообщений из контекста?",
            parse_mode=None,
            reply_markup=_confirm_clear_keyboard(),
        )

    elif data == "clear_yes":
        count = db.clear_context(user.id)
        await query.message.edit_text(
            f"🗑 Контекст очищен ({count} сообщений удалено).",
            parse_mode=None,
        )

    elif data == "clear_no":
        await query.message.edit_text("Отменено.", parse_mode=None)

    elif data == "menu_voice":
        current = db.get_voice_responses(user.id)
        new_state = not current
        db.set_voice_responses(user.id, new_state)
        icon = "🔊" if new_state else "🔇"
        status = "включены" if new_state else "выключены"
        await query.message.reply_text(
            f"{icon} Голосовые ответы {status}.", parse_mode=None,
        )

    elif data == "menu_system":
        current = db.get_system_prompt(user.id) or DEFAULT_SYSTEM_PROMPT
        label = "(пользовательский)" if db.get_system_prompt(user.id) else "(по умолчанию)"
        display = current[:500] + "..." if len(current) > 500 else current
        await query.message.reply_text(
            f"⚙️ Системный промпт {label}:\n\n{display}\n\n"
            "Чтобы сменить: /system <текст>\n"
            "Чтобы сбросить: /system reset",
            parse_mode=None,
        )

    elif data == "menu_export":
        history = db.export_history(user.id)
        if not history:
            await query.message.reply_text("История чата пуста.", parse_mode=None)
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
                await query.message.reply_document(
                    document=f,
                    filename=f"analytik_chat_{user.id}.txt",
                    caption=f"📤 История чата ({len(history)} сообщений)",
                )
        finally:
            export_path.unlink(missing_ok=True)

    elif data == "menu_help":
        await query.message.reply_text(
            "📖 Analytik Bot — справка\n\n"
            "Как пользоваться:\n"
            "— Просто напишите текст или отправьте голосовое\n"
            "— Бот помнит контекст беседы\n"
            "— Спрашивайте про SQL, Python, данные, статистику\n\n"
            "Команды:\n"
            "/start — приветствие и меню\n"
            "/clear — очистить контекст\n"
            "/system — показать / сменить промпт\n"
            "/export — экспорт истории чата\n"
            "/voice — вкл/выкл голосовые ответы\n"
            "/help — эта справка",
            parse_mode=None,
        )

    elif data == "retry":
        question = context.user_data.get("last_question", "")
        if question:
            await _process_chat(
                update, context, question=question, source="retry",
                target_message=query.message,
            )
        else:
            await query.message.reply_text("Нет вопроса для повтора.", parse_mode=None)


# ---- message handlers --------------------------------------------------

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not _is_allowed(context, user.id):
        await _deny(update)
        return

    rate_limiter: RateLimiter = context.bot_data["rate_limiter"]
    if not rate_limiter.is_allowed(user.id):
        await update.effective_message.reply_text(
            "⏳ Слишком много запросов. Подождите минуту.", parse_mode=None,
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
            "⏳ Слишком много запросов. Подождите минуту.", parse_mode=None,
        )
        return

    cfg: Config = context.bot_data["cfg"]
    stt: GroqSTT = context.bot_data["stt"]
    msg = update.effective_message
    voice = msg.voice or msg.audio
    if voice is None:
        return

    if voice.file_size and voice.file_size > cfg.max_audio_mb * 1024 * 1024:
        await msg.reply_text(
            f"❌ Аудио слишком большое (лимит {cfg.max_audio_mb} МБ).", parse_mode=None,
        )
        return

    placeholder = await msg.reply_text("🎙 Распознаю голос...", parse_mode=None)
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    ogg_path = None
    wav_path = None
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
        await placeholder.edit_text(f"❌ Не удалось распознать речь: {e}")
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("STT failed")
        await placeholder.edit_text(f"❌ Ошибка распознавания ({type(e).__name__}).")
        return
    finally:
        for p in (ogg_path, wav_path):
            if p is not None:
                try:
                    p.unlink(missing_ok=True)
                except Exception:  # noqa: BLE001
                    pass

    display_transcript = transcript[:200] + "..." if len(transcript) > 200 else transcript
    await placeholder.edit_text(f"🎙 Распознано: {display_transcript}")
    await _process_chat(
        update, context, question=transcript, source="voice", target_message=placeholder,
    )


async def _process_chat(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    question: str,
    source: str,
    target_message=None,
) -> None:
    """Основной пайплайн: получает вопрос, добавляет контекст, отправляет LLM, сохраняет ответ."""
    cfg: Config = context.bot_data["cfg"]
    db: Database = context.bot_data["db"]
    llm: GroqLLM = context.bot_data["llm"]
    user = update.effective_user
    msg = update.effective_message

    db.upsert_user(user.id, user.username, user.first_name, user.last_name)

    chat_context = db.get_context(user.id, limit=cfg.max_context_messages)
    system_prompt = db.get_system_prompt(user.id) or DEFAULT_SYSTEM_PROMPT

    if target_message is None:
        target_message = await msg.reply_text("⏳ Думаю...", parse_mode=None)
    else:
        try:
            await target_message.edit_text("⏳ Думаю...")
        except Exception:  # noqa: BLE001
            target_message = await msg.reply_text("⏳ Думаю...", parse_mode=None)
    await context.bot.send_chat_action(chat_id=msg.chat_id, action=ChatAction.TYPING)

    context.user_data["last_question"] = question

    try:
        answer = await llm.chat(question, system_prompt, context=chat_context)
    except LLMError as e:
        await target_message.edit_text(
            f"❌ Не смог ответить: {e}",
            parse_mode=None,
            reply_markup=_retry_keyboard(),
        )
        return
    except Exception as e:  # noqa: BLE001
        logger.exception("LLM chat failed")
        await target_message.edit_text(
            f"❌ Ошибка ИИ ({type(e).__name__}). Попробуйте ещё раз.",
            parse_mode=None,
            reply_markup=_retry_keyboard(),
        )
        return

    db.add_message(user.id, "user", question)
    db.add_message(user.id, "assistant", answer)

    chunks = _split_for_telegram(answer)
    for i, chunk in enumerate(chunks):
        if i == 0:
            await _safe_reply(target_message, chunk)
        else:
            try:
                escaped = _escape_md(chunk)
                await msg.reply_text(escaped, parse_mode=ParseMode.MARKDOWN_V2)
            except Exception:  # noqa: BLE001
                await msg.reply_text(chunk, parse_mode=None)

    voice_enabled = db.get_voice_responses(user.id)
    if voice_enabled:
        await _tts_reply(msg, answer)

    logger.info(
        "chat user=%d source=%s q_len=%d a_len=%d ctx=%d",
        user.id, source, len(question), len(answer), len(chat_context),
    )


# ---- bootstrap ---------------------------------------------------------

BOT_COMMANDS: list[tuple[str, str]] = [
    ("start", "Приветствие и меню"),
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

    app.add_handler(CallbackQueryHandler(on_callback))

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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _handle_signal(sig: int, _frame) -> None:
        logger.info("Получен сигнал %s, завершаю...", signal.Signals(sig).name)
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
