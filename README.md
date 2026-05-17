# Analytik Bot

Telegram чат-бот аналитик данных на базе Groq LLM (бесплатная модель).

## Возможности

- Чат с ИИ-аналитиком (SQL, Python, данные, статистика)
- Голосовой ввод (Groq Whisper — бесплатно)
- Голосовые ответы (gTTS — бесплатно)
- Запоминание контекста беседы (SQLite)
- Настраиваемый системный промпт прямо в чате
- Экспорт истории чата в файл

## Команды

| Команда | Описание |
|---------|----------|
| `/start` | Приветствие |
| `/help` | Справка |
| `/clear` | Очистить контекст беседы |
| `/system` | Показать текущий системный промпт |
| `/system <текст>` | Установить новый системный промпт |
| `/system reset` | Вернуть промпт по умолчанию |
| `/export` | Экспорт истории чата в .txt файл |
| `/voice` | Вкл/выкл голосовые ответы |

## Установка

```bash
# Клонировать репозиторий
git clone https://github.com/Mirzoevmurad/analytik.git
cd analytik

# Создать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# Установить зависимости
pip install -r requirements.txt

# Скопировать и заполнить конфиг
cp .env.example .env
# Отредактировать .env — указать токен бота и ключ Groq

# Запустить
python bot.py
```

## Переменные окружения

| Переменная | Описание | По умолчанию |
|-----------|----------|-------------|
| `ANALYTIK_BOT_TOKEN` | Токен Telegram бота | — |
| `GROQ_API_KEY` | API ключ Groq | — |
| `ALLOWED_USER_IDS` | ID пользователей через запятую | — |
| `LLM_MODEL` | Модель LLM | `llama-3.3-70b-versatile` |
| `STT_MODEL` | Модель STT | `whisper-large-v3-turbo` |
| `MAX_CONTEXT_MESSAGES` | Макс. сообщений в контексте | `20` |
| `VOICE_RESPONSES` | Голосовые ответы по умолчанию | `false` |

## Требования

- Python 3.10+
- ffmpeg (для конвертации аудио)
- Бесплатный аккаунт [Groq](https://console.groq.com)
- Telegram Bot Token (через [@BotFather](https://t.me/BotFather))
