"""Groq LLM интеграция с поддержкой контекста."""
from __future__ import annotations

import logging

from groq import AsyncGroq


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """Ты — опытный аналитик данных и ИИ-ассистент. Ты помогаешь пользователю с анализом данных, SQL-запросами, Python-кодом (pandas, numpy, matplotlib), статистикой, визуализацией, построением отчётов и любыми вопросами, связанными с данными.

Твои сильные стороны:
— Написание и оптимизация SQL-запросов (PostgreSQL, MySQL, SQLite, BigQuery)
— Python-код для анализа данных (pandas, numpy, scipy, sklearn)
— Визуализация (matplotlib, seaborn, plotly)
— Статистический анализ и интерпретация результатов
— Формулы Excel / Google Sheets
— Объяснение сложных концепций простым языком
— Построение ETL-пайплайнов и архитектуры данных

Правила:
— Отвечай на ТОМ ЖЕ языке, на котором задан вопрос.
— Не добавляй преамбул типа «Конечно!», «Хороший вопрос!». Сразу по делу.
— Длина ответа пропорциональна сложности вопроса.
— Код оформляй аккуратно с комментариями.
— Если не знаешь точного ответа — честно скажи и предложи направление.
— Учитывай контекст предыдущих сообщений в беседе.
— Будь дружелюбным, но профессиональным."""


class LLMError(Exception):
    pass


class GroqLLM:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def chat(
        self,
        question: str,
        system_prompt: str,
        context: list[dict[str, str]] | None = None,
    ) -> str:
        """Отправляет вопрос с контекстом и возвращает ответ."""
        if not question.strip():
            raise LLMError("Пустой вопрос")

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": question})

        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.5,
            max_tokens=4000,
        )
        out = (completion.choices[0].message.content or "").strip()
        if not out:
            raise LLMError("LLM вернул пустой ответ")
        return out
