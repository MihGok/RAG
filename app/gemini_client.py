"""
gemini_client.py
────────────────
Обёртка над Google Generative AI SDK для работы с Gemini 2.5 Pro.

Функции:
    run_gemini()      — запрос с опциональной JSON-схемой ответа
    run_gemini_async() — асинхронная версия для FastAPI

Поддерживается:
  • Системный промпт
  • Структурированный JSON-ответ (через response_mime_type + response_schema)
  • Запись метрик инференса в metrics_store
  • Передача истории диалога (multi-turn)
"""

import os
import time
import json
import logging
from typing import Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

import metrics_store

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────
#  Конфигурация
# ────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str    = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL:   str    = "gemini-2.5-pro-preview-06-05"   # актуальная версия 2.5 Pro

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning(
        "GEMINI_API_KEY не задан. Запросы к Gemini будут завершаться ошибкой."
    )

# ────────────────────────────────────────────────────────────────────────────
#  Синглтон модели
# ────────────────────────────────────────────────────────────────────────────
_gemini_model: genai.GenerativeModel | None = None


def _get_model(system_prompt: str | None = None) -> genai.GenerativeModel:
    """
    Создать или вернуть инстанс GenerativeModel.
    Системный промпт встраивается на уровне модели.
    """
    global _gemini_model
    # Пересоздаём если меняется system_prompt (редко, но возможно)
    _gemini_model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=system_prompt,
    )
    return _gemini_model


# ────────────────────────────────────────────────────────────────────────────
#  Вспомогательные
# ────────────────────────────────────────────────────────────────────────────

def _build_generation_config(
    response_schema: dict | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> GenerationConfig:
    cfg: dict = {
        "max_output_tokens": max_tokens,
        "temperature":       temperature,
        "top_p":             top_p,
    }
    if response_schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"]    = response_schema
    return GenerationConfig(**cfg)


def _parse_gemini_response(text: str, response_schema: dict | None) -> Any:
    if response_schema is None:
        return {"response": text.strip()}
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        logger.warning("Gemini: не удалось распарсить JSON ответ.")
        return {"response": text.strip(), "_parse_error": True}


# ────────────────────────────────────────────────────────────────────────────
#  Публичный API
# ────────────────────────────────────────────────────────────────────────────

def run_gemini(
    prompt: str,
    system_prompt: str | None = None,
    response_schema: dict | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.95,
    chat_history: list[dict] | None = None,
) -> Any:
    """
    Синхронный запрос к Gemini 2.5 Pro.

    Args:
        prompt:          Текст запроса.
        system_prompt:   Системная инструкция (задаётся на уровне модели).
        response_schema: JSON Schema для принудительного структурированного вывода.
                         Пример: get_schema("summary") из schemas.py.
        max_tokens:      Максимум токенов в ответе.
        temperature:     Температура (0 = детерминировано, 1 = креативно).
        top_p:           Nucleus sampling.
        chat_history:    История диалога для multi-turn:
                         [{"role": "user"/"model", "parts": ["текст"]}, ...]

    Returns:
        dict с полями по схеме, или {"response": str} без схемы.

    Raises:
        RuntimeError: если GEMINI_API_KEY не задан.
        google.api_core.exceptions.*: при ошибках API.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY не задан. Укажите ключ в .env файле."
        )

    model = _get_model(system_prompt)
    config = _build_generation_config(response_schema, max_tokens, temperature, top_p)

    t0 = time.perf_counter()

    if chat_history:
        # Multi-turn: создаём сессию с историей
        chat = model.start_chat(history=chat_history)
        response = chat.send_message(prompt, generation_config=config)
    else:
        response = model.generate_content(prompt, generation_config=config)

    elapsed = time.perf_counter() - t0

    # Токены (Gemini возвращает usage_metadata)
    usage      = response.usage_metadata
    prompt_t   = getattr(usage, "prompt_token_count",      0) or 0
    compl_t    = getattr(usage, "candidates_token_count",  0) or 0
    total_t    = getattr(usage, "total_token_count",       0) or 0
    tps        = compl_t / elapsed if elapsed > 0 else 0.0

    metrics_store.record_inference(
        model_name=GEMINI_MODEL,
        prompt_tokens=prompt_t,
        completion_tokens=compl_t,
        total_tokens=total_t,
        latency_sec=elapsed,
        tokens_per_sec=tps,
    )
    logger.info(
        "Gemini | %.2fs | prompt=%d compl=%d | %.1f tok/s",
        elapsed, prompt_t, compl_t, tps,
    )

    raw = response.text
    return _parse_gemini_response(raw, response_schema)


async def run_gemini_async(
    prompt: str,
    system_prompt: str | None = None,
    response_schema: dict | None = None,
    max_tokens: int = 8192,
    temperature: float = 0.7,
    top_p: float = 0.95,
    chat_history: list[dict] | None = None,
) -> Any:
    """
    Асинхронная обёртка над run_gemini для использования в FastAPI.
    Запускает синхронный вызов в ThreadPoolExecutor.
    """
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_gemini(
            prompt=prompt,
            system_prompt=system_prompt,
            response_schema=response_schema,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            chat_history=chat_history,
        ),
    )
